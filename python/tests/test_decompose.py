"""Opt-in decomposition of vLLM custom operators into ATen operators.

Graphs are plain Python functions traced with ``make_fx`` on fake tensors.
"""

import sys

import pytest
import torch
import vllm._custom_ops  # noqa # registers torch.ops._C
from kashi.decompose import DECOMPOSED_FROM_KEY, decompose_custom_ops
from kashi.importer import import_fx
from kashi.iree_partition import partition_at_custom_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

C = torch.ops._C
EPS = 1e-6
T, HEAD = 5, 16


def _trace(program, *inputs):
    """Trace ``program`` into an ATen graph with FakeTensor ``val`` metadata."""
    # The varargs wrapper stops make_fx from counting keyword-only parameters.
    return make_fx(lambda *args: program(*args), tracing_mode="fake")(*inputs)


def _custom_ops(gm):
    return [
        n
        for n in gm.graph.nodes
        if isinstance(n.target, torch._ops.OpOverload) and n.target.namespace != "aten"
    ]


def _decompose(program, *inputs):
    """Trace and decompose ``program``; a leftover ``_C`` op would run on CPU."""
    gm = _trace(program, *inputs)
    decompose_custom_ops(gm.graph)
    gm.recompile()
    assert not _custom_ops(gm)
    return gm


# --- Programs shared by the CPU, wiring and CUDA parity tests. ---------------


def _rms_norm(x, w):
    out = torch.empty_like(x)
    C.rms_norm(out, x, w, EPS)
    return out


def _fused_add_rms_norm(x, res, w):
    x, res = x.clone(), res.clone()
    C.fused_add_rms_norm(x, res, w, EPS)
    return x, res


def _silu_and_mul(x):
    out = x.new_empty(x.shape[0], x.shape[1] // 2)
    C.silu_and_mul(out, x)
    return out


def _rope(
    pos, q, k, cache, *, key=True, is_neox=True, rope_dim_offset=0, inverse=False
):
    q, k = q.clone(), (k.clone() if key else None)
    C.rotary_embedding(pos, q, k, HEAD, cache, is_neox, rope_dim_offset, inverse)
    return (q, k) if key else q


def _two_norms_then_silu(x, w):
    return _silu_and_mul(_rms_norm(_rms_norm(x, w), w))


def _rope_inputs(cache_width=HEAD):
    positions = torch.tensor([0, 3, 1, 9, 4])
    return (
        positions,
        torch.randn(T, 2 * HEAD),
        torch.randn(T, HEAD),
        torch.randn(10, cache_width),
    )


# --- Independent references (vLLM's CUDA kernels cannot run on CPU). ---------


def _ref_rms(x, w):
    x32 = x.float()
    out = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype)
    return out if w is None else out * w


def _ref_silu_and_mul(x):
    half = x.shape[-1] // 2
    return torch.nn.functional.silu(x[:, :half]) * x[:, half:]


def _ref_rope(positions, x, cache):
    cos, sin = (part.unsqueeze(-2) for part in cache[positions].chunk(2, -1))
    x1, x2 = x.view(x.shape[0], -1, HEAD).chunk(2, -1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).view(x.shape)


# --- Numerics of each rule. ---------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_weight", [True, False])
def test_rms_norm(dtype, with_weight):
    """Later reads see the normalized values in the input dtype, weight optional."""
    x, w = torch.randn(3, 8, dtype=dtype), torch.randn(8, dtype=dtype)
    gm = _decompose(lambda x, w: _rms_norm(x, w if with_weight else None), x, w)

    result = gm(x, w)
    assert result.dtype == dtype
    tol = {"rtol": 2e-2, "atol": 2e-2} if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(result, _ref_rms(x, w if with_weight else None), **tol)


def test_fused_add_rms_norm_updates_both_buffers_in_order():
    """Both buffers are updated for later reads; earlier reads keep old values."""

    def program(x, y, res, w):
        i, r = x.clone(), res.clone()
        before = i * 2
        C.fused_add_rms_norm(i, r, w, EPS)
        i2 = y + i
        C.fused_add_rms_norm(i2, r, w, EPS)
        return before, i2, r

    inputs = x, y, res, w = (
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.randn(8),
    )
    before, i2, r = _decompose(program, *inputs)(*inputs)

    r1 = x + res
    r2 = y + _ref_rms(r1, w) + r1
    torch.testing.assert_close(before, 2 * x)
    torch.testing.assert_close(r, r2)
    torch.testing.assert_close(i2, _ref_rms(r2, w))


def test_silu_and_mul():
    """Output is silu(first half) * second half."""
    x = torch.randn(3, 8)
    torch.testing.assert_close(_decompose(_silu_and_mul, x)(x), _ref_silu_and_mul(x))


def test_rotary_embedding_neox():
    """Query and key are both rotated by the cached cos/sin at ``positions``."""
    inputs = pos, q, k, cache = _rope_inputs()
    q_out, k_out = _decompose(_rope, *inputs)(*inputs)

    torch.testing.assert_close(q_out, _ref_rope(pos, q, cache))
    torch.testing.assert_close(k_out, _ref_rope(pos, k, cache))


def test_rotary_views_keep_token_dim_symbolic():
    """Reshapes use -1 for tokens so the traced token count isn't baked in."""
    gm = _decompose(_rope, *_rope_inputs())

    views = [n for n in gm.graph.nodes if n.target is torch.ops.aten.view.default]
    assert views and all(n.args[1][0] == -1 for n in views)


def test_rms_norm_then_rotary_through_views():
    """Qwen3 q-norm: mutations through views reach reads through later views."""

    def program(pos, q, k, cache, w):
        per_head = q.view(-1, 2, HEAD)
        normed = torch.empty_like(per_head)
        C.rms_norm(normed, per_head, w, EPS)
        flat = normed.view(-1, 2 * HEAD)
        C.rotary_embedding(pos, flat, k.clone(), HEAD, cache, True)
        return flat.view(-1, 2, HEAD)

    inputs = pos, q, _, cache, w = (*_rope_inputs(), torch.randn(HEAD))
    result = _decompose(program, *inputs)(*inputs)

    normed = _ref_rms(q.view(T, 2, HEAD), w).view(T, 2 * HEAD)
    torch.testing.assert_close(result, _ref_rope(pos, normed, cache).view(T, 2, HEAD))


@pytest.mark.parametrize(
    "variant, cache_width",
    [
        ({"is_neox": False}, HEAD),
        ({"rope_dim_offset": 1}, HEAD),
        ({"inverse": True}, HEAD),
        ({"key": False}, HEAD),
        ({}, HEAD // 2),  # partial rotary
    ],
    ids=["not_neox", "rope_dim_offset", "inverse", "no_key", "partial_rotary"],
)
def test_unsupported_rotary_variants_raise(variant, cache_width):
    """Variants other than full-head NeoX with a key raise."""
    gm = _trace(lambda *args: _rope(*args, **variant), *_rope_inputs(cache_width))

    with pytest.raises(NotImplementedError, match="rotary_embedding"):
        decompose_custom_ops(gm.graph)


# --- Writes the rewrite cannot express. --------------------------------------


def _writes_graph_input(out, x):
    C.rms_norm(out, x, None, EPS)
    return out


def _writes_slice_of_graph_input(buf, x):
    C.rms_norm(buf[:3], x, None, EPS)
    return buf


def _reads_view_base_after_write(x):
    base = torch.empty_like(x).unsqueeze(0)
    C.rms_norm(base.squeeze(0), x, None, EPS)
    return -base


def _reads_split_base_after_write(x, w):
    qkv = x.clone()
    q, _ = qkv.split([8, 8], -1)
    C.rms_norm(q, q, w, EPS)
    return -qkv


@pytest.mark.parametrize(
    "program, inputs, match",
    [
        (_writes_graph_input, (torch.empty(3, 8), torch.randn(3, 8)), "graph input"),
        (
            _writes_slice_of_graph_input,
            (torch.empty(5, 8), torch.randn(3, 8)),
            "graph input",
        ),
        (_reads_view_base_after_write, (torch.randn(3, 8),), "alias"),
        (_reads_split_base_after_write, (torch.randn(3, 16), torch.randn(8)), "alias"),
    ],
    ids=["graph_input", "slice_of_graph_input", "view_base", "split_base"],
)
def test_unrepresentable_writes_raise(program, inputs, match):
    """Writes visible to the caller or through another alias raise."""
    gm = _trace(program, *inputs)

    with pytest.raises(NotImplementedError, match=match):
        decompose_custom_ops(gm.graph)


# --- Tagging and graph hygiene. -----------------------------------------------


def test_each_call_gets_its_own_tag():
    """Each call's nodes get a unique ``<op>#<n>`` tag; existing nodes don't."""
    gm = _trace(_two_norms_then_silu, torch.randn(3, 8), torch.randn(8))
    untouched = [n for n in gm.graph.nodes if n not in _custom_ops(gm)]
    decompose_custom_ops(gm.graph)

    tags = {n.meta.get(DECOMPOSED_FROM_KEY) for n in gm.graph.nodes} - {None}
    assert tags == {"_C.rms_norm#0", "_C.rms_norm#1", "_C.silu_and_mul#0"}
    assert all(DECOMPOSED_FROM_KEY not in n.meta for n in untouched)


def test_uses_the_graph_fake_mode_inside_a_dynamo_backend():
    """Uses the graph's fake mode, not the Dynamo TracingContext's."""
    from torch._guards import TracingContext, tracing

    gm = _trace(_rms_norm, torch.randn(3, 8), torch.randn(8))

    with tracing(TracingContext(FakeTensorMode())):
        decompose_custom_ops(gm.graph)

    assert not _custom_ops(gm)


def test_graph_without_custom_ops_is_unchanged():
    """Graphs without custom ops are left untouched."""
    gm = _trace(torch.neg, torch.randn(3))
    before = str(gm.graph)

    decompose_custom_ops(gm.graph)

    assert str(gm.graph) == before


# --- Wiring: import_fx, MLIR locations, partitioning and compile_iree_fx. ----


def _import(decompose):
    gm = _trace(_two_norms_then_silu, torch.randn(3, 8), torch.randn(8))
    return import_fx(gm, decompose_custom_ops=decompose).module


def test_import_turns_tags_into_name_locations():
    """import_fx decomposes all custom ops and turns tags into NameLocs."""
    module = _import(decompose=True)

    assert module.operation.verify()
    asm = module.operation.get_asm(enable_debug_info=True)
    assert "torch.operator" not in asm
    for tag in ("_C.rms_norm#0", "_C.rms_norm#1", "_C.silu_and_mul#0"):
        assert f'"kashi.decomposed:{tag}"' in asm


def test_import_without_flag_keeps_custom_ops():
    """Decomposition is opt-in."""
    asm = _import(decompose=False).operation.get_asm(enable_debug_info=True)

    assert 'torch.operator "torch._C.rms_norm"' in asm
    assert "kashi.decomposed" not in asm


def test_decomposed_graph_is_a_single_iree_segment():
    """With no custom ops left, IREE gets one segment."""
    assert len(partition_at_custom_ops(_import(decompose=True)).segments) == 1


def test_compile_iree_fx_forwards_the_flag(monkeypatch, tmp_path):
    """compile_iree_fx passes ``decompose_custom_ops`` to import_fx."""
    import kashi.iree_backend

    seen = {}

    def record(gm, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after import")

    monkeypatch.setattr(kashi.iree_backend, "import_fx", record)
    gm = _trace(_rms_norm, torch.randn(3, 8), torch.randn(8))
    with pytest.raises(RuntimeError, match="stop after import"):
        kashi.iree_backend.compile_iree_fx(
            gm, output_dir=tmp_path, decompose_custom_ops=True
        )
    assert seen == {"decompose_custom_ops": True}


@pytest.mark.skipif(sys.platform == "darwin", reason="IREE CUDA compilation")
def test_partitioned_artifact_keeps_tags(tmp_path):
    """Tags survive the IREE pipeline into ``partitioned.mlir``."""
    from kashi.iree_backend import compile_iree_fx

    gm = _trace(_two_norms_then_silu, torch.randn(3, 8), torch.randn(8))
    result = compile_iree_fx(
        gm, output_dir=tmp_path, cuda_target="sm_80", decompose_custom_ops=True
    )

    partitioned = (result.artifact_dir / "partitioned.mlir").read_text()
    assert "kashi.decomposed:_C.rms_norm#0" in partitioned


# --- Parity with vLLM's real kernels (CUDA only). -----------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM _C kernels need CUDA")
@pytest.mark.parametrize(
    "program, inputs",
    [
        (_rms_norm, lambda: (torch.randn(3, 8), torch.randn(8))),
        (
            _fused_add_rms_norm,
            lambda: (torch.randn(3, 8), torch.randn(3, 8), torch.randn(8)),
        ),
        (_silu_and_mul, lambda: (torch.randn(3, 8),)),
        (_rope, _rope_inputs),
    ],
    ids=["rms_norm", "fused_add_rms_norm", "silu_and_mul", "rotary_embedding"],
)
def test_decomposition_matches_vllm_kernel_on_cuda(program, inputs):
    """Decomposition matches vLLM's real kernel."""
    inputs = [t.cuda() for t in inputs()]

    torch.testing.assert_close(
        _decompose(program, *inputs)(*inputs), program(*inputs), rtol=1e-4, atol=1e-4
    )


# --- Schema argument names differ between vLLM builds. ------------------------

# vLLM's CPU build names the output buffer "out" where the CUDA build says
# "result"; positions are the same. These copies use the CPU wheel's schemas.
_CPU_SCHEMAS = torch.library.Library("kashi_cpu_schema", "DEF")
for _schema in (
    "rms_norm(Tensor(a!) out, Tensor input, Tensor? weight, float epsilon) -> ()",
    "silu_and_mul(Tensor(a!) out, Tensor input) -> ()",
):
    _CPU_SCHEMAS.define(_schema)
    _CPU_SCHEMAS.impl(_schema.split("(")[0], lambda *args: None, "Meta")


@pytest.mark.parametrize("op", ["rms_norm", "silu_and_mul"])
def test_rules_do_not_depend_on_schema_argument_names(op, monkeypatch):
    """Rules bind by position, so CPU (``out``) and CUDA (``result``) builds work."""
    import kashi.decompose

    rules = dict(kashi.decompose._DECOMPOSITIONS)
    rules[f"kashi_cpu_schema::{op}"] = rules[f"_C::{op}"]
    monkeypatch.setattr(kashi.decompose, "_DECOMPOSITIONS", rules)
    monkeypatch.setitem(globals(), "C", torch.ops.kashi_cpu_schema)
    x = torch.randn(3, 8)

    if op == "rms_norm":
        gm = _decompose(lambda x: _rms_norm(x, None), x)
        torch.testing.assert_close(gm(x), _ref_rms(x, None))
    else:
        torch.testing.assert_close(
            _decompose(_silu_and_mul, x)(x), _ref_silu_and_mul(x)
        )
