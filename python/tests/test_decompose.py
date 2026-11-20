"""Opt-in decomposition of vLLM custom operators into ATen operators."""

import pytest
import torch
from fx_test_utils import RecordExamples
from torch._subclasses.fake_tensor import FakeTensorMode

pytest.importorskip("vllm._custom_ops")  # registers torch.ops._C

from kashi.decompose import DECOMPOSED_FROM_KEY, decompose_custom_ops

aten = torch.ops.aten
EPS = 1e-6


def _fake(graph, *inputs):
    """Wrap ``graph`` in a module and record FakeTensor ``val`` metadata."""
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode() as mode:
        RecordExamples(gm, "val").run(*[mode.from_tensor(x) for x in inputs])
    return gm


def _decompose(gm):
    decompose_custom_ops(gm.graph)
    gm.recompile()
    return gm


def _is_c_op(node):
    return node.op == "call_function" and str(node.target).startswith("_C.")


def _empty(graph, shape, dtype=torch.float32):
    return graph.call_function(
        aten.empty.memory_format,
        (shape,),
        {"dtype": dtype, "device": torch.device("cpu")},
    )


def _ref_rms(x, w, eps):
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    out = out.to(x.dtype)
    return out if w is None else out * w


def _tol(dtype):
    return {"rtol": 2e-2, "atol": 2e-2} if dtype == torch.bfloat16 else {}


# --- _C.rms_norm and the shared rewriting machinery. ------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_weight", [True, False])
def test_rms_norm_result_reaches_later_reads(dtype, with_weight):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    out = _empty(graph, [3, 8], dtype)
    graph.call_function(
        torch.ops._C.rms_norm.default, (out, x, w if with_weight else None, EPS)
    )
    graph.output(graph.call_function(aten.neg.default, (out,)))
    x_val, w_val = torch.randn(3, 8, dtype=dtype), torch.randn(8, dtype=dtype)
    gm = _decompose(_fake(graph, x_val, w_val))

    assert not [n for n in gm.graph.nodes if _is_c_op(n)]
    result = gm(x_val, w_val)
    assert result.dtype == dtype
    expected = -_ref_rms(x_val, w_val if with_weight else None, EPS)
    torch.testing.assert_close(result, expected, **_tol(dtype))


def test_each_call_gets_its_own_tag():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    first, second = _empty(graph, [3, 8]), _empty(graph, [3, 8])
    graph.call_function(torch.ops._C.rms_norm.default, (first, x, w, EPS))
    graph.call_function(torch.ops._C.rms_norm.default, (second, first, w, EPS))
    graph.output((second,))
    gm = _fake(graph, torch.randn(3, 8), torch.randn(8))
    untouched = [n for n in gm.graph.nodes if not _is_c_op(n)]
    _decompose(gm)

    tags = {n.meta.get(DECOMPOSED_FROM_KEY) for n in gm.graph.nodes} - {None}
    assert tags == {"_C.rms_norm#0", "_C.rms_norm#1"}
    assert all(DECOMPOSED_FROM_KEY not in n.meta for n in untouched)


def test_mutating_a_graph_input_raises():
    graph = torch.fx.Graph()
    out = graph.placeholder("out")
    x = graph.placeholder("x")
    graph.call_function(torch.ops._C.rms_norm.default, (out, x, None, EPS))
    graph.output((out,))
    gm = _fake(graph, torch.empty(3, 8), torch.randn(3, 8))

    with pytest.raises(NotImplementedError, match="graph input"):
        decompose_custom_ops(gm.graph)


def test_reading_base_of_mutated_view_after_call_raises():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    out = _empty(graph, [3, 8])
    view = graph.call_function(aten.view.default, (out, [3, 8]))
    graph.call_function(torch.ops._C.rms_norm.default, (view, x, None, EPS))
    graph.output((graph.call_function(aten.neg.default, (out,)),))
    gm = _fake(graph, torch.randn(3, 8))

    with pytest.raises(NotImplementedError, match="alias"):
        decompose_custom_ops(gm.graph)


def test_graph_without_custom_ops_is_unchanged():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output((graph.call_function(aten.neg.default, (x,)),))
    gm = _fake(graph, torch.randn(3))
    before = str(gm.graph)

    decompose_custom_ops(gm.graph)

    assert str(gm.graph) == before


# --- _C.fused_add_rms_norm and _C.silu_and_mul. -----------------------------


def test_fused_add_rms_norm_updates_input_and_residual():
    graph = torch.fx.Graph()
    x, res, w = (graph.placeholder(n) for n in ("x", "res", "w"))
    i = graph.call_function(aten.clone.default, (x,))
    r = graph.call_function(aten.clone.default, (res,))
    pre = graph.call_function(aten.mul.Tensor, (i, 2.0))
    graph.call_function(torch.ops._C.fused_add_rms_norm.default, (i, r, w, EPS))
    graph.output((pre, i, r))
    inputs = torch.randn(3, 8), torch.randn(3, 8), torch.randn(8)
    x_val, res_val, w_val = inputs
    gm = _decompose(_fake(graph, *inputs))

    pre_out, i_out, r_out = gm(*inputs)
    torch.testing.assert_close(pre_out, 2 * x_val)
    torch.testing.assert_close(r_out, x_val + res_val)
    torch.testing.assert_close(i_out, _ref_rms(x_val + res_val, w_val, EPS))


def test_chained_fused_add_rms_norm_threads_the_residual():
    graph = torch.fx.Graph()
    x, y, res, w = (graph.placeholder(n) for n in ("x", "y", "res", "w"))
    i1 = graph.call_function(aten.clone.default, (x,))
    r = graph.call_function(aten.clone.default, (res,))
    graph.call_function(torch.ops._C.fused_add_rms_norm.default, (i1, r, w, EPS))
    i2 = graph.call_function(aten.add.Tensor, (y, i1))
    graph.call_function(torch.ops._C.fused_add_rms_norm.default, (i2, r, w, EPS))
    graph.output((i2, r))
    inputs = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8), torch.randn(8)
    x_val, y_val, res_val, w_val = inputs
    gm = _decompose(_fake(graph, *inputs))

    r1 = x_val + res_val
    r2 = (y_val + _ref_rms(r1, w_val, EPS)) + r1
    i2_out, r_out = gm(*inputs)
    torch.testing.assert_close(r_out, r2)
    torch.testing.assert_close(i2_out, _ref_rms(r2, w_val, EPS))


def test_silu_and_mul_writes_result():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    out = _empty(graph, [3, 4])
    graph.call_function(torch.ops._C.silu_and_mul.default, (out, x))
    graph.output((out,))
    x_val = torch.randn(3, 8)
    gm = _decompose(_fake(graph, x_val))

    (result,) = gm(x_val)
    expected = torch.nn.functional.silu(x_val[:, :4]) * x_val[:, 4:]
    torch.testing.assert_close(result, expected)


# --- _C.rotary_embedding (NeoX style only). ---------------------------------

T, HEAD = 5, 16


def _ref_rope(positions, x, head_size, cache):
    cos, sin = (part.unsqueeze(-2) for part in cache[positions].chunk(2, -1))
    x1, x2 = x.view(x.shape[0], -1, head_size).chunk(2, -1)
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)
    return rotated.view(x.shape)


def _rope_inputs(cache_width=HEAD):
    return (
        torch.tensor([0, 3, 1, 9, 4]),
        torch.randn(T, 2 * HEAD),
        torch.randn(T, HEAD),
        torch.randn(10, cache_width),
    )


def _rope_graph(*, key=True, is_neox=True, extra=()):
    graph = torch.fx.Graph()
    pos, qin, kin, cache = (graph.placeholder(n) for n in ("pos", "q", "k", "cache"))
    q = graph.call_function(aten.clone.default, (qin,))
    k = graph.call_function(aten.clone.default, (kin,)) if key else None
    graph.call_function(
        torch.ops._C.rotary_embedding.default,
        (pos, q, k, HEAD, cache, is_neox, *extra),
    )
    graph.output((q, k) if key else (q,))
    return graph


def test_rotary_embedding_neox_matches_reference():
    inputs = _rope_inputs()
    pos, qin, kin, cache = inputs
    gm = _decompose(_fake(_rope_graph(), *inputs))

    q, k = gm(*inputs)
    torch.testing.assert_close(q, _ref_rope(pos, qin, HEAD, cache))
    torch.testing.assert_close(k, _ref_rope(pos, kin, HEAD, cache))


def test_rotary_views_keep_token_dim_symbolic():
    gm = _decompose(_fake(_rope_graph(), *_rope_inputs()))

    views = [
        n
        for n in gm.graph.nodes
        if n.target is aten.view.default and DECOMPOSED_FROM_KEY in n.meta
    ]
    assert views and all(n.args[1][0] == -1 for n in views)


def test_qwen_rms_norm_then_rotary_chain():
    graph = torch.fx.Graph()
    pos, qin, kin, cache, w = (
        graph.placeholder(n) for n in ("pos", "q", "k", "cache", "w")
    )
    qv = graph.call_function(aten.view.default, (qin, [T, 2, HEAD]))
    out = _empty(graph, [T, 2, HEAD])
    graph.call_function(torch.ops._C.rms_norm.default, (out, qv, w, EPS))
    q1 = graph.call_function(aten.view.default, (out, [T, 2 * HEAD]))
    k = graph.call_function(aten.clone.default, (kin,))
    graph.call_function(
        torch.ops._C.rotary_embedding.default, (pos, q1, k, HEAD, cache, True)
    )
    graph.output((graph.call_function(aten.view.default, (q1, [-1, 2, HEAD])),))
    inputs = (*_rope_inputs(), torch.randn(HEAD))
    pos_val, q_val, _, cache_val, w_val = inputs
    gm = _decompose(_fake(graph, *inputs))

    (query,) = gm(*inputs)
    normed = _ref_rms(q_val.view(T, 2, HEAD), w_val, EPS).view(T, 2 * HEAD)
    expected = _ref_rope(pos_val, normed, HEAD, cache_val).view(-1, 2, HEAD)
    torch.testing.assert_close(query, expected)


@pytest.mark.parametrize(
    "variant",
    ["not_neox", "rope_dim_offset", "inverse", "no_key", "partial_rotary"],
)
def test_unsupported_rotary_variants_raise(variant):
    graph = _rope_graph(
        key=variant != "no_key",
        is_neox=variant != "not_neox",
        extra={"rope_dim_offset": (1, False), "inverse": (0, True)}.get(variant, ()),
    )
    width = HEAD // 2 if variant == "partial_rotary" else HEAD
    gm = _fake(graph, *_rope_inputs(width))

    with pytest.raises(NotImplementedError, match="rotary_embedding"):
        decompose_custom_ops(gm.graph)
