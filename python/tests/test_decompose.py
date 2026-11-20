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
