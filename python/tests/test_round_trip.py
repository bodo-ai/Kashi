"""FX -> MLIR -> FX round trips: the rebuilt graph must compute what the
original computes, with the same output structure."""

import pytest
torch = pytest.importorskip("torch")

import operator

from torch.utils import _pytree as pytree

from kashi.importer import import_fx
from kashi.mlir_to_fx import rebuild_fx


class _RecordVals(torch.fx.Interpreter):
    def run_node(self, node):
        result = super().run_node(node)
        node.meta["val"] = result
        return result


def _graph_module(build, inputs):
    graph = torch.fx.Graph()
    graph.output(build(graph))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    _RecordVals(gm).run(*inputs)
    return gm, inputs


def _linear(target):
    def build(g):
        x, w, b = g.placeholder("x"), g.placeholder("w"), g.placeholder("b")
        return g.call_function(target, (x, w, b))

    return lambda: _graph_module(
        build, [torch.randn(2, 4), torch.randn(3, 4), torch.randn(3)]
    )


def _aten_embedding():
    def build(g):
        weight, indices = g.placeholder("weight"), g.placeholder("indices")
        return g.call_function(torch.ops.aten.embedding.default, (weight, indices))

    return _graph_module(build, [torch.randn(10, 4), torch.tensor([1, 3, 5])])


def _functional_embedding():
    def build(g):
        weight, indices = g.placeholder("weight"), g.placeholder("indices")
        return g.call_function(
            torch.nn.functional.embedding,
            (indices, weight),
            {
                "padding_idx": None,
                "max_norm": None,
                "norm_type": 2.0,
                "scale_grad_by_freq": False,
                "sparse": False,
            },
        )

    return _graph_module(build, [torch.randn(10, 4), torch.tensor([1, 3, 5])])


def _aten_view():
    def build(g):
        return g.call_function(torch.ops.aten.view.default, (g.placeholder("x"), [3, 2]))

    return _graph_module(build, [torch.randn(2, 3)])


def _method_view_one_tuple():
    def build(g):
        return (g.call_method("view", (g.placeholder("x"), 3, 2)),)

    return _graph_module(build, [torch.randn(2, 3)])


def _chunk(op, target):
    def build(g):
        chunks = g.create_node(op, target, (g.placeholder("x"), 2), {"dim": -1})
        first = g.call_function(operator.getitem, (chunks, 0))
        second = g.call_function(operator.getitem, (chunks, 1))
        return (first, second)

    return lambda: _graph_module(build, [torch.randn(2, 6)])


def _empty(target):
    def build(g):
        return g.call_function(
            target, ([2, 3],), {"dtype": torch.float16, "device": torch.device("cpu")}
        )

    return lambda: _graph_module(build, [])


def _size_feeding_empty():
    def build(g):
        size = g.call_function(torch.Size, ([2, 3],))
        return g.call_function(
            torch.empty, (size,), {"dtype": torch.float32, "device": torch.device("cpu")}
        )

    return _graph_module(build, [])


CASES = {
    "aten.linear.default": _linear(torch.ops.aten.linear.default),
    "torch._C._nn.linear": _linear(torch._C._nn.linear),
    "aten.embedding.default": _aten_embedding,
    "torch.nn.functional.embedding": _functional_embedding,
    "aten.view.default": _aten_view,
    "Tensor.view-one-tuple": _method_view_one_tuple,
    "aten.chunk.default": _chunk("call_function", torch.ops.aten.chunk.default),
    "torch.chunk": _chunk("call_function", torch.chunk),
    "Tensor.chunk": _chunk("call_method", "chunk"),
    "aten.empty.memory_format": _empty(torch.ops.aten.empty.memory_format),
    "torch.empty": _empty(torch.empty),
    "torch.Size-feeding-empty": _size_feeding_empty,
}

# torch.empty returns uninitialized memory: only shape and dtype are comparable.
UNINITIALIZED = {"aten.empty.memory_format", "torch.empty", "torch.Size-feeding-empty"}


@pytest.mark.parametrize("case", CASES)
def test_round_trip_matches_original(case):
    gm, inputs = CASES[case]()

    imported = import_fx(gm)
    rebuilt = rebuild_fx(imported.module, imported.out_spec)

    expected, actual = gm(*inputs), rebuilt(*inputs)
    assert pytree.tree_structure(actual) == pytree.tree_structure(expected)
    for e, a in zip(pytree.tree_leaves(expected), pytree.tree_leaves(actual)):
        if case in UNINITIALIZED:
            assert (a.shape, a.dtype) == (e.shape, e.dtype)
        else:
            torch.testing.assert_close(a, e)
