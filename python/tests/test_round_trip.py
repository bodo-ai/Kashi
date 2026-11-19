"""FX -> MLIR -> FX round trips: the rebuilt graph must compute what the
original computes, with the same output structure."""

import pytest

torch = pytest.importorskip("torch")

import operator

from fx_test_utils import _round_trip
from kashi.importer import import_fx
from kashi.mlir_to_fx import rebuild_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree


class _RecordVals(torch.fx.Interpreter):
    def run_node(self, node):
        result = super().run_node(node)
        node.meta["val"] = result
        return result


def _graph_module(build, inputs):
    graph = torch.fx.Graph()
    graph.output(build(graph))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    fake_mode = FakeTensorMode()
    fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
    with fake_mode:
        _RecordVals(gm).run(*fake_inputs)
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
        return g.call_function(
            torch.ops.aten.view.default, (g.placeholder("x"), [3, 2])
        )

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
            torch.empty,
            (size,),
            {"dtype": torch.float32, "device": torch.device("cpu")},
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


def test_data_attr_and_long_round_trip_detach_and_convert():
    """Data access detaches gradients and long conversion preserves values."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    data = graph.call_function(torch._C._autograd._get_data_attr, (x,))
    converted = graph.call_method("long", (data,))
    graph.output((data, converted))
    x_value = torch.tensor([[1.5, -2.3]], requires_grad=True)
    original, rebuilt = _round_trip(graph, x_value)

    expected_data, expected_long = original(x_value)
    actual_data, actual_long = rebuilt(x_value)
    torch.testing.assert_close(actual_data, expected_data)
    torch.testing.assert_close(actual_long, expected_long)
    assert not actual_data.requires_grad
    assert actual_long.dtype == torch.int64


@pytest.mark.parametrize(
    "dtype,keyword,copy",
    [
        (torch.float32, False, False),
        (torch.float32, True, True),
        (torch.float16, False, False),
    ],
)
def test_to_dtype_round_trip(dtype, keyword, copy):
    """Dtype conversion retains values and requested copy semantics."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    args = (x,) if keyword else (x, dtype)
    kwargs = {"dtype": dtype} if keyword else {}
    if copy:
        kwargs.update(non_blocking=True, copy=True)
    converted = graph.call_method("to", args, kwargs)
    graph.output((converted, x))
    x_value = torch.tensor([1.25, -2.5], dtype=torch.float32)
    _, rebuilt = _round_trip(graph, x_value)

    result, original_input = rebuilt(x_value)
    torch.testing.assert_close(result, torch.tensor([1.25, -2.5], dtype=dtype))
    assert original_input is x_value
    assert (result.data_ptr() != x_value.data_ptr()) == (copy or dtype != x_value.dtype)


@pytest.mark.parametrize("dim,keyword", [(0, False), (-1, True)])
def test_size_dim_round_trip(dim, keyword):
    """Dimension-specific size queries use the runtime input shape."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    args, kwargs = ((x,), {"dim": dim}) if keyword else ((x, dim), {})
    size = graph.call_method("size", args, kwargs)
    graph.output(size)
    original, rebuilt = _round_trip(graph, torch.randn(2, 3))

    for x_value in (torch.randn(2, 3), torch.randn(4, 5)):
        assert rebuilt(x_value) == original(x_value)


@pytest.mark.parametrize("shape_source", ["method", "aten"])
@pytest.mark.parametrize("shape", [(2, 3), ()])
def test_view_consumes_runtime_size_list(shape_source, shape):
    """View accepts runtime shape lists, including scalar tensor shapes."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    if shape_source == "method":
        size = graph.call_method("size", (x,))
    else:
        size = graph.call_function(torch.ops.aten.size.default, (x,))
    viewed = graph.call_method("view", (x, size))
    graph.output(viewed)
    original, rebuilt = _round_trip(graph, torch.randn(shape))

    inputs = (torch.randn(shape), torch.randn(3, 2)) if shape else (torch.randn(()),)
    for x_value in inputs:
        torch.testing.assert_close(rebuilt(x_value), original(x_value))


def test_view_single_runtime_dimension_is_not_treated_as_shape_list():
    """A scalar size node is treated as one dimension rather than a shape list."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    size = graph.call_method("size", (x, 0))
    graph.output(graph.call_method("view", (x, size)))
    original, rebuilt = _round_trip(graph, torch.arange(5))

    for x_value in (torch.arange(5), torch.arange(8)):
        torch.testing.assert_close(rebuilt(x_value), original(x_value))


@pytest.mark.parametrize("whole_shape", [True, False])
def test_empty_consumes_runtime_size(whole_shape):
    """Empty allocation respects runtime whole-shape and scalar-size inputs."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    size = graph.call_method("size", (x,) if whole_shape else (x, 0))
    graph.output(graph.call_function(torch.empty, (size,), {"dtype": torch.float32}))
    original, rebuilt = _round_trip(graph, torch.randn(2, 3))

    for x_value in (torch.randn(2, 3), torch.randn(5, 7)):
        expected, actual = original(x_value), rebuilt(x_value)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype


def test_torch_size_constructor_feeding_view():
    """A torch.Size constructor can supply a view's shape."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    size = graph.call_function(torch.Size, ([3, 2],))
    graph.output(graph.call_method("view", (x, size)))
    x_value = torch.arange(6).reshape(2, 3)
    original, rebuilt = _round_trip(graph, x_value)

    torch.testing.assert_close(rebuilt(x_value), original(x_value))


@pytest.mark.parametrize("shape", [(2, 3), ()])
def test_size_result_round_trip_including_scalar_shape(shape):
    """Returned size values retain all dimensions, including zero dimensions."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_method("size", (x,)))
    x_value = torch.randn(shape)
    original, rebuilt = _round_trip(graph, x_value)

    # ATen represents the Torch Size result as a list in the rebuilt graph.
    assert tuple(rebuilt(x_value)) == tuple(original(x_value))


@pytest.mark.parametrize(
    "sections,dim,keyword",
    [
        (2, -1, False),
        ([1, 2, 1], -1, True),
        ((1, 2, 1), 1, False),
        ([0, 4], -1, True),
    ],
)
def test_split_elements_round_trip(sections, dim, keyword):
    """Scalar and section-based splits retain their element values."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    args = (x, sections) if keyword else (x, sections, dim)
    chunks = graph.call_method("split", args, {"dim": dim} if keyword else {})
    count = 2 if isinstance(sections, int) else len(sections)
    graph.output(
        tuple(graph.call_function(operator.getitem, (chunks, i)) for i in range(count))
    )
    x_value = torch.arange(8).reshape(2, 4)
    original, rebuilt = _round_trip(graph, x_value)

    expected, actual = original(x_value), rebuilt(x_value)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)


def test_split_default_dim_consumed_by_getitem():
    """Indexed scalar splits use dimension zero when no dimension is supplied."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    chunks = graph.call_method("split", (x, 2))
    graph.output(graph.call_function(operator.getitem, (chunks, 1)))
    x_value = torch.arange(15).reshape(5, 3)
    original, rebuilt = _round_trip(graph, x_value)

    for x_value in (torch.arange(15).reshape(5, 3), torch.arange(12).reshape(4, 3)):
        torch.testing.assert_close(rebuilt(x_value), original(x_value))


def test_split_keyword_sections_and_default_dim():
    """Keyword split sections are accepted with the default dimension."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    chunks = graph.call_method("split", (x,), {"split_size": [1, 3]})
    graph.output(graph.call_function(operator.getitem, (chunks, 1)))
    x_value = torch.arange(8).reshape(4, 2)
    original, rebuilt = _round_trip(graph, x_value)

    torch.testing.assert_close(rebuilt(x_value), original(x_value))


def test_native_aten_split_output_remains_a_list():
    """Native ATen split outputs retain their dynamic-length list structure."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(torch.ops.aten.split.Tensor, (x, 2, 0)))
    x_value = torch.arange(15).reshape(5, 3)
    original, rebuilt = _round_trip(graph, x_value)

    for x_value in (torch.arange(15).reshape(5, 3), torch.arange(12).reshape(4, 3)):
        expected, actual = original(x_value), rebuilt(x_value)
        assert isinstance(expected, list)
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference)
