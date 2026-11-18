"""Qwen3 FX normalization through the real importer and reconstructor."""

import operator

import pytest

torch = pytest.importorskip("torch")

from torch._subclasses.fake_tensor import FakeTensorMode

from kashi.importer import import_fx
from kashi.rebuild import rebuild_fx


class _RecordExamples(torch.fx.Interpreter):
    """Record fake execution results as Dynamo-style example metadata."""

    def run_node(self, node):
        """Execute a node and attach its result for subsequent FX import."""
        value = super().run_node(node)
        node.meta["example_value"] = value
        return value


def _round_trip(graph, *inputs):
    """Import and rebuild a graph with fake examples, without mutating its nodes."""
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode() as mode:
        fake_inputs = [mode.from_tensor(value) for value in inputs]
        _RecordExamples(gm).run(*fake_inputs)

    original_graph = str(gm.graph)
    imported = import_fx(gm)
    # Import must not rewrite the caller's nodes or add normalized metadata.
    assert str(gm.graph) == original_graph
    assert all("val" not in node.meta for node in gm.graph.nodes)
    return gm, rebuild_fx(imported.module, imported.out_spec)


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


def test_cpu_rms_normalization_round_trip():
    """CPU RMS normalization preserves reduction and broadcast arithmetic."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    squared = graph.call_method("pow", (x, 2))
    variance = graph.call_method("mean", (squared,), {"dim": -1, "keepdim": True})
    stabilized = graph.call_function(operator.add, (variance, 1e-6))
    scale = graph.call_function(torch.rsqrt, (stabilized,))
    normalized = graph.call_function(operator.mul, (x, scale))
    graph.output(graph.call_function(operator.mul, (normalized, weight)))
    inputs = (torch.tensor([[1.0, -2.0], [3.0, 4.0]]), torch.tensor([0.5, 2.0]))
    original, rebuilt = _round_trip(graph, *inputs)

    torch.testing.assert_close(rebuilt(*inputs), original(*inputs))


def test_cpu_gated_activation_round_trip():
    """CPU gating keeps reshape, trailing slices and SiLU semantics."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    reshaped = graph.call_method("reshape", (x, -1, 4))
    gate = graph.call_function(operator.getitem, (reshaped, (Ellipsis, slice(None, 2))))
    value = graph.call_function(operator.getitem, (reshaped, (Ellipsis, slice(2, None))))
    activated = graph.call_function(torch.nn.functional.silu, (gate,))
    graph.output(graph.call_function(operator.mul, (activated, value)))
    x_value = torch.tensor([[-2.0, -0.5, 1.5, 2.0], [0.0, 3.0, -1.0, 4.0]])
    original, rebuilt = _round_trip(graph, x_value)

    torch.testing.assert_close(rebuilt(x_value), original(x_value))


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


def test_split_literal_sections_preserve_nested_tuple_output():
    """Fixed split sections preserve tuple outputs nested in other containers."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    chunks = graph.call_method("split", (x, [1, 2, 1]), {"dim": -1})
    graph.output({"parts": [chunks, x]})
    original, rebuilt = _round_trip(graph, torch.arange(8).reshape(2, 4))

    for x_value in (torch.arange(8).reshape(2, 4), torch.arange(12).reshape(3, 4)):
        expected, actual = original(x_value), rebuilt(x_value)
        assert isinstance(actual, dict)
        assert isinstance(actual["parts"], list)
        assert isinstance(actual["parts"][0], tuple)
        assert len(actual["parts"][0]) == len(expected["parts"][0]) == 3
        for result, reference in zip(actual["parts"][0], expected["parts"][0], strict=True):
            torch.testing.assert_close(result, reference)
        torch.testing.assert_close(actual["parts"][1], expected["parts"][1])


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


def test_scalar_split_output_does_not_freeze_example_arity():
    """Direct scalar split output must not specialize its length to an example."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_method("split", (x, 2)))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode() as mode:
        _RecordExamples(gm).run(mode.from_tensor(torch.randn(5, 3)))

    with pytest.raises(NotImplementedError, match="statically sized sections"):
        import_fx(gm)


@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("long", (), {"unexpected": True}),
        ("size", (0,), {"dim": 1}),
        ("split", (2, 0), {"dim": 1}),
        ("view", (2, 2), {"unexpected": True}),
    ],
)
def test_invalid_method_arguments_are_not_silently_discarded(method, args, kwargs):
    """Unsupported method arguments fail rather than changing semantics."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    with FakeTensorMode() as mode:
        x.meta["example_value"] = mode.from_tensor(torch.randn(2, 2))
    graph.output(graph.call_method(method, (x, *args), kwargs))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    with pytest.raises((NotImplementedError, TypeError, ValueError)):
        import_fx(gm)
