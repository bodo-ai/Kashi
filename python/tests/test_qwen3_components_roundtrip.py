"""Qwen3 FX normalization through the real importer and reconstructor."""

import operator

import pytest
import torch
from fx_test_utils import RecordExamples, assert_tree_close, round_trip
from kashi.importer import import_fx
from torch._subclasses.fake_tensor import FakeTensorMode


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
    original, rebuilt = round_trip(graph, *inputs)

    torch.testing.assert_close(rebuilt(*inputs), original(*inputs))


def test_float_method_preserves_bfloat16_values():
    """Qwen3.8 promotes normalization weights to float32 before arithmetic."""
    graph = torch.fx.Graph()
    weight = graph.placeholder("weight")
    graph.output(graph.call_method("float", (weight,)))
    source = torch.tensor([1.25, -2.5, 0.125], dtype=torch.bfloat16)
    _, rebuilt = round_trip(graph, source)
    actual = rebuilt(source)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, source.float())


@pytest.mark.parametrize("dims", [(0, 2, 1), (-1, -2, 0), ((2, 0, 1),)])
def test_permute_method_preserves_layout_and_values(dims):
    """FP8 activation scales are permuted into the kernel's expected layout."""
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    graph.output(graph.call_method("permute", (source, *dims)))
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    original, rebuilt = round_trip(graph, value)
    expected, actual = original(value), rebuilt(value)
    torch.testing.assert_close(actual, expected)
    assert actual.stride() == expected.stride()


def test_numpy_t_packet_preserves_transposed_weight():
    """FP8 matmul transposes its weight and scale through an ATen packet."""
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    graph.output(graph.call_function(torch.ops.aten.numpy_T, (source,)))
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    _, rebuilt = round_trip(graph, value)
    result = rebuilt(value)
    torch.testing.assert_close(result, value.T)
    assert result.stride() == value.T.stride()


def test_contiguous_method_copies_noncontiguous_input():
    """GDN's gate vectors require dense contiguous buffers before its custom op."""
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    graph.output(graph.call_method("contiguous", (source,)))
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    _, rebuilt = round_trip(graph, value)
    result = rebuilt(value)
    torch.testing.assert_close(result, value)
    assert result.is_contiguous()
    assert result.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()


def test_flatten_method_preserves_values_and_shape():
    """GDN reshapes its head output before the output projection."""
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    graph.output(graph.call_method("flatten", (source, -2)))
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    _, rebuilt = round_trip(graph, value)
    torch.testing.assert_close(rebuilt(value), value.reshape(2, 12))


def test_zeros_uses_runtime_shape_and_dtype():
    """GDN's zero-initialized output follows the current token count."""
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    rows = graph.call_method("size", (source, 0))
    result = graph.call_function(torch.zeros, ((rows, 3),), {"dtype": torch.bfloat16})
    graph.output(result)
    _, rebuilt = round_trip(graph, torch.ones(2, 3))
    for rows in (2, 5):
        value = rebuilt(torch.ones(rows, 3))
        torch.testing.assert_close(value, torch.zeros(rows, 3, dtype=torch.bfloat16))


def test_zeros_keyword_size_preserves_shape():
    """Keyword-provided dimensions must not be discarded during import."""
    graph = torch.fx.Graph()
    graph.output(graph.call_function(torch.zeros, (), {"size": [2, 3]}))
    _, rebuilt = round_trip(graph)
    torch.testing.assert_close(rebuilt(), torch.zeros(2, 3))


def test_zeros_out_is_not_silently_discarded():
    """A mutating out= allocation must be preserved or explicitly rejected."""
    graph = torch.fx.Graph()
    output = graph.placeholder("output")
    graph.call_function(torch.zeros, ([2, 3],), {"out": output})
    graph.output(output)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode() as mode:
        RecordExamples(gm).run(mode.from_tensor(torch.ones(2, 3)))
    with pytest.raises(NotImplementedError, match="out"):
        import_fx(gm)


def test_cpu_gated_activation_round_trip():
    """CPU gating keeps reshape, trailing slices and SiLU semantics."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    reshaped = graph.call_method("reshape", (x, -1, 4))
    gate = graph.call_function(operator.getitem, (reshaped, (Ellipsis, slice(None, 2))))
    value = graph.call_function(
        operator.getitem, (reshaped, (Ellipsis, slice(2, None)))
    )
    activated = graph.call_function(torch.nn.functional.silu, (gate,))
    graph.output(graph.call_function(operator.mul, (activated, value)))
    x_value = torch.tensor([[-2.0, -0.5, 1.5, 2.0], [0.0, 3.0, -1.0, 4.0]])
    original, rebuilt = round_trip(graph, x_value)

    torch.testing.assert_close(rebuilt(x_value), original(x_value))


def test_split_literal_sections_preserve_nested_tuple_output():
    """Fixed split sections preserve tuple outputs nested in other containers."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    chunks = graph.call_method("split", (x, [1, 2, 1]), {"dim": -1})
    graph.output({"parts": [chunks, x]})
    original, rebuilt = round_trip(graph, torch.arange(8).reshape(2, 4))

    for x_value in (torch.arange(8).reshape(2, 4), torch.arange(12).reshape(3, 4)):
        expected = original(x_value)
        assert len(expected["parts"][0]) == 3
        # Tree comparison also checks the dict/list/tuple container types.
        assert_tree_close(rebuilt(x_value), expected)


def test_scalar_split_output_does_not_freeze_example_arity():
    """Direct scalar split output must not specialize its length to an example."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_method("split", (x, 2)))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode() as mode:
        RecordExamples(gm).run(mode.from_tensor(torch.randn(5, 3)))

    with pytest.raises(NotImplementedError, match="statically sized sections"):
        import_fx(gm)


@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("long", (), {"unexpected": True}),
        ("size", (0,), {"dim": 1}),
        ("split", (2, 0), {"dim": 1}),
        ("view", (2, 2), {"unexpected": True}),
        ("contiguous", (1,), {}),
        ("flatten", (0, 1), {"end_dim": 0}),
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
