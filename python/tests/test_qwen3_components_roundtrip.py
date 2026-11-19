"""Qwen3 FX normalization through the real importer and reconstructor."""

import operator

import pytest
import torch
from fx_test_utils import RecordExamples, assert_tree_close, round_trip
from kashi.importer import import_fx
from torch._subclasses.fake_tensor import FakeTensorMode


def test_cpu_rms_normalizationround_trip():
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


def test_cpu_gated_activationround_trip():
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
