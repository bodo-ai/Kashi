"""Helpers for FX importer and reconstruction tests."""

import torch
from kashi.importer import import_fx
from kashi.mlir_to_fx import rebuild_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree

# The importer accepts both metadata keys: ``val`` (AOT/export graphs) and
# ``example_value`` (graphs handed to a Dynamo backend).
META_KEYS = ("val", "example_value")


class RecordExamples(torch.fx.Interpreter):
    """Record fake execution results on each node under ``meta_key``."""

    def __init__(self, gm, meta_key="example_value"):
        super().__init__(gm)
        self.meta_key = meta_key

    def run_node(self, node):
        """Execute a node and attach its result for subsequent FX import."""
        value = super().run_node(node)
        node.meta[self.meta_key] = value
        return value


def round_trip(graph, *inputs, meta_key="example_value"):
    """Import and rebuild a graph with fake examples, without mutating its nodes.

    Returns ``(original_gm, rebuilt_gm)``.
    """
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode() as mode:
        fake_inputs = [mode.from_tensor(value) for value in inputs]
        RecordExamples(gm, meta_key).run(*fake_inputs)

    original_graph = str(gm.graph)
    original_meta_keys = [set(node.meta) for node in gm.graph.nodes]
    imported = import_fx(gm)
    # Import must not rewrite the caller's nodes or add normalized metadata.
    assert str(gm.graph) == original_graph
    assert [set(node.meta) for node in gm.graph.nodes] == original_meta_keys
    return gm, rebuild_fx(imported.module, imported.out_spec)


def call_targets(gm, only=None):
    """Return ``call_function`` targets in graph order, optionally filtered to ``only``."""
    return [
        node.target
        for node in gm.graph.nodes
        if node.op == "call_function" and (only is None or node.target in only)
    ]


def assert_tree_close(actual, expected, *, uninitialized=()):
    """Assert two output pytrees have the same containers and close leaves.

    ``uninitialized`` lists leaf indices produced by ``torch.empty``; their
    memory is arbitrary, so only shape and dtype are compared.
    """
    assert pytree.tree_structure(actual) == pytree.tree_structure(expected)
    leaves = zip(pytree.tree_leaves(actual), pytree.tree_leaves(expected), strict=True)
    for i, (got, want) in enumerate(leaves):
        if i in uninitialized:
            assert (got.shape, got.dtype) == (want.shape, want.dtype)
        else:
            torch.testing.assert_close(got, want)
