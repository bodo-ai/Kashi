"""Helpers for FX importer and reconstruction tests."""

import torch
from kashi.importer import import_fx
from kashi.mlir_to_fx import rebuild_fx
from torch._subclasses.fake_tensor import FakeTensorMode


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
