"""FX to MLIR importer built on IREE's packaged Torch FX importer."""

import operator
from dataclasses import dataclass

import torch
from iree.compiler import ir
from iree.compiler.extras.fx_importer import FxImporter
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree


@dataclass(frozen=True)
class ImportedModule:
    """A Torch-dialect module and the pytree structure of the graph output,
    which the importer flattens into ``func.return`` operands."""

    module: ir.Module
    out_spec: pytree.TreeSpec


def import_fx(gm: torch.fx.GraphModule) -> ImportedModule:
    """Import ``gm`` as a Torch-dialect MLIR module.

    The input graph is not modified; normalization runs on a structural copy.
    """
    normalized = _normalize(gm)
    (output,) = normalized.graph.find_nodes(op="output")
    out_spec = pytree.tree_flatten(output.args[0])[1]
    importer = FxImporter()
    importer.import_graph_module(normalized)
    return ImportedModule(importer.module, out_spec)


def _normalize(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    output = graph.graph_copy(gm.graph, {})
    graph.output(output)

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    for node in list(graph.nodes):
        _fake_meta(node, fake_mode)
        if node.op in ("call_function", "call_method"):
            _rewrite(graph, node)

    graph.lint()
    return torch.fx.GraphModule(gm, graph)


def _fake_meta(node: torch.fx.Node, fake_mode: FakeTensorMode) -> None:
    """Set ``meta["val"]`` from Dynamo's ``example_value`` when missing, and
    replace real tensors with fake tensors of the same type."""
    if "val" not in node.meta:
        if "example_value" not in node.meta:
            return
        node.meta["val"] = node.meta["example_value"]
    val = node.meta["val"]
    if isinstance(val, (tuple, list)):
        node.meta["val"] = [_to_fake(v, fake_mode) for v in val]
    else:
        node.meta["val"] = _to_fake(val, fake_mode)


def _to_fake(val, fake_mode: FakeTensorMode):
    if isinstance(val, torch.Tensor) and not isinstance(
        val, torch._subclasses.FakeTensor
    ):
        return fake_mode.from_tensor(val)
    return val


def _rewrite(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    target = node.target
    if node.op == "call_function":
        if isinstance(target, torch._ops.OpOverload) or target is operator.getitem:
            return
        if target is torch._C._nn.linear:
            _replace(
                graph, node, torch.ops.aten.linear.default, node.args, node.kwargs
            )
            return
        if target is torch.nn.functional.embedding:
            _rewrite_embedding(graph, node)
            return
        if target is torch.chunk:
            _rewrite_chunk(graph, node)
            return
        if target is torch.empty:
            size = node.args[0] if len(node.args) == 1 else node.args
            _replace(
                graph,
                node,
                torch.ops.aten.empty.memory_format,
                (list(size),),
                node.kwargs,
            )
            return
        if target is torch.Size:
            _fold_size(node)
            return
    elif node.op == "call_method":
        if target == "view":
            x, *shape = node.args
            if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
                shape = shape[0]
            _replace(graph, node, torch.ops.aten.view.default, (x, list(shape)), {})
            return
        if target == "chunk":
            _rewrite_chunk(graph, node)
            return

    raise NotImplementedError(
        f"Unsupported FX node {node.format_node()} ({node.op} {target!r})"
    )


def _rewrite_embedding(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    bound = torch.fx.operator_schemas.normalize_function(
        torch.nn.functional.embedding,
        node.args,
        node.kwargs,
        normalize_to_only_use_kwargs=True,
    ).kwargs
    if bound.get("max_norm") is not None:
        raise NotImplementedError(
            f"embedding with max_norm mutates its weight: {node.format_node()}"
        )
    padding_idx = bound.get("padding_idx")
    _replace(
        graph,
        node,
        torch.ops.aten.embedding.default,
        (
            bound["weight"],
            bound["input"],
            -1 if padding_idx is None else padding_idx,
            bound.get("scale_grad_by_freq", False),
            bound.get("sparse", False),
        ),
        {},
    )


def _rewrite_chunk(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    x, chunks, *rest = node.args
    dim = rest[0] if rest else node.kwargs.get("dim", 0)
    _replace(graph, node, torch.ops.aten.chunk.default, (x, chunks, dim), {})


def _fold_size(node: torch.fx.Node) -> None:
    (dims,) = node.args
    if not all(isinstance(d, int) for d in dims):
        raise NotImplementedError(
            f"torch.Size with non-constant dimensions: {node.format_node()}"
        )
    for user in list(node.users):
        user.replace_input_with(node, list(dims))
    node.graph.erase_node(node)


def _replace(graph: torch.fx.Graph, node: torch.fx.Node, target, args, kwargs) -> None:
    with graph.inserting_before(node):
        new = graph.call_function(target, tuple(args), dict(kwargs))
    new.meta = node.meta
    node.replace_all_uses_with(new)
    graph.erase_node(node)
