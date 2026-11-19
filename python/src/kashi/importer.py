"""FX to MLIR importer built on IREE's packaged Torch FX importer."""

import operator
from dataclasses import dataclass

import torch
from iree.compiler import ir
from iree.compiler.extras.fx_importer import ContextCache, FxImporter, IrType
from torch._library.fake_class_registry import FakeScriptObject
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
    # FxImporter represents FX graph outputs as a sequence of MLIR function results.
    # For example: x, (x,), and [x] all lower to the same `return %x` in MLIR.
    # We preserve the original output structure as a TreeSpec so it can be
    # restored during FX graph reconstruction.
    out_spec = pytree.tree_flatten(output.args[0])[1]
    importer = FxImporter()
    _use_opaque_types(importer)
    importer.import_graph_module(normalized)
    return ImportedModule(importer.module, out_spec)


class _OpaqueContextCache(ContextCache):
    """Types hoisted opaque objects (``FakeScriptObject`` metadata, e.g.
    vLLM's ``LayerName``) as ``!torch.any``; defers everything else."""

    def value_info_to_type(self, val, *, tensor_meta=None, mutable=False):
        if isinstance(val, FakeScriptObject):
            return IrType.parse("!torch.any", self._c)
        return super().value_info_to_type(val, tensor_meta=tensor_meta, mutable=mutable)


def _use_opaque_types(importer: FxImporter) -> None:
    """Adapter: install the opaque-aware cache on this importer only. Uses
    IREE's private ``_cc``/``_c``/``_py_attr_tracker`` fields."""
    importer._cc = _OpaqueContextCache(
        importer._c, py_attr_tracker=importer._py_attr_tracker
    )


def _normalize(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Normalize a raw Dynamo FX graph into a form supported by FxImporter.

    FxImporter expects a functionalized graph consisting of ATen operators. Such graphs
    are typically obtained through AOTAutograd, which may also apply unwanted
    decompositions of higher-level operators such as linear.

    This function makes the minimal changes required for compatibility with FxImporter,
    including rewriting operators to their ATen equivalents and normalizing node
    metadata. It does not perform functionalization.
    """
    graph = torch.fx.Graph()
    output = graph.graph_copy(gm.graph, {})
    graph.output(output)

    for node in list(graph.nodes):
        _normalize_meta(node)
        if node.op in ("call_function", "call_method"):
            _rewrite(graph, node)

    graph.lint()
    return torch.fx.GraphModule(gm, graph)


def _normalize_meta(node: torch.fx.Node) -> None:
    """Populate the metadata fields expected by FxImporter."""
    if "val" not in node.meta and "example_value" in node.meta:
        node.meta["val"] = node.meta["example_value"]

    # FxImporter expects lists rather than tuples for tensor sequences.
    if isinstance(node.meta.get("val"), tuple):
        node.meta["val"] = list(node.meta["val"])


def _rewrite(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """Rewrite a Dynamo FX node into a form supported by FxImporter.

    Dynamo graphs may contain Python functions, tensor method calls, and operator
    overload packets rather than concrete ATen operators.
    """
    target = node.target
    if node.op == "call_function":
        if isinstance(target, torch._ops.OpOverloadPacket):
            _resolve_packet(node)
            return
        if isinstance(target, torch._ops.OpOverload) or target is operator.getitem:
            return
        if target is torch._C._nn.linear:
            _replace(graph, node, torch.ops.aten.linear.default, node.args, node.kwargs)
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


def _resolve_packet(node: torch.fx.Node) -> None:
    """Resolve an operator overload packet to a concrete operator overload.

    FxImporter expects an OpOverload with a specific operator schema rather than an
    OpOverloadPacket. Dynamo may emit overload packets, particularly for custom
    operators such as vllm.unified_kv_cache_update.
    """
    packet = node.target
    overloads = packet.overloads()
    if len(overloads) != 1:
        raise NotImplementedError(
            f"Ambiguous overload packet {packet} (overloads {overloads}): "
            f"{node.format_node()}"
        )
    node.target = getattr(packet, overloads[0])


def _rewrite_embedding(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """Rewrite torch.nn.functional.embedding to its ATen equivalent.

    The functional and ATen operators have different argument orders and default values.
    Normalize the arguments to match the ATen schema.
    """
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
    """Rewrite torch.chunk and Tensor.chunk to their ATen equivalent.

    Normalize positional and keyword arguments to match the ATen schema.
    """
    x, chunks, *rest = node.args
    dim = rest[0] if rest else node.kwargs.get("dim", 0)
    _replace(graph, node, torch.ops.aten.chunk.default, (x, chunks, dim), {})


def _fold_size(node: torch.fx.Node) -> None:
    """Eliminate torch.Size constructors, which FxImporter does not support.

    Replace uses of the constructor with an equivalent list of dimensions, preserving
    both static and symbolic sizes.
    """
    (dims,) = node.args
    if not all(isinstance(d, int) or _is_symint(d) for d in dims):
        raise NotImplementedError(
            f"torch.Size with unsupported dimensions: {node.format_node()}"
        )
    for user in list(node.users):
        user.replace_input_with(node, list(dims))
    node.graph.erase_node(node)


def _is_symint(value) -> bool:
    return isinstance(value, torch.fx.Node) and isinstance(
        value.meta.get("val"), torch.SymInt
    )


def _replace(graph: torch.fx.Graph, node: torch.fx.Node, target, args, kwargs) -> None:
    """Replace an FX node with a call_function node targeting an ATen operator."""
    with graph.inserting_before(node):
        # FxImporter does not support call_method nodes.
        new = graph.call_function(target, tuple(args), dict(kwargs))
    new.meta = node.meta
    node.replace_all_uses_with(new)
    graph.erase_node(node)
