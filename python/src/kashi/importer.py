"""FX to MLIR importer built on IREE's packaged Torch FX importer."""

import operator
from dataclasses import dataclass

import torch
from iree.compiler import ir
from iree.compiler.extras.fx_importer import ContextCache, FxImporter, IrType
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import get_opaque_type_name, register_opaque_type
from torch._opaque_base import OpaqueBase
from torch.utils import _pytree as pytree

# vLLM's attention writes its result into ``output`` and returns nothing, which
# value-semantic MLIR cannot express. ``import_fx`` replaces it with the
# functional ``kashi::unified_attention``, which returns the result, and
# ``rebuild_fx`` expands that back into the vLLM op writing a fresh buffer.
_VLLM_ATTENTION = "vllm::unified_attention_with_output"
_LIB = torch.library.Library("kashi", "FRAGMENT")


class _LayerName(OpaqueBase):
    """Names vLLM's opaque ``layer_name`` type in the schema below. Schemas
    report every opaque type as ``PyObject``, which cannot be parsed back."""


register_opaque_type(_LayerName, typ="reference")
_LIB.define(
    "unified_attention(Tensor query, Tensor key, Tensor value, "
    f"{get_opaque_type_name(_LayerName)} layer_name, "
    "Tensor? kv_cache_dummy_dep=None) -> Tensor"
)


@torch.library.register_fake("kashi::unified_attention", lib=_LIB)
def _(query, key, value, layer_name, kv_cache_dummy_dep=None):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


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
    leaves, out_spec = pytree.tree_flatten(output.args[0])
    # IREE's graph importer flattens only the outermost container. Flatten all
    # output pytrees here so reconstruction can restore nested structures.
    output.args = (tuple(leaves),)
    importer = FxImporter()
    _use_opaque_types(importer)
    importer.import_graph_module(normalized)
    return ImportedModule(importer.module, out_spec)


class _OpaqueContextCache(ContextCache):
    """Type opaque objects and the shape/tensor sequences used by Dynamo."""

    def value_info_to_type(self, val, *, tensor_meta=None, mutable=False):
        if isinstance(val, FakeScriptObject):
            return IrType.parse("!torch.any", self._c)
        # torch.Size is a tuple subclass, including for a rank-zero tensor.
        if isinstance(val, torch.Size):
            return IrType.parse("!torch.list<int>", self._c)
        if isinstance(val, (tuple, list)):
            if all(isinstance(v, torch._subclasses.FakeTensor) for v in val):
                return IrType.parse("!torch.list<vtensor>", self._c)
            if val and all(isinstance(v, (int, torch.SymInt)) for v in val):
                return IrType.parse("!torch.list<int>", self._c)
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
    _expand_split_outputs(graph)

    for node in list(graph.nodes):
        _normalize_meta(node)
        if node.op in ("call_function", "call_method"):
            _rewrite(graph, node)

    graph.eliminate_dead_code(is_impure_node=_is_impure)
    graph.lint()
    return torch.fx.GraphModule(gm, graph)


def _is_impure(node: torch.fx.Node) -> bool:
    """Keep every non-ATen operator during dead-code elimination: custom ops can
    have side effects their schemas do not declare, e.g. vLLM's KV-cache update
    and the attention that consumes its dependency."""
    target = node.target
    if isinstance(target, torch._ops.OpOverload) and target.namespace != "aten":
        return True
    return node.is_impure()


def _expand_split_outputs(graph: torch.fx.Graph) -> None:
    """Preserve statically sized Tensor.split tuples before the ATen rewrite.

    An FX node alone cannot distinguish the method's tuple from a native ATen
    split's list. Only literal sections guarantee a fixed output arity; scalar
    split sizes can yield a different number of pieces at runtime.
    """
    (output,) = graph.find_nodes(op="output")

    def expand(value):
        """Expand an output split node into individually indexed tuple elements."""
        if not isinstance(value, torch.fx.Node):
            return value
        if value.op != "call_method" or value.target != "split":
            return value
        sections = (
            value.args[1] if len(value.args) >= 2 else value.kwargs.get("split_size")
        )
        if not isinstance(sections, (list, tuple)):
            raise NotImplementedError(
                f"Tensor.split output requires statically sized sections: {value.format_node()}"
            )
        example = value.meta.get("val", value.meta.get("example_value"))
        if not isinstance(example, (tuple, list)) or len(example) != len(sections):
            raise NotImplementedError(
                f"Tensor.split output requires matching example metadata: {value.format_node()}"
            )
        items = []
        with graph.inserting_before(output):
            for index, item in enumerate(example):
                element = graph.call_function(operator.getitem, (value, index))
                element.meta["val"] = item
                items.append(element)
        return tuple(items)

    output.args = (pytree.tree_map(expand, output.args[0]),)


def _normalize_meta(node: torch.fx.Node) -> None:
    """Populate the metadata fields expected by FxImporter."""
    if "val" not in node.meta:
        if "example_value" not in node.meta:
            return
        node.meta["val"] = node.meta["example_value"]

    val = node.meta["val"]
    # An empty plain list is ambiguous with an empty list of tensors. Size's
    # operator schema identifies it as a shape even for rank-zero tensors.
    if (
        node.op == "call_function"
        and node.target is torch.ops.aten.size.default
        and isinstance(val, (list, tuple))
        and all(isinstance(dim, (int, torch.SymInt)) for dim in val)
    ):
        val = node.meta["val"] = torch.Size(val)
    if isinstance(val, torch.Size):
        return

    # FxImporter expects lists rather than tuples for tensor sequences.
    if isinstance(val, tuple):
        node.meta["val"] = list(val)


def _rewrite(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """Rewrite a Dynamo FX node into a form supported by FxImporter.

    Dynamo graphs may contain Python functions, tensor method calls, and operator
    overload packets rather than concrete ATen operators.
    """
    target = node.target
    if node.op == "call_function":
        if isinstance(target, torch._ops.OpOverloadPacket):
            _resolve_packet(node)
            target = node.target
        if isinstance(target, torch._ops.OpOverload):
            if target._schema.name == _VLLM_ATTENTION:
                _functionalize_attention(graph, node)
            return
        if target is operator.getitem:
            source, index = node.args
            if isinstance(source, torch.fx.Node) and isinstance(
                source.meta.get("val"), torch.Tensor
            ):
                _rewrite_tensor_slice(graph, node, source, index)
            return
        if target in (operator.add, operator.mul):
            _rewrite_tensor_binary(graph, node)
            return
        if target is torch.rsqrt:
            _replace(graph, node, torch.ops.aten.rsqrt.default, node.args, node.kwargs)
            return
        if target is torch.nn.functional.silu:
            if len(node.args) != 1 or node.kwargs not in ({}, {"inplace": False}):
                raise NotImplementedError(f"Unsupported SiLU: {node.format_node()}")
            _replace(graph, node, torch.ops.aten.silu.default, node.args, {})
            return
        if target is torch._C._nn.linear:
            _replace(graph, node, torch.ops.aten.linear.default, node.args, node.kwargs)
            return
        if target is torch._C._autograd._get_data_attr:
            _replace(graph, node, torch.ops.aten.detach.default, node.args, node.kwargs)
            return
        if target is torch.nn.functional.embedding:
            _rewrite_embedding(graph, node)
            return
        if target is torch.chunk:
            _rewrite_chunk(graph, node)
            return
        if target is torch.empty:
            size = node.args[0] if len(node.args) == 1 else node.args
            # A size() result is already a runtime list; an individual
            # symbolic dimension instead needs to be wrapped as a list.
            if not _is_shape_list_node(size):
                size = list(size) if isinstance(size, (list, tuple)) else [size]
            _replace(
                graph,
                node,
                torch.ops.aten.empty.memory_format,
                (size,),
                node.kwargs,
            )
            return
        if target is torch.Size:
            _fold_size(node)
            return
    elif node.op == "call_method":
        if target == "to":
            dtype = node.args[1] if len(node.args) > 1 else node.kwargs.get("dtype")
            # Tensor.to has device and tensor overloads too; only a dtype
            # selects aten.to.dtype without changing the original semantics.
            if isinstance(dtype, torch.dtype):
                _replace(graph, node, torch.ops.aten.to.dtype, node.args, node.kwargs)
                return
        if target == "long":
            if len(node.args) != 1 or node.kwargs:
                raise NotImplementedError(
                    f"Unsupported Tensor.long: {node.format_node()}"
                )
            _replace(
                graph, node, torch.ops.aten.to.dtype, (node.args[0], torch.int64), {}
            )
            return
        if (
            target == "pow"
            and len(node.args) == 2
            and isinstance(node.args[1], (int, float))
        ):
            _replace(
                graph, node, torch.ops.aten.pow.Tensor_Scalar, node.args, node.kwargs
            )
            return
        if target == "mean":
            _rewrite_mean(graph, node)
            return
        if target == "size":
            _rewrite_size(graph, node)
            return
        if target == "split":
            _rewrite_split(graph, node)
            return
        if target in ("view", "reshape"):
            if not node.args or node.kwargs:
                raise NotImplementedError(
                    f"Unsupported Tensor.{target}: {node.format_node()}"
                )
            x, *shape = node.args
            if len(shape) == 1 and _is_shape_list_node(shape[0]):
                # A runtime shape list is already the complete shape operand.
                shape = shape[0]
            elif len(shape) == 1 and isinstance(shape[0], (list, tuple)):
                shape = list(shape[0])
            op = (
                torch.ops.aten.view.default
                if target == "view"
                else torch.ops.aten.reshape.default
            )
            _replace(graph, node, op, (x, shape), {})
            return
        if target == "chunk":
            _rewrite_chunk(graph, node)
            return

    raise NotImplementedError(
        f"Unsupported FX node {node.format_node()} ({node.op} {target!r})"
    )


def _rewrite_tensor_binary(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """Select the ATen tensor or scalar overload for tensor-first arithmetic."""
    left, right = node.args
    if not isinstance(left, torch.fx.Node) or not isinstance(
        left.meta.get("val"), torch.Tensor
    ):
        raise NotImplementedError(
            f"Unsupported tensor arithmetic: {node.format_node()}"
        )
    packet = torch.ops.aten.add if node.target is operator.add else torch.ops.aten.mul
    if isinstance(right, torch.fx.Node) and isinstance(
        right.meta.get("val"), torch.Tensor
    ):
        target = packet.Tensor
    elif isinstance(right, (int, float, complex)):
        target = packet.Scalar
    else:
        raise NotImplementedError(
            f"Unsupported tensor arithmetic: {node.format_node()}"
        )
    _replace(graph, node, target, node.args, node.kwargs)


def _rewrite_tensor_slice(
    graph: torch.fx.Graph, node: torch.fx.Node, source, index
) -> None:
    """Lower first- or last-dimension tensor slices to ATen without copying."""
    if isinstance(index, slice):
        dim = 0
    elif (
        isinstance(index, tuple)
        and len(index) == 2
        and index[0] is Ellipsis
        and isinstance(index[1], slice)
    ):
        dim, index = -1, index[1]
    else:
        raise NotImplementedError(f"Unsupported tensor indexing: {node.format_node()}")
    step = 1 if index.step is None else index.step
    _replace(
        graph,
        node,
        torch.ops.aten.slice.Tensor,
        (source, dim, index.start, index.stop, step),
        {},
    )


def _rewrite_mean(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """Normalize Tensor.mean's scalar dimension to the ATen dimension-list form."""
    if len(node.args) > 1 and "dim" in node.kwargs:
        raise NotImplementedError(f"Unsupported Tensor.mean: {node.format_node()}")
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if dim is None:
        raise NotImplementedError(f"Unsupported Tensor.mean: {node.format_node()}")
    dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
    kwargs = {key: value for key, value in node.kwargs.items() if key != "dim"}
    _replace(
        graph,
        node,
        torch.ops.aten.mean.dim,
        (node.args[0], dims, *node.args[2:]),
        kwargs,
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


def _functionalize_attention(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """Replace vLLM's ``unified_attention_with_output`` call with equivalent
    functional version ``kashi.unified_attention`` (a dummy without implementation).

    Only safe when nothing else can observe the write: ``output`` must be a
    contiguous, query-shaped view of a fresh ``empty`` that nothing reads
    after the call through another alias.
    """

    def reject(reason):
        raise NotImplementedError(
            f"Cannot functionalize attention, {reason}: {node.format_node()}"
        )

    # Check output shape and other properties to match pattern
    bound = torch.fx.operator_schemas.normalize_function(
        node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    ).kwargs
    output, query = bound["output"], bound["query"]
    if bound["output_scale"] is not None or bound["output_block_scale"] is not None:
        reject("output_scale and output_block_scale are unsupported")
    out, q = output.meta["val"], query.meta["val"]
    if not out.is_contiguous():
        reject("output is not contiguous")
    if (out.shape, out.dtype) != (q.shape, q.dtype):
        reject("output and query differ in shape or dtype")
    base = output
    while base.target is torch.ops.aten.view.default:
        base = base.args[0]
    if base.target is not torch.ops.aten.empty.memory_format:
        reject(f"output is not a fresh buffer (it comes from {base.name})")

    node_order = {n: i for i, n in enumerate(graph.nodes)}
    for alias in _derived_before(base, node, node_order) - {output}:
        if any(node_order[user] > node_order[node] for user in alias.users):
            reject(f"{alias.name} aliases output and is used after the call")

    output_users = [
        user for user in output.users if node_order[user] > node_order[node]
    ]
    with graph.inserting_before(node):
        result = graph.call_function(
            torch.ops.kashi.unified_attention.default,
            (query, bound["key"], bound["value"], bound["layer_name"]),
            {"kv_cache_dummy_dep": bound["kv_cache_dummy_dep"]},
        )
    result.meta = dict(output.meta)
    for user in output_users:
        user.replace_input_with(output, result)
    graph.erase_node(node)


def _derived_before(base: torch.fx.Node, call: torch.fx.Node, node_order) -> set:
    """``base`` and every node computed from it before ``call``."""
    found, stack = {base}, [base]
    while stack:
        for user in stack.pop().users:
            if (
                user is not call
                and node_order[user] < node_order[call]
                and user not in found
            ):
                found.add(user)
                stack.append(user)
    return found


def _is_shape_list_node(value) -> bool:
    """Return whether an FX node's example metadata describes a shape list."""
    if not isinstance(value, torch.fx.Node):
        return False
    example = value.meta.get("val", value.meta.get("example_value"))
    return isinstance(example, (torch.Size, list, tuple)) and all(
        isinstance(dim, (int, torch.SymInt)) for dim in example
    )


def _rewrite_size(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """Replace a Tensor.size call with its matching ATen overload in graph."""
    if (
        not 1 <= len(node.args) <= 2
        or set(node.kwargs) - {"dim"}
        or len(node.args) == 2
        and "dim" in node.kwargs
    ):
        raise NotImplementedError(f"Unsupported Tensor.size: {node.format_node()}")
    x = node.args[0]
    if len(node.args) == 1 and "dim" not in node.kwargs:
        _replace(graph, node, torch.ops.aten.size.default, (x,), {})
    else:
        dim = node.args[1] if len(node.args) == 2 else node.kwargs["dim"]
        if dim is None:
            raise NotImplementedError(f"Unsupported Tensor.size: {node.format_node()}")
        _replace(graph, node, torch.ops.aten.size.int, (x, dim), {})


def _rewrite_split(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    """Replace Tensor.split with the scalar-size or section-list ATen overload."""
    if (
        not 1 <= len(node.args) <= 3
        or set(node.kwargs) - {"split_size", "dim"}
        or len(node.args) >= 2
        and "split_size" in node.kwargs
        or len(node.args) == 3
        and "dim" in node.kwargs
        or len(node.args) == 1
        and "split_size" not in node.kwargs
    ):
        raise NotImplementedError(f"Unsupported Tensor.split: {node.format_node()}")
    x = node.args[0]
    sections = node.args[1] if len(node.args) >= 2 else node.kwargs["split_size"]
    dim = node.args[2] if len(node.args) == 3 else node.kwargs.get("dim", 0)
    target = (
        torch.ops.aten.split_with_sizes.default
        if isinstance(sections, (list, tuple)) or _is_shape_list_node(sections)
        else torch.ops.aten.split.Tensor
    )
    # IREE's importer expects list-valued schema operands as Python lists;
    # tuple sections are otherwise treated as multiple independent operands.
    if isinstance(sections, tuple):
        sections = list(sections)
    _replace(graph, node, target, (x, sections, dim), {})


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
