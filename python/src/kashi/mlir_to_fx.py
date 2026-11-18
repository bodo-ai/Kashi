"""MLIR to FX reconstruction for Torch-dialect modules produced by ``import_fx``."""

import operator

import torch
from iree.compiler import ir
from iree.compiler.extras.fx_importer import TORCH_DTYPE_TO_INT
from torch.utils import _pytree as pytree


# --- Adapter: private PyTorch APIs used for schema-driven binding. ----------


def _resolve(op, name: str) -> torch._ops.OpOverload:
    """Resolve an MLIR operator name to a callable PyTorch OpOverload.

    For example:
        "torch.aten.linear" -> torch.ops.aten.linear.default
        "torch.aten.add.Tensor" -> torch.ops.aten.add.Tensor
        "torch.vllm.unified_kv_cache_update" ->
            torch.ops.vllm.unified_kv_cache_update.default

    If no overload is specified, assume "default".
    """
    if not name.startswith("torch."):
        raise _error(op, f"malformed operator name {name!r}")
    namespace, _, rest = name.removeprefix("torch.").partition(".")
    base, _, overload = rest.partition(".")
    overload = overload or "default"
    # torch.ops creates namespaces on demand, and a namespace object has its
    # own attributes (e.g. ``name``); only an OpOverloadPacket is an operator.
    packet = getattr(getattr(torch.ops, namespace), base, None)
    if (
        not isinstance(packet, torch._ops.OpOverloadPacket)
        or overload not in packet.overloads()
    ):
        raise _error(op, f"no torch.ops.{namespace}.{base}.{overload}")
    return getattr(packet, overload)


# The importer encodes dtypes with this table; decoding with its inverse keeps
# both directions in step.
_INT_TO_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_INT.items()}


def _decode(argument, value, op):
    """Decode literals representing PyTorch-specific argument types.

    For example (MLIR types omitted):
        %dtype = torch.constant.int 5 
        %cpu = torch.constant.device "cpu" 
        %y = torch.aten.empty.memory_format %size, %dtype, %none,
            %cpu, %none, %none

    The operator schema tells us to interpret 5 as torch.float16 and "cpu" as
    torch.device("cpu").
    """
    # Use real_type because type represents enums such as ScalarType as int.
    kind = str(argument.real_type)
    if kind.startswith("Optional[") and kind.endswith("]"):
        kind = kind[len("Optional[") : -1]
    if kind == "ScalarType" and isinstance(value, int):
        if value not in _INT_TO_DTYPE:
            raise _error(op, f"{argument.name}: unknown ScalarType {value}")
        return _INT_TO_DTYPE[value]
    if kind == "Device" and isinstance(value, str):
        return torch.device(value)
    if kind in ("Layout", "MemoryFormat") and value is not None:
        raise _error(op, f"{argument.name}: only None is supported for {kind}")
    return value


# --- Reconstruction. --------------------------------------------------------


def rebuild_fx(
    module: ir.Module, out_spec: pytree.TreeSpec | None = None
) -> torch.fx.GraphModule:
    """Reconstruct an executable FX graph from a Torch-dialect MLIR module.

    MLIR block arguments become FX placeholders. Operations are translated in program
    order, mapping SSA values to FX nodes, Python literals, or containers to preserve
    data dependencies.

    *out_spec* restores the original FX output structure (e.g. x vs. (x,)). Without it,
    results are returned as a tuple unless there is exactly one.
    """
    block = _main_block(module)
    graph = torch.fx.Graph()
    values = {}

    for i, arg in enumerate(block.arguments):
        if not _supported_argument_type(str(arg.type)):
            raise NotImplementedError(
                f"Unsupported block argument %arg{i} of type {arg.type}: "
                "only !torch.vtensor, !torch.int and !torch.any arguments "
                "are supported"
            )
        values[arg] = graph.placeholder(f"arg{i}")

    for op in block.operations:
        name = op.operation.name
        operands = [values[v] for v in op.operands]
        if name == "func.return":
            graph.output(_output(op, operands, out_spec))
            continue
        if name in _STRUCTURAL:
            results = _STRUCTURAL[name](graph, op, operands)
        elif name.startswith("torch.aten."):
            results = _call(graph, op, name, operands, max_results=1, min_results=1)
        elif name == "torch.operator":
            qualified = ir.StringAttr(op.attributes["name"]).value
            results = _call(graph, op, qualified, operands, max_results=1)
        else:
            raise _error(op, "unsupported operation")
        for result, value in zip(op.results, results, strict=True):
            values[result] = value

    graph.lint()
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _main_block(module: ir.Module) -> ir.Block:
    """Extract the single block of the main function.

    Only straight-line programs with the following structure are supported:

        module {
            func.func @main(<args>) -> <result_types> {
                <ops>
                return <results>
            }
        }
    """
    ops = list(module.body.operations)
    if (
        len(ops) != 1
        or ops[0].operation.name != "func.func"
        or ir.StringAttr(ops[0].attributes["sym_name"]).value != "main"
    ):
        raise NotImplementedError(
            "Expected a module containing exactly one func.func @main"
        )
    (region,) = ops[0].regions
    blocks = list(region.blocks)
    if len(blocks) != 1:
        raise _error(ops[0], f"expected 1 block, found {len(blocks)}")
    return blocks[0]


def _supported_argument_type(type_str: str) -> bool:
    return type_str.startswith("!torch.vtensor") or type_str in (
        "!torch.int",
        "!torch.any",
    )


def _call(graph, op, name, operands, *, max_results, min_results=0):
    """Reconstruct an MLIR operator as an FX call_function node.

    Use the operator schema to decode literals and distinguish positional and
    keyword-only arguments, since MLIR represents all operands positionally.

    For example (MLIR types omitted):
        %z = torch.aten.add.Tensor %x, %y, %alpha

    Becomes:
        z = torch.ops.aten.add.Tensor(x, y, alpha=alpha)
    """
    target = _resolve(op, name)
    num_results = len(op.results)
    if not min_results <= num_results <= max_results:
        expected = (
            f"{max_results}" if min_results == max_results else f"at most {max_results}"
        )
        raise _error(op, f"expected {expected} result, found {num_results}")
    arguments = target._schema.arguments
    if len(operands) != len(arguments):
        raise _error(
            op, f"expected {len(arguments)} operands, found {len(operands)}"
        )
    args, kwargs = [], {}
    for argument, value in zip(arguments, operands):
        value = _decode(argument, value, op)
        if argument.kwarg_only:
            kwargs[argument.name] = value
        else:
            args.append(value)
    node = graph.call_function(target, tuple(args), kwargs)
    return [node] if num_results else []


def _output(op, operands, out_spec):
    """Restore the original FX output structure from MLIR return operands.

    For example:
        return %x

    May be reconstructed as x, (x,), or [x], depending on out_spec.
    """
    if out_spec is None:
        return operands[0] if len(operands) == 1 else tuple(operands)
    if out_spec.num_leaves != len(operands):
        raise _error(
            op,
            f"out_spec expects {out_spec.num_leaves} values, found {len(operands)}",
        )
    return pytree.tree_unflatten(operands, out_spec)


def _constant(attr_type):
    """Create a handler that converts MLIR constants to Python literals.

    Unlike MLIR, FX does not require separate nodes for literal arguments.

    For example (MLIR types omitted):
        %dim = torch.constant.int -1
        %n = torch.constant.int 3
        %y = torch.aten.chunk %x, %n, %dim

    Becomes:
        y = torch.ops.aten.chunk.default(x, 3, -1)
    """
    def convert(graph, op, operands):
        return [attr_type(op.attributes["value"]).value]

    return convert


def _list_unpack(graph, op, operands):
    """Rewrite MLIR ListUnpack operations as individual FX getitem calls.

    For example (MLIR types omitted):
        %chunks = torch.aten.chunk %x, %n, %dim
        %1:2 = torch.prim.ListUnpack %chunks

    Becomes:
        chunks = torch.ops.aten.chunk.default(x, n, dim)
        a = operator.getitem(chunks, 0)
        b = operator.getitem(chunks, 1)
    """
    (src,) = operands
    if not isinstance(src, torch.fx.Node):
        raise _error(op, "only the result of an operation can be unpacked")
    return [
        graph.call_function(operator.getitem, (src, i))
        for i in range(len(op.results))
    ]


# Structural operations have no corresponding PyTorch dispatcher operator
# and are instead reconstructed as Python literals, lists, or getitem calls.
_STRUCTURAL = {
    "torch.prim.ListUnpack": _list_unpack,
    "torch.constant.int": _constant(ir.IntegerAttr),
    "torch.constant.float": _constant(ir.FloatAttr),
    "torch.constant.bool": _constant(ir.BoolAttr),
    "torch.constant.device": _constant(ir.StringAttr),
    "torch.constant.none": lambda graph, op, operands: [None],
    # %size = torch.prim.ListConstruct %a, %b -> [a, b]
    "torch.prim.ListConstruct": lambda graph, op, operands: [list(operands)],
}


def _error(op, reason: str) -> NotImplementedError:
    return NotImplementedError(
        f"Unsupported MLIR op {op.operation.name} at {op.location}: {reason}"
    )
