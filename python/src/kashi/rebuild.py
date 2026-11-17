"""MLIR to FX reconstruction for Torch-dialect modules produced by ``import_fx``."""

import operator

import torch
from iree.compiler import ir
from iree.compiler.extras.fx_importer import TORCH_DTYPE_TO_INT
from torch.utils import _pytree as pytree


# --- Adapter: private PyTorch APIs used for schema-driven binding. ----------


def _resolve_aten(op, name: str) -> torch._ops.OpOverload:
    """``torch.aten.<name>[.<overload>]`` -> ``torch.ops.aten.<name>.<overload>``."""
    base, _, overload = name.removeprefix("torch.aten.").partition(".")
    overload = overload or "default"
    packet = getattr(torch.ops.aten, base, None)
    # The namespace object has its own attributes (e.g. ``name``); only an
    # OpOverloadPacket is an operator.
    if (
        not isinstance(packet, torch._ops.OpOverloadPacket)
        or overload not in packet.overloads()
    ):
        raise _error(op, f"no torch.ops.aten.{base}.{overload}")
    return getattr(packet, overload)


def _schema_arguments(overload: torch._ops.OpOverload):
    return overload._schema.arguments


# The importer encodes dtypes with this table; decoding with its inverse keeps
# both directions in step.
_INT_TO_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_INT.items()}


def _decode(argument, value, op):
    """Decode an operand by its schema type. ``Argument.type`` reports enum
    types such as ``ScalarType`` as ``int``; ``real_type`` keeps them."""
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
    """Rebuild an executable FX graph from a Torch-dialect ``func.func @main``.

    ``out_spec`` restores the original output structure, which the importer
    flattens. Without it, one result is returned bare and several as a tuple.
    ``module`` is not modified.
    """
    block = _main_block(module)
    graph = torch.fx.Graph()
    values = {}

    for i, arg in enumerate(block.arguments):
        if not str(arg.type).startswith("!torch.vtensor"):
            raise NotImplementedError(
                f"Unsupported block argument %arg{i} of type {arg.type}: "
                "only !torch.vtensor arguments are supported"
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
            results = [_aten(graph, op, name, operands)]
        else:
            raise _error(op, "unsupported operation")
        for result, value in zip(op.results, results, strict=True):
            values[result] = value

    graph.lint()
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _main_block(module: ir.Module) -> ir.Block:
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


def _aten(graph, op, name, operands):
    target = _resolve_aten(op, name)
    if len(op.results) != 1:
        raise _error(op, f"expected 1 result, found {len(op.results)}")
    arguments = _schema_arguments(target)
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
    return graph.call_function(target, tuple(args), kwargs)


def _output(op, operands, out_spec):
    if out_spec is None:
        return operands[0] if len(operands) == 1 else tuple(operands)
    if out_spec.num_leaves != len(operands):
        raise _error(
            op,
            f"out_spec expects {out_spec.num_leaves} values, found {len(operands)}",
        )
    return pytree.tree_unflatten(operands, out_spec)


def _constant(attr_type):
    def convert(graph, op, operands):
        return [attr_type(op.attributes["value"]).value]

    return convert


def _list_unpack(graph, op, operands):
    (src,) = operands
    if not isinstance(src, torch.fx.Node):
        raise _error(op, "only the result of an operation can be unpacked")
    return [
        graph.call_function(operator.getitem, (src, i))
        for i in range(len(op.results))
    ]


_STRUCTURAL = {
    "torch.prim.ListUnpack": _list_unpack,
    "torch.constant.int": _constant(ir.IntegerAttr),
    "torch.constant.float": _constant(ir.FloatAttr),
    "torch.constant.bool": _constant(ir.BoolAttr),
    "torch.constant.device": _constant(ir.StringAttr),
    "torch.constant.none": lambda graph, op, operands: [None],
    "torch.prim.ListConstruct": lambda graph, op, operands: [list(operands)],
}


def _error(op, reason: str) -> NotImplementedError:
    return NotImplementedError(
        f"Unsupported MLIR op {op.operation.name} at {op.location}: {reason}"
    )
