"""MLIR to FX reconstruction for Torch-dialect modules produced by ``import_fx``."""

import operator

import torch
from iree.compiler import ir
from iree.compiler.extras.fx_importer import TORCH_DTYPE_TO_INT
from torch.utils import _pytree as pytree

from kashi.functional import functional_original


# --- Adapter: private PyTorch APIs used for schema-driven binding. ----------


def _resolve(op, name: str) -> torch._ops.OpOverload:
    """``torch.<ns>.<name>[.<overload>]`` -> ``torch.ops.<ns>.<name>.<overload>``."""
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
            original = _functional_original(qualified)
            if original is not None:
                results = _expand_functional(graph, op, *original, operands)
            else:
                results = _call(graph, op, qualified, operands, max_results=1)
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


def _supported_argument_type(type_str: str) -> bool:
    return type_str.startswith("!torch.vtensor") or type_str in (
        "!torch.int",
        "!torch.any",
    )


def _call(graph, op, name, operands, *, max_results, min_results=0):
    """Emit a call to the dispatcher operator ``name``, binding operands by
    schema. An operator with no results is still emitted, exactly once."""
    target = _resolve(op, name)
    num_results = _check_results(op, min_results, max_results)
    node = graph.call_function(target, *_bind(op, target, operands))
    return [node] if num_results else []


def _check_results(op, min_results, max_results) -> int:
    num_results = len(op.results)
    if not min_results <= num_results <= max_results:
        expected = (
            f"{max_results}" if min_results == max_results else f"at most {max_results}"
        )
        raise _error(op, f"expected {expected} result, found {num_results}")
    return num_results


def _bind(op, target, operands):
    """Bind MLIR operands to ``target``'s schema as ``(args, kwargs)``."""
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
    return tuple(args), kwargs


def _functional_original(qualified: str):
    """``(original op name, form)`` when ``qualified`` names a kashi
    functional dummy, else ``None``."""
    if not qualified.startswith("torch.kashi."):
        return None
    dummy = qualified.removeprefix("torch.kashi.")
    if dummy.endswith(".default"):
        dummy = dummy.removesuffix(".default")
    return functional_original(dummy)


def _expand_functional(graph, op, original, form, operands):
    """Expand a functional dummy into a fresh contiguous buffer written by
    the original destination-passing op; the buffer is the result."""
    target = _resolve(op, "torch." + original.replace("::", "."))
    _check_results(op, 1, 1)
    names = [a.name for a in _schema_arguments(target)]
    dummy_names = [n for n in names if n != form.output]
    if len(operands) != len(dummy_names):
        raise _error(
            op, f"expected {len(dummy_names)} operands, found {len(operands)}"
        )
    like = operands[dummy_names.index(form.like)]
    buffer = graph.call_function(
        torch.ops.aten.empty_like.default,
        (like,),
        {"memory_format": torch.contiguous_format},
    )
    full = list(operands)
    full.insert(names.index(form.output), buffer)
    graph.call_function(target, *_bind(op, target, full))
    return [buffer]


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
