"""Split Torch-dialect graphs into IREE segments and a mutation-safe driver."""

from dataclasses import dataclass
from functools import cache

import torch
from iree.compiler import ir
from iree.compiler.dialects import func


@dataclass
class SegmentInfo:
    """Describe a run's signature and any values it cannot return to its caller."""

    index: int
    func_name: str
    live_in_types: list[str]
    live_out_types: list[str]
    unsupported_live_outs: list[str]


@dataclass
class PartitionResult:
    """The independent compiled module, optional driver, and source device.

    A missing driver means at least one segment has an unsupported live-out;
    its type is recorded in that segment's ``unsupported_live_outs``.
    """

    module: ir.Module
    driver: ir.Module | None
    device: str | None
    segments: list[SegmentInfo]


def _constant(name: str) -> bool:
    """Identify pure, operand-free Torch constants suitable for duplication."""
    return name.startswith("torch.constant.") or name == "torch.vtensor.literal"


def _supported(type_: ir.Type) -> bool:
    """Identify values supported by the compiled segment interface."""
    text = str(type_)
    return (
        text.startswith(("!torch.vtensor", "!torch.tensor", "!torch.list"))
        or text == "!torch.int"
    )


@cache
def _driver_boundary(name: str) -> bool:
    """Keep custom, aliasing and mutating operations in the PyTorch driver.

    IREE's tensor value semantics can materialize a view or elide a copy.
    Keeping these operators outside compiled segments preserves the storage
    identity that later in-place custom operations and graph outputs observe.
    """
    # Shape lists are PyTorch-side values: IREE's exported function ABI cannot
    # pass !torch.list<int> across segment boundaries. Allocations consume
    # those lists and should use the source framework's allocator and device.
    if name in (
        "torch.operator",
        "torch.prim.ListUnpack",
        "torch.aten.clone",
        "torch.aten._to_copy",
        "torch.aten.size",
        "torch.aten.size.int",
        "torch.aten.sym_size.int",
        "torch.aten.empty.memory_format",
    ):
        return True
    if not name.startswith("torch.aten."):
        return False
    base, _, overload = name.removeprefix("torch.aten.").partition(".")
    packet = getattr(torch.ops.aten, base, None)
    overload = overload or "default"
    if (
        not isinstance(packet, torch._ops.OpOverloadPacket)
        or overload not in packet.overloads()
    ):
        raise NotImplementedError(f"Cannot determine alias semantics for {name}")
    schema = getattr(packet, overload)._schema
    return any(result.alias_info is not None for result in schema.returns) or any(
        arg.alias_info is not None and arg.alias_info.is_write
        for arg in schema.arguments
    )


def _is_boundary(operation) -> bool:
    """Keep dynamic list construction in the driver; constants are cloned."""
    if operation.name == "torch.prim.ListConstruct":
        return True
    return _driver_boundary(operation.name)


def _clone(operation, mapping, *, rewrite_device: bool):
    """Clone a region-free operation, translating SSA values into its new block."""
    attributes = {name: operation.attributes[name] for name in operation.attributes}
    # Device constants in IREE segments are normalized for the Torch frontend;
    # the driver and source still carry the original CUDA device.
    if (
        rewrite_device
        and operation.name == "torch.constant.device"
        and ir.StringAttr(operation.attributes["value"]).value.startswith("cuda")
    ):
        attributes["value"] = ir.StringAttr.get("cpu")
    cloned = ir.Operation.create(
        operation.name,
        results=[value.type for value in operation.results],
        operands=[mapping[value] for value in operation.operands],
        attributes=attributes,
        loc=operation.location,
    )
    mapping.update(zip(operation.results, cloned.results, strict=True))


def _source_block(module: ir.Module):
    """Find the single supported entry block or reject ambiguous source modules."""
    functions = list(module.body.operations)
    if (
        len(functions) != 1
        or functions[0].operation.name != "func.func"
        or ir.StringAttr(functions[0].attributes["sym_name"]).value != "main"
    ):
        raise NotImplementedError("expected a module containing only func.func @main")
    blocks = list(functions[0].regions[0].blocks)
    if len(blocks) != 1:
        raise NotImplementedError("expected a single block in func.func @main")
    operations = [op.operation for op in blocks[0].operations]
    if not operations or operations[-1].name != "func.return":
        raise NotImplementedError("expected func.return at the end of @main")
    for operation in operations:
        if len(operation.regions):
            raise NotImplementedError(
                f"nested regions are not supported: {operation.name}"
            )
        if operation.name == "func.return" and operation != operations[-1]:
            raise NotImplementedError("unexpected early func.return in @main")
        if operation.name != "func.return" and not operation.name.startswith("torch."):
            raise NotImplementedError(f"unsupported source operation: {operation.name}")
    return blocks[0], operations


def _driver(block, operations, clonable, segment_io):
    """Replace each run by a call while retaining mutation and alias boundaries.

    Declarations make calls verifiable without embedding segment bodies. Runs
    are called immediately before their following boundary (or the return),
    so all their inputs are available and mutation ordering remains intact.
    """
    driver = ir.Module.create()
    with ir.InsertionPoint(driver.body):
        for index, (live_in, live_out) in segment_io.items():
            func.FuncOp(
                f"segment_{index}",
                ir.FunctionType.get(
                    [v.type for v in live_in], [v.type for v in live_out]
                ),
                visibility="private",
            )
        main = func.FuncOp(
            "main",
            ir.FunctionType.get(
                [v.type for v in block.arguments],
                [v.type for v in operations[-1].operands],
            ),
        )
        entry = main.add_entry_block()
    mapping = dict(zip(block.arguments, entry.arguments, strict=True))

    def emit_call(index):
        """Map a segment's results back to the corresponding source values."""
        if index not in segment_io:
            return
        live_in, live_out = segment_io[index]
        call = ir.Operation.create(
            "func.call",
            operands=[mapping[value] for value in live_in],
            results=[value.type for value in live_out],
            attributes={"callee": ir.FlatSymbolRefAttr.get(f"segment_{index}")},
        )
        mapping.update(zip(live_out, call.results, strict=True))

    index = 0
    with ir.InsertionPoint(entry):
        for operation in operations:
            if operation.name == "func.return":
                emit_call(index)
                func.ReturnOp([mapping[value] for value in operation.operands])
            elif operation.results and operation.results[0] in clonable:
                _clone(operation, mapping, rewrite_device=False)
            elif _is_boundary(operation):
                emit_call(index)
                _clone(operation, mapping, rewrite_device=False)
                index += 1
    return driver


def partition_at_custom_ops(module: ir.Module) -> PartitionResult:
    """Partition @main at custom, aliasing and mutating operations.

    Each maximal ordinary-op run becomes a public segment function. Constants
    and lists made entirely from constants are duplicated in runs that use them;
    all other dependencies cross segment interfaces as SSA values. The driver
    retains alias-sensitive operators and custom calls in their original order.
    Unsupported live-out types are recorded and prevent driver construction.
    """
    block, operations = _source_block(module)
    groups = [[]]
    boundaries = []
    clonable_ops = []
    clonable = {}
    devices = set()

    for operation in operations:
        name = operation.name
        if name == "func.return":
            continue
        if _constant(name) or (
            name == "torch.prim.ListConstruct"
            and all(value in clonable for value in operation.operands)
        ):
            if _constant(name) and operation.operands:
                raise NotImplementedError(f"constant with operands: {name}")
            if name == "torch.constant.device":
                devices.add(ir.StringAttr(operation.attributes["value"]).value)
            for value in operation.results:
                clonable[value] = len(clonable_ops)
            clonable_ops.append(operation)
        elif _is_boundary(operation):
            boundaries.append(operation)
            groups.append([])
        else:
            groups[-1].append(operation)
    if len(devices) > 1:
        raise NotImplementedError(
            f"multiple source devices are unsupported: {sorted(devices)}"
        )
    produced = {value for group in groups for op in group for value in op.results}

    # Consumers distinguish local uses from values that must survive a call.
    # A custom call, later segment, or function return is outside this run.
    consumers = {}
    for index, group in enumerate(groups):
        for operation in group:
            for value in operation.operands:
                consumers.setdefault(value, set()).add(("segment", index))
    for index, operation in enumerate(boundaries):
        for value in operation.operands:
            consumers.setdefault(value, set()).add(("boundary", index))
    for value in operations[-1].operands:
        consumers.setdefault(value, set()).add(("return", None))

    segment_io = {}
    segments = []
    with module.context, ir.Location.unknown():
        compiled = ir.Module.create()

        def need_constant(value, needed):
            """Collect a constant-built list and its transitive dependencies."""
            position = clonable[value]
            if position in needed:
                return
            needed.add(position)
            for operand in clonable_ops[position].operands:
                need_constant(operand, needed)

        for index, group in enumerate(groups):
            local = {value for operation in group for value in operation.results}
            live_in = []
            needed = set()

            for operation in group:
                for value in operation.operands:
                    if value in clonable:
                        need_constant(value, needed)
                    elif value not in local and value not in live_in:
                        if not _supported(value.type) and value not in produced:
                            raise NotImplementedError(
                                f"unsupported live-in for segment_{index}: {value.type}"
                            )
                        live_in.append(value)

            live_out = []
            unsupported = []
            for operation in group:
                for value in operation.results:
                    if not consumers.get(value, set()) - {("segment", index)}:
                        continue
                    if _supported(value.type):
                        live_out.append(value)
                    else:
                        unsupported.append(str(value.type))
            segments.append(
                SegmentInfo(
                    index=index,
                    func_name=f"segment_{index}",
                    live_in_types=[str(v.type) for v in live_in],
                    live_out_types=[str(v.type) for v in live_out],
                    unsupported_live_outs=unsupported,
                )
            )
            if not group:
                continue
            segment_io[index] = (live_in, live_out)
            with ir.InsertionPoint(compiled.body):
                function = func.FuncOp(
                    f"segment_{index}",
                    ir.FunctionType.get(
                        [v.type for v in live_in], [v.type for v in live_out]
                    ),
                )
                entry = function.add_entry_block()
            mapping = dict(zip(live_in, entry.arguments, strict=True))
            with ir.InsertionPoint(entry):
                for position in sorted(needed):
                    _clone(clonable_ops[position], mapping, rewrite_device=True)
                for operation in group:
                    _clone(operation, mapping, rewrite_device=True)
                func.ReturnOp([mapping[value] for value in live_out])

        driver = None
        if not any(segment.unsupported_live_outs for segment in segments):
            driver = _driver(block, operations, clonable, segment_io)
    return PartitionResult(
        module=compiled,
        driver=driver,
        device=next(iter(devices), None),
        segments=segments,
    )
