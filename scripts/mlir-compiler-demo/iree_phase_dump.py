"""Partition a Torch-MLIR module at vLLM custom ops so IREE can compile the rest.

IREE cannot ingest `torch.operator` custom ops, and the ops that carry the
dataflow (in-place `Tensor!` kernels) cannot simply be dropped. This module
splits a copy of the module at those ops: every maximal run of ordinary ops
becomes its own `segment_<N>` function (live-ins as arguments, live-outs as
results) and the custom ops stay outside, to become opaque FX calls later.

Besides the module IREE compiles, the partitioner builds a *driver* module (the
source `@main` with each segment collapsed into a call) that keeps the wiring
between segments and custom ops. `run_iree_compile` compiles the segments,
dumping per-phase IR and the generated kernels, and `dump_iree_phases` ties the
steps together. Rebuilding an FX graph from the results is
`iree_fx_reassembly`'s job.
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch_mlir import ir
from torch_mlir.dialects import func as func_d
from torch_mlir.dialects import torch as torch_d

CUSTOM_OP = "torch.operator"
RETURN_OP = "func.return"
NON_COMPUTE_OPS = {"torch.prim.ListConstruct", "torch.prim.ListUnpack", RETURN_OP}
VIEW_OPS = {
    "torch.aten.view",
    "torch.aten.reshape",
    "torch.aten.unsqueeze",
    "torch.aten.squeeze",
    "torch.aten.transpose.int",
    "torch.aten.permute",
    "torch.aten.flatten.using_ints",
    "torch.aten.expand",
}


@dataclass
class SegmentInfo:
    """Describe one maximal ordinary-op run between custom-op boundaries.

    Attributes:
        index: Position of the run among all custom-op-separated segments.
        func_name: Name assigned to the emitted segment function.
        op_names: Original operation names in graph order.
        compute_op_count: Operations counted as compute, excluding constants
            and list/return plumbing.
        live_in_types: Types passed into the segment from outside it.
        live_out_types: Supported types returned to external consumers.
        unsupported_live_outs: Types observed crossing the boundary but not
            emitted as results.
        view_live_outs: Count of view-like live-outs aliasing another live-out.
        before_op: Custom op immediately before this segment, if any.
        after_op: Custom op immediately after this segment, if any.
    """
    index: int
    func_name: str
    op_names: list[str]
    compute_op_count: int
    live_in_types: list[str]
    live_out_types: list[str]
    unsupported_live_outs: list[str]
    view_live_outs: int
    before_op: str | None
    after_op: str | None


@dataclass
class BoundaryInfo:
    """Describe a retained custom-op boundary and its Torch schema metadata.

    `mutated_operands` is `None` when no registered schema was available.
    """
    index: int
    name: str
    operand_types: list[str]
    result_types: list[str]
    mutated_operands: list[int] | None


@dataclass
class PartitionResult:
    """Hold the partitioned module, per-segment metadata, and operation totals.

    Attributes:
        module: Copy containing only ordinary-op segments; this is what IREE
            compiles. Custom operations are not emitted into it.
        segments / boundaries: Per-segment and per-custom-op analysis.
        total_ops / total_compute_ops: Operation totals of the source function.
        driver: Torch-MLIR module whose `@main` is the source function with each
            segment collapsed into a `func.call @segment_<N>` and every custom
            op/constant kept in its original place. It records the wiring
            between segments and custom ops that `module` alone loses, and is
            `None` when some cross-segment value cannot be passed through a
            segment signature (see `SegmentInfo.unsupported_live_outs`).
        device: The CUDA device string (e.g. `cuda:0`) the source graph used,
            which the CPU rewrite in `_clone` hides from the compiled segments.
    """
    module: ir.Module
    segments: list[SegmentInfo] = field(default_factory=list)
    boundaries: list[BoundaryInfo] = field(default_factory=list)
    total_ops: int = 0
    total_compute_ops: int = 0
    driver: ir.Module | None = None
    device: str | None = None


def copy_module(module: ir.Module) -> ir.Module:
    """Reparse an MLIR module into a fresh context to isolate later edits.

    Args:
        module: Source module to copy.

    Returns:
        An independent module; operations in the caller's module are unchanged.
    """
    context = ir.Context()
    torch_d.register_dialect(context)
    return ir.Module.parse(str(module), context)


def type_kind(type_text: str) -> str:
    """Classify a Torch-MLIR type string for boundary-traffic statistics.

    Args:
        type_text: Printed Torch-MLIR type.

    Returns:
        One of `tensor`, `int`, `list`, or `other`.
    """
    if type_text.startswith(("!torch.vtensor", "!torch.tensor")):
        return "tensor"
    if type_text == "!torch.int":
        return "int"
    if type_text.startswith("!torch.list"):
        return "list"
    return "other"


def _is_constant(name: str) -> bool:
    return name.startswith("torch.constant.") or name == "torch.vtensor.literal"


def _is_compute(name: str) -> bool:
    return not (_is_constant(name) or name in NON_COMPUTE_OPS)


def _string_attr(operation, key: str) -> str:
    return ir.StringAttr(operation.attributes[key]).value


def _clone(operation, mapping, *, rewrite_device=True):
    """Clone `operation` at the current insertion point, remapping operands.

    `mapping` translates source values to their clones and is extended with
    the new results. `rewrite_device` applies the CUDA-to-CPU device-constant
    workaround below; the driver, which IREE never sees, keeps the original.
    """
    attributes = {key: operation.attributes[key] for key in operation.attributes}
    # `torch.aten.empty.memory_format` in this IREE frontend cannot be
    # legalized when its device operand is vLLM's `cuda:N` value. Normalize
    # only the disposable partition copy; the IREE HAL target is selected
    # separately as CUDA in `run_iree_compile`. This is a compile-time
    # workaround, not a claim about runtime device semantics.
    if rewrite_device and operation.name == "torch.constant.device":
        if _string_attr(operation, "value").startswith("cuda"):
            attributes["value"] = ir.StringAttr.get("cpu")
    cloned = ir.Operation.create(
        operation.name,
        results=[result.type for result in operation.results],
        operands=[mapping[operand] for operand in operation.operands],
        attributes=attributes,
    )
    for old, new in zip(operation.results, cloned.results):
        mapping[old] = new


def _build_driver(block, return_op, clonable_index, segment_io) -> ir.Module:
    """Build the driver module: the source `@main` with segments as calls.

    Args:
        block: Entry block of the source function.
        return_op: The source `func.return`.
        clonable_index: Values defined by constants / constant-built lists.
        segment_io: `{segment index: (live_in values, live_out values)}` for
            every non-empty segment.

    Segment `g` holds the ordinary ops between custom op `g - 1` and custom op
    `g`, so its call is emitted right before custom op `g` (or the return):
    every live-in is defined by then, and every consumer comes after. Custom
    ops and constants are cloned unchanged, so the later MLIR-to-FX conversion
    reproduces them exactly as it does for the unpartitioned graph.
    """
    driver = ir.Module.create()
    with ir.InsertionPoint(driver.body):
        # Private declarations keep driver.mlir verifiable on its own (a
        # `func.call` must resolve its callee) although the bodies live in the
        # separate module handed to IREE.
        for g, (live_in, live_out) in segment_io.items():
            func_d.FuncOp(
                f"segment_{g}",
                ir.FunctionType.get([v.type for v in live_in], [v.type for v in live_out]),
                visibility="private",
            )
        main = func_d.FuncOp(
            "main",
            ir.FunctionType.get(
                [arg.type for arg in block.arguments],
                [v.type for v in return_op.operands],
            ),
        )
        entry = main.add_entry_block()
    mapping = dict(zip(block.arguments, entry.arguments))

    def emit_segment_call(g):
        if g not in segment_io:  # empty segment: nothing was sent to IREE
            return
        live_in, live_out = segment_io[g]
        call = ir.Operation.create(
            "func.call",
            results=[v.type for v in live_out],
            operands=[mapping[v] for v in live_in],
            attributes={"callee": ir.FlatSymbolRefAttr.get(f"segment_{g}")},
        )
        for old, new in zip(live_out, call.results):
            mapping[old] = new

    custom_ops_seen = 0
    with ir.InsertionPoint(entry):
        for op in block.operations:
            operation = op.operation
            if operation.name == RETURN_OP:
                emit_segment_call(custom_ops_seen)
                func_d.ReturnOp([mapping[v] for v in operation.operands])
            elif operation.name == CUSTOM_OP:
                emit_segment_call(custom_ops_seen)
                _clone(operation, mapping, rewrite_device=False)
                custom_ops_seen += 1
            elif operation.results and operation.results[0] in clonable_index:
                _clone(operation, mapping, rewrite_device=False)
            # Every other op lives inside a segment and is represented by a call.
    return driver


def partition_at_custom_ops(module: ir.Module) -> PartitionResult:
    """Split the entry-block graph into maximal runs separated by custom ops.

    Args:
        module: A Torch-MLIR module containing a single-block `func.func`.

    Returns:
        A new module with ordinary operations in `segment_<N>` functions,
        plus metadata describing custom-op boundaries and cross-segment values.

    Constants and constant-built lists are cloned into each segment that uses
    them. Tensor/int/list values crossing a segment boundary become function
    arguments or results; custom ops remain only in the analysis metadata.
    """
    source = copy_module(module)
    main = next(
        op for op in source.body.operations if op.operation.name == "func.func"
    )
    block = main.regions[0].blocks[0]

    groups: list[list] = [[]]
    custom_ops: list = []
    return_op = None
    clonable_ops: list = []
    clonable_index: dict = {}
    producer: dict = {arg: ("arg", None) for arg in block.arguments}
    segment_io: dict = {}  # segment index -> (live-in values, live-out values)
    source_device = None

    # Preserve graph order: each custom op closes the preceding run and starts
    # the next, while constants are cloned later into whichever runs need them.
    for op in block.operations:
        operation = op.operation
        name = operation.name
        if name == RETURN_OP:
            return_op = operation
        elif name == CUSTOM_OP:
            custom_ops.append(operation)
            groups.append([])
            for result in operation.results:
                producer[result] = ("custom", len(custom_ops) - 1)
        elif _is_constant(name) or (
            name == "torch.prim.ListConstruct"
            and all(operand in clonable_index for operand in operation.operands)
        ):
            if name == "torch.constant.device" and source_device is None:
                device_value = _string_attr(operation, "value")
                if device_value.startswith("cuda"):
                    source_device = device_value
            for result in operation.results:
                clonable_index[result] = len(clonable_ops)
            clonable_ops.append(operation)
        else:
            groups[-1].append(operation)
            for result in operation.results:
                producer[result] = ("seg", len(groups) - 1)

    # A value is live out when any consumer is outside its defining segment,
    # including a custom op or the original function return.
    consumers: dict = {}
    for g, group in enumerate(groups):
        for operation in group:
            for operand in operation.operands:
                consumers.setdefault(operand, set()).add(("seg", g))
    for b, operation in enumerate(custom_ops):
        for operand in operation.operands:
            consumers.setdefault(operand, set()).add(("custom", b))
    if return_op is not None:
        for operand in return_op.operands:
            consumers.setdefault(operand, set()).add(("return", None))

    with source.context, ir.Location.unknown():
        new_module = ir.Module.create()
        result = PartitionResult(module=new_module)
        for b, operation in enumerate(custom_ops):
            result.boundaries.append(
                BoundaryInfo(
                    index=b,
                    name=_string_attr(operation, "name"),
                    operand_types=[str(v.type) for v in operation.operands],
                    result_types=[str(r.type) for r in operation.results],
                    mutated_operands=mutated_operand_indices(_string_attr(operation, "name")),
                )
            )

        # Resolve each segment's interface before cloning it; constants are local
        # to the segment rather than passed across the boundary.
        for g, group in enumerate(groups):
            local = {r for operation in group for r in operation.results}
            live_in: list = []
            needed: set[int] = set()

            # Constants can depend on other clonable constants (notably list builds),
            # so recursively collect the full dependency chain used by this segment.
            def need_clonable(value):
                index = clonable_index[value]
                if index in needed:
                    return
                needed.add(index)
                for operand in clonable_ops[index].operands:
                    need_clonable(operand)

            for operation in group:
                for operand in operation.operands:
                    if operand in clonable_index:
                        need_clonable(operand)
                    elif operand not in local and operand not in live_in:
                        live_in.append(operand)

            # Only values crossing this segment boundary need to appear in its result
            # type; scalar-like types the Torch function interface cannot return are
            # reported separately below.
            live_out: list = []
            defining: dict = {}
            unsupported: list[str] = []
            for operation in group:
                for r in operation.results:
                    outside = consumers.get(r, set()) - {("seg", g)}
                    if not outside:
                        continue
                    if type_kind(str(r.type)) == "other":
                        unsupported.append(str(r.type))
                    else:
                        live_out.append(r)
                        defining[r] = operation
            # A view result can alias a value that also crosses the boundary; count
            # these separately because they affect buffer traffic interpretation.
            view_live_outs = sum(
                1
                for r in live_out
                if defining[r].name in VIEW_OPS and defining[r].operands[0] in live_out
            )
            if group:
                segment_io[g] = (live_in, live_out)

            func_name = f"segment_{g}"
            if group:
                with ir.InsertionPoint(new_module.body):
                    func_op = func_d.FuncOp(
                        func_name,
                        ir.FunctionType.get(
                            [v.type for v in live_in], [v.type for v in live_out]
                        ),
                    )
                    entry = func_op.add_entry_block()
                mapping = dict(zip(live_in, entry.arguments))
                with ir.InsertionPoint(entry):
                    for index in sorted(needed):
                        _clone(clonable_ops[index], mapping)
                    for operation in group:
                        _clone(operation, mapping)
                    func_d.ReturnOp([mapping[v] for v in live_out])

            result.segments.append(
                SegmentInfo(
                    index=g,
                    func_name=func_name,
                    op_names=[operation.name for operation in group],
                    compute_op_count=sum(
                        _is_compute(operation.name) for operation in group
                    ),
                    live_in_types=[str(v.type) for v in live_in],
                    live_out_types=[str(v.type) for v in live_out],
                    unsupported_live_outs=unsupported,
                    view_live_outs=view_live_outs,
                    before_op=result.boundaries[g - 1].name if g > 0 else None,
                    after_op=result.boundaries[g].name if g < len(custom_ops) else None,
                )
            )

        # The driver is only valid if every value that crosses a segment boundary
        # could be passed through a segment signature.
        if return_op is not None and not any(
            info.unsupported_live_outs for info in result.segments
        ):
            result.driver = _build_driver(block, return_op, clonable_index, segment_io)
        result.device = source_device

    all_ops = [op.operation for op in block.operations if op.operation.name != RETURN_OP]
    result.total_ops = len(all_ops)
    result.total_compute_ops = sum(_is_compute(operation.name) for operation in all_ops)
    return result


def mutated_operand_indices(mlir_op_name: str) -> list[int] | None:
    """Return writable operand indices from the registered Torch schema.

    Args:
        mlir_op_name: Fully qualified custom-op name such as
            `torch._C.rms_norm`.

    Returns:
        A list of zero-based writable operand indices, or `None` when the
        schema cannot be discovered so unknown mutation remains explicit.
    """
    namespace, _, op_name = mlir_op_name.removeprefix("torch.").partition(".")
    try:
        schema = getattr(getattr(torch.ops, namespace), op_name).default._schema
    except (AttributeError, RuntimeError):
        return None
    return [
        index
        for index, argument in enumerate(schema.arguments)
        if argument.alias_info is not None and argument.alias_info.is_write
    ]


def compute_partition_stats(result: PartitionResult) -> dict:
    """Summarize partition fragmentation and values crossing boundaries.

    Args:
        result: Partition analysis produced by `partition_at_custom_ops`.

    Returns:
        A JSON-serializable dictionary of operation, segment, and traffic stats.
    """
    segments = result.segments
    op_counts = [len(segment.op_names) for segment in segments]
    compute_in_segments = sum(segment.compute_op_count for segment in segments)
    custom_ops_by_name: dict[str, int] = {}
    for boundary in result.boundaries:
        custom_ops_by_name[boundary.name] = custom_ops_by_name.get(boundary.name, 0) + 1
    return {
        "total_ops": result.total_ops,
        "total_compute_ops": result.total_compute_ops,
        "custom_ops_by_name": custom_ops_by_name,
        "segment_count": len(segments),
        "ops_per_segment": {
            "min": min(op_counts),
            "median": statistics.median(op_counts),
            "mean": round(statistics.mean(op_counts), 2),
            "max": max(op_counts),
        },
        "empty_segments": sum(segment.compute_op_count == 0 for segment in segments),
        "compute_ops_in_segments_pct": (
            round(100.0 * compute_in_segments / result.total_compute_ops, 1)
            if result.total_compute_ops
            else 0.0
        ),
        "boundary_traffic": {
            "live_in_values": sum(len(s.live_in_types) for s in segments),
            "live_out_values": sum(len(s.live_out_types) for s in segments),
            "distinct_tensors_crossing": sum(
                type_kind(t) == "tensor" for s in segments for t in s.live_out_types
            ),
        },
        "distinct_segment_structures": len({tuple(s.op_names) for s in segments}),
        "unsupported_live_outs": sum(len(s.unsupported_live_outs) for s in segments),
        "view_live_outs": sum(s.view_live_outs for s in segments),
    }


def _kind_counts(types: list[str]) -> dict[str, int]:
    counts = {"tensor": 0, "int": 0, "list": 0, "other": 0}
    for type_text in types:
        counts[type_kind(type_text)] += 1
    return counts


def write_partition_dump(result: PartitionResult, out_dir: Path) -> dict:
    """Write the partitioned module, driver, boundary/segment reports, and JSON stats.

    Args:
        result: Partition module and analysis to serialize.
        out_dir: Directory receiving `partitioned.mlir`, `driver.mlir` (when
            the driver could be built), and the report files.

    Returns:
        The statistics dictionary also written to `partition_stats.json`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_partition_stats(result)

    (out_dir / "partitioned.mlir").write_text(str(result.module))
    if result.driver is not None:
        (out_dir / "driver.mlir").write_text(str(result.driver))
    (out_dir / "boundaries.txt").write_text(
        "".join(
            f"{b.index} {b.name} operands={b.operand_types} results={b.result_types} "
            f"mutated_operands={'unknown' if b.mutated_operands is None else b.mutated_operands}\n"
            for b in result.boundaries
        )
    )
    (out_dir / "segments.txt").write_text(
        "".join(
            f"{s.func_name}: ops={len(s.op_names)} compute={s.compute_op_count} "
            f"live_in={_kind_counts(s.live_in_types)} live_out={_kind_counts(s.live_out_types)} "
            f"unsupported={len(s.unsupported_live_outs)} views={s.view_live_outs} "
            f"before={s.before_op} after={s.after_op}\n"
            for s in result.segments
        )
    )
    (out_dir / "partition_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True)
    )

    traffic = stats["boundary_traffic"]
    per_segment = stats["ops_per_segment"]
    print(
        f"IREE partition: {stats['segment_count']} segments, "
        f"{len(result.boundaries)} custom-op boundaries, "
        f"{stats['compute_ops_in_segments_pct']}% of compute ops in segments"
    )
    print(
        f"  ops per segment: min={per_segment['min']} median={per_segment['median']} "
        f"mean={per_segment['mean']} max={per_segment['max']}"
    )
    print(
        f"  empty segments: {stats['empty_segments']}, "
        f"distinct segment structures: {stats['distinct_segment_structures']}"
    )
    print(
        f"  boundary traffic: live-in values={traffic['live_in_values']} "
        f"live-out values={traffic['live_out_values']} "
        f"distinct tensors crossing={traffic['distinct_tensors_crossing']}"
    )
    print(
        f"  unsupported live-outs: {stats['unsupported_live_outs']}, "
        f"view live-outs: {stats['view_live_outs']}"
    )
    return stats


def cuda_target_from_capability(capability: tuple[int, int]) -> str:
    """Format a CUDA compute-capability pair as an IREE target such as `sm_89`.

    Args:
        capability: Major/minor CUDA compute capability.

    Returns:
        IREE's `sm_<major><minor>` target string.
    """
    return f"sm_{capability[0]}{capability[1]}"


@dataclass
class IreeRunResult:
    """Report one IREE compile attempt without raising on expected failures.

    Attributes:
        ok: Whether the compile succeeded.
        message: Human-readable outcome or failure detail.
        phases_dir: Phase dump directory, when IREE created one.
        log_path: Captured compiler log, when the compiler ran.
        kernels_path: Consolidated PTX listing of every generated kernel, when
            the compile succeeded and produced at least one kernel.
    """
    ok: bool
    message: str
    phases_dir: Path | None
    log_path: Path | None
    kernels_path: Path | None = None


# IREE names each unlinked CUDA executable's PTX `module_<executable>_cuda_nvptx_fb.ptx`.
_PTX_FILE = re.compile(r"module_(?P<executable>.+)_cuda_nvptx_fb\.ptx")


def write_kernel_listing(kernel_dir: Path, output_path: Path) -> Path | None:
    """Concatenate the per-executable PTX files IREE dumped into one readable file.

    Args:
        kernel_dir: Directory written by `--iree-hal-dump-executable-files-to`.
        output_path: Listing to create.

    Returns:
        `output_path`, or `None` if IREE generated no kernels (e.g. a module
        with no dispatches). Executables are ordered by their numeric indices
        (`segment_2` before `segment_10`) and each is introduced by a banner
        naming the executable, which is the name the FX kernel calls carry.
    """
    executables = []
    for ptx_path in Path(kernel_dir).glob("module_*.ptx"):
        match = _PTX_FILE.fullmatch(ptx_path.name)
        if match:
            name = match.group("executable")
            executables.append(([int(n) for n in re.findall(r"\d+", name)], name, ptx_path))
    if not executables:
        return None

    banner = "// " + "=" * 76
    sections = [
        f"{banner}\n// IREE executable: {name}\n// source file: {ptx_path.name}\n{banner}\n"
        f"{ptx_path.read_text()}"
        for _, name, ptx_path in sorted(executables)
    ]
    Path(output_path).write_text("\n".join(sections))
    return Path(output_path)


def run_iree_compile(
    mlir_path: Path,
    out_dir: Path,
    *,
    compiler: str | None = None,
    cuda_target: str | None = None,
) -> IreeRunResult:
    """Compile a partition with IREE, dumping per-phase IR and generated kernels.

    The whole pipeline runs (not just up to `executable-targets`) because
    kernels only exist once IREE serializes each executable to PTX.
    `--iree-hal-link-executables=false` keeps one PTX per executable so each can
    be matched to the `flow.dispatch` exports that call it. The compiled binary
    itself is discarded; only the dumps are kept.

    Args:
        mlir_path: Input partitioned MLIR module.
        out_dir: Directory for the compiler log, `phases/`, `kernel_files/`
            (raw IREE dumps) and `kernels.ptx` (consolidated listing).
        compiler: Optional `iree-compile` path; otherwise use the environment
            override or `PATH`.
        cuda_target: Optional `sm_XX` target; otherwise use the environment
            override or the current CUDA device's compute capability.

    Returns:
        An `IreeRunResult`; compile and setup errors are reported in the result
        instead of raised so the caller can keep the FX round trip alive.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "iree-compile.log"
    phases_dir = out_dir / "phases"
    kernel_dir = out_dir / "kernel_files"

    compiler = compiler or os.environ.get("IREE_COMPILE") or shutil.which("iree-compile")
    if not compiler:
        return IreeRunResult(
            False, "iree-compile not found; set IREE_COMPILE or add it to PATH", None, None
        )
    target = cuda_target or os.environ.get("IREE_CUDA_TARGET")
    if not target:
        try:
            target = cuda_target_from_capability(torch.cuda.get_device_capability())
        except Exception as error:
            return IreeRunResult(
                False, f"cannot determine CUDA target ({error}); set IREE_CUDA_TARGET", None, None
            )

    command = [
        compiler,
        str(mlir_path),
        "--iree-hal-target-device=cuda",
        f"--iree-cuda-target={target}",
        "--iree-hal-link-executables=false",
        f"--iree-hal-dump-executable-files-to={kernel_dir}",
        f"--dump-compilation-phases-to={phases_dir}",
        "-o",
        os.devnull,
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return IreeRunResult(False, f"cannot run {compiler}: {error}", None, None)

    log_path.write_text(completed.stdout)
    dumps = phases_dir if phases_dir.exists() else None
    if completed.returncode != 0:
        return IreeRunResult(
            False,
            f"iree-compile exited with {completed.returncode}; see {log_path}",
            dumps,
            log_path,
        )
    kernels_path = write_kernel_listing(kernel_dir, out_dir / "kernels.ptx")
    return IreeRunResult(True, f"compiled for {target}", dumps, log_path, kernels_path)


def add_iree_stats(stats_path: Path, phases_dir: Path) -> dict:
    """Add static dispatch-site and executable counts from IREE phase dumps.

    Args:
        stats_path: Existing partition-statistics JSON file to update.
        phases_dir: Directory containing numbered IREE phase MLIR dumps.

    Returns:
        Updated statistics. Dispatches count `flow.dispatch` call sites in
        `segment_N$async` functions, not runtime executions.
    """
    stats_path = Path(stats_path)
    stats = json.loads(stats_path.read_text())

    dispatches: dict[str, int] = {}
    flow_files = list(Path(phases_dir).glob("*.6.flow.mlir"))
    if flow_files:
        text = flow_files[0].read_text()
        functions = list(re.finditer(r"util\.func (?:public|private) @([\w$.]+)\(", text))
        for index, match in enumerate(functions):
            end = functions[index + 1].start() if index + 1 < len(functions) else len(text)
            segment = re.fullmatch(r"(segment_\d+)\$async", match.group(1))
            if segment:
                dispatches[segment.group(1)] = len(
                    re.findall(r"\bflow\.dispatch\b", text[match.start() : end])
                )

    unique_executables = 0
    target_files = list(Path(phases_dir).glob("*.10.executable-targets.mlir"))
    if target_files:
        unique_executables = len(
            re.findall(
                r"^\s*hal\.executable (?:private |public )?@",
                target_files[0].read_text(),
                re.MULTILINE,
            )
        )

    stats["iree"] = {
        "unique_executables": unique_executables,
        "dispatches_per_segment": dispatches,
        "total_dispatch_call_sites": sum(dispatches.values()),
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    return stats


@dataclass
class IreeDump:
    """Everything a successful dump run produced, for rebuilding an FX graph.

    Attributes:
        out_dir: Numbered directory holding every artifact of this run.
        partition: Segments, boundaries and the driver module.
        run: The compile outcome, including the kernel listing path.
        flow_ir: The flow-phase IR, where each segment is a function whose
            kernels appear as `flow.dispatch` ops.
    """
    out_dir: Path
    partition: PartitionResult
    run: IreeRunResult
    flow_ir: Path


def dump_iree_phases(module: ir.Module, base_dir: Path) -> IreeDump | None:
    """Partition `module`, save it under the next numbered directory, and compile.

    Args:
        module: Source Torch-MLIR module; partitioning operates on a copy.
        base_dir: Parent directory for numbered dump runs.

    Returns:
        An `IreeDump` when the compile succeeded, else `None`. All failures are
        printed and swallowed so this optional diagnostic path cannot prevent
        the original FX round trip.
    """
    try:
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        numbers = [int(p.name) for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        out_dir = base_dir / str(max(numbers, default=-1) + 1)

        result = partition_at_custom_ops(module)
        write_partition_dump(result, out_dir)
        run = run_iree_compile(out_dir / "partitioned.mlir", out_dir)
        flow_files = list(run.phases_dir.glob("*.6.flow.mlir")) if run.phases_dir else []
        if not (run.ok and flow_files):
            print(f"IREE: {run.message}")
            return None

        stats = add_iree_stats(out_dir / "partition_stats.json", run.phases_dir)
        print(
            f"IREE: {run.message}; phase dumps in {run.phases_dir} "
            f"({stats['iree']['unique_executables']} executables, "
            f"{stats['iree']['total_dispatch_call_sites']} dispatch call-sites); "
            f"kernels in {run.kernels_path}"
        )
        return IreeDump(out_dir, result, run, flow_files[0])
    except Exception:
        print("IREE phase dump failed:")
        print(traceback.format_exc())
        return None
