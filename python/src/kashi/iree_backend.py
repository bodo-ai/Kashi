"""Opt-in FX to IREE CUDA compilation and opaque-kernel graph reconstruction."""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from iree.compiler.tools.binaries import find_tool

from kashi.importer import import_fx
from kashi.iree_partition import partition_at_custom_ops
from kashi.iree_reassembly import flow_segment_handlers
from kashi.mlir_to_fx import rebuild_fx


@dataclass(frozen=True)
class IreeCompilation:
    """An IREE-reassembled FX graph and the artifacts from its compilation.

    ``graph`` preserves the original graph's inputs and output pytree but its
    IREE kernel calls are opaque and cannot execute. ``artifact_dir`` contains
    the partitioned and driver MLIR, compiler log, phase dumps and raw kernel
    files. ``flow_ir`` and ``ptx_files`` identify the corresponding artifacts;
    ``kernels_ptx`` is a combined readable listing, or None if no PTX was emitted.
    """

    graph: torch.fx.GraphModule
    artifact_dir: Path
    flow_ir: Path
    ptx_files: tuple[Path, ...]
    kernels_ptx: Path | None


def _source_device(
    gm: torch.fx.GraphModule, partition_device: str | None
) -> str | None:
    """Find the source tensor device so flow allocations use the original device."""
    devices = set()
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        value = node.meta.get("val", node.meta.get("example_value"))
        if isinstance(value, torch.Tensor):
            devices.add(str(value.device))
    if partition_device is not None:
        devices.add(partition_device)
    if len(devices) > 1:
        raise NotImplementedError(
            f"IREE reassembly requires one source tensor device, got {sorted(devices)}"
        )
    return next(iter(devices), None)


def _cuda_target(cuda_target: str | None) -> str:
    """Resolve an explicit, environment-provided or current CUDA architecture."""
    target = cuda_target or os.environ.get("IREE_CUDA_TARGET")
    if target:
        return target
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as error:
        raise ValueError(
            "Set cuda_target or IREE_CUDA_TARGET when no CUDA device is available"
        ) from error
    return f"sm_{major}{minor}"


def compile_iree_fx(
    gm: torch.fx.GraphModule,
    *,
    output_dir: str | os.PathLike,
    cuda_target: str | None = None,
    compiler: str | os.PathLike | None = None,
) -> IreeCompilation:
    """Compile an FX graph to CUDA PTX and reconstruct its opaque-kernel FX graph.

    Args:
        gm: Graph with Dynamo/FakeTensor metadata suitable for ``import_fx``.
        output_dir: Parent directory in which a unique artifact directory is created.
        cuda_target: IREE CUDA architecture such as ``sm_80``. If omitted, use
            ``IREE_CUDA_TARGET`` or the current CUDA device's capability.
        compiler: Optional path to ``iree-compile``; by default use the IREE
            package's tool discovery.

    Returns:
        The reassembled graph and paths to the flow IR and PTX artifacts. Kernel
        targets are intentionally opaque: attempting to execute them raises.

    Raises:
        NotImplementedError: If a segment cannot be represented or rebuilt.
        RuntimeError: If IREE compilation or its expected phase dump fails.
    """
    imported = import_fx(gm)
    if not imported.module.operation.verify():
        raise RuntimeError(
            f"Verification failed for imported module:\n{imported.module}"
        )
    partition = partition_at_custom_ops(imported.module)
    if partition.driver is None:
        unsupported = [
            kind
            for segment in partition.segments
            for kind in segment.unsupported_live_outs
        ]
        raise NotImplementedError(
            f"Cannot build IREE driver; unsupported segment live-outs: {unsupported}"
        )
    device = _source_device(gm, partition.device)
    target = _cuda_target(cuda_target)
    compiler_path = (
        os.fspath(compiler) if compiler is not None else find_tool("iree-compile")
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix="kashi-iree-", dir=output_dir))
    partitioned = artifact_dir / "partitioned.mlir"
    partitioned.write_text(str(partition.module))
    (artifact_dir / "driver.mlir").write_text(str(partition.driver))
    phases_dir = artifact_dir / "phases"
    kernels_dir = artifact_dir / "kernel_files"
    command = [
        compiler_path,
        str(partitioned),
        "--iree-hal-target-device=cuda",
        f"--iree-cuda-target={target}",
        "--iree-hal-link-executables=false",
        f"--iree-hal-dump-executable-files-to={kernels_dir}",
        f"--dump-compilation-phases-to={phases_dir}",
        "-o",
        os.devnull,
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"Could not run IREE compiler {compiler_path}: {error}"
        ) from error
    log_path = artifact_dir / "iree-compile.log"
    log_path.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"IREE compiler exited with {completed.returncode}; see {log_path}"
        )

    flow_files = sorted(phases_dir.glob("*.flow.mlir"))
    if len(flow_files) != 1:
        raise RuntimeError(
            f"Expected one IREE flow phase in {phases_dir}, found {len(flow_files)}"
        )
    flow_ir = flow_files[0]
    ptx_files = tuple(sorted(kernels_dir.glob("*.ptx")))
    kernels_ptx = None
    if ptx_files:
        kernels_ptx = artifact_dir / "kernels.ptx"
        kernels_ptx.write_text(
            "\n".join(f"// {path.name}\n{path.read_text()}" for path in ptx_files)
        )

    handlers = flow_segment_handlers(flow_ir.read_text(), device=device)
    graph = rebuild_fx(partition.driver, imported.out_spec, segment_handlers=handlers)
    return IreeCompilation(graph, artifact_dir, flow_ir, ptx_files, kernels_ptx)
