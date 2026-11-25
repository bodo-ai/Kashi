"""Opt-in FX to IREE CUDA compilation and opaque-kernel graph reconstruction."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from iree.compiler import ir

from kashi import _native
from kashi.importer import import_fx
from kashi.iree_reassembly import IreeKernel, flow_segment_handlers
from kashi.mlir_to_fx import rebuild_fx


@dataclass(frozen=True)
class IreeCompilation:
    """An IREE-reassembled FX graph and the artifacts from its compilation.

    ``graph`` preserves the original graph's inputs and output pytree but its
    IREE kernel calls are opaque and cannot execute. ``artifact_dir`` contains
    the partitioned, driver and Flow MLIR, compiler log, VM bytecode and raw PTX.
    ``kernel_manifest`` maps Flow exports to their linked PTX entry points.
    ``kernels_ptx`` is a combined readable listing, or None if no PTX was emitted.
    """

    graph: torch.fx.GraphModule
    artifact_dir: Path
    flow_ir: Path
    ptx_files: tuple[Path, ...]
    kernels_ptx: Path | None
    kernel_manifest: Path


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


def _annotate_ptx_metadata(
    graph: torch.fx.GraphModule,
    artifact_dir: Path,
    manifest: dict[str, dict[str, dict[str, str]]],
) -> None:
    """Attach the linked PTX image and entry to each opaque kernel call."""
    for node in graph.graph.nodes:
        target = node.target
        if not isinstance(target, IreeKernel):
            continue
        try:
            kernel = manifest[target.executable][target.export]
        except KeyError as error:
            raise RuntimeError(f"Missing linked CUDA kernel: {target}") from error
        node.meta["iree_ptx_file"] = artifact_dir / kernel["file"]
        node.meta["iree_ptx_entry"] = kernel["entry"]


def compile_iree_fx(
    gm: torch.fx.GraphModule,
    *,
    output_dir: str | os.PathLike,
    cuda_target: str | None = None,
    decompose_custom_ops: bool = False,
) -> IreeCompilation:
    """Compile an FX graph to CUDA PTX and reconstruct its opaque-kernel FX graph.

    Args:
        gm: Graph with Dynamo/FakeTensor metadata suitable for ``import_fx``.
        output_dir: Parent directory in which a unique artifact directory is created.
        cuda_target: IREE CUDA architecture such as ``sm_80``. If omitted, use
            ``IREE_CUDA_TARGET`` or the current CUDA device's capability.
        decompose_custom_ops: Rewrite supported vLLM custom ops into ATen ops
            before import. Their MLIR locations name the original call, and the
            MLIR artifacts are written with locations so IREE sees them.

    Returns:
        The reassembled graph and paths to the flow IR and PTX artifacts. Kernel
        targets are intentionally opaque: attempting to execute them raises.

    Raises:
        NotImplementedError: If a segment cannot be represented or rebuilt.
        RuntimeError: If IREE compilation or its expected phase dump fails.
    """
    imported = import_fx(gm, decompose_custom_ops=decompose_custom_ops)
    if not imported.module.operation.verify():
        raise RuntimeError(
            f"Verification failed for imported module:\n{imported.module}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix="kashi-iree-", dir=output_dir))
    module = imported.module.operation
    partition_device = _native.compile_iree(
        module.get_asm(enable_debug_info=True),
        artifact_dir,
        target="cuda",
        cuda_target=_cuda_target(cuda_target),
    )
    device = _source_device(gm, partition_device)
    flow_ir = artifact_dir / "flow.mlir"
    kernel_manifest = artifact_dir / "kernel_manifest.json"
    manifest = json.loads(kernel_manifest.read_text())
    ptx_files = tuple(sorted(artifact_dir.glob("kernel_*.ptx")))
    kernels_ptx = None
    if ptx_files:
        kernels_ptx = artifact_dir / "kernels.ptx"
        kernels_ptx.write_text(
            "\n".join(f"// {path.name}\n{path.read_text()}" for path in ptx_files)
        )

    handlers = flow_segment_handlers(flow_ir.read_text(), device=device)
    driver = ir.Module.parse((artifact_dir / "driver.mlir").read_text(), ir.Context())
    graph = rebuild_fx(driver, imported.out_spec, segment_handlers=handlers)
    _annotate_ptx_metadata(graph, artifact_dir, manifest)
    return IreeCompilation(
        graph, artifact_dir, flow_ir, ptx_files, kernels_ptx, kernel_manifest
    )
