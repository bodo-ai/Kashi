"""Opt-in FX to IREE CUDA compilation and executable graph reconstruction."""

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

_UNSIGNED_RESULT_DTYPES = {
    "ui8": torch.uint8,
    "ui16": torch.uint16,
    "ui32": torch.uint32,
    "ui64": torch.uint64,
}


@dataclass(frozen=True)
class IreeCompilation:
    """An IREE-reassembled FX graph and the artifacts from its compilation.

    ``graph`` preserves the original graph's inputs and output pytree.
    ``execution_mode`` identifies the CUDA kernel or IREE segment executor.
    ``artifact_dir`` contains partitioned and driver MLIR and, when there are
    compiled segments, the VMFB module, compiler log, phase dumps and raw
    kernel files. ``flow_ir`` and ``vmfb_path`` are None when no segments
    require compilation. ``kernels_ptx`` is None when no PTX was emitted.
    """

    graph: torch.fx.GraphModule
    artifact_dir: Path
    flow_ir: Path | None
    ptx_files: tuple[Path, ...]
    kernels_ptx: Path | None
    vmfb_path: Path | None
    execution_mode: str


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


def _cuda_target(cuda_target: str | None, source_device: str | None = None) -> str:
    """Resolve an explicit target or the source CUDA tensor's architecture."""
    target = cuda_target or os.environ.get("IREE_CUDA_TARGET")
    if target:
        return target
    try:
        device = torch.device(source_device) if source_device else None
        if device is not None and device.type != "cuda":
            device = None
        major, minor = torch.cuda.get_device_capability(device)
    except Exception as error:
        raise ValueError(
            "Set cuda_target or IREE_CUDA_TARGET when no CUDA device is available"
        ) from error
    return f"sm_{major}{minor}"


def _asm(module, with_locations: bool) -> str:
    """Print a module, keeping locations only when they carry decomposition tags."""
    if with_locations:
        return module.operation.get_asm(enable_debug_info=True)
    return str(module)


def _view_root(value):
    """Find the storage-producing FX value beneath IREE's reconstructed views."""
    while (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target in (torch.ops.aten.view.default, torch.ops.aten.view.dtype)
    ):
        value = value.args[0]
    return value


def _preserve_segment_results(handlers, segments):
    """Prevent IREE value deduplication from creating new PyTorch aliases.

    Alias-producing source operators stay in the driver, so distinct tensor
    live-outs of a compiled segment have independent PyTorch storage. IREE can
    nevertheless return the same value, or a view of an input, for more than
    one live-out. Clone only those results whose reconstructed storage roots
    collide with an input or a previous live-out.
    """

    def wrap(handler, result_types):
        """Add copies only when a source-distinct tensor would share storage."""

        def splice(graph, args):
            """Splice a segment and enforce its source-level storage identities."""
            results = list(handler(graph, args))
            seen = {_view_root(arg) for arg in args if isinstance(arg, torch.fx.Node)}
            for index, (value, type_text) in enumerate(
                zip(results, result_types, strict=True)
            ):
                if not type_text.startswith(("!torch.vtensor", "!torch.tensor")):
                    continue
                root = _view_root(value)
                if root in seen:
                    value = graph.call_function(torch.ops.aten.clone.default, (value,))
                    results[index] = value
                    root = value
                seen.add(root)
                # Flow integers are signless; restore the Torch result dtype
                # by reinterpreting the same bits rather than converting them.
                element = type_text.rsplit(",", 1)[-1].removesuffix(">").strip()
                dtype = _UNSIGNED_RESULT_DTYPES.get(element)
                if dtype is not None:
                    results[index] = graph.call_function(
                        torch.ops.aten.view.dtype, (value, dtype)
                    )
            return results

        return splice

    types = {segment.func_name: segment.live_out_types for segment in segments}
    return {name: wrap(handler, types[name]) for name, handler in handlers.items()}


def compile_iree_fx(
    gm: torch.fx.GraphModule,
    *,
    output_dir: str | os.PathLike,
    cuda_target: str | None = None,
    compiler: str | os.PathLike | None = None,
    decompose_custom_ops: bool = False,
) -> IreeCompilation:
    """Compile an FX graph to CUDA PTX and reconstruct an executable FX graph.

    Args:
        gm: Graph with Dynamo/FakeTensor metadata suitable for ``import_fx``.
        output_dir: Parent directory in which a unique artifact directory is created.
        cuda_target: IREE CUDA architecture such as ``sm_80``. If omitted, use
            ``IREE_CUDA_TARGET`` or the current CUDA device's capability.
        compiler: Optional path to ``iree-compile``; by default use the IREE
            package's tool discovery.
        decompose_custom_ops: Rewrite supported vLLM custom ops into ATen ops
            before import. Their MLIR locations name the original call, and the
            MLIR artifacts are written with locations so IREE sees them.

    Returns:
        The executable graph and paths to the flow IR, VMFB and PTX artifacts.
        ``KASHI_IREE_EXECUTION`` selects ``cuda`` (default) or ``iree``.

    Raises:
        NotImplementedError: If a segment cannot be represented or rebuilt.
        RuntimeError: If IREE compilation or its expected phase dump fails.
    """
    mode = os.environ.get("KASHI_IREE_EXECUTION", "cuda")
    if mode not in ("cuda", "iree"):
        raise ValueError(f"KASHI_IREE_EXECUTION must be 'cuda' or 'iree', got {mode!r}")
    imported = import_fx(gm, decompose_custom_ops=decompose_custom_ops)
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix="kashi-iree-", dir=output_dir))
    partitioned = artifact_dir / "partitioned.mlir"
    partitioned.write_text(_asm(partition.module, decompose_custom_ops))
    (artifact_dir / "driver.mlir").write_text(
        _asm(partition.driver, decompose_custom_ops)
    )
    if not list(partition.module.body.operations):
        # A graph made entirely of alias-sensitive/custom operations has no
        # compiled work. Execute its driver directly instead of feeding IREE
        # an empty module, which cannot produce a VMFB.
        graph = rebuild_fx(partition.driver, imported.out_spec, segment_handlers={})
        return IreeCompilation(graph, artifact_dir, None, (), None, None, mode)

    device = _source_device(gm, partition.device)
    target = _cuda_target(cuda_target, device)
    compiler_path = (
        os.fspath(compiler) if compiler is not None else find_tool("iree-compile")
    )
    phases_dir = artifact_dir / "phases"
    vmfb_path = artifact_dir / "module.vmfb"
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
        str(vmfb_path),
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

    if mode == "cuda":
        from kashi.iree_cuda import create_cuda_kernel

        def kernel_factory(executable, export):
            """Bind a flow dispatch to its CUDA executable and launch metadata."""
            return create_cuda_kernel(executable, export, artifact_dir=artifact_dir)

        handlers = flow_segment_handlers(
            flow_ir.read_text(), device=device, kernel_factory=kernel_factory
        )
        handlers = _preserve_segment_results(handlers, partition.segments)
    else:
        from kashi.iree_runtime import runtime_segment_handlers

        # Segments with no tensor live-ins still need the source device.
        partition.device = device
        handlers = runtime_segment_handlers(vmfb_path, partition)
    graph = rebuild_fx(partition.driver, imported.out_spec, segment_handlers=handlers)
    return IreeCompilation(
        graph, artifact_dir, flow_ir, ptx_files, kernels_ptx, vmfb_path, mode
    )


class _IreeBackendCallable:
    """Keep generated artifacts alive as long as Dynamo owns the callable."""

    def __init__(self, compilation: IreeCompilation, workspace):
        """Retain the executable graph and its temporary artifact directory."""
        self.compilation = compilation
        self._workspace = workspace

    def __call__(self, *args, **kwargs):
        """Execute the compiled FX graph with the caller's arguments."""
        return self.compilation.graph(*args, **kwargs)


def compile_iree_backend(gm: torch.fx.GraphModule, example_inputs):
    """Compile an FX graph for ``torch.compile`` or a vLLM backend setting.

    ``KASHI_IREE_EXECUTION`` selects direct CUDA kernels by default or ``iree``
    for VMFB segment calls. Artifacts remain available for the lifetime of the
    returned callable and are removed when that callable is released.
    ``example_inputs`` is provided by Dynamo; input metadata is read from ``gm``.
    """
    workspace = tempfile.TemporaryDirectory(prefix="kashi-iree-backend-")
    try:
        compilation = compile_iree_fx(gm, output_dir=workspace.name)
    except BaseException:
        workspace.cleanup()
        raise
    return _IreeBackendCallable(compilation, workspace)
