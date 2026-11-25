"""Opt-in FX to IREE CUDA compilation and executable graph reconstruction."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from iree.compiler import ir

from kashi import _native
from kashi.importer import import_fx
from kashi.iree_reassembly import (
    IreeKernel,
    _segment_functions,
    _segment_output_aliases,
    _segment_result_types,
    _view_root,
    flow_segment_handlers,
)
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
    ``kernel_manifest`` maps Flow exports to their linked PTX entry points.
    """

    graph: torch.fx.GraphModule
    artifact_dir: Path
    flow_ir: Path | None
    ptx_files: tuple[Path, ...]
    kernels_ptx: Path | None
    vmfb_path: Path | None
    execution_mode: str
    kernel_manifest: Path | None


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


def _imported_device(module) -> str | None:
    """Find a device literal before target selection for factory-only graphs."""
    for function in module.body.operations:
        if function.operation.name != "func.func" or not function.regions[0].blocks:
            continue
        for op in function.regions[0].blocks[0].operations:
            if op.operation.name == "torch.constant.device":
                return ir.StringAttr(op.attributes["value"]).value
    return None


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
        node.meta["iree_stable_export"] = kernel["stable"]


def _preserve_segment_results(handlers, functions):
    """Restore source SSA aliases and independent storage for fresh results."""

    def wrap(handler, result_types, aliases):
        """Add copies only when a source-distinct tensor would share storage."""

        def splice(graph, args):
            """Splice a segment and enforce its source-level storage identities."""
            results = list(handler(graph, args))
            seen = {_view_root(arg) for arg in args if isinstance(arg, torch.fx.Node)}
            for index, (value, type_text, alias) in enumerate(
                zip(results, result_types, aliases, strict=True)
            ):
                if alias is not None:
                    results[index] = (
                        args[alias[1]] if alias[0] == "input" else results[alias[1]]
                    )
                    continue
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

    return {
        name: wrap(
            handler,
            _segment_result_types(functions[name]),
            _segment_output_aliases(functions[name]),
        )
        for name, handler in handlers.items()
    }


def compile_iree_fx(
    gm: torch.fx.GraphModule,
    *,
    output_dir: str | os.PathLike,
    cuda_target: str | None = None,
    decompose_custom_ops: bool = False,
) -> IreeCompilation:
    """Compile an FX graph to CUDA PTX and reconstruct an executable FX graph.

    Args:
        gm: Graph with Dynamo/FakeTensor metadata suitable for ``import_fx``.
        output_dir: Parent directory in which a unique artifact directory is created.
        cuda_target: IREE CUDA architecture such as ``sm_80``. If omitted, use
            ``IREE_CUDA_TARGET`` or the current CUDA device's capability.
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix="kashi-iree-", dir=output_dir))
    target_error = None
    target = None
    try:
        device_hint = _source_device(gm, _imported_device(imported.module))
        target = _cuda_target(cuda_target, device_hint)
    except (ValueError, NotImplementedError) as error:
        # A driver-only graph needs neither a unique device nor a CUDA target.
        target_error = error
    try:
        partition_device = _native.compile_iree(
            imported.module.operation.get_asm(enable_debug_info=True),
            artifact_dir,
            target="cuda",
            cuda_target=target,
        )
    except RuntimeError as error:
        if (
            target_error is not None
            and "IREE configuration failed" in str(error)
            and "cuda_target is required only for CUDA"
            in (artifact_dir / "iree-compile.log").read_text()
        ):
            raise target_error from error
        raise
    partitioned = ir.Module.parse(
        (artifact_dir / "partitioned.mlir").read_text(), ir.Context()
    )
    driver = ir.Module.parse((artifact_dir / "driver.mlir").read_text(), ir.Context())
    functions = _segment_functions(partitioned)
    if not functions:
        graph = rebuild_fx(driver, imported.out_spec, segment_handlers={})
        return IreeCompilation(graph, artifact_dir, None, (), None, None, mode, None)

    device = _source_device(gm, partition_device)
    flow_ir = artifact_dir / "flow.mlir"
    vmfb_path = artifact_dir / "module.vmfb"
    kernel_manifest = artifact_dir / "kernel_manifest.json"
    manifest = json.loads(kernel_manifest.read_text())
    ptx_files = tuple(sorted(artifact_dir.glob("kernel_*.ptx")))
    kernels_ptx = None
    if ptx_files:
        kernels_ptx = artifact_dir / "kernels.ptx"
        kernels_ptx.write_text(
            "\n".join(f"// {path.name}\n{path.read_text()}" for path in ptx_files)
        )

    if mode == "cuda":
        from kashi.iree_cuda import create_cuda_kernel

        def kernel_factory(executable, export):
            """Bind a flow dispatch to its linked CUDA launch metadata."""
            return create_cuda_kernel(executable, export, artifact_dir=artifact_dir)

        handlers = flow_segment_handlers(
            flow_ir.read_text(), device=device, kernel_factory=kernel_factory
        )
        handlers = _preserve_segment_results(handlers, functions)
    else:
        from kashi.iree_runtime import runtime_segment_handlers

        handlers = runtime_segment_handlers(vmfb_path, partitioned, device=device)
    graph = rebuild_fx(driver, imported.out_spec, segment_handlers=handlers)
    if mode == "cuda":
        _annotate_ptx_metadata(graph, artifact_dir, manifest)
    return IreeCompilation(
        graph,
        artifact_dir,
        flow_ir,
        ptx_files,
        kernels_ptx,
        vmfb_path,
        mode,
        kernel_manifest,
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
