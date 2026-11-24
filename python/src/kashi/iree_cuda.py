"""Launch IREE CUDA executable exports directly on PyTorch's current stream.

The flow phase describes logical arguments, stream lowering describes the
physical binding/constant ABI, and executable-targets describes the launch.
No IREE runtime or host-device copy is involved.
"""

import ctypes
import re
import threading
from pathlib import Path

from kashi import iree_gpu_dispatch as dispatch

_LOCK = threading.RLock()
_DRIVER = None
_MODULES = {}
_FUNCTIONS = {}


def _metadata(directory, executable, export):
    """Validate CUDA PTX against the common flow and stream ABI."""
    metadata = dispatch._metadata(directory, executable, export)
    binding_count, constant_count = metadata[1:3]
    ptx_files = list(
        (directory / "kernel_files").glob(f"module_{executable}_cuda_nvptx_fb.ptx")
    )
    if len(ptx_files) != 1:
        raise RuntimeError(
            f"Expected one CUDA PTX for {executable} in {directory}, found {len(ptx_files)}"
        )
    ptx = ptx_files[0].read_text()
    symbol = export.replace("$", "_")
    match = re.search(rf"\.visible\s+\.entry\s+{re.escape(symbol)}\s*\(", ptx)
    if not match:
        dispatch._unsupported(export, f"PTX entry symbol {symbol}")
    params, _ = dispatch._balanced(ptx, match.end() - 1, "(", ")")
    params = dispatch._split(params)
    if len(params) != binding_count + constant_count:
        dispatch._unsupported(export, "PTX parameter count")
    if any(
        not re.search(r"\.param\s+\.u64\s+\.ptr\s+\.global", p)
        for p in params[:binding_count]
    ) or any(not re.search(r"\.param\s+\.b32\b", p) for p in params[binding_count:]):
        dispatch._unsupported(export, "PTX pointer or constant parameter types")
    aligns = []
    for param in params[:binding_count]:
        align = re.search(r"\.align\s+(\d+)", param)
        aligns.append(int(align.group(1)) if align else 1)
    return metadata, ptx, symbol, aligns


def _driver():
    """Initialize the CUDA driver using the process's existing CUDA context."""
    global _DRIVER
    with _LOCK:
        if _DRIVER is not None:
            return _DRIVER
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            raise RuntimeError(
                "IREE direct CUDA execution requires libcuda.so.1"
            ) from exc
        signatures = {
            "cuInit": ([ctypes.c_uint], ctypes.c_int),
            "cuCtxGetCurrent": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "cuModuleLoadData": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
                ctypes.c_int,
            ),
            "cuModuleGetFunction": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
                ctypes.c_int,
            ),
            "cuFuncSetAttribute": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
                ctypes.c_int,
            ),
            "cuLaunchKernel": (
                [
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.c_uint,
                    ctypes.c_uint,
                    ctypes.c_uint,
                    ctypes.c_uint,
                    ctypes.c_uint,
                    ctypes.c_uint,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
            "cuGetErrorString": (
                [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
                ctypes.c_int,
            ),
        }
        for name, (args, result) in signatures.items():
            func = getattr(lib, name)
            func.argtypes, func.restype = args, result
        _check(lib, "cuInit", lib.cuInit(0))
        _DRIVER = lib
        return lib


def _check(lib, name, status):
    """Turn a CUDA driver result into a diagnostic without hiding failures."""
    if status:
        message = ctypes.c_char_p()
        lib.cuGetErrorString(status, ctypes.byref(message))
        detail = (
            message.value.decode(errors="replace")
            if message.value
            else "unknown CUDA error"
        )
        raise RuntimeError(f"{name} failed ({status}): {detail}")


def _function_handle(lib, context, ptx, symbol, shared):
    """Cache module/function handles separately for each active CUDA context."""
    with _LOCK:
        key = (context, ptx)
        module = _MODULES.get(key)
        if module is None:
            image = ctypes.create_string_buffer(ptx.encode() + b"\0")
            module = ctypes.c_void_p()
            _check(
                lib,
                "cuModuleLoadData",
                lib.cuModuleLoadData(ctypes.byref(module), image),
            )
            _MODULES[key] = module
        fkey = (key, symbol)
        function = _FUNCTIONS.get(fkey)
        if function is None:
            function = ctypes.c_void_p()
            _check(
                lib,
                "cuModuleGetFunction",
                lib.cuModuleGetFunction(
                    ctypes.byref(function), module, symbol.encode()
                ),
            )
            _FUNCTIONS[fkey] = function
        if shared > 49152:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8. This opts in
            # to dynamic shared memory above CUDA's default per-block limit.
            _check(
                lib, "cuFuncSetAttribute", lib.cuFuncSetAttribute(function, 8, shared)
            )
        return function


def create_cuda_kernel(executable: str, export: str, *, artifact_dir: Path):
    """Create an FX-compatible PTX dispatch on the current CUDA stream."""
    metadata, ptx, symbol, aligns = _metadata(Path(artifact_dir), executable, export)

    def launch(device, stream, bindings, constants, grid, block, shared):
        lib = _driver()
        context = ctypes.c_void_p()
        _check(lib, "cuCtxGetCurrent", lib.cuCtxGetCurrent(ctypes.byref(context)))
        if not context.value:
            raise RuntimeError(f"IREE CUDA export {export}: no current CUDA context")
        function = _function_handle(lib, context.value, ptx, symbol, shared)
        # Driver kernelParams points to per-argument host storage.
        values = [ctypes.c_uint64(t.data_ptr()) for t in bindings]
        values.extend(ctypes.c_uint32(c) for c in constants)
        params = (ctypes.c_void_p * len(values))(
            *(ctypes.addressof(value) for value in values)
        )
        _check(
            lib,
            "cuLaunchKernel",
            lib.cuLaunchKernel(
                function,
                *grid,
                *block,
                shared,
                ctypes.c_void_p(stream.cuda_stream),
                params,
                None,
            ),
        )

    kernel = dispatch.create_gpu_kernel(
        executable, export, metadata, aligns, launch, vendor="cuda"
    )
    kernel.__module__ = __name__
    return kernel
