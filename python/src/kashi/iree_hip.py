"""Direct IREE HIP dispatch with validated AMDGPU code-object metadata."""

import ctypes
import io
import re
import threading
from pathlib import Path

import msgpack
import torch
from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile

from kashi import iree_gpu_dispatch as dispatch

_LOCK = threading.RLock()
_DRIVER = None
_MODULES = {}
_FUNCTIONS = {}
_PRIMARY_CONTEXTS = {}

# ELF EF_AMDGPU_MACH values for concrete AMDGPU processors. Unknown and
# generic code objects are rejected rather than equated with a physical GPU.
_ELF_MACHINES = {
    0x20: "gfx600",
    0x21: "gfx601",
    0x22: "gfx700",
    0x23: "gfx701",
    0x24: "gfx702",
    0x25: "gfx703",
    0x26: "gfx704",
    0x28: "gfx801",
    0x29: "gfx802",
    0x2A: "gfx803",
    0x2B: "gfx810",
    0x2C: "gfx900",
    0x2D: "gfx902",
    0x2E: "gfx904",
    0x2F: "gfx906",
    0x30: "gfx908",
    0x31: "gfx909",
    0x32: "gfx90c",
    0x33: "gfx1010",
    0x34: "gfx1011",
    0x35: "gfx1012",
    0x36: "gfx1030",
    0x37: "gfx1031",
    0x38: "gfx1032",
    0x39: "gfx1033",
    0x3A: "gfx602",
    0x3B: "gfx705",
    0x3C: "gfx805",
    0x3D: "gfx1035",
    0x3E: "gfx1034",
    0x3F: "gfx90a",
    0x40: "gfx940",
    0x41: "gfx1100",
    0x42: "gfx1013",
    0x43: "gfx1150",
    0x44: "gfx1103",
    0x45: "gfx1036",
    0x46: "gfx1101",
    0x47: "gfx1102",
    0x48: "gfx1200",
    0x49: "gfx1250",
    0x4A: "gfx1151",
    0x4B: "gfx941",
    0x4C: "gfx942",
    0x4E: "gfx1201",
    0x4F: "gfx950",
    0x50: "gfx1310",
    0x55: "gfx1152",
    0x58: "gfx1153",
    0x5A: "gfx1251",
    0x5C: "gfx1172",
    0x5D: "gfx1170",
    0x5E: "gfx1171",
}


def _target_id(value):
    """Parse an AMD target ID without silently dropping target features."""
    if not isinstance(value, str):
        raise ValueError("Invalid AMDGPU target metadata")  # noqa: TRY004
    if value.startswith("amdgcn-amd-amdhsa--"):
        value = value[len("amdgcn-amd-amdhsa--") :]
    elif value.startswith("amdgpu"):
        match = re.fullmatch(
            r"amdgpu(\d+)\.([0-9a-f]{2})-amd-amdhsa-unknown-(.+)", value
        )
        if not match:
            raise ValueError(f"Invalid AMDGPU target triple {value!r}")
        major, minor, value = match.groups()
        processor = value.split(":", 1)[0]
        if processor != f"gfx{major}{minor}":
            raise ValueError(
                f"AMDGPU target triple disagrees with processor {processor!r}"
            )
    parts = value.split(":")
    if not re.fullmatch(r"gfx[0-9a-f]+", parts[0]):
        raise ValueError(f"Invalid AMDGPU target {value!r}")
    features = {}
    for feature in parts[1:]:
        match = re.fullmatch(r"(sramecc|xnack)([+-])", feature)
        if not match or match.group(1) in features:
            raise ValueError(f"Invalid AMDGPU target feature {feature!r}")
        features[match.group(1)] = match.group(2)
    return parts[0], features


def _elf_target(elf, note_target):
    """Cross-check code-object ISA and feature bits against its metadata note."""
    arch, note_features = _target_id(note_target)
    ident = elf.header["e_ident"]
    if ident["EI_ABIVERSION"] not in (2, 3, 4):
        raise ValueError("Unsupported AMDGPU target ELF ABI version")
    flags = elf.header["e_flags"]
    if _ELF_MACHINES.get(flags & 0xFF) != arch:
        raise ValueError(f"AMDGPU ELF target does not match metadata target {arch}")
    # HSA code object v4+ uses two bits per feature: unsupported, any,
    # off, on. An explicit feature must agree across the ELF and note.
    features = {}
    for name, shift in (("sramecc", 10), ("xnack", 8)):
        state = (flags >> shift) & 3
        encoded = {2: "-", 3: "+"}.get(state)
        if name in note_features and note_features[name] != encoded:
            raise ValueError(f"AMDGPU ELF target {name} disagrees with metadata target")
        if encoded:
            features[name] = encoded
    return arch + "".join(f":{name}{value}" for name, value in features.items())


def _check_target(artifact_target, device_target):
    """An unspecified artifact feature is portable; a specified one must match."""
    try:
        arch, features = _target_id(artifact_target)
        device_arch, device_features = _target_id(device_target)
    except ValueError as exc:
        raise RuntimeError(f"Invalid HIP target: {exc}") from exc
    if arch != device_arch or any(
        device_features.get(k) != v for k, v in features.items()
    ):
        raise RuntimeError(
            f"HIP target {artifact_target} is incompatible with device target {device_target}"
        )


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate AMDGPU metadata key {key!r}")
        result[key] = value
    return result


def _hsaco_metadata(image, export, binding_count, constant_count, block):
    """Validate the ELF note, exported descriptor and complete explicit ABI.

    HIP takes the reflection name, whereas the ELF symbol is its .kd kernel
    descriptor. Offsets describe the kernarg buffer HIP constructs from the
    per-argument pointers; no hidden or dynamic-LDS arguments are supported.
    """
    try:
        elf = ELFFile(io.BytesIO(image))
        if (
            elf.elfclass != 64
            or not elf.little_endian
            or elf.header["e_machine"] != "EM_AMDGPU"
            or image[7] != 64
            or elf.header["e_type"] != "ET_DYN"
        ):
            raise ValueError("HSACO is not an AMDGPU ELF64 shared object")
        notes = []
        for segment in elf.iter_segments():
            if segment["p_type"] != "PT_NOTE":
                continue
            start, size = segment["p_offset"], segment["p_filesz"]
            if start + size > len(image):
                raise ValueError("Truncated HSACO note segment")
            for note in segment.iter_notes():
                if note["n_offset"] + note["n_size"] > start + size:
                    raise ValueError("Truncated HSACO note")
                if note["n_name"] == "AMDGPU" and note["n_type"] == 32:
                    if len(note["n_descdata"]) != note["n_descsz"]:
                        raise ValueError("Truncated AMDGPU metadata")
                    notes.append(note["n_descdata"])
        if len(notes) != 1:
            raise ValueError("Expected exactly one AMDGPU metadata note")
        metadata = msgpack.unpackb(notes[0], raw=False, object_pairs_hook=_unique_pairs)
        if not isinstance(metadata, dict):
            raise ValueError("AMDGPU metadata is not a map")  # noqa: TRY004
        version = metadata.get("amdhsa.version")
        if (
            not isinstance(version, list)
            or len(version) != 2
            or version[0] != 1
            or not isinstance(version[1], int)
            or not 0 <= version[1] <= 2
        ):
            raise ValueError("Unsupported AMDGPU metadata version")
        target = metadata.get("amdhsa.target")
        if not isinstance(target, str) or not target.startswith(
            ("amdgcn-amd-amdhsa--", "amdgpu")
        ):
            raise ValueError("Missing or invalid AMDGPU target")
        target = _elf_target(elf, target)
        kernels = metadata.get("amdhsa.kernels")
        if not isinstance(kernels, list) or any(
            not isinstance(k, dict) for k in kernels
        ):
            raise ValueError("Invalid AMDGPU kernel metadata")
        matches = [k for k in kernels if k.get(".name") == export]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous HIP export {export}")
        kernel = matches[0]
        descriptor = export + ".kd"
        if kernel.get(".symbol") != descriptor:
            raise ValueError(f"Invalid descriptor for HIP export {export}")
        symbols = elf.get_section_by_name(".dynsym")
        if symbols is None or not any(
            sym.name == descriptor and sym["st_shndx"] != "SHN_UNDEF"
            for sym in symbols.iter_symbols()
        ):
            raise ValueError(f"Missing HIP kernel descriptor {descriptor}")
        args = kernel.get(".args", [])
        if not isinstance(args, list) or len(args) != binding_count + constant_count:
            raise ValueError(f"HIP export {export}: parameter count mismatch")
        segment_size = kernel.get(".kernarg_segment_size")
        segment_align = kernel.get(".kernarg_segment_align")
        group_size = kernel.get(".group_segment_fixed_size")
        if any(type(x) is not int or x < 0 for x in (segment_size, group_size)):
            raise ValueError("Invalid HSACO segment size")
        if (
            type(segment_align) is not int
            or segment_align <= 0
            or segment_align & (segment_align - 1)
        ):
            raise ValueError("Invalid HSACO kernarg alignment")
        required = kernel.get(".reqd_workgroup_size")
        if required is not None and required != list(block):
            raise ValueError(
                f"HIP export {export}: incompatible required workgroup size"
            )
        maximum = kernel.get(".max_flat_workgroup_size")
        if maximum is not None and (
            type(maximum) is not int or maximum < block[0] * block[1] * block[2]
        ):
            raise ValueError(f"HIP export {export}: workgroup exceeds kernel limit")
        if len(block) != 3 or any(type(n) is not int or n <= 0 for n in block):
            raise ValueError(f"HIP export {export}: invalid workgroup size")
        offset = 0
        alignments = []
        for index, arg in enumerate(args):
            if not isinstance(arg, dict):
                raise ValueError("Invalid HIP argument metadata")  # noqa: TRY004
            pointer = index < binding_count
            size = 8 if pointer else 4
            kind = "global_buffer" if pointer else "by_value"
            if (
                arg.get(".value_kind") != kind
                or arg.get(".size") != size
                or arg.get(".offset") != offset
                or offset + size > segment_size
            ):
                raise ValueError(
                    f"HIP export {export}: unsupported argument {index} ABI"
                )
            if pointer and arg.get(".address_space", "generic") not in (
                "global",
                "generic",
            ):
                raise ValueError(
                    f"HIP export {export}: unsupported pointer address space"
                )
            alignment = (
                arg.get(".pointee_align", arg.get(".align", 1)) if pointer else 1
            )
            if (
                type(alignment) is not int
                or alignment <= 0
                or alignment & (alignment - 1)
            ):
                raise ValueError(f"HIP export {export}: invalid binding alignment")
            if pointer:
                alignments.append(alignment)
            offset += size
        return target, alignments
    except (
        ELFError,
        EOFError,
        IndexError,
        KeyError,
        TypeError,
        UnicodeError,
        msgpack.UnpackException,
        OSError,
    ) as exc:
        raise ValueError(f"Invalid AMDGPU HSACO metadata for {export}: {exc}") from exc


def _check(lib, name, status):
    if status:
        message = lib.hipGetErrorString(status)
        detail = message.decode(errors="replace") if message else "unknown HIP error"
        raise RuntimeError(f"{name} failed ({status}): {detail}")


def _driver():
    """Load HIP lazily: offline compilation does not need a driver or GPU."""
    global _DRIVER
    with _LOCK:
        if _DRIVER is not None:
            return _DRIVER
        try:
            # The packaged ROCm wheel may expose only libamdhip64.so.7. Reuse
            # the already-loaded PyTorch runtime to preserve context identity.
            lib = ctypes.CDLL(dispatch._hip_runtime_path())
        except OSError as exc:
            raise RuntimeError(
                "IREE direct HIP execution requires PyTorch's HIP runtime library"
            ) from exc
        signatures = {
            "hipGetErrorString": ([ctypes.c_int], ctypes.c_char_p),
            "hipGetDevice": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "hipCtxGetCurrent": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "hipDevicePrimaryCtxRetain": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
                ctypes.c_int,
            ),
            "hipModuleLoadData": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
                ctypes.c_int,
            ),
            "hipModuleGetFunction": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
                ctypes.c_int,
            ),
            "hipModuleLaunchKernel": (
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
        }
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result
        _DRIVER = lib
        return lib


def _check_context(lib, device, context):
    # PyTorch's allocations and streams belong to the device primary context.
    # Retain it along with our cached modules for process lifetime; using a
    # different current context on the same GPU is not enough for safe access.
    with _LOCK:
        primary = _PRIMARY_CONTEXTS.get(device)
        if primary is None:
            primary = ctypes.c_void_p()
            _check(
                lib,
                "hipDevicePrimaryCtxRetain",
                lib.hipDevicePrimaryCtxRetain(ctypes.byref(primary), device),
            )
            if not primary.value:
                raise RuntimeError("HIP returned a null primary context")
            _PRIMARY_CONTEXTS[device] = primary
        if context != primary.value:
            raise RuntimeError(
                "Current HIP context does not match PyTorch device primary context"
            )


def _function_handle(lib, device, context, image, symbol):
    # Contexts are not interchangeable, even on the same physical device.
    # Retain loaded modules for process lifetime so asynchronous kernels can
    # never outlive their code, including after an FX graph is released.
    with _LOCK:
        key = (device, context, image)
        module = _MODULES.get(key)
        if module is None:
            storage = ctypes.create_string_buffer(image)
            module = ctypes.c_void_p()
            _check(
                lib,
                "hipModuleLoadData",
                lib.hipModuleLoadData(ctypes.byref(module), storage),
            )
            _MODULES[key] = module
        fkey = (key, symbol)
        function = _FUNCTIONS.get(fkey)
        if function is None:
            function = ctypes.c_void_p()
            _check(
                lib,
                "hipModuleGetFunction",
                lib.hipModuleGetFunction(
                    ctypes.byref(function), module, symbol.encode()
                ),
            )
            _FUNCTIONS[fkey] = function
        return function


def _validate_hal_layout(targets, executable, export, bindings, constants):
    """The serialized HAL layout must agree with stream and ELF parameters."""
    target = dispatch._executable(targets, "hal", executable)
    match = re.search(
        rf"\bhal\.executable\.export\s+(?:public|private)\s+@{re.escape(export)}\b"
        r"[^\n]*?\blayout\((#[\w.$-]+)\)",
        target,
    )
    if not match:
        raise ValueError(f"HIP export {export}: missing HAL pipeline layout")
    alias = re.search(
        rf"^{re.escape(match.group(1))}\s*=\s*#hal\.pipeline\.layout<",
        targets,
        re.MULTILINE,
    )
    if not alias:
        raise ValueError(f"HIP export {export}: missing HAL layout definition")
    layout, _ = dispatch._balanced(targets, alias.end() - 1, "<", ">")
    count = re.search(r"\bconstants\s*=\s*(\d+)", layout)
    fields = re.search(r"\bbindings\s*=\s*\[([^]]*)\]", layout)
    if fields is None:
        raise ValueError(f"HIP export {export}: missing HAL binding layout")
    entries = dispatch._split(fields.group(1))
    if (int(count.group(1)) if count else 0) != constants or len(entries) != bindings:
        raise ValueError(f"HIP export {export}: HAL layout count mismatch")
    if any(
        not re.match(r"#hal\.pipeline\.binding<storage_buffer\s*,", e) for e in entries
    ):
        raise ValueError(f"HIP export {export}: unsupported HAL binding layout")


def _binding_alignments(stream_args, elf_alignments, export):
    """Honor IREE's base-pointer assumptions in addition to ELF metadata."""
    if len(stream_args) != len(elf_alignments):
        raise ValueError(f"HIP export {export}: binding alignment count mismatch")
    result = []
    for arg, elf_alignment in zip(stream_args, elf_alignments):
        match = re.search(r"\bstream\.alignment\s*=\s*(\d+)\s*:\s*index\b", arg)
        if "stream.alignment" in arg and not match:
            raise ValueError(f"HIP export {export}: invalid stream binding alignment")
        alignment = int(match.group(1)) if match else 1
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"HIP export {export}: invalid stream binding alignment")
        # LLVM GPU kernel arguments are at least 16-byte aligned; stream
        # annotations can impose stronger assumptions on the base address.
        result.append(max(16, elf_alignment, alignment))
    return result


def create_hip_kernel(executable: str, export: str, *, artifact_dir: Path):
    """Create an FX-compatible direct HIP callable, validating before loading."""
    directory = Path(artifact_dir)
    metadata = dispatch._metadata(directory, executable, export)
    paths = list(
        (directory / "kernel_files").glob(f"module_{executable}_rocm_hsaco_fb.hsaco")
    )
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one HIP HSACO for {executable} in {directory}, found {len(paths)}"
        )
    image = paths[0].read_bytes()
    target, alignments = _hsaco_metadata(
        image, export, metadata[1], metadata[2], metadata[7]
    )
    stream_ir = dispatch._executable(
        dispatch._phase(directory, "stream"), "stream", executable
    )
    stream_args, _ = dispatch._function(stream_ir, export)
    alignments = _binding_alignments(stream_args[: metadata[1]], alignments, export)
    targets_ir = dispatch._phase(directory, "executable-targets")
    _validate_hal_layout(targets_ir, executable, export, metadata[1], metadata[2])
    architectures = set(re.findall(r'\barch\s*=\s*"(gfx[0-9a-f]+)"', targets_ir))
    if len(architectures) != 1 or target.split(":")[0] not in architectures:
        raise ValueError(
            f"HIP HSACO target {target} does not match executable target metadata"
        )

    def validate_device(device):
        if not torch.version.hip or torch.version.cuda:
            raise RuntimeError("Direct HIP execution requires a ROCm PyTorch build")
        properties = torch.cuda.get_device_properties(device)
        _check_target(target, getattr(properties, "gcnArchName", None))

    def launch(device, stream, bindings, constants, grid, block, shared):
        lib = _driver()
        current_device = ctypes.c_int()
        _check(lib, "hipGetDevice", lib.hipGetDevice(ctypes.byref(current_device)))
        if current_device.value != device.index:
            raise RuntimeError("Current HIP device does not match tensor device")
        context = ctypes.c_void_p()
        _check(lib, "hipCtxGetCurrent", lib.hipCtxGetCurrent(ctypes.byref(context)))
        if not context.value:
            raise RuntimeError(f"IREE HIP export {export}: no current HIP context")
        _check_context(lib, device.index, context.value)
        function = _function_handle(lib, device.index, context.value, image, export)
        values = [ctypes.c_uint64(t.data_ptr()) for t in bindings]
        values.extend(ctypes.c_uint32(c) for c in constants)
        params = (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))
        # IREE's HIP executable stores LDS statically in the code object. Unlike
        # CUDA's launch ABI it always passes zero dynamic shared-memory bytes.
        _check(
            lib,
            "hipModuleLaunchKernel",
            lib.hipModuleLaunchKernel(
                function,
                *grid,
                *block,
                0,
                ctypes.c_void_p(stream.cuda_stream),
                params,
                None,
            ),
        )

    kernel = dispatch.create_gpu_kernel(
        executable,
        export,
        metadata,
        alignments,
        launch,
        vendor="hip",
        validate_device=validate_device,
    )
    kernel.__module__ = __name__
    return kernel
