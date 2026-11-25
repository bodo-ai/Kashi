"""Execute partitioned VMFB exports on CPU tensors using IREE local-task.

The local VM invocation is synchronous. CPU DLPack imports retain their Torch
owners during the call. Fresh outputs are copied to independently owned host
storage because a retained HAL buffer can borrow a device-owned pool slab.
Exact source aliases are returned directly without copying.
"""

import operator
import re
import threading
from collections.abc import Callable
from pathlib import Path

import iree.runtime as rt
import torch
from iree.compiler import ir

from kashi.iree_partition import PartitionResult

_DTYPES = {
    "i1": torch.bool,
    "si8": torch.int8,
    "si16": torch.int16,
    "si32": torch.int32,
    "si64": torch.int64,
    "ui8": torch.uint8,
    "ui16": torch.uint16,
    "ui32": torch.uint32,
    "ui64": torch.uint64,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}
_INTEGER_REINTERPRET = {
    torch.int8: torch.uint8,
    torch.uint8: torch.int8,
    torch.int16: torch.uint16,
    torch.uint16: torch.int16,
    torch.int32: torch.uint32,
    torch.uint32: torch.int32,
    torch.int64: torch.uint64,
    torch.uint64: torch.int64,
}
_HAL_DTYPES = {
    rt.HalElementType.BOOL_8: torch.bool,
    rt.HalElementType.INT_8: torch.int8,
    rt.HalElementType.SINT_8: torch.int8,
    rt.HalElementType.UINT_8: torch.uint8,
    rt.HalElementType.INT_16: torch.int16,
    rt.HalElementType.SINT_16: torch.int16,
    rt.HalElementType.UINT_16: torch.uint16,
    rt.HalElementType.INT_32: torch.int32,
    rt.HalElementType.SINT_32: torch.int32,
    rt.HalElementType.UINT_32: torch.uint32,
    rt.HalElementType.INT_64: torch.int64,
    rt.HalElementType.SINT_64: torch.int64,
    rt.HalElementType.UINT_64: torch.uint64,
    rt.HalElementType.FLOAT_16: torch.float16,
    rt.HalElementType.BFLOAT_16: torch.bfloat16,
    rt.HalElementType.FLOAT_32: torch.float32,
    rt.HalElementType.FLOAT_64: torch.float64,
}
_TORCH_HAL_TYPES = {dtype: element for element, dtype in _HAL_DTYPES.items()}
_TENSOR_TYPE = re.compile(r"!torch\.(?:v)?tensor<\[([^]]*)\],\s*([a-z0-9]+)>")


def _tensor_spec(type_text):
    """Decode ranked Torch tensor metadata without guessing missing dtypes."""
    if not type_text.startswith(("!torch.vtensor", "!torch.tensor")):
        return None
    match = _TENSOR_TYPE.fullmatch(type_text)
    if match is None or match.group(2) not in _DTYPES:
        raise NotImplementedError(f"Unsupported IREE CPU tensor type: {type_text}")
    dimensions = match.group(1).strip()
    shape = []
    if dimensions:
        for dim in dimensions.split(","):
            dim = dim.strip()
            if dim == "?":
                shape.append(None)
            elif dim.isdecimal():
                shape.append(int(dim))
            else:
                raise NotImplementedError(
                    f"Unsupported IREE CPU tensor shape: {type_text}"
                )
    return _DTYPES[match.group(2)], tuple(shape)


def _validate_tensor(tensor, spec, where):
    """Enforce the dtype and known dimensions of a partition boundary."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"IREE CPU {where} must be a tensor")
    dtype, shape = spec
    if tensor.dtype != dtype:
        raise TypeError(f"IREE CPU {where} has dtype {tensor.dtype}, expected {dtype}")
    if len(tensor.shape) != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(tensor.shape, shape)
    ):
        raise ValueError(
            f"IREE CPU {where} has shape {tuple(tensor.shape)}, expected {shape}"
        )


def _output_aliases(function):
    """Record exact source SSA identities, before IREE can deduplicate values."""
    block = function.regions[0].blocks[0]
    inputs = {value: index for index, value in enumerate(block.arguments)}
    seen = {}
    aliases = []
    for index, value in enumerate(block.operations[-1].operation.operands):
        if value in inputs:
            aliases.append(("input", inputs[value]))
        elif value in seen:
            aliases.append(("output", seen[value]))
        else:
            aliases.append(None)
        seen.setdefault(value, index)
    return tuple(aliases)


class _RuntimeModule:
    """Own one lazily initialized local device, VM context and export table."""

    def __init__(self, path, names):
        self.path = Path(path)
        self.names = names
        self._instance = None
        self._lock = threading.Lock()

    def instance(self):
        with self._lock:
            if self._instance is None:
                device = rt.get_driver("local-task").create_default_device()
                config = rt.Config(device=device)
                module = rt.VmModule.mmap(config.vm_instance, str(self.path))
                bound = rt.load_vm_module(module, config=config)
                functions = {name: bound[name] for name in self.names}
                self._instance = (device, bound, functions)
            return self._instance


def _import_argument(value, device, imports):
    """Import dense CPU memory; retain packed and original tensors until return."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise RuntimeError(
                f"IREE CPU runtime requires CPU tensors, got {value.device}"
            )
        if value.layout != torch.strided:
            raise NotImplementedError(
                f"IREE cannot import tensor layout {value.layout}"
            )
        if value.numel() == 0:
            element_type = _TORCH_HAL_TYPES.get(value.dtype)
            if element_type is None:
                raise TypeError(f"Unsupported IREE CPU tensor dtype: {value.dtype}")
            # PyTorch's DLPack data pointer is NULL for every empty tensor.
            # local-task uses a NULL fixup pointer as the *indirect binding*
            # sentinel, so importing that pointer as a direct zero-byte buffer
            # can dereference a nonexistent binding table. Supply a real aligned
            # HAL allocation while preserving the original zero-element shape.
            buffer = device.allocator.allocate_buffer(
                memory_type=rt.MemoryType.DEVICE_LOCAL,
                allowed_usage=rt.BufferUsage.DEFAULT,
                allocation_size=64,
            )
            imports.append((value, value))
            return rt.HalBufferView(buffer, tuple(value.shape), element_type)
        # LLVM-CPU dispatch bindings assume IREE's default 64-byte resource
        # alignment. A contiguous offset view is not necessarily aligned, and
        # marking it UNALIGNED for HAL does not relax the generated code ABI.
        packed = value.detach().resolve_conj().resolve_neg()
        if not packed.is_contiguous() or packed.data_ptr() % 64:
            size = packed.numel() * packed.element_size()
            storage = torch.empty(size + 63, dtype=torch.uint8, device="cpu")
            offset = -storage.data_ptr() % 64
            dense = (
                storage[offset : offset + size].view(packed.dtype).view(packed.shape)
            )
            dense.copy_(packed)
            packed = dense
        imports.append((value, packed))
        return device.from_dlpack_capsule(
            torch.utils.dlpack.to_dlpack(packed),
            memory_type=rt.MemoryType.HOST_LOCAL | rt.MemoryType.DEVICE_VISIBLE,
            memory_access=rt.MemoryAccess.READ | rt.MemoryAccess.WRITE,
        )
    if isinstance(value, (tuple, list)):
        return type(value)(_import_argument(item, device, imports) for item in value)
    if isinstance(value, (int, torch.SymInt)):
        return int(value)
    raise TypeError(f"Unsupported IREE CPU segment argument: {type(value).__name__}")


def _result_view(value, spec):
    """Validate a result's HAL dtype and shape without exporting its storage."""
    if isinstance(value, rt.VmRef):
        value = value.deref(rt.HalBufferView, True)
    if isinstance(value, rt.DeviceArray):
        value = value._buffer_view
    if not isinstance(value, rt.HalBufferView):
        raise TypeError(f"Expected IREE CPU tensor result, got {type(value).__name__}")
    dtype, shape = spec
    actual_dtype = _HAL_DTYPES.get(value.element_type)
    if actual_dtype != dtype and _INTEGER_REINTERPRET.get(actual_dtype) != dtype:
        raise TypeError(
            f"IREE CPU returned element type {value.element_type}, expected {dtype}"
        )
    actual_shape = tuple(value.shape)
    if len(actual_shape) != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(actual_shape, shape)
    ):
        raise ValueError(f"IREE CPU result has shape {actual_shape}, expected {shape}")
    return value


def _export_tensor(value, spec, device):
    """Copy a CPU result before its device-owned pool can be destroyed."""
    value = _result_view(value, spec)
    # kDLCPU = 1, device id = 0. DLPack retains the HAL buffer but not the
    # local-task device that owns the backing pool. It is only a temporary view
    # while we hold the device alive and synchronously copy into Torch storage.
    try:
        capsule = device.create_dlpack_capsule(value, 1, 0)
    except (ValueError, RuntimeError) as error:
        if not any(
            message in str(error)
            for message in (
                "Legacy dlpack cannot preserve read-only or write-only buffer access",
                "Cannot export device buffer",
            )
        ):
            raise
        # A read-only or unexportable local allocation cannot be handed to
        # mutable PyTorch. Mapping is synchronous after the VM call; copy its
        # bytes once into writable Python-owned storage, including bf16 which
        # NumPy cannot represent on all supported platforms.
        dtype, _ = spec
        mapped = memoryview(value.map())
        if len(mapped) != value.byte_length:
            raise RuntimeError(
                "IREE CPU result has an incompatible buffer length"
            ) from error
        if len(mapped):
            tensor = torch.frombuffer(bytearray(mapped), dtype=dtype).reshape(
                tuple(value.shape)
            )
        else:
            tensor = torch.empty(tuple(value.shape), dtype=dtype)
        needs_copy = False
    else:
        tensor = torch.utils.dlpack.from_dlpack(capsule)
        needs_copy = True
    dtype, _ = spec
    if tensor.dtype != dtype:
        if _INTEGER_REINTERPRET.get(tensor.dtype) != dtype:
            raise TypeError(f"IREE CPU returned {tensor.dtype}, expected {dtype}")
        tensor = tensor.reshape(-1).view(dtype).reshape(tensor.shape)
    _validate_tensor(tensor, spec, "result")
    return tensor.clone() if needs_copy else tensor


def _export_scalar(value):
    if isinstance(value, rt.VmRef):
        items = value.deref(rt.VmVariantList, True)
        if items is None:
            raise TypeError("Unsupported IREE CPU VM result reference")
        return [_export_scalar(items.get_variant(i)) for i in range(len(items))]
    if isinstance(value, (tuple, list)):
        return type(value)(_export_scalar(item) for item in value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported IREE CPU result: {type(value).__name__}")


class _SegmentTarget:
    """An FX-callable export retaining the shared local runtime and signature."""

    def __init__(self, runtime, segment, aliases):
        self.runtime = runtime
        self.name = segment.func_name
        self.input_types = tuple(segment.live_in_types)
        self.output_types = tuple(segment.live_out_types)
        self.input_specs = tuple(_tensor_spec(t) for t in self.input_types)
        self.output_specs = tuple(_tensor_spec(t) for t in self.output_types)
        self.aliases = aliases
        self.__name__ = self.name
        self.__module__ = __name__

    def __call__(self, *args):
        if len(args) != len(self.input_types):
            raise RuntimeError(
                f"{self.name} expected {len(self.input_types)} arguments, got {len(args)}"
            )
        # Validate before initializing a local runtime or importing any buffer.
        for index, (arg, spec) in enumerate(zip(args, self.input_specs, strict=True)):
            if isinstance(arg, torch.Tensor) and arg.device.type != "cpu":
                raise RuntimeError(
                    f"IREE CPU runtime requires CPU tensors, got {arg.device}"
                )
            if spec is not None:
                _validate_tensor(arg, spec, f"argument {index}")
        device, _bound, functions = self.runtime.instance()
        imports = []
        converted = [_import_argument(arg, device, imports) for arg in args]
        result = functions[self.name](*converted)
        count = len(self.output_types)
        if count == 0:
            if result is not None:
                raise RuntimeError(f"{self.name} returned an unexpected result")
            return ()
        if count == 1:
            result = (result,)
        elif not isinstance(result, tuple) or len(result) != count:
            raise RuntimeError(f"{self.name} returned an incorrect number of results")

        # Fresh values may share IREE's pooled allocation with input or other
        # results, and that pool can die with the device. Copy fresh results;
        # source SSA aliases must instead preserve their original identities.
        outputs = []
        for value, spec, alias in zip(
            result, self.output_specs, self.aliases, strict=True
        ):
            if alias is not None:
                source, index = alias
                output = args[index] if source == "input" else outputs[index]
                if spec is not None:
                    _result_view(value, spec)
                    _validate_tensor(output, spec, "aliased result")
            elif spec is None:
                output = _export_scalar(value)
            else:
                output = _export_tensor(value, spec, device)
            outputs.append(output)
        return tuple(outputs)


class _SegmentHandler:
    """Splice a segment export and its flat results into an FX graph."""

    def __init__(self, target):
        self.target = target

    def __call__(self, graph, args) -> list:
        if len(args) != len(self.target.input_types):
            raise ValueError(f"{self.target.name} has an incorrect live-in count")
        call = graph.call_function(self.target, tuple(args))
        return [
            graph.call_function(operator.getitem, (call, i))
            for i in range(len(self.target.output_types))
        ]


def runtime_segment_handlers(
    vmfb_path: Path, partition: PartitionResult
) -> dict[str, Callable]:
    """Create lazy CPU VMFB handlers in the partition's segment convention.

    Each handler accepts ``(graph, args)`` and returns flat FX live-outs. All
    segments share one local-task VM context, including scalar-only segments.
    Missing exports, device mismatches and runtime errors are never hidden.
    """
    if partition.device is not None and torch.device(partition.device).type != "cpu":
        raise ValueError(
            f"IREE CPU runtime requires a CPU partition, got {partition.device}"
        )
    functions = {
        ir.StringAttr(op.attributes["sym_name"]).value: op
        for op in partition.module.body.operations
        if op.operation.name == "func.func"
    }
    segments = [s for s in partition.segments if s.func_name in functions]
    runtime = _RuntimeModule(vmfb_path, tuple(s.func_name for s in segments))
    return {
        segment.func_name: _SegmentHandler(
            _SegmentTarget(
                runtime, segment, _output_aliases(functions[segment.func_name])
            )
        )
        for segment in segments
    }
