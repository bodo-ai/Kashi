"""Execute partitioned IREE functions as FX calls on PyTorch CUDA tensors.

IREE's Python DLPack bindings only exchange memory; they do not establish
stream dependencies. The synchronous VM export waits for IREE's completion,
while we explicitly wait for the caller's PyTorch stream before entering it.
Fresh results are copied device-to-device into PyTorch-owned storage before
returning: a foreign DLPack allocation has no PyTorch allocator stream tracking
and can otherwise be freed while an asynchronous consumer still uses it.
"""

import operator
import re
import threading
from collections.abc import Callable
from pathlib import Path

import iree.runtime as rt
import torch

from kashi.iree_reassembly import (
    _segment_functions,
    _segment_output_aliases,
    _segment_result_types,
)

_TENSOR_DTYPES = {
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


def _result_dtype(type_text: str):
    """Extract a Torch tensor dtype; refuse incomplete or unsupported metadata."""
    if not type_text.startswith(("!torch.vtensor", "!torch.tensor")):
        return None
    match = re.fullmatch(r"!torch\.(?:v)?tensor<.*?,\s*([a-z0-9]+)>", type_text)
    if match is None or match.group(1) not in _TENSOR_DTYPES:
        raise NotImplementedError(f"Unsupported IREE output tensor type: {type_text}")
    return _TENSOR_DTYPES[match.group(1)]


class _RuntimeModule:
    """Share per-device VM contexts and exported functions among all segments."""

    def __init__(self, path: Path, device: str | None, names: tuple[str, ...]):
        """Record the persistent VMFB, device hint, and compiled exports."""
        self.path = Path(path)
        self.device = torch.device(device) if device is not None else None
        self.names = names
        self._instances = {}
        self._lock = threading.Lock()

    def instance(self, index: int):
        """Get a cached IREE device and bound exports for a PyTorch CUDA index.

        Match physical UUIDs rather than assuming IREE and PyTorch enumerate
        devices identically, particularly with CUDA_VISIBLE_DEVICES or MIG.
        IREE CUDA and PyTorch both use the CUDA device's primary context.
        """
        with self._lock:
            if index in self._instances:
                return self._instances[index]
            properties = torch.cuda.get_device_properties(index)
            uuid = getattr(properties, "uuid", None)
            if uuid is None:
                raise RuntimeError(f"PyTorch did not report a UUID for cuda:{index}")

            def normalize_uuid(value):
                """Normalize CUDA's optional GPU prefix and UUID separators."""
                return str(value).lower().removeprefix("gpu-").replace("-", "")

            driver = rt.get_driver("cuda")
            devices = [
                info
                for info in driver.query_available_devices()
                if normalize_uuid(info["path"]) == normalize_uuid(uuid)
            ]
            if len(devices) != 1:
                raise RuntimeError(
                    f"Cannot uniquely match PyTorch cuda:{index} ({uuid}) "
                    "to an IREE CUDA device"
                )
            device = driver.create_device(devices[0], params=None)
            config = rt.Config(device=device)
            module = rt.VmModule.mmap(config.vm_instance, str(self.path))
            bound = rt.load_vm_module(module, config=config)
            functions = {name: bound[name] for name in self.names}
            instance = (device, bound, functions)
            self._instances[index] = instance
            return instance

    def device_index(self, args) -> int:
        """Choose the actual input tensor device, or require an explicit hint."""
        devices = set()

        def visit(value):
            """Collect tensor devices from nested segment arguments."""
            if isinstance(value, torch.Tensor):
                if value.device.type != "cuda" or value.device.index is None:
                    raise RuntimeError(
                        f"IREE runtime requires CUDA tensors, got {value.device}"
                    )
                devices.add(value.device.index)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        for arg in args:
            visit(arg)
        if len(devices) > 1:
            raise RuntimeError(
                f"IREE segment received tensors on multiple CUDA devices: {devices}"
            )
        if devices:
            return next(iter(devices))
        if (
            self.device is None
            or self.device.type != "cuda"
            or self.device.index is None
        ):
            raise RuntimeError(
                "IREE segment without CUDA tensor inputs requires an explicit CUDA device index"
            )
        return self.device.index


class _SegmentTarget:
    """An FX-callable target retaining its shared VM runtime and signature."""

    def __init__(
        self,
        runtime: _RuntimeModule,
        name: str,
        inputs: int,
        output_types,
        output_aliases,
    ):
        """Record the export signature and its source-level result identities."""
        self.runtime = runtime
        self.name = name
        self.inputs = inputs
        self.output_types = tuple(output_types)
        self.output_aliases = tuple(output_aliases)
        self.outputs = len(self.output_types)
        self.__name__ = name
        self.__module__ = __name__

    def __call__(self, *args):
        """Execute the synchronous VM export and return its flat live-outs."""
        if len(args) != self.inputs:
            raise RuntimeError(
                f"{self.name} expected {self.inputs} arguments, got {len(args)}"
            )
        index = self.runtime.device_index(args)
        with torch.cuda.device(index):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "IREE runtime execution does not support CUDA graph capture"
                )
            device, _bound, functions = self.runtime.instance(index)
            stream = torch.cuda.current_stream(index)
            imports = []
            converted = [_import_argument(arg, device, imports) for arg in args]
            # Imported buffers do not carry a producer fence. This wait also
            # covers any contiguous materializations enqueued above, and all
            # preceding custom-op mutations on this current PyTorch stream.
            stream.synchronize()
            result = functions[self.name](*converted)
            if self.outputs == 0:
                if result is not None:
                    raise RuntimeError(f"{self.name} returned an unexpected result")
                return ()
            if self.outputs == 1:
                result = (result,)
            elif not isinstance(result, tuple) or len(result) != self.outputs:
                raise RuntimeError(
                    f"{self.name} returned an incorrect number of results"
                )
            return _export_results(
                result,
                device,
                index,
                imports,
                stream,
                self.output_types,
                output_aliases=self.output_aliases,
                source_args=args,
            )


class _SegmentHandler:
    """Splice one executable segment call into the reconstructed FX graph."""

    def __init__(self, target: _SegmentTarget):
        """Retain the executable call target and its result count."""
        self.target = target

    def __call__(self, graph, args) -> list:
        """Add the executable call and expose each live-out in source order."""
        if len(args) != self.target.inputs:
            raise ValueError(f"{self.target.name} has an incorrect live-in count")
        call = graph.call_function(self.target, tuple(args))
        return [
            graph.call_function(operator.getitem, (call, i))
            for i in range(self.target.outputs)
        ]


def _import_argument(value, device, imports):
    """Import tensors with DLPack and recursively preserve integer lists.

    IREE's HAL buffer views require dense row-major storage. A non-contiguous
    tensor is packed on the GPU; the original and packed views are retained so
    returned input aliases can be restored rather than silently detached.
    """
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise NotImplementedError(
                f"IREE cannot import tensor layout {value.layout}"
            )
        packed = value.detach().resolve_conj().resolve_neg().contiguous()
        capsule = torch.utils.dlpack.to_dlpack(packed)
        buffer_view = device.from_dlpack_capsule(capsule)
        imports.append((value, packed))
        return buffer_view
    if isinstance(value, (tuple, list)):
        return type(value)(_import_argument(item, device, imports) for item in value)
    if isinstance(value, (int, torch.SymInt)):
        return int(value)
    raise TypeError(f"Unsupported IREE segment argument: {type(value).__name__}")


def _input_alias(result, imports):
    """Restore a VM result that refers to an imported PyTorch tensor's bytes."""
    if result.numel() == 0:
        return None
    pointer = result.data_ptr()
    size = result.numel() * result.element_size()
    for original, packed in imports:
        start = packed.data_ptr()
        end = start + packed.numel() * packed.element_size()
        if pointer + size <= start or pointer >= end:
            continue
        if pointer < start or pointer + size > end:
            raise NotImplementedError(
                "IREE returned an alias extending beyond an imported tensor view"
            )
        offset = pointer - start
        if (
            offset == 0
            and result.dtype == original.dtype
            and tuple(result.shape) == tuple(original.shape)
        ):
            return original
        # A dense view over a packed tensor aliases the original only if
        # PyTorch can express that same view without copying. Refuse an
        # unrepresentable view rather than losing custom-op mutation semantics.
        try:
            if (
                offset == 0
                and result.dtype == original.dtype
                and result.numel() == original.numel()
            ):
                return original.view(result.shape)
            flat_bytes = original.view(-1).view(torch.uint8)
            return (
                flat_bytes[offset : offset + size].view(result.dtype).view(result.shape)
            )
        except RuntimeError as error:
            raise NotImplementedError(
                "IREE returned an input alias that cannot be represented as a PyTorch view"
            ) from error
    return None


def _export_results(
    results,
    device,
    index: int,
    imports,
    stream,
    output_types=None,
    *,
    output_aliases=None,
    source_args=None,
):
    """Export VM results, preserving aliases and safe CUDA allocation lifetime.

    Exact source SSA identities restore repeated results and input aliases.
    Fresh outputs are copied into PyTorch-owned CUDA storage and synchronized:
    PyTorch's allocator cannot track asynchronous uses of IREE-owned memory.
    No tensor data passes through host memory.
    """
    foreign = []
    pooled = []

    def convert(value, dtype, metadata_required):
        """Convert IREE values to scalars, containers, or foreign tensors."""
        if isinstance(value, rt.VmRef):
            view = value.deref(rt.HalBufferView, True)
            if view is not None:
                value = view
            else:
                vm_list = value.deref(rt.VmVariantList, True)
                if vm_list is None:
                    raise TypeError("Unsupported IREE VM result reference")
                return [
                    convert(vm_list.get_variant(i), dtype, metadata_required)
                    for i in range(len(vm_list))
                ]
        if isinstance(value, rt.DeviceArray):
            value = value._buffer_view
        if isinstance(value, rt.HalBufferView):
            if output_aliases is not None:
                # Source SSA identities were resolved above. A fresh result
                # needs PyTorch-owned storage regardless of IREE's allocator;
                # pooled CUDA buffers cannot be exported through DLPack at all.
                if dtype is None:
                    raise NotImplementedError(
                        "Missing Torch dtype metadata for IREE tensor result"
                    )
                position = len(pooled)
                pooled.append((value, dtype))
                return _PooledResult(position)
            try:
                capsule = device.create_dlpack_capsule(
                    value, 2, index
                )  # DLPack kDLCUDA
            except RuntimeError as error:
                # IREE's stream-ordered CUDA pool does not support DLPack
                # export. Its HAL buffer can still be copied device-to-device
                # into an imported, PyTorch-owned allocation.
                if "CUDA buffer type is not supported for export" not in str(error):
                    raise
                if dtype is None:
                    raise NotImplementedError(
                        "Missing Torch dtype metadata for pooled IREE tensor result"
                    ) from error
                position = len(pooled)
                pooled.append((value, dtype))
                return _PooledResult(position)
            tensor = torch.utils.dlpack.from_dlpack(capsule)
            if dtype is None and metadata_required:
                raise NotImplementedError(
                    "Missing Torch dtype metadata for IREE tensor result"
                )
            if dtype is not None and tensor.dtype != dtype:
                if _INTEGER_REINTERPRET.get(tensor.dtype) != dtype:
                    raise TypeError(
                        f"IREE returned {tensor.dtype} where {dtype} was expected"
                    )
                # Signless IREE integers are exported as signed DLPack types.
                # Reinterpret the same bits for an unsigned Torch result.
                tensor = tensor.reshape(-1).view(dtype).reshape(tensor.shape)
            if output_aliases is None:
                alias = _input_alias(tensor, imports)
                if alias is not None:
                    return alias
            position = len(foreign)
            foreign.append(tensor)
            return _ForeignResult(position)
        if isinstance(value, (tuple, list)):
            return type(value)(
                convert(item, dtype, metadata_required) for item in value
            )
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        raise TypeError(f"Unsupported IREE result: {type(value).__name__}")

    if output_types is None:
        dtypes = [None] * len(results)
    else:
        dtypes = [_result_dtype(type_text) for type_text in output_types]
    converted = []
    aliases = output_aliases if output_aliases is not None else (None,) * len(results)
    for result, dtype, alias in zip(results, dtypes, aliases, strict=True):
        if alias is not None and alias[0] == "input":
            converted.append(source_args[alias[1]])
        elif alias is not None and alias[0] == "output":
            converted.append(converted[alias[1]])
        else:
            converted.append(convert(result, dtype, output_types is not None))
    owned = [None] * len(foreign)
    intervals = []
    for position, tensor in enumerate(foreign):
        if tensor.numel() == 0:
            owned[position] = torch.empty_like(tensor)
        else:
            start = tensor.data_ptr()
            intervals.append(
                (start, start + tensor.numel() * tensor.element_size(), position)
            )
    intervals.sort()
    groups = []
    for start, end, position in intervals:
        if output_aliases is None and groups and start < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append((start, end, position))
        else:
            groups.append([start, end, [(start, end, position)]])
    copied = False
    try:
        for start, end, members in groups:
            alignment = max(
                foreign[position].element_size() for _, _, position in members
            )
            base = start - start % alignment
            storage = torch.empty(end - base, dtype=torch.uint8, device=f"cuda:{index}")
            for member_start, member_end, position in members:
                tensor = foreign[position]
                part = storage[member_start - base : member_end - base]
                part.copy_(tensor.reshape(-1).view(torch.uint8))
                copied = True
                owned[position] = part.view(tensor.dtype).view(tensor.shape)
    finally:
        if copied:
            # Keep all foreign buffers alive until even partially enqueued
            # copies complete, including if view construction fails.
            stream.synchronize()

    pooled_outputs = []
    for view, dtype in pooled:
        output = torch.empty(tuple(view.shape), dtype=dtype, device=f"cuda:{index}")
        if view.byte_length != output.numel() * output.element_size():
            raise RuntimeError("IREE pooled result has an incompatible byte length")
        # The IREE copy runs on its own stream, so first complete work ordered
        # before this allocation on PyTorch's stream and then wait for the copy.
        stream.synchronize()
        imported = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(output))
        source = view.get_buffer()
        if source.byte_length() != view.byte_length:
            raise NotImplementedError(
                "IREE pooled result uses a nontrivial buffer view"
            )
        semaphore = device.create_semaphore(0)
        device.queue_copy(
            source, imported.get_buffer(), [(semaphore, 0)], [(semaphore, 1)]
        )
        rt.HalFence.create_at(semaphore, 1).wait()
        pooled_outputs.append(output)

    def replace(value):
        """Replace foreign-output markers by their PyTorch-owned views."""
        if isinstance(value, _ForeignResult):
            return owned[value.position]
        if isinstance(value, _PooledResult):
            return pooled_outputs[value.position]
        if isinstance(value, (tuple, list)):
            return type(value)(replace(item) for item in value)
        return value

    return tuple(replace(result) for result in converted)


class _ForeignResult:
    """Mark an IREE result until its overlapping allocation is reconstructed."""

    def __init__(self, position: int):
        """Record the foreign tensor's position in the conversion batch."""
        self.position = position


class _PooledResult:
    """Mark an IREE pool allocation until copied into PyTorch-owned storage."""

    def __init__(self, position: int):
        """Record the pooled buffer's position in the conversion batch."""
        self.position = position


def runtime_segment_handlers(
    vmfb_path: Path, partitioned_module, *, device: str | None = None
) -> dict[str, Callable]:
    """Create VMFB handlers using signatures and aliases from native IR.

    Each handler accepts ``(graph, args)`` and returns the segment's flat FX
    live-outs. The first call on a CUDA device loads and caches its VM context.
    Scalar-only segments require ``device`` to name a CUDA index. IREE's
    synchronous ABI and stream waits preclude CUDA graph capture.
    """
    functions = _segment_functions(partitioned_module)
    runtime = _RuntimeModule(vmfb_path, device, tuple(functions))
    return {
        name: _SegmentHandler(
            _SegmentTarget(
                runtime,
                name,
                len(function.regions[0].blocks[0].arguments),
                _segment_result_types(function),
                _segment_output_aliases(function),
            )
        )
        for name, function in functions.items()
    }
