"""Splice IREE flow-phase segments into FX with kernel dispatch calls.

Flow is the last tensor-level phase with segment boundaries and dispatches.
The coarse-fences ABI's imports, exports and barriers only change IREE's
runtime representation or ordering; they are aliases in the reconstructed FX
graph. A compiler-provided executor makes each kernel call executable.
"""

import operator
import re
from collections.abc import Callable
from typing import Any

import torch
from iree.compiler import ir

ASYNC_SUFFIX = "$async"
_ELEMENT_TYPE_TO_DTYPE = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i1": torch.bool,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
}
_ENTRY_POINT = re.compile(r"@(?P<executable>[^:]+)::@(?P<export>.+)")


class IreeKernel:
    """FX call target identifying one IREE executable export.

    A target built without an executor is inspection-only. Compiled graphs
    attach an executor that loads and launches the corresponding kernel.
    """

    def __init__(self, executable: str, export: str, executor=None):
        """Record the executable, export and optional kernel executor."""
        self.executable = executable
        self.export = export
        self.__name__ = export
        self.__module__ = "iree_kernel"
        self._executor = executor

    def __call__(self, *args, **kwargs):
        """Run the configured kernel, or reject an inspection-only target."""
        if self._executor is None:
            raise NotImplementedError(
                f"{self.export}: inspection-only opaque target has no executor"
            )
        return self._executor(*args, **kwargs)

    def __repr__(self):
        """Display the export when inspecting FX graphs."""
        return f"IreeKernel({self.export})"


def _dtype(element_type) -> torch.dtype:
    """Map signless IREE tensor element types to PyTorch dtypes."""
    try:
        return _ELEMENT_TYPE_TO_DTYPE[str(element_type)]
    except KeyError as exc:
        raise NotImplementedError(
            f"No torch dtype for IREE element type {element_type}"
        ) from exc


class _SegmentConverter:
    """Translate one async function's SSA values into FX values.

    Values may be FX nodes, scalar constants, or aliases for ABI operations.
    Fence arguments are synchronization plumbing, not segment data inputs.
    """

    def __init__(self, func, graph, args, kernels, device, kernel_factory):
        """Bind segment data arguments to the caller's FX values."""
        self.graph = graph
        self.kernels = kernels
        self.device = device
        self.kernel_factory = kernel_factory
        self.name = ir.StringAttr(func.attributes["sym_name"]).value
        self.block = func.regions[0].blocks[0]
        self.values: dict[Any, Any] = {}
        data_args = [
            arg for arg in self.block.arguments if str(arg.type) != "!hal.fence"
        ]
        if len(data_args) != len(args):
            raise ValueError(
                f"{self.name}: called with {len(args)} operands but the compiled "
                f"function takes {len(data_args)} data arguments"
            )
        self.values.update(zip(data_args, args))

    def convert(self) -> list:
        """Append the function's operations and return its data results."""
        for op in self.block.operations:
            operation = op.operation
            if operation.name == "util.return":
                return [
                    self.values[v]
                    for v in operation.operands
                    if str(v.type) != "!hal.fence"
                ]
            method = _CONVERTERS.get(operation.name)
            if method is None:
                raise NotImplementedError(
                    f"No FX conversion for IREE operation '{operation.name}' in {self.name}"
                )
            getattr(self, method)(operation)
        raise ValueError(f"{self.name}: function has no util.return")

    def _alias_first_operand(self, op):
        """Erase tensor import/export and integer-to-index representation changes."""
        self.values[op.results[0]] = self.values[op.operands[0]]

    def _barrier(self, op):
        """Forward each joined tensor; the final fence operand only orders work."""
        for result, source in zip(op.results, op.operands):
            self.values[result] = self.values[source]

    def _buffer_view_dim(self, op):
        """Recover a dynamic input dimension via an FX symbolic size query."""
        dim = ir.IntegerAttr(op.attributes["index"]).value
        self.values[op.results[0]] = self.graph.call_function(
            torch.ops.aten.sym_size.int, (self.values[op.operands[0]], dim)
        )

    def _shape(self, tensor_type, dynamic_dims) -> list:
        """Interleave dynamic dimension values with static tensor dimensions."""
        dims = iter(dynamic_dims)
        shape = [
            next(dims) if ir.ShapedType.is_dynamic_size(d) else d
            for d in ir.RankedTensorType(tensor_type).shape
        ]
        if next(dims, None) is not None:
            raise ValueError(
                f"{self.name}: too many dynamic dimensions for {tensor_type}"
            )
        return shape

    def _index_constant(self, op):
        """Represent integer dimension constants directly as Python integers."""
        try:
            value = ir.IntegerAttr(op.attributes["value"]).value
        except ValueError as exc:
            raise NotImplementedError(
                f"arith.constant with non-integer value {op.attributes['value']} in {self.name}"
            ) from exc
        self.values[op.results[0]] = value

    def _index_mul(self, op):
        """Multiply dimension values, folding fully static calculations."""
        lhs, rhs = (self.values[v] for v in op.operands)
        self.values[op.results[0]] = (
            lhs * rhs
            if isinstance(lhs, int) and isinstance(rhs, int)
            else self.graph.call_function(operator.mul, (lhs, rhs))
        )

    def _tensor_empty(self, op):
        """Create an ATen empty tensor with the flow result's shape and dtype."""
        result_type = op.results[0].type
        size = self._shape(result_type, [self.values[v] for v in op.operands])
        kwargs = {"dtype": _dtype(ir.RankedTensorType(result_type).element_type)}
        if self.device is not None:
            kwargs["device"] = torch.device(self.device)
        self.values[op.results[0]] = self.graph.call_function(
            torch.ops.aten.empty.memory_format, (size,), kwargs
        )

    def _tensor_reshape(self, op):
        """Translate a flow reshape into a no-copy ATen view.

        Source dynamic dimensions precede result dynamic dimensions among the
        operands; only the latter specify the resulting view shape.
        """
        source_type = ir.RankedTensorType(op.operands[0].type)
        source_dims = sum(ir.ShapedType.is_dynamic_size(d) for d in source_type.shape)
        result_dims = [self.values[v] for v in op.operands[1 + source_dims :]]
        size = self._shape(op.results[0].type, result_dims)
        self.values[op.results[0]] = self.graph.call_function(
            torch.ops.aten.view.default, (self.values[op.operands[0]], size)
        )

    def _dispatch(self, op):
        """Emit a kernel call and unpack its results when necessary.

        Operand segments are workload, arguments, argument dimensions, and
        result dimensions. Only arguments are kernel inputs; workload and
        result types remain attached to the call as launch/result metadata.
        Tied outputs are represented as call results with their input ties in
        ``iree_tied_operands`` metadata. Inspection-only targets cannot execute;
        executable targets also receive the mapping so they can reuse the
        correct input buffers.
        """
        sizes = list(ir.DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
        num_workload, num_args, num_argument_dims, num_result_dims = sizes
        tied = [
            ir.IntegerAttr(a).value
            for a in ir.ArrayAttr(op.attributes["tied_operands"])
        ]
        entries = ir.ArrayAttr(op.attributes["entry_points"])
        if len(entries) != 1:
            raise NotImplementedError(
                f"flow.dispatch with multiple entry points in {self.name}"
            )
        entry = _ENTRY_POINT.fullmatch(str(entries[0]))
        if entry is None:
            raise NotImplementedError(
                f"Unsupported flow.dispatch entry point {entries[0]} in {self.name}"
            )
        key = (entry.group("executable"), entry.group("export"))
        if key not in self.kernels:
            executor = (
                self.kernel_factory(*key) if self.kernel_factory is not None else None
            )
            self.kernels[key] = IreeKernel(*key, executor=executor)
        operands = [self.values[v] for v in op.operands]
        workload = operands[:num_workload]
        args_start = num_workload
        args_end = args_start + num_args
        dims_end = args_end + num_argument_dims
        result_end = dims_end + num_result_dims
        args = operands[args_start:args_end]
        kwargs = {"workload": workload}
        if self.kernel_factory is not None:
            # Flow separates dynamic result dimensions from kernel arguments.
            # Preserve them in the FX call so the launcher can allocate outputs.
            result_dims = operands[dims_end:result_end]
            shapes, dtypes = [], []
            offset = 0
            for result in op.results:
                tensor_type = ir.RankedTensorType(result.type)
                count = sum(
                    ir.ShapedType.is_dynamic_size(dim) for dim in tensor_type.shape
                )
                shapes.append(
                    self._shape(result.type, result_dims[offset : offset + count])
                )
                dtypes.append(_dtype(tensor_type.element_type))
                offset += count
            if offset != len(result_dims):
                raise ValueError(
                    f"{self.name}: unexpected result dimensions in flow.dispatch"
                )
            kwargs.update(
                argument_dims=operands[args_end:dims_end],
                result_shapes=shapes,
                result_dtypes=dtypes,
                tied_operands=tied,
            )
        node = self.graph.call_function(self.kernels[key], tuple(args), kwargs)
        node.meta["iree_result_types"] = [str(result.type) for result in op.results]
        if any(t != -1 for t in tied):
            node.meta["iree_tied_operands"] = tied
        if len(op.results) == 1:
            self.values[op.results[0]] = node
        else:
            for index, result in enumerate(op.results):
                self.values[result] = self.graph.call_function(
                    operator.getitem, (node, index)
                )


_CONVERTERS = {
    "hal.tensor.import": "_alias_first_operand",
    "hal.tensor.export": "_alias_first_operand",
    "arith.index_cast": "_alias_first_operand",
    "hal.tensor.barrier": "_barrier",
    "hal.buffer_view.dim": "_buffer_view_dim",
    "arith.constant": "_index_constant",
    "arith.muli": "_index_mul",
    "flow.tensor.empty": "_tensor_empty",
    "flow.tensor.reshape": "_tensor_reshape",
    "flow.dispatch": "_dispatch",
}


class _SegmentHandler:
    """Keep the parsed module alive while splicing one function into FX."""

    def __init__(self, module, func, kernels, device, kernel_factory):
        """Retain the function, module, shared kernels and source device."""
        self._module = module
        self._func = func
        self._kernels = kernels
        self._device = device
        self._kernel_factory = kernel_factory

    def __call__(self, graph, args) -> list:
        """Append this segment's FX nodes and return its live-out values."""
        return _SegmentConverter(
            self._func, graph, args, self._kernels, self._device, self._kernel_factory
        ).convert()


def flow_segment_handlers(
    flow_ir: str, *, device: str | None, kernel_factory=None
) -> dict[str, Callable]:
    """Parse flow IR and return segment-name to FX-splicing callable mappings.

    Each handler accepts ``(graph, args)`` with live-ins in source order and
    returns live-outs in order. All handlers share kernel target identities.
    ``device`` restores the source device for flow tensor allocations; None
    leaves PyTorch's default device in effect.
    ``kernel_factory``, when provided, accepts an executable and export name
    and returns the callable used to execute that kernel.
    Tied dispatches retain ``iree_tied_operands`` node metadata in both modes;
    without a factory their targets remain inspection-only and cannot execute.
    """
    module = ir.Module.parse(flow_ir, ir.Context())
    kernels: dict[tuple[str, str], IreeKernel] = {}
    handlers = {}
    for op in module.body.operations:
        if op.operation.name != "util.func":
            continue
        name = ir.StringAttr(op.attributes["sym_name"]).value
        if name.endswith(ASYNC_SUFFIX):
            handlers[name.removesuffix(ASYNC_SUFFIX)] = _SegmentHandler(
                module, op, kernels, device, kernel_factory
            )
    return handlers
