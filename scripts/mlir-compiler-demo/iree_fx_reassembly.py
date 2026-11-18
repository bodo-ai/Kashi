"""Convert IREE-compiled segments back into torch.fx nodes.

Pipeline context: the partitioner (`iree_phase_dump`) sends the ordinary ops
between vLLM custom ops to `iree-compile` as `segment_<N>` functions. This
module reads those functions back from IREE's *flow* phase and turns each one
into FX nodes so they can be spliced into a single graph together with the
custom-op calls (see `simple_llm.mlir_to_fx`, `segment_handlers`).

Why the flow phase
------------------
It is the last phase that is still tensor-level SSA with the original function
boundaries, and it is where IREE has already decided what becomes a kernel:

* every kernel is a `flow.dispatch @executable::@export(...)` call, which maps
  one-to-one onto an opaque FX call (`IreeKernel`);
* whatever IREE did *not* turn into a kernel stays as a handful of small ops
  (`flow.tensor.reshape`, `flow.tensor.empty`, dimension queries and the index
  arithmetic on them), which must be converted back to ATen / Python here;
* the segment's ABI wrapper (`hal.tensor.import/export/barrier`, fences) is pure
  plumbing for IREE's runtime and maps to identity.

Later phases (stream, hal) have already lowered to buffers, allocations and
command buffers, which have no sensible FX counterpart.

Anything not in `_CONVERTERS` raises `NotImplementedError` naming the op rather
than being silently dropped, so a new IREE version or a new model that emits an
unseen op fails loudly.
"""

import operator
import re
from typing import Any, Callable

import torch
from iree.compiler import ir

# IREE's "coarse-fences" ABI exports every segment as `<name>$async`, taking a
# wait fence and a signal fence after its data arguments. The plain `<name>`
# wrapper only creates those fences and calls the async function.
ASYNC_SUFFIX = "$async"

# Element types that appear as tensor dtypes in IREE IR. IREE integers are
# signless, so `i8` is mapped to the signed torch type.
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

# `flow.dispatch`'s `entry_points` attribute prints as `@executable::@export`.
_ENTRY_POINT = re.compile(r"@(?P<executable>[^:]+)::@(?P<export>.+)")


class IreeKernel:
    """Opaque FX call target standing for one IREE-generated GPU kernel.

    One instance exists per `flow.executable` export, shared by every dispatch
    site of that kernel. FX derives the node's printed name from `__name__` and
    `__module__`, so the node shows up as `iree_kernel.<export>`; the export
    name encodes the kernel's shape and dtype, e.g.
    `segment_1$async_dispatch_0_matmul_Dx4096x1024_bf16xbf16xf32`. The same name
    labels the PTX in `kernels.ptx`.

    Attributes:
        executable: Name of the containing `flow.executable`.
        export: Name of the kernel entry point (the PTX `.entry` name, with `$`
            sanitized to `_`).
    """

    def __init__(self, executable: str, export: str):
        self.executable = executable
        self.export = export
        self.__name__ = export
        self.__module__ = "iree_kernel"

    def __call__(self, *args, **kwargs):
        """Executing a kernel needs IREE's runtime, which is not wired up yet."""
        raise NotImplementedError(
            f"{self.export}: the reassembled graph carries kernel calls as opaque "
            "targets; launching them through the IREE runtime is not implemented"
        )

    def __repr__(self):
        return f"IreeKernel({self.export})"


def _dtype(element_type) -> torch.dtype:
    """Map an IREE tensor element type to the matching torch dtype."""
    try:
        return _ELEMENT_TYPE_TO_DTYPE[str(element_type)]
    except KeyError:
        raise NotImplementedError(f"No torch dtype for IREE element type {element_type}")


class _SegmentConverter:
    """Convert the body of one `segment_<N>$async` function into FX nodes.

    Every MLIR value the function defines is mapped, in `values`, to the FX
    value that now stands for it: an FX node, a placeholder handed in by the
    caller, a Python int (dimension sizes), or (for ABI plumbing) simply the
    value it wraps.
    """

    def __init__(self, func, graph, args, kernels, device):
        self.func = func
        self.graph = graph
        self.kernels = kernels
        self.device = device
        self.name = ir.StringAttr(func.attributes["sym_name"]).value
        self.block = func.regions[0].blocks[0]
        self.values: dict[Any, Any] = {}

        # Fence arguments carry synchronization only; data arguments appear in
        # the same order as the live-ins of the original torch segment.
        data_args = [a for a in self.block.arguments if str(a.type) != "!hal.fence"]
        if len(data_args) != len(args):
            raise ValueError(
                f"{self.name}: called with {len(args)} operands but the compiled "
                f"function takes {len(data_args)} data arguments"
            )
        self.values.update(zip(data_args, args))

    def convert(self) -> list:
        """Emit FX nodes for the body; return the FX values of the results."""
        for op in self.block.operations:
            operation = op.operation
            if operation.name == "util.return":
                return [
                    self.values[v] for v in operation.operands
                    if str(v.type) != "!hal.fence"
                ]
            method = _CONVERTERS.get(operation.name)
            if method is None:
                raise NotImplementedError(
                    f"No FX conversion for IREE operation '{operation.name}' "
                    f"in {self.name}"
                )
            getattr(self, method)(operation)
        raise ValueError(f"{self.name}: function has no util.return")

    # --- ABI plumbing: no computation, the value is just re-labelled ----------

    def _alias_first_operand(self, op):
        """`hal.tensor.import/export`, `arith.index_cast`: result is operand 0.

        A buffer_view <-> tensor change of representation, or an i64 -> index
        cast, is invisible in FX where tensors are tensors and ints are Python
        ints. Trailing operands (dynamic dims, wait fence) are metadata.
        """
        self.values[op.results[0]] = self.values[op.operands[0]]

    def _barrier(self, op):
        """`hal.tensor.barrier join(a, b, ...) => fence`: results are the joined values.

        The barrier only orders the tensors against the signal fence (last
        operand); nothing is computed.
        """
        for result, source in zip(op.results, op.operands):
            self.values[result] = self.values[source]

    # --- shape handling -------------------------------------------------------

    def _buffer_view_dim(self, op):
        """`hal.buffer_view.dim<%bv>[i]` -> `aten.sym_size.int(tensor, i)`.

        This is how IREE reads a dynamic dimension off an incoming buffer; in
        FX the same query is a size lookup on the tensor itself.
        """
        dim = ir.IntegerAttr(op.attributes["index"]).value
        self.values[op.results[0]] = self.graph.call_function(
            torch.ops.aten.sym_size.int, (self.values[op.operands[0]], dim)
        )

    def _shape(self, tensor_type, dynamic_dims) -> list:
        """Build a size list: static dims as ints, `?` dims from `dynamic_dims` in order."""
        dims = iter(dynamic_dims)
        return [
            next(dims) if ir.ShapedType.is_dynamic_size(d) else d
            for d in ir.RankedTensorType(tensor_type).shape
        ]

    # IREE computes flattened sizes (e.g. the 1-D view an elementwise kernel
    # runs over is `dim0 * 8` long) with scalar index ops. These are plain
    # Python integer arithmetic in FX.

    def _index_constant(self, op):
        """`arith.constant N : index` -> the Python int `N`.

        Only integer/index constants are supported; they are dimension sizes,
        which FX represents as plain ints rather than nodes.
        """
        try:
            value = ir.IntegerAttr(op.attributes["value"]).value
        except ValueError:  # float / dense constants are not dimension arithmetic
            raise NotImplementedError(
                f"arith.constant with non-integer value {op.attributes['value']} in {self.name}"
            )
        self.values[op.results[0]] = value

    def _index_mul(self, op):
        """`arith.muli a, b` -> `a * b`, folded when both are known ints.

        Otherwise (a dynamic dimension is involved) an `operator.mul` node is
        emitted. Python ints do not overflow, so the `nsw` flag is irrelevant.
        """
        lhs, rhs = (self.values[v] for v in op.operands)
        if isinstance(lhs, int) and isinstance(rhs, int):
            self.values[op.results[0]] = lhs * rhs
        else:
            self.values[op.results[0]] = self.graph.call_function(operator.mul, (lhs, rhs))

    # --- non-kernel tensor ops IREE keeps in the flow phase --------------------

    def _tensor_empty(self, op):
        """`flow.tensor.empty : tensor<?x1024xbf16>{%d}` -> `aten.empty.memory_format`.

        The operands are the dynamic dimensions. `flow.tensor.empty` has no
        device (IREE tensors live on the compile target), so the device of the
        source graph is restored; without one, torch's default applies.
        """
        result_type = op.results[0].type
        size = self._shape(result_type, [self.values[v] for v in op.operands])
        kwargs = {"dtype": _dtype(ir.RankedTensorType(result_type).element_type)}
        if self.device is not None:
            kwargs["device"] = torch.device(self.device)
        self.values[op.results[0]] = self.graph.call_function(
            torch.ops.aten.empty.memory_format, (size,), kwargs
        )

    def _tensor_reshape(self, op):
        """`flow.tensor.reshape` -> `aten.view`: a no-copy change of shape.

        Operands are `[source, source dynamic dims..., result dynamic dims...]`,
        so the result's dynamic dims follow the source's.
        """
        source_type = ir.RankedTensorType(op.operands[0].type)
        num_source_dynamic = sum(ir.ShapedType.is_dynamic_size(d) for d in source_type.shape)
        result_dims = [self.values[v] for v in op.operands[1 + num_source_dynamic:]]
        size = self._shape(op.results[0].type, result_dims)
        self.values[op.results[0]] = self.graph.call_function(
            torch.ops.aten.view.default, (self.values[op.operands[0]], size)
        )

    # --- kernels -----------------------------------------------------------------

    def _dispatch(self, op):
        """`flow.dispatch @exe::@export[workload](args...)` -> one opaque kernel call.

        Operands are grouped by `operandSegmentSizes` as
        `[workload, arguments, argument dims, result dims]`. The kernel receives
        `arguments` (tensors and scalar index values, exactly as the kernel
        signature lists them); the dimension groups only describe the tensor
        types and are not kernel inputs. `workload` sizes the launch grid, so it
        is kept as a keyword. Result types are kept in `node.meta` because the
        FX value itself carries no shape.
        """
        num_workload, num_args = list(ir.DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))[:2]
        tied = [ir.IntegerAttr(a).value for a in ir.ArrayAttr(op.attributes["tied_operands"])]
        if any(t != -1 for t in tied):
            # A tied result aliases an operand (in-place kernel); that needs
            # mutation semantics FX nodes here do not model.
            raise NotImplementedError(f"flow.dispatch with tied operands in {self.name}")

        entry = _ENTRY_POINT.fullmatch(str(ir.ArrayAttr(op.attributes["entry_points"])[0]))
        key = (entry.group("executable"), entry.group("export"))
        kernel = self.kernels.setdefault(key, IreeKernel(*key))

        operands = [self.values[v] for v in op.operands]
        workload = operands[:num_workload]
        args = operands[num_workload : num_workload + num_args]
        node = self.graph.call_function(kernel, tuple(args), {"workload": workload})
        node.meta["iree_result_types"] = [str(r.type) for r in op.results]

        if len(op.results) == 1:
            self.values[op.results[0]] = node
        else:
            # Several results come back as a tuple, unpacked the same way
            # `torch.prim.ListUnpack` is handled elsewhere in the demo.
            for index, result in enumerate(op.results):
                self.values[result] = self.graph.call_function(operator.getitem, (node, index))


# IREE op name -> `_SegmentConverter` method. `util.return` is handled inline.
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
    """Callable that splices one compiled segment into an FX graph.

    Holds the parsed MLIR module so the operations it points into stay alive.
    """

    def __init__(self, module, func, kernels, device):
        self._module = module
        self._func = func
        self._kernels = kernels
        self._device = device

    def __call__(self, graph, args) -> list:
        return _SegmentConverter(self._func, graph, args, self._kernels, self._device).convert()


def flow_segment_handlers(flow_ir: str, *, device: str | None) -> dict[str, Callable]:
    """Parse IREE flow-phase IR into one FX-splicing handler per segment.

    Args:
        flow_ir: Text of the flow-phase dump (`*.6.flow.mlir`).
        device: Device of the source graph (e.g. `cuda:0`), restored on tensors
            IREE allocates (`flow.tensor.empty`), or `None` to leave it default.

    Returns:
        `{"segment_<N>": handler}`. `handler(graph, args)` appends the segment's
        FX nodes to `graph`, where `args` are the FX values of the segment's
        live-ins in order, and returns the FX values of its live-outs in order.
        All handlers share one `IreeKernel` per export.
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
                module, op, kernels, device
            )
    return handlers
