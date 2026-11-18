"""Tests for stitching IREE-compiled segments and vLLM custom ops into one FX graph.

The pipeline under test: partition at custom ops -> real `iree-compile` ->
flow-phase IR of each segment -> FX nodes (opaque kernel calls plus the
non-kernel ops IREE left behind) spliced into the driver's graph.
"""

import contextlib
import io
import operator
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

import iree_fx_reassembly
import iree_phase_dump
import simple_llm
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d

IREE_COMPILE = os.environ.get("IREE_COMPILE") or shutil.which("iree-compile")

# `empty` (dynamic size), a kernel-worthy elementwise op, a real vLLM custom op
# that mutates `%out` in place, and a consumer of the mutated buffer. The
# custom op splits the graph into two IREE segments.
RMS_NORM_SOURCE = """
module {
  func.func @main(%arg0: !torch.vtensor<[?,8],bf16>, %arg1: !torch.vtensor<[8],bf16>) -> !torch.vtensor<[?,8],bf16> {
    %int0 = torch.constant.int 0
    %int8 = torch.constant.int 8
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %dev = torch.constant.device "cuda:0"
    %eps = torch.constant.float 1.000000e-06
    %n = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,8],bf16>, !torch.int -> !torch.int
    %size = torch.prim.ListConstruct %n, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
    %out = torch.aten.empty.memory_format %size, %int15, %none, %dev, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,8],bf16>
    %a = torch.aten.relu %arg0 : !torch.vtensor<[?,8],bf16> -> !torch.vtensor<[?,8],bf16>
    torch.operator "torch._C.rms_norm"(%out, %a, %arg1, %eps) : (!torch.vtensor<[?,8],bf16>, !torch.vtensor<[?,8],bf16>, !torch.vtensor<[8],bf16>, !torch.float) -> ()
    %b = torch.aten.neg %out : !torch.vtensor<[?,8],bf16> -> !torch.vtensor<[?,8],bf16>
    return %b : !torch.vtensor<[?,8],bf16>
  }
}
"""


def _parse(source):
    context = ir.Context()
    torch_d.register_dialect(context)
    return ir.Module.parse(source, context)


def _compile_partition(source, directory):
    """Partition `source` and compile it with the real IREE compiler."""
    result = iree_phase_dump.partition_at_custom_ops(_parse(source))
    out_dir = Path(directory)
    with contextlib.redirect_stdout(io.StringIO()):
        iree_phase_dump.write_partition_dump(result, out_dir)
    run = iree_phase_dump.run_iree_compile(
        out_dir / "partitioned.mlir", out_dir, compiler=IREE_COMPILE, cuda_target="sm_80"
    )
    assert run.ok, run.message
    return result, run


def _reassemble(result, run):
    """Rebuild one FX graph from the driver and the compiled flow-phase segments."""
    flow_ir = next(run.phases_dir.glob("*.6.flow.mlir")).read_text()
    handlers = iree_fx_reassembly.flow_segment_handlers(flow_ir, device=result.device)
    return simple_llm.mlir_to_fx(result.driver, segment_handlers=handlers)


@unittest.skipUnless(IREE_COMPILE, "iree-compile not found (set IREE_COMPILE)")
class ReassemblyTest(unittest.TestCase):
    """Stitch real IREE output for a graph with one in-place vLLM custom op."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        result, run = _compile_partition(RMS_NORM_SOURCE, cls.directory.name)
        cls.graph = _reassemble(result, run).graph

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _calls(self):
        return [node for node in self.graph.nodes if node.op == "call_function"]

    def test_single_graph_keeps_original_signature(self):
        placeholders = [node.name for node in self.graph.nodes if node.op == "placeholder"]
        outputs = [node for node in self.graph.nodes if node.op == "output"]

        self.assertEqual(placeholders, ["arg0", "arg1"])
        self.assertEqual(len(outputs), 1)

    def test_custom_op_keeps_its_original_call_form(self):
        rms_norm = [n for n in self._calls() if n.target is torch.ops._C.rms_norm.default]

        self.assertEqual(len(rms_norm), 1)
        buffer, normed_input, weight, eps = rms_norm[0].args
        self.assertEqual(buffer.target, torch.ops.aten.empty.memory_format)
        # The kernel computes on a flattened view and IREE reshapes its result
        # back (`flow.tensor.reshape`), so kernel -> view -> custom op.
        self.assertEqual(normed_input.target, torch.ops.aten.view.default)
        self.assertIsInstance(normed_input.args[0].target, iree_fx_reassembly.IreeKernel)
        self.assertEqual(weight.name, "arg1")
        self.assertEqual(eps, 1e-6)

    def test_kernels_surround_the_custom_op_in_execution_order(self):
        order = [
            "kernel" if isinstance(n.target, iree_fx_reassembly.IreeKernel)
            else "custom" if n.target is torch.ops._C.rms_norm.default
            else None
            for n in self._calls()
        ]
        order = [kind for kind in order if kind]

        self.assertEqual(order[0], "kernel")
        self.assertEqual(order[-1], "kernel")
        self.assertEqual(order.count("custom"), 1)
        self.assertLess(order.index("custom"), len(order) - 1)

    def test_non_kernel_flow_ops_are_converted_to_aten(self):
        empty = next(n for n in self._calls() if n.target == torch.ops.aten.empty.memory_format)

        # `flow.tensor.empty` has no device; it is restored from the source graph.
        self.assertEqual(empty.kwargs["device"], torch.device("cuda:0"))
        self.assertEqual(empty.kwargs["dtype"], torch.bfloat16)
        sym_sizes = [n for n in self._calls() if n.target == torch.ops.aten.sym_size.int]
        self.assertTrue(sym_sizes, "hal.buffer_view.dim should become aten.sym_size.int")
        # The flattened kernel size `dim0 * 8` is index arithmetic on that dim.
        products = [n for n in self._calls() if n.target is operator.mul]
        self.assertTrue(products)
        self.assertIn(8, products[0].args)

    def test_no_iree_dialect_operation_leaks_into_the_graph(self):
        for node in self._calls():
            self.assertTrue(
                node.target is operator.mul
                or isinstance(node.target, (torch._ops.OpOverload, iree_fx_reassembly.IreeKernel)),
            )

    def test_kernel_nodes_carry_operands_and_workload(self):
        kernel = next(
            n for n in self._calls() if isinstance(n.target, iree_fx_reassembly.IreeKernel)
        )

        self.assertTrue(kernel.target.export.startswith("segment_"))
        self.assertIn("workload", kernel.kwargs)
        self.assertGreaterEqual(len(kernel.args), 1)


class UnsupportedOpTest(unittest.TestCase):
    """Unseen IREE ops must fail with their name, never be skipped."""

    def test_unconverted_flow_ops_fail_loudly_with_the_op_name(self):
        flow_ir = """
        module {
          util.func public @segment_0$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view {
            %d = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
            %s = arith.addi %d, %d : index
            util.return %arg0 : !hal.buffer_view
          }
        }
        """
        handlers = iree_fx_reassembly.flow_segment_handlers(flow_ir, device="cuda:0")

        with self.assertRaisesRegex(NotImplementedError, "arith.addi"):
            handlers["segment_0"](simple_llm.fx.Graph(), [object()])


def _relu_rms_norm_neg_graph():
    """A Dynamo-like graph: ordinary op, in-place vLLM custom op, ordinary op."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    out = graph.placeholder("out")
    relu = graph.call_function(torch.ops.aten.relu.default, (x,))
    graph.call_function(torch.ops._C.rms_norm.default, (out, relu, weight, 1e-6))
    graph.output((graph.call_function(torch.ops.aten.neg.default, (out,)),))
    gm = fx.GraphModule(nn.Module(), graph)

    # The importer types values from fake-tensor metadata.
    mode = FakeTensorMode()
    matrix = mode.from_tensor(torch.empty((4, 8), dtype=torch.bfloat16))
    vector = mode.from_tensor(torch.empty((8,), dtype=torch.bfloat16))
    for node in gm.graph.nodes:
        if node.op == "placeholder" or node.op == "call_function":
            value = vector if node.name == "weight" else matrix
            if node.target is torch.ops._C.rms_norm.default:
                value = None  # in-place op: no result
            node.meta["val"] = value
            node.meta["example_value"] = value
    return gm


@unittest.skipUnless(IREE_COMPILE, "iree-compile not found (set IREE_COMPILE)")
class BackendReassemblyTest(unittest.TestCase):
    """The backend shows the reassembled graph but must not change its own result."""

    def test_backend_prints_reassembled_graph_but_returns_roundtrip_graph(self):
        environment = {
            "IREE_PHASE_DUMP_DIR": "",
            "IREE_COMPILE": IREE_COMPILE,
            "IREE_CUDA_TARGET": "sm_80",
        }
        with tempfile.TemporaryDirectory() as directory:
            environment["IREE_PHASE_DUMP_DIR"] = directory
            captured = io.StringIO()
            with mock.patch.dict(os.environ, environment), contextlib.redirect_stdout(captured):
                forward = simple_llm.mlir_roundtrip_backend(_relu_rms_norm_neg_graph(), [])
            saved = (Path(directory) / "0" / "reassembled_fx.txt").read_text()

        reassembled = captured.getvalue().split("Reassembled FX", 1)[1]
        self.assertIn(saved.strip(), reassembled)
        self.assertIn("opcode", reassembled)  # tabular form
        self.assertIn("graph():", reassembled)  # graph form
        self.assertIn("iree_kernel.", reassembled)
        self.assertIn("rms_norm", reassembled)

        # What the backend returns is still the plain round-trip graph.
        returned = forward.__self__.graph
        self.assertFalse(
            any(isinstance(n.target, iree_fx_reassembly.IreeKernel) for n in returned.nodes)
        )


if __name__ == "__main__":
    unittest.main()
