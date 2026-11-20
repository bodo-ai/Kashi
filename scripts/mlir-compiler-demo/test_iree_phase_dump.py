import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d

_TEST_LIBRARY = torch.library.Library("iree_test", "FRAGMENT")
_TEST_LIBRARY.define("mutate_it(Tensor! x, Tensor y) -> ()")


# The mutation and allocation force a live value to cross the custom-op boundary.
TWO_SEGMENT_SOURCE = """
module {
  func.func @main(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %int1 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int6 = torch.constant.int 6
    %none = torch.constant.none
    %dev = torch.constant.device "cuda:0"
    %size = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
    %a = torch.aten.relu %arg0 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %s = torch.aten.add.Tensor %a, %arg0, %int1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    %buf = torch.aten.empty.memory_format %size, %int6, %none, %dev, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[4],f32>
    torch.operator "torch._C.test_mutate"(%buf, %s) : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> ()
    %b = torch.aten.neg %buf : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %c = torch.aten.add.Tensor %b, %s, %int1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    return %c : !torch.vtensor<[4],f32>
  }
}
"""

# Repeated boundaries, including one with no surrounding compute, exercise fragmentation.
FRAGMENTED_SOURCE = """
module {
  func.func @main(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %a = torch.aten.relu %arg0 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    torch.operator "torch._C.test_a"(%a) : (!torch.vtensor<[4],f32>) -> ()
    torch.operator "torch._C.test_b"(%a) : (!torch.vtensor<[4],f32>) -> ()
    %b = torch.aten.relu %a : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    torch.operator "torch._C.test_a"(%b) : (!torch.vtensor<[4],f32>) -> ()
    %c = torch.aten.relu %b : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    return %c : !torch.vtensor<[4],f32>
  }
}
"""

# The view result is an alias-like live-out, counted separately from ordinary tensors.
VIEW_SOURCE = """
module {
  func.func @main(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %int0 = torch.constant.int 0
    %a = torch.aten.relu %arg0 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %v = torch.aten.unsqueeze %a, %int0 : !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[1,4],f32>
    torch.operator "torch._C.test_view"(%a, %v) : (!torch.vtensor<[4],f32>, !torch.vtensor<[1,4],f32>) -> ()
    return %arg0 : !torch.vtensor<[4],f32>
  }
}
"""

# A scalar live-out cannot become a segment's tensor return value.
UNRETURNABLE_SOURCE = """
module {
  func.func @main(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
    %f = torch.aten.Float.Tensor %arg0 : !torch.vtensor<[],f32> -> !torch.float
    torch.operator "torch._C.test_scalar"(%f) : (!torch.float) -> ()
    return %arg0 : !torch.vtensor<[],f32>
  }
}
"""

TENSOR = "!torch.vtensor<[4],f32>"


def _parse(source):
    """Parse Torch-dialect MLIR text using a context with its dialect registered."""
    context = ir.Context()
    torch_d.register_dialect(context)
    return ir.Module.parse(source, context)


def _phase_dump():
    assert importlib.util.find_spec("iree_phase_dump") is not None, (
        "iree_phase_dump module is missing"
    )
    import iree_phase_dump

    return iree_phase_dump


def _function_text(module_text, name):
    """Return one function body from printed MLIR, through the next function."""
    start = module_text.index(f"func.func @{name}(")
    end = module_text.find("func.func @", start + 1)
    return module_text[start : end if end != -1 else len(module_text)]




class PartitionTest(unittest.TestCase):
    """Verify partition boundaries, value flow, and input-module preservation."""

    def test_partition_splits_at_custom_ops(self):
        phase_dump = _phase_dump()
        result = phase_dump.partition_at_custom_ops(_parse(TWO_SEGMENT_SOURCE))

        self.assertEqual(
            [segment.func_name for segment in result.segments],
            ["segment_0", "segment_1"],
        )
        self.assertNotIn("torch.operator", str(result.module))
        first, second = result.segments
        self.assertEqual(first.live_in_types, [TENSOR])
        self.assertEqual(first.live_out_types, [TENSOR, TENSOR])
        self.assertEqual(second.live_in_types, [TENSOR, TENSOR])
        self.assertEqual(second.live_out_types, [TENSOR])
        self.assertEqual(len(result.boundaries), 1)
        self.assertEqual(result.boundaries[0].name, "torch._C.test_mutate")
        self.assertEqual(result.boundaries[0].operand_types, [TENSOR, TENSOR])

    def test_partition_clones_constants_and_rewrites_device(self):
        phase_dump = _phase_dump()
        result = phase_dump.partition_at_custom_ops(_parse(TWO_SEGMENT_SOURCE))
        text = str(result.module)

        self.assertIn("torch.constant.int 1", _function_text(text, "segment_0"))
        self.assertIn("torch.constant.int 1", _function_text(text, "segment_1"))
        self.assertEqual(len(result.segments[1].live_in_types), 2)
        self.assertNotIn("cuda:0", text)
        self.assertIn('torch.constant.device "cpu"', text)

    def test_partition_leaves_original_module_unchanged(self):
        phase_dump = _phase_dump()
        module = _parse(TWO_SEGMENT_SOURCE)
        before = str(module)

        phase_dump.partition_at_custom_ops(module)

        self.assertEqual(str(module), before)

    def test_partition_reports_unreturnable_live_outs(self):
        phase_dump = _phase_dump()
        result = phase_dump.partition_at_custom_ops(_parse(UNRETURNABLE_SOURCE))

        self.assertEqual(result.segments[0].unsupported_live_outs, ["!torch.float"])

    def test_partition_emits_driver_that_calls_segments_in_graph_order(self):
        phase_dump = _phase_dump()
        result = phase_dump.partition_at_custom_ops(_parse(TWO_SEGMENT_SOURCE))
        driver = str(result.driver)

        # The driver is the original function with each segment collapsed into
        # one call, so wiring between segments and custom ops is preserved.
        call_0 = driver.index("call @segment_0")
        custom_op = driver.index("torch._C.test_mutate")
        call_1 = driver.index("call @segment_1")
        self.assertLess(call_0, custom_op)
        self.assertLess(custom_op, call_1)
        self.assertNotIn("torch.aten.", driver)
        self.assertEqual(result.device, "cuda:0")

    def test_partition_has_no_driver_when_a_live_out_cannot_cross(self):
        phase_dump = _phase_dump()
        result = phase_dump.partition_at_custom_ops(_parse(UNRETURNABLE_SOURCE))

        self.assertIsNone(result.driver)



class StatisticsTest(unittest.TestCase):
    """Check partition statistics and the emitted summary artifacts."""

    def test_mutated_operand_indices_from_registered_schema(self):
        phase_dump = _phase_dump()

        self.assertEqual(
            phase_dump.mutated_operand_indices("torch.iree_test.mutate_it"), [0]
        )
        self.assertIsNone(phase_dump.mutated_operand_indices("torch._C.no_such_op"))

    def test_partition_stats_counts_fragmentation(self):
        phase_dump = _phase_dump()
        result = phase_dump.partition_at_custom_ops(_parse(FRAGMENTED_SOURCE))
        stats = phase_dump.compute_partition_stats(result)

        self.assertEqual(stats["segment_count"], 4)
        self.assertEqual(stats["empty_segments"], 1)
        self.assertEqual(
            stats["custom_ops_by_name"], {"torch._C.test_a": 2, "torch._C.test_b": 1}
        )
        self.assertEqual(stats["distinct_segment_structures"], 2)
        self.assertEqual(stats["compute_ops_in_segments_pct"], 50.0)
        self.assertEqual(
            stats["boundary_traffic"],
            {"live_in_values": 3, "live_out_values": 3, "distinct_tensors_crossing": 3},
        )
        self.assertEqual(
            stats["ops_per_segment"], {"min": 0, "median": 1, "mean": 0.75, "max": 1}
        )

    def test_partition_counts_view_live_outs(self):
        phase_dump = _phase_dump()
        result = phase_dump.partition_at_custom_ops(_parse(VIEW_SOURCE))

        self.assertEqual(result.segments[0].view_live_outs, 1)
        self.assertEqual(phase_dump.compute_partition_stats(result)["view_live_outs"], 1)

    def test_write_partition_dump_writes_all_files(self):
        phase_dump = _phase_dump()
        source = TWO_SEGMENT_SOURCE.replace(
            "torch._C.test_mutate", "torch.iree_test.mutate_it"
        )
        result = phase_dump.partition_at_custom_ops(_parse(source))

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                stats = phase_dump.write_partition_dump(result, out_dir)

            self.assertEqual((out_dir / "partitioned.mlir").read_text(), str(result.module))
            self.assertEqual((out_dir / "driver.mlir").read_text(), str(result.driver))
            self.assertEqual(
                json.loads((out_dir / "partition_stats.json").read_text()), stats
            )
            segments_text = (out_dir / "segments.txt").read_text()
            self.assertIn("segment_0", segments_text)
            self.assertIn("segment_1", segments_text)
            boundaries_text = (out_dir / "boundaries.txt").read_text()
            self.assertIn("torch.iree_test.mutate_it", boundaries_text)
            self.assertIn("mutated_operands=[0]", boundaries_text)

        self.assertIn(
            "IREE partition: 2 segments, 1 custom-op boundaries, "
            "83.3% of compute ops in segments",
            captured.getvalue(),
        )



IREE_COMPILE = os.environ.get("IREE_COMPILE") or shutil.which("iree-compile")


class IreeRunTest(unittest.TestCase):
    """Exercise compiler invocation, phase dumps, and CUDA target selection."""

    def test_cuda_target_from_capability(self):
        phase_dump = _phase_dump()

        self.assertEqual(phase_dump.cuda_target_from_capability((12, 0)), "sm_120")
        self.assertEqual(phase_dump.cuda_target_from_capability((8, 9)), "sm_89")

    def test_run_iree_reports_missing_compiler_without_raising(self):
        phase_dump = _phase_dump()
        missing = "/nonexistent/iree-compile"
        with tempfile.TemporaryDirectory() as directory:
            mlir_path = Path(directory) / "partitioned.mlir"
            mlir_path.write_text("module {}")

            result = phase_dump.run_iree_compile(
                mlir_path, Path(directory) / "out", compiler=missing, cuda_target="sm_80"
            )

        self.assertFalse(result.ok)
        self.assertIn(missing, result.message)

    def test_dump_iree_phases_never_raises_and_numbers_directories(self):
        phase_dump = _phase_dump()
        module = _parse(TWO_SEGMENT_SOURCE)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"IREE_COMPILE": "/nonexistent/iree-compile"}
        ), contextlib.redirect_stdout(io.StringIO()):
            phase_dump.dump_iree_phases(module, Path(directory))
            phase_dump.dump_iree_phases(module, Path(directory))

            for number in ("0", "1"):
                self.assertTrue((Path(directory) / number / "partitioned.mlir").exists())
                self.assertTrue((Path(directory) / number / "partition_stats.json").exists())

    @unittest.skipUnless(IREE_COMPILE, "iree-compile not found (set IREE_COMPILE)")
    def test_run_iree_dumps_phases_for_partitioned_module(self):
        phase_dump = _phase_dump()
        result = phase_dump.partition_at_custom_ops(_parse(TWO_SEGMENT_SOURCE))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            io.StringIO()
        ):
            out_dir = Path(directory)
            phase_dump.write_partition_dump(result, out_dir)

            run = phase_dump.run_iree_compile(
                out_dir / "partitioned.mlir",
                out_dir,
                compiler=IREE_COMPILE,
                cuda_target="sm_80",
            )

            self.assertTrue(run.ok, run.message)
            phase_names = [
                "input", "abi", "preprocessing", "global-optimization",
                "dispatch-creation", "flow", "stream", "executable-sources",
                "executable-configurations", "executable-targets",
            ]
            for number, name in enumerate(phase_names, start=1):
                self.assertEqual(
                    len(list(run.phases_dir.glob(f"*.{number}.{name}.mlir"))), 1, name
                )
            last_phase = next(run.phases_dir.glob("*.10.executable-targets.mlir"))
            self.assertIn("hal.executable.export", last_phase.read_text())

            # Kernels are dumped at the end of the same compile: raw files from
            # IREE plus one consolidated, human-readable PTX listing.
            self.assertEqual(run.kernels_path, out_dir / "kernels.ptx")
            kernels = run.kernels_path.read_text()
            self.assertIn(".visible .entry", kernels)
            self.assertIn("segment_0$async_dispatch_0", kernels)
            self.assertTrue(any((out_dir / "kernel_files").glob("*.ptx")))

            stats = phase_dump.add_iree_stats(out_dir / "partition_stats.json", run.phases_dir)
            self.assertGreaterEqual(stats["iree"]["unique_executables"], 1)
            self.assertGreaterEqual(stats["iree"]["dispatches_per_segment"]["segment_0"], 1)
            self.assertIn(
                "iree", json.loads((out_dir / "partition_stats.json").read_text())
            )

if __name__ == "__main__":
    unittest.main()
