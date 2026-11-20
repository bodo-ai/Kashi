"""Exercise the runtime segment bridge with compiled CUDA VM functions."""

import gc
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
compiler = pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

from kashi.iree_partition import PartitionResult, SegmentInfo
from kashi.iree_runtime import runtime_segment_handlers

MODULE = r"""
module {
  func.func @segment_0(%x: tensor<?x?xf32>, %n: i64)
      -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %x, %c0 : tensor<?x?xf32>
    %d1 = tensor.dim %x, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%d0, %d1) : tensor<?x?xf32>
    %one = arith.constant 1.0 : f32
    %y = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %one : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %y, %y, %x, %n : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, i64
  }
  func.func @segment_1(%n: i64) -> i64 {
    return %n : i64
  }
}
"""


@pytest.fixture(scope="module")
def compiled_module(tmp_path_factory):
    """Compile a real VMFB with dynamic tensors and multiple result types."""
    from iree.compiler import ir

    major, minor = torch.cuda.get_device_capability()
    blob = compiler.compile_str(
        MODULE,
        input_type="none",
        target_backends=["cuda"],
        extra_args=[f"--iree-cuda-target=sm_{major}{minor}"],
    )
    path = tmp_path_factory.mktemp("runtime") / "module.vmfb"
    path.write_bytes(blob)
    module = ir.Module.parse(MODULE, ir.Context())
    partition = PartitionResult(
        module=module,
        driver=None,
        device=None,
        segments=[
            SegmentInfo(
                0,
                "segment_0",
                ["!torch.vtensor<[?,?],f32>", "!torch.int"],
                ["!torch.vtensor<[?,?],f32>"] * 3 + ["!torch.int"],
                [],
            ),
            SegmentInfo(1, "segment_1", ["!torch.int"], ["!torch.int"], []),
            SegmentInfo(2, "segment_2", [], [], []),
        ],
    )
    return path, partition


def _build_graph(handler, arity):
    """Build a runnable FX graph around the real segment handler."""
    graph = torch.fx.Graph()
    inputs = [graph.placeholder(f"arg{i}") for i in range(arity)]
    results = handler(graph, inputs)
    graph.output(tuple(results))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_runtime_dynamic_results_alias_and_lifetime(compiled_module):
    """Dynamic results, input aliases, and retained IREE storage survive the call."""
    path, partition = compiled_module
    handlers = runtime_segment_handlers(path, partition)
    assert "segment_2" not in handlers
    graph = _build_graph(handlers["segment_0"], 2)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for rows in (7, 3):
            backing = torch.arange(rows * 5 + 2, dtype=torch.float32, device="cuda")
            source = backing[1:-1].reshape(rows, 5)
            result, result_alias, alias, scalar = graph(source, rows)
            assert result.device == source.device and result.dtype == torch.float32
            assert result.shape == source.shape and scalar == rows
            torch.testing.assert_close(result, source + 1)
            result_alias.add_(2)
            torch.testing.assert_close(result, source + 3)
            result_alias.sub_(2)
            alias.add_(100)
            torch.testing.assert_close(source, alias)
        del graph, handlers, backing, source, alias, result_alias
        gc.collect()
        torch.testing.assert_close(
            result,
            torch.arange(2, 17, device="cuda", dtype=torch.float32).reshape(3, 5),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_runtime_noncontiguous_passthrough_preserves_mutation(compiled_module):
    """Passing an imported strided tensor through a segment retains its alias."""
    path, partition = compiled_module
    graph = _build_graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    source = torch.arange(24, dtype=torch.float32, device="cuda").reshape(4, 6).t()
    result, _, alias, _ = graph(source, 5)
    torch.testing.assert_close(result, source + 1)
    alias.fill_(9)
    torch.testing.assert_close(source, torch.full_like(source, 9))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_runtime_preserves_overlapping_result_offsets(compiled_module):
    """Overlapping IREE output ranges remain overlapping PyTorch views."""
    from kashi.iree_runtime import _export_results

    path, partition = compiled_module
    handler = runtime_segment_handlers(path, partition)["segment_0"]
    index = torch.cuda.current_device()
    device, _, _ = handler.target.runtime.instance(index)
    stream = torch.cuda.current_stream(index)
    source = torch.arange(20, device="cuda", dtype=torch.float32)
    stream.synchronize()
    first = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(source[2:12]))
    second = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(source[7:17]))
    left, right = _export_results((first, second), device, index, [], stream)
    assert right.data_ptr() - left.data_ptr() == 5 * source.element_size()
    left[5:].fill_(42)
    torch.testing.assert_close(right[:5], torch.full((5,), 42.0, device="cuda"))
    torch.testing.assert_close(
        right[5:], torch.arange(12, 17, device="cuda", dtype=torch.float32)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_runtime_preserves_input_alias_offset(compiled_module):
    """An IREE result that points inside an input remains a view of that input."""
    from kashi.iree_runtime import _export_results

    path, partition = compiled_module
    handler = runtime_segment_handlers(path, partition)["segment_0"]
    index = torch.cuda.current_device()
    device, _, _ = handler.target.runtime.instance(index)
    stream = torch.cuda.current_stream(index)
    source = torch.arange(20, device="cuda", dtype=torch.float32)
    stream.synchronize()
    view = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(source[4:12]))
    (result,) = _export_results((view,), device, index, [(source, source)], stream)
    result.fill_(17)
    torch.testing.assert_close(source[4:12], torch.full((8,), 17.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_runtime_restores_unsigned_tensor_dtype(compiled_module):
    """Signless IREE bytes retain the unsigned Torch dtype and bit patterns."""
    from kashi.iree_runtime import _export_results

    path, partition = compiled_module
    handler = runtime_segment_handlers(path, partition)["segment_0"]
    index = torch.cuda.current_device()
    device, _, _ = handler.target.runtime.instance(index)
    stream = torch.cuda.current_stream(index)
    signed = torch.tensor([-1, 0, 1], dtype=torch.int8, device="cuda")
    stream.synchronize()
    view = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(signed))
    (result,) = _export_results(
        (view,), device, index, [], stream, ["!torch.vtensor<[3],ui8>"]
    )
    assert result.dtype == torch.uint8
    torch.testing.assert_close(
        result, torch.tensor([255, 0, 1], dtype=torch.uint8, device="cuda")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_runtime_requires_device_for_scalar_only_segment(compiled_module):
    """A scalar-only invocation must not pick an unrelated default GPU."""
    path, partition = compiled_module
    graph = _build_graph(runtime_segment_handlers(path, partition)["segment_1"], 1)
    with pytest.raises(RuntimeError, match="device"):
        graph(12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_runtime_uses_explicit_device_for_scalar_only_segment(compiled_module):
    """Scalar-only segments can run when their device is explicitly known."""
    path, partition = compiled_module
    partition = replace(partition, device=f"cuda:{torch.cuda.current_device()}")
    graph = _build_graph(runtime_segment_handlers(path, partition)["segment_1"], 1)
    assert graph(12) == (12,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_runtime_rejects_cuda_graph_capture(compiled_module):
    """The synchronous runtime bridge must fail before capture can deadlock."""
    path, partition = compiled_module
    graph = _build_graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    source = torch.ones((2, 2), device="cuda")
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture), pytest.raises(RuntimeError, match="capture"):
        graph(source, 2)
