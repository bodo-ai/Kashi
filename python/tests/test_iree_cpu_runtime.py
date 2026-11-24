"""Exercise the local CPU VMFB bridge with real IREE modules and tensors."""

import gc
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
compiler = pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

from iree.compiler import ir
from kashi.iree_cpu_runtime import runtime_segment_handlers
from kashi.iree_partition import PartitionResult, SegmentInfo, partition_at_custom_ops

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
  func.func @segment_2(%x: tensor<?xi8>) -> tensor<?xi8> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %x, %c0 : tensor<?xi8>
    %empty = tensor.empty(%d0) : tensor<?xi8>
    %one = arith.constant 1 : i8
    %y = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x : tensor<?xi8>) outs(%empty : tensor<?xi8>) {
    ^bb0(%a: i8, %b: i8):
      %sum = arith.addi %a, %one : i8
      linalg.yield %sum : i8
    } -> tensor<?xi8>
    return %y : tensor<?xi8>
  }
  func.func @segment_3(%x: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    %a = tensor.extract_slice %x[1] [4] [1] : tensor<?xf32> to tensor<4xf32>
    %b = tensor.cast %a : tensor<4xf32> to tensor<?xf32>
    %c = tensor.extract_slice %x[3] [4] [1] : tensor<?xf32> to tensor<4xf32>
    %d = tensor.cast %c : tensor<4xf32> to tensor<?xf32>
    return %b, %d : tensor<?xf32>, tensor<?xf32>
  }
  func.func @segment_5(%n: i64) {
    return
  }
  func.func @segment_6(%x: tensor<?xf32>) -> tensor<f32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<f32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<f32>) -> tensor<f32>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%x : tensor<?xf32>) outs(%init : tensor<f32>) {
    ^bb0(%a: f32, %b: f32):
      %next = arith.addf %a, %b : f32
      linalg.yield %next : f32
    } -> tensor<f32>
    return %sum : tensor<f32>
  }
  func.func @segment_7(%n: i64) -> i64 {
    %one = arith.constant 1 : i64
    %result = arith.addi %n, %one : i64
    return %result : i64
  }
}
"""


@pytest.fixture(scope="module")
def compiled_module(tmp_path_factory):
    blob = compiler.compile_str(
        MODULE,
        input_type="none",
        target_backends=["llvm-cpu"],
        extra_args=["--iree-llvmcpu-target-cpu=generic"],
    )
    path = tmp_path_factory.mktemp("cpu_runtime") / "module.vmfb"
    path.write_bytes(blob)
    module = ir.Module.parse(MODULE, ir.Context())
    f32_2d = "!torch.vtensor<[?,?],f32>"
    partition = PartitionResult(
        module,
        None,
        "cpu",
        [
            SegmentInfo(
                0,
                "segment_0",
                [f32_2d, "!torch.int"],
                [f32_2d, f32_2d, f32_2d, "!torch.int"],
                [],
            ),
            SegmentInfo(1, "segment_1", ["!torch.int"], ["!torch.int"], []),
            SegmentInfo(
                2,
                "segment_2",
                ["!torch.vtensor<[?],ui8>"],
                ["!torch.vtensor<[?],ui8>"],
                [],
            ),
            SegmentInfo(
                3,
                "segment_3",
                ["!torch.vtensor<[?],f32>"],
                ["!torch.vtensor<[?],f32>"] * 2,
                [],
            ),
            SegmentInfo(4, "segment_4", [], [], []),
            SegmentInfo(5, "segment_5", ["!torch.int"], [], []),
            SegmentInfo(
                6,
                "segment_6",
                ["!torch.vtensor<[?],f32>"],
                ["!torch.vtensor<[],f32>"],
                [],
            ),
            SegmentInfo(7, "segment_7", ["!torch.int"], ["!torch.int"], []),
        ],
    )
    return path, partition


def _graph(handler, arity):
    graph = torch.fx.Graph()
    args = [graph.placeholder(f"arg{i}") for i in range(arity)]
    graph.output(tuple(handler(graph, args)))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_real_local_runtime_multires_result_alias_and_lifetime(compiled_module):
    path, partition = compiled_module
    handlers = runtime_segment_handlers(path, partition)
    assert "segment_4" not in handlers
    graph = _graph(handlers["segment_0"], 2)
    source = torch.arange(37, dtype=torch.float32)[1:36].reshape(7, 5)
    result, repeated, alias, scalar = graph(source, 19)
    assert result.device.type == "cpu" and result.dtype == torch.float32
    assert result.shape == (7, 5) and scalar == 19
    torch.testing.assert_close(result, source + 1)
    assert repeated is result and alias is source
    repeated.add_(2)
    torch.testing.assert_close(result, source + 3)
    del graph, handlers, source, repeated, alias
    gc.collect()
    torch.testing.assert_close(result, torch.arange(4, 39).reshape(7, 5).float())


def test_fresh_results_survive_runtime_and_view_owners(compiled_module):
    path, partition = compiled_module
    handlers = runtime_segment_handlers(path, partition)
    graph = _graph(handlers["segment_0"], 2)
    source = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    first, _, _, _ = graph(source, 1)
    view = first[:, 1:4]
    for i in range(8):
        later, _, _, _ = graph(torch.full((4, 5), float(i)), i)
        torch.testing.assert_close(later, torch.full((4, 5), float(i + 1)))
    del first, graph, handlers, source, later
    gc.collect()
    torch.testing.assert_close(
        view,
        torch.tensor(
            [[2.0, 3.0, 4.0], [7.0, 8.0, 9.0], [12.0, 13.0, 14.0], [17.0, 18.0, 19.0]]
        ),
    )
    view.add_(10)
    assert view[0, 0].item() == 12


def test_noncontiguous_input_is_packed_but_passthrough_alias_is_preserved(
    compiled_module,
):
    path, partition = compiled_module
    graph = _graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    result, _, alias, _ = graph(source, 5)
    torch.testing.assert_close(result, source + 1)
    assert alias is source
    alias.fill_(9)
    torch.testing.assert_close(source, torch.full_like(source, 9))


def test_scalar_only_segment_and_dynamic_empty_tensors(compiled_module):
    path, partition = compiled_module
    handlers = runtime_segment_handlers(path, partition)
    assert _graph(handlers["segment_1"], 1)(-17) == (-17,)
    assert _graph(handlers["segment_7"], 1)(-17) == (-16,)
    source = torch.empty(0, 5)
    result, repeated, alias, value = _graph(handlers["segment_0"], 2)(source, 0)
    assert result.shape == (0, 5) and result.dtype == torch.float32
    assert repeated is result and alias is source and value == 0


def test_empty_input_still_executes_reduction(compiled_module):
    path, partition = compiled_module
    graph = _graph(runtime_segment_handlers(path, partition)["segment_6"], 1)
    (result,) = graph(torch.empty(0, dtype=torch.float32))
    assert result.shape == torch.Size([]) and result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor(0.0))
    (result,) = graph(torch.tensor([2.0, -3.0, 5.0]))
    torch.testing.assert_close(result, torch.tensor(4.0))


@pytest.mark.parametrize("shape", [(0,), (0, 5), (5, 0), (2, 0, 3)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool]
)
def test_empty_hal_import_has_valid_backing_and_logical_shape(
    compiled_module, shape, dtype
):
    from kashi.iree_cpu_runtime import _import_argument

    path, partition = compiled_module
    handler = runtime_segment_handlers(path, partition)["segment_0"]
    device, _, _ = handler.target.runtime.instance()
    source = torch.empty(shape, dtype=dtype)
    imports = []
    view = _import_argument(source, device, imports)
    assert tuple(view.shape) == shape and view.byte_length == 0
    assert view.get_buffer().byte_length() > 0
    assert imports[0][0] is source


def test_unsigned_output_keeps_bits_and_dtype(compiled_module):
    path, partition = compiled_module
    graph = _graph(runtime_segment_handlers(path, partition)["segment_2"], 1)
    (result,) = graph(torch.tensor([0, 127, 254, 255], dtype=torch.uint8))
    assert result.dtype == torch.uint8
    torch.testing.assert_close(
        result, torch.tensor([1, 128, 255, 0], dtype=torch.uint8)
    )


def test_empty_result_and_empty_partition(compiled_module, tmp_path):
    path, partition = compiled_module
    handlers = runtime_segment_handlers(path, replace(partition, device=None))
    assert _graph(handlers["segment_5"], 1)(42) == ()
    module = ir.Module.parse("module {}", ir.Context())
    empty = PartitionResult(module, None, "cpu", [])
    assert runtime_segment_handlers(tmp_path / "absent.vmfb", empty) == {}


def test_read_only_local_buffer_is_copied_with_dtype_and_lifetime(compiled_module):
    import iree.runtime as rt
    from kashi.iree_cpu_runtime import _export_tensor

    path, partition = compiled_module
    handler = runtime_segment_handlers(path, partition)["segment_0"]
    device, _, _ = handler.target.runtime.instance()
    source = torch.tensor([1.5, -2.0], dtype=torch.bfloat16)
    view = device.from_dlpack_capsule(
        torch.utils.dlpack.to_dlpack(source),
        memory_type=rt.MemoryType.HOST_LOCAL | rt.MemoryType.DEVICE_VISIBLE,
        memory_access=rt.MemoryAccess.READ | rt.MemoryAccess.UNALIGNED,
    )
    result = _export_tensor(view, (torch.bfloat16, (2,)), device)
    del view, source, handler, device
    gc.collect()
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.tensor([1.5, -2.0], dtype=torch.bfloat16))
    result.fill_(3)


def test_distinct_outputs_do_not_acquire_accidental_input_aliases(compiled_module):
    path, partition = compiled_module
    graph = _graph(runtime_segment_handlers(path, partition)["segment_3"], 1)
    source = torch.arange(9, dtype=torch.float32)
    left, right = graph(source)
    torch.testing.assert_close(left, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(right, torch.tensor([3.0, 4.0, 5.0, 6.0]))
    left.fill_(22)
    torch.testing.assert_close(source, torch.arange(9, dtype=torch.float32))
    torch.testing.assert_close(right, torch.tensor([3.0, 4.0, 5.0, 6.0]))


def test_cpu_import_uses_aligned_storage_without_unnecessary_copy(compiled_module):
    from kashi.iree_cpu_runtime import _import_argument

    path, partition = compiled_module
    handler = runtime_segment_handlers(path, partition)["segment_0"]
    device, _, _ = handler.target.runtime.instance()
    storage = torch.arange(80, dtype=torch.float32)
    offset = (-storage.data_ptr() % 64) // storage.element_size()
    aligned = storage[offset : offset + 16]
    imports = []
    view = _import_argument(aligned, device, imports)
    exported = torch.utils.dlpack.from_dlpack(device.create_dlpack_capsule(view, 1, 0))
    assert exported.data_ptr() == aligned.data_ptr()
    assert imports[0][1].data_ptr() == aligned.data_ptr()

    imports = []
    unaligned = storage[offset + 1 : offset + 17]
    _import_argument(unaligned, device, imports)
    packed = imports[0][1]
    assert packed.data_ptr() % 64 == 0 and packed.data_ptr() != unaligned.data_ptr()
    torch.testing.assert_close(packed, unaligned)


def test_rejects_non_cpu_tensor_before_execution(compiled_module):
    path, partition = compiled_module
    graph = _graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    with pytest.raises(RuntimeError, match="CPU"):
        graph(torch.empty((2, 2), device="meta"), 2)


def test_rejects_incompatible_layout_and_argument_count(compiled_module):
    path, partition = compiled_module
    handler = runtime_segment_handlers(path, partition)["segment_0"]
    with pytest.raises(ValueError, match="live-in"):
        _graph(handler, 1)
    graph = _graph(handler, 2)
    with pytest.raises(NotImplementedError, match="layout"):
        graph(torch.eye(4).to_sparse(), 3)
    with pytest.raises(TypeError, match="argument"):
        graph(torch.eye(4), object())
    with pytest.raises(TypeError, match="dtype"):
        graph(torch.ones(4, 4, dtype=torch.int32), 3)
    with pytest.raises(ValueError, match="shape"):
        graph(torch.ones(4), 3)


@pytest.mark.parametrize(
    "dtype,element",
    [
        (torch.float16, "f16"),
        (torch.bfloat16, "bf16"),
        (torch.float32, "f32"),
    ],
)
def test_floating_dtype_and_rank_zero_result(tmp_path, dtype, element):
    module_text = f"""
module {{
  func.func @segment_0(%x: tensor<{element}>) -> tensor<{element}> {{
    %empty = tensor.empty() : tensor<{element}>
    %one = arith.constant 1.0 : {element}
    %y = linalg.generic {{
      indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
      iterator_types = []
    }} ins(%x : tensor<{element}>) outs(%empty : tensor<{element}>) {{
    ^bb0(%a: {element}, %b: {element}):
      %sum = arith.addf %a, %one : {element}
      linalg.yield %sum : {element}
    }} -> tensor<{element}>
    return %y : tensor<{element}>
  }}
}}
"""
    path = tmp_path / "scalar.vmfb"
    path.write_bytes(
        compiler.compile_str(
            module_text,
            input_type="none",
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
    )
    module = ir.Module.parse(module_text, ir.Context())
    typ = f"!torch.vtensor<[],{element}>"
    partition = PartitionResult(
        module,
        None,
        "cpu",
        [
            SegmentInfo(0, "segment_0", [typ], [typ], []),
        ],
    )
    graph = _graph(runtime_segment_handlers(path, partition)["segment_0"], 1)
    (result,) = graph(torch.tensor(2, dtype=dtype))
    assert result.dtype == dtype and result.shape == torch.Size([])
    torch.testing.assert_close(result, torch.tensor(3, dtype=dtype))


def test_runtime_rejects_mismatched_output_metadata(compiled_module):
    path, partition = compiled_module
    segment = replace(
        partition.segments[0],
        live_out_types=[
            "!torch.vtensor<[9,5],f32>",
            *partition.segments[0].live_out_types[1:],
        ],
    )
    partition = replace(partition, segments=[segment])
    graph = _graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    with pytest.raises((RuntimeError, ValueError), match="shape"):
        graph(torch.ones(7, 5), 1)


@pytest.mark.parametrize(
    "type_text", ["!torch.vtensor<[?],complex64>", "!torch.vtensor<[bad],f32>"]
)
def test_cpu_rejects_unrepresentable_tensor_signature(compiled_module, type_text):
    """Unknown tensor dimensions and dtypes must not acquire a guessed ABI."""
    path, partition = compiled_module
    segment = replace(partition.segments[0], live_in_types=[type_text, "!torch.int"])
    with pytest.raises(NotImplementedError, match="tensor"):
        runtime_segment_handlers(path, replace(partition, segments=[segment]))


def test_cpu_rejects_non_tensor_input_without_invoking_vm(compiled_module):
    """A scalar cannot be imported as a tensor binding."""
    path, partition = compiled_module
    graph = _graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    with pytest.raises(TypeError, match="must be a tensor"):
        graph(1, 2)


def test_cpu_rejects_unsupported_empty_dtype(compiled_module):
    """Empty buffers still need a representable HAL element type."""
    from kashi.iree_cpu_runtime import _import_argument

    path, partition = compiled_module
    handler = runtime_segment_handlers(path, partition)["segment_0"]
    device, _, _ = handler.target.runtime.instance()
    with pytest.raises(TypeError, match="dtype"):
        _import_argument(torch.empty(0, dtype=torch.complex64), device, [])


def test_cpu_rejects_unsupported_scalar_output():
    """An unknown VM result cannot silently escape as a Python value."""
    from kashi.iree_cpu_runtime import _export_scalar

    assert _export_scalar([3, -2]) == [3, -2]
    with pytest.raises(TypeError, match="result"):
        _export_scalar(object())


def test_cpu_execution_never_initializes_cuda(compiled_module, monkeypatch):
    path, partition = compiled_module

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU execution attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    graph = _graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    result, _, _, _ = graph(torch.ones(2, 2), 1)
    torch.testing.assert_close(result, torch.full((2, 2), 2.0))


def test_missing_or_invalid_vmfb_propagates_error(tmp_path, compiled_module):
    _, partition = compiled_module
    for path in (tmp_path / "missing.vmfb", tmp_path / "invalid.vmfb"):
        if path.name.startswith("invalid"):
            path.write_bytes(b"not a VMFB")
        graph = _graph(runtime_segment_handlers(path, partition)["segment_1"], 1)
        with pytest.raises((OSError, ValueError, RuntimeError)):
            graph(2)


def test_missing_vm_export_fails_without_fallback(compiled_module):
    path, _ = compiled_module
    module = ir.Module.parse(
        "module { func.func @missing(%n: i64) -> i64 { return %n : i64 } }",
        ir.Context(),
    )
    partition = PartitionResult(
        module,
        None,
        "cpu",
        [
            SegmentInfo(0, "missing", ["!torch.int"], ["!torch.int"], []),
        ],
    )
    graph = _graph(runtime_segment_handlers(path, partition)["missing"], 1)
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        graph(12)


def test_rejects_incompatible_partition_device(compiled_module):
    path, partition = compiled_module
    with pytest.raises((RuntimeError, ValueError), match="CPU"):
        runtime_segment_handlers(path, replace(partition, device="cuda:0"))


def test_full_cpu_partition_and_rebuilt_fx_graph(tmp_path):
    """Compile a metadata-bearing FX graph without backend target selection."""
    from kashi.importer import import_fx
    from kashi.mlir_to_fx import rebuild_fx
    from torch._subclasses.fake_tensor import FakeTensorMode

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    relu = graph.call_function(torch.ops.aten.relu.default, (x,))
    neg = graph.call_function(torch.ops.aten.neg.default, (relu,))
    graph.output((relu, neg))
    with FakeTensorMode() as mode:
        example = mode.from_tensor(torch.empty(3, 5))
        for node in (x, relu, neg):
            node.meta["val"] = example
    original = torch.fx.GraphModule(torch.nn.Module(), graph)
    imported = import_fx(original)
    partition = partition_at_custom_ops(imported.module)
    assert partition.driver is not None and partition.segments
    path = tmp_path / "partition.vmfb"
    path.write_bytes(
        compiler.compile_str(
            str(partition.module),
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
    )
    rebuilt = rebuild_fx(
        partition.driver,
        imported.out_spec,
        segment_handlers=runtime_segment_handlers(path, partition),
    )
    source = torch.linspace(-5, 5, 15).reshape(3, 5)
    actual = rebuilt(source)
    expected = original(source)
    torch.testing.assert_close(actual, expected)
    actual[0].add_(100)
    torch.testing.assert_close(actual[1], expected[1])
