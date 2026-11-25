"""Exercise the runtime segment bridge with compiled CUDA and HIP VM functions."""

import gc
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
compiler = pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

from kashi.iree_partition import PartitionResult, SegmentInfo
from kashi.iree_runtime import runtime_segment_handlers

NVIDIA_AVAILABLE = bool(
    torch.cuda.is_available() and torch.version.cuda and not torch.version.hip
)
AMD_AVAILABLE = bool(
    torch.cuda.is_available() and torch.version.hip and not torch.version.cuda
)

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


@pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA device required")
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


@pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA device required")
def test_runtime_noncontiguous_passthrough_preserves_mutation(compiled_module):
    """Passing an imported strided tensor through a segment retains its alias."""
    path, partition = compiled_module
    graph = _build_graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    source = torch.arange(24, dtype=torch.float32, device="cuda").reshape(4, 6).t()
    result, _, alias, _ = graph(source, 5)
    torch.testing.assert_close(result, source + 1)
    alias.fill_(9)
    torch.testing.assert_close(source, torch.full_like(source, 9))


@pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA device required")
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


@pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA device required")
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


@pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA device required")
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


@pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA device required")
def test_runtime_requires_device_for_scalar_only_segment(compiled_module):
    """A scalar-only invocation must not pick an unrelated default GPU."""
    path, partition = compiled_module
    graph = _build_graph(runtime_segment_handlers(path, partition)["segment_1"], 1)
    with pytest.raises(RuntimeError, match="device"):
        graph(12)


@pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA device required")
def test_runtime_uses_explicit_device_for_scalar_only_segment(compiled_module):
    """Scalar-only segments can run when their device is explicitly known."""
    path, partition = compiled_module
    partition = replace(partition, device=f"cuda:{torch.cuda.current_device()}")
    graph = _build_graph(runtime_segment_handlers(path, partition)["segment_1"], 1)
    assert graph(12) == (12,)


@pytest.mark.skipif(not NVIDIA_AVAILABLE, reason="NVIDIA CUDA device required")
def test_runtime_rejects_cuda_graph_capture(compiled_module):
    """The synchronous runtime bridge must fail before capture can deadlock."""
    path, partition = compiled_module
    graph = _build_graph(runtime_segment_handlers(path, partition)["segment_0"], 2)
    source = torch.ones((2, 2), device="cuda")
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture), pytest.raises(RuntimeError, match="capture"):
        graph(source, 2)


@pytest.fixture(scope="module")
def compiled_hip_module(tmp_path_factory):
    """Compile an actual VMFB for the active AMD GPU."""
    from iree.compiler import ir

    target = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    architecture, *features = target.split(":")
    args = ["--iree-hal-target-device=hip", f"--iree-rocm-target={architecture}"]
    if features:
        args.append(
            "--iree-rocm-target-features="
            + ",".join(f"{feature[-1]}{feature[:-1]}" for feature in features)
        )
    blob = compiler.compile_str(MODULE, input_type="none", extra_args=args)
    path = tmp_path_factory.mktemp("hip_runtime") / "module.vmfb"
    path.write_bytes(blob)
    partition = PartitionResult(
        module=ir.Module.parse(MODULE, ir.Context()),
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
        ],
    )
    return path, partition


@pytest.mark.skipif(not AMD_AVAILABLE, reason="AMD ROCm device required")
def test_hip_runtime_vmfb_stream_aliases_and_lifetime(compiled_hip_module):
    path, partition = compiled_hip_module
    handlers = runtime_segment_handlers(path, partition, target_device="hip")
    graph = _build_graph(handlers["segment_0"], 2)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for stream in streams:
        with torch.cuda.stream(stream):
            source = torch.arange(20, device="cuda", dtype=torch.float32).reshape(4, 5)
            source.add_(3)
            result, repeated, alias, scalar = graph(source, 7)
            assert scalar == 7 and alias is source and repeated is result
            alias.add_(1)
            torch.testing.assert_close(
                result,
                torch.arange(20, device="cuda", dtype=torch.float32).reshape(4, 5) + 4,
            )
            results.append(result)
    del graph, handlers, source, alias, repeated, result
    gc.collect()
    for result in results:
        torch.testing.assert_close(
            result,
            torch.arange(20, device="cuda", dtype=torch.float32).reshape(4, 5) + 4,
        )


@pytest.mark.skipif(not AMD_AVAILABLE, reason="AMD ROCm device required")
def test_hip_runtime_dlpack_export_uses_rocm_device_type(compiled_hip_module):
    from kashi.iree_runtime import _export_results

    path, partition = compiled_hip_module
    handler = runtime_segment_handlers(path, partition, target_device="hip")[
        "segment_0"
    ]
    index = torch.cuda.current_device()
    stream = torch.cuda.current_stream(index)
    device, _, _ = handler.target.runtime.instance(index, stream)
    source = torch.arange(8, device="cuda", dtype=torch.float32)
    stream.synchronize()
    view = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(source))
    (result,) = _export_results(
        (view,), device, index, [], stream, dlpack_device_type=10
    )
    torch.testing.assert_close(result, source)


@pytest.mark.parametrize("family", ["cuda", "hip"])
def test_runtime_matches_uuid_not_enumeration_and_uses_current_stream(
    monkeypatch, family
):
    from kashi import iree_runtime

    monkeypatch.setattr(torch.version, "cuda", "12" if family == "cuda" else None)
    monkeypatch.setattr(torch.version, "hip", "6" if family == "hip" else None)
    uuid = "12345678-9abc-def0-1234-56789abcdef0"
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid=uuid)
    )
    selected = []

    class Driver:
        def query_available_devices(self):
            return [
                {
                    "device_id": 1,
                    "path": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "name": "other",
                },
                {"device_id": 2, "path": "GPU-" + uuid.upper(), "name": "selected"},
            ]

        def create_device(self, info, params):
            selected.append((info["device_id"], params))
            return object()

    monkeypatch.setattr(
        iree_runtime.rt, "get_driver", lambda name: Driver() if name == family else None
    )
    monkeypatch.setattr(iree_runtime, "_hip_driver", lambda: Driver())
    monkeypatch.setattr(
        iree_runtime.rt, "Config", lambda device: SimpleNamespace(vm_instance=object())
    )
    monkeypatch.setattr(
        iree_runtime.rt,
        "VmModule",
        SimpleNamespace(mmap=lambda instance, path: object()),
    )
    monkeypatch.setattr(iree_runtime.rt, "load_vm_module", lambda module, config: {})
    runtime = iree_runtime._RuntimeModule("module.vmfb", None, (), target_device=family)
    first = SimpleNamespace(cuda_stream=1234)
    second = SimpleNamespace(cuda_stream=5678)
    runtime.instance(0, first)
    runtime.instance(0, second)
    runtime.instance(0, first)
    if family == "hip":
        assert selected == [
            (2, {"hip_external_stream": "1234"}),
            (2, {"hip_external_stream": "5678"}),
        ]
    else:
        assert selected == [(2, None)]


@pytest.mark.parametrize(
    "paths", [[], ["GPU-12345678-9abc-def0-1234-56789abcdef0"] * 2]
)
def test_runtime_rejects_missing_or_ambiguous_uuid(monkeypatch, paths):
    from kashi import iree_runtime

    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "6")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid="12345678-9abc-def0-1234-56789abcdef0"),
    )
    driver = SimpleNamespace(
        query_available_devices=lambda: [{"path": path} for path in paths]
    )
    monkeypatch.setattr(iree_runtime.rt, "get_driver", lambda name: driver)
    monkeypatch.setattr(iree_runtime, "_hip_driver", lambda: driver)
    runtime = iree_runtime._RuntimeModule("module.vmfb", None, (), target_device="hip")
    with pytest.raises(RuntimeError, match="uniquely match"):
        runtime.instance(0, SimpleNamespace(cuda_stream=1234))


@pytest.mark.parametrize(
    "cuda,hip,target",
    [
        ("12", None, "hip"),
        (None, "6", "cuda"),
        ("12", "6", "hip"),
        (None, None, "cuda"),
    ],
)
def test_runtime_rejects_wrong_or_ambiguous_torch_build(monkeypatch, cuda, hip, target):
    from kashi import iree_runtime

    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "hip", hip)
    runtime = iree_runtime._RuntimeModule("module.vmfb", None, (), target_device=target)
    with pytest.raises(RuntimeError, match="PyTorch"):
        runtime.instance(0, SimpleNamespace(cuda_stream=1))


def test_runtime_rejects_invalid_target():
    from kashi import iree_runtime

    with pytest.raises(ValueError, match="target_device"):
        iree_runtime._RuntimeModule("module.vmfb", None, (), target_device="cpu")


@pytest.mark.parametrize("family", ["cuda", "hip"])
def test_runtime_rejects_non_gpu_tensor(family):
    from kashi import iree_runtime

    runtime = iree_runtime._RuntimeModule(
        "module.vmfb", "cuda:2", (), target_device=family
    )
    message = (
        f"IREE {family.upper()} runtime requires tensors on a PyTorch "
        "cuda:<index> device, got cpu"
    )
    with pytest.raises(RuntimeError, match=message):
        runtime.device_index([torch.empty(1)])


def test_public_runtime_rejects_invalid_target():
    partition = SimpleNamespace(
        module=SimpleNamespace(body=SimpleNamespace(operations=[])),
        segments=[],
        device=None,
    )
    with pytest.raises(ValueError, match="target_device"):
        runtime_segment_handlers("module.vmfb", partition, target_device="cpu")


@pytest.mark.parametrize("uuid", [None, "not-a-uuid", "GPU-0000"])
def test_runtime_rejects_unverifiable_uuid(monkeypatch, uuid):
    from kashi import iree_runtime

    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "6")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid=uuid)
    )
    runtime = iree_runtime._RuntimeModule("module.vmfb", None, (), target_device="hip")
    with pytest.raises(RuntimeError, match="UUID"):
        runtime.instance(0, SimpleNamespace(cuda_stream=1))


def test_runtime_propagates_driver_errors(monkeypatch):
    from kashi import iree_runtime

    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "6")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid="12345678-9abc-def0-1234-56789abcdef0"),
    )

    def unavailable(name):
        raise RuntimeError("HIP driver unavailable")

    monkeypatch.setattr(iree_runtime.rt, "get_driver", unavailable)
    monkeypatch.setattr(iree_runtime, "_hip_driver", lambda: unavailable("hip"))
    runtime = iree_runtime._RuntimeModule("module.vmfb", None, (), target_device="hip")
    with pytest.raises(RuntimeError, match="HIP driver unavailable"):
        runtime.instance(0, SimpleNamespace(cuda_stream=1))


@pytest.mark.skipif(not AMD_AVAILABLE, reason="AMD ROCm device required")
def test_hip_runtime_rejects_different_partition_device(compiled_hip_module):
    path, partition = compiled_hip_module
    index = torch.cuda.current_device()
    partition = replace(partition, device=f"cuda:{index + 1}")
    graph = _build_graph(
        runtime_segment_handlers(path, partition, target_device="hip")["segment_0"], 2
    )
    with pytest.raises(RuntimeError, match="device"):
        graph(torch.ones((2, 2), device=f"cuda:{index}"), 2)


def test_hip_driver_rejects_mismatched_runtime_libraries(monkeypatch):
    from kashi import iree_runtime

    paths = iter(
        [{"/torch/libamdhip64.so"}, {"/torch/libamdhip64.so", "/iree/libamdhip64.so"}]
    )
    monkeypatch.setattr(iree_runtime, "_mapped_hip_libraries", lambda: next(paths))
    monkeypatch.setattr(iree_runtime, "_hip_library_path", None)
    monkeypatch.setattr(iree_runtime, "parse_flags", lambda *args: None)
    monkeypatch.setattr(iree_runtime.rt, "get_driver", lambda name: object())
    with pytest.raises(RuntimeError, match="HIP librar"):
        iree_runtime._hip_driver()


@pytest.mark.parametrize(
    "reflection,accepted",
    [
        ({"iree.abi.declaration": "sync func @segment_0() -> ()"}, True),
        ({"iree.abi.model": "coarse-fences"}, False),
        ({}, False),
    ],
)
def test_hip_runtime_requires_a_verified_synchronous_abi(reflection, accepted):
    from kashi.iree_runtime import _require_synchronous_abi

    function = SimpleNamespace(vm_function=SimpleNamespace(reflection=reflection))
    if accepted:
        _require_synchronous_abi(function)
    else:
        with pytest.raises(RuntimeError, match="synchronous"):
            _require_synchronous_abi(function)


def test_hip_driver_selects_pytorch_library(monkeypatch):
    from kashi import iree_runtime

    path = "/torch/libamdhip64.so"
    flags = []
    driver = object()
    monkeypatch.setattr(iree_runtime, "_mapped_hip_libraries", lambda: {path})
    monkeypatch.setattr(iree_runtime, "_hip_library_path", None)
    monkeypatch.setattr(iree_runtime, "parse_flags", lambda flag: flags.append(flag))
    monkeypatch.setattr(iree_runtime.rt, "get_driver", lambda name: driver)
    assert iree_runtime._hip_driver() is driver
    assert flags == [f"--hip_dylib_path=file:{path}"]


def test_hip_driver_rejects_unidentifiable_library(monkeypatch):
    from kashi import iree_runtime

    monkeypatch.setattr(iree_runtime, "_mapped_hip_libraries", lambda: set())
    with pytest.raises(RuntimeError, match="HIP library"):
        iree_runtime._hip_driver()


@pytest.mark.skipif(not AMD_AVAILABLE, reason="AMD ROCm device required")
def test_hip_runtime_preserves_overlapping_result_offsets(compiled_hip_module):
    from kashi.iree_runtime import _export_results

    path, partition = compiled_hip_module
    handler = runtime_segment_handlers(path, partition, target_device="hip")[
        "segment_0"
    ]
    index = torch.cuda.current_device()
    stream = torch.cuda.current_stream(index)
    device, _, _ = handler.target.runtime.instance(index, stream)
    source = torch.arange(20, device="cuda", dtype=torch.float32)
    stream.synchronize()
    first = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(source[2:12]))
    second = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(source[7:17]))
    left, right = _export_results(
        (first, second), device, index, [], stream, dlpack_device_type=10
    )
    assert right.data_ptr() - left.data_ptr() == 5 * source.element_size()
    left[5:].fill_(42)
    torch.testing.assert_close(right[:5], torch.full((5,), 42.0, device="cuda"))


def test_runtime_rejects_dlpack_from_wrong_device():
    from kashi.iree_runtime import _import_argument

    with pytest.raises(RuntimeError, match="DLPack device"):
        _import_argument(torch.ones(2), object(), [], 10, 0)


@pytest.mark.skipif(
    not (AMD_AVAILABLE or NVIDIA_AVAILABLE), reason="CUDA or ROCm device required"
)
def test_runtime_realigns_contiguous_offset_input():
    from kashi.iree_runtime import _import_argument

    class Device:
        def from_dlpack_capsule(self, capsule):
            return torch.utils.dlpack.from_dlpack(capsule)

    index = torch.cuda.current_device()
    source = torch.arange(65, dtype=torch.float32, device=f"cuda:{index}")[1:]
    assert source.is_contiguous() and source.data_ptr() % 64 != 0
    imports = []
    result = _import_argument(
        source, Device(), imports, 10 if AMD_AVAILABLE else 2, index
    )
    original, packed = imports[0]
    assert original is source and packed.data_ptr() % 64 == 0
    assert packed.data_ptr() != source.data_ptr()
    torch.testing.assert_close(result, source)


def test_runtime_realigns_cpu_offset_before_import():
    from kashi.iree_runtime import _import_argument

    class Device:
        def from_dlpack_capsule(self, capsule):
            return torch.utils.dlpack.from_dlpack(capsule)

    source = torch.arange(65, dtype=torch.float32)[1:]
    assert source.is_contiguous() and source.data_ptr() % 64 != 0
    imports = []
    result = _import_argument(source, Device(), imports, 1, 0)
    original, packed = imports[0]
    assert original is source and packed.data_ptr() % 64 == 0
    assert packed.data_ptr() != source.data_ptr()
    torch.testing.assert_close(result, source)


@pytest.mark.parametrize("aliases_input", [False, True])
def test_hip_ssa_results_remain_distinct_when_iree_reuses_storage(
    monkeypatch, aliases_input
):
    import iree.runtime as rt
    from kashi.iree_runtime import _export_results

    cpu_device = rt.get_device("local-sync")
    source = torch.arange(16, dtype=torch.float32)
    view = cpu_device.from_dlpack_capsule(
        torch.utils.dlpack.to_dlpack(source),
        memory_type=rt.MemoryType.HOST_LOCAL | rt.MemoryType.DEVICE_VISIBLE,
        memory_access=rt.MemoryAccess.READ | rt.MemoryAccess.WRITE,
    )

    class Device:
        def create_dlpack_capsule(self, value, device_type, index):
            assert (device_type, index) == (10, 0)
            return cpu_device.create_dlpack_capsule(value, 1, 0)

    original_empty = torch.empty

    def empty_on_cpu(*args, **kwargs):
        if kwargs.get("device") == "cuda:0":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_on_cpu)
    first, second, repeated = _export_results(
        (view, view, view),
        Device(),
        0,
        [(source, source)] if aliases_input else [],
        SimpleNamespace(synchronize=lambda: None),
        ["!torch.vtensor<[16],f32>"] * 3,
        output_aliases=[None, None, ("output", 0)],
        source_args=(source,),
        dlpack_device_type=10,
    )
    assert repeated is first
    first.fill_(42)
    torch.testing.assert_close(second, torch.arange(16, dtype=torch.float32))
    torch.testing.assert_close(source, torch.arange(16, dtype=torch.float32))


@pytest.mark.skipif(not AMD_AVAILABLE, reason="AMD ROCm device required")
@pytest.mark.parametrize("aliases_input", [False, True])
def test_hip_runtime_distinct_ssa_results_do_not_alias(
    compiled_hip_module, aliases_input
):
    from kashi.iree_runtime import _export_results

    path, partition = compiled_hip_module
    handler = runtime_segment_handlers(path, partition, target_device="hip")[
        "segment_0"
    ]
    index = torch.cuda.current_device()
    stream = torch.cuda.current_stream(index)
    device, _, _ = handler.target.runtime.instance(index, stream)
    source = torch.arange(16, device=f"cuda:{index}", dtype=torch.float32)
    stream.synchronize()
    view = device.from_dlpack_capsule(torch.utils.dlpack.to_dlpack(source))
    first, second, repeated = _export_results(
        (view, view, view),
        device,
        index,
        [(source, source)] if aliases_input else [],
        stream,
        ["!torch.vtensor<[16],f32>"] * 3,
        output_aliases=[None, None, ("output", 0)],
        source_args=(source,),
        dlpack_device_type=10,
    )
    assert repeated is first
    first.fill_(42)
    torch.testing.assert_close(second, source)
    torch.testing.assert_close(
        source, torch.arange(16, device=f"cuda:{index}", dtype=torch.float32)
    )


@pytest.mark.skipif(not AMD_AVAILABLE, reason="AMD ROCm device required")
def test_hip_runtime_realigns_offset_and_preserves_input_alias(compiled_hip_module):
    path, partition = compiled_hip_module
    handler = runtime_segment_handlers(path, partition, target_device="hip")[
        "segment_0"
    ]
    graph = _build_graph(handler, 2)
    source = torch.arange(21, device="cuda", dtype=torch.float32)[1:].reshape(4, 5)
    assert source.data_ptr() % 64 != 0
    result, _, alias, _ = graph(source, 5)
    torch.testing.assert_close(result, source + 1)
    assert alias is source
    alias.fill_(9)
    torch.testing.assert_close(source, torch.full_like(source, 9))


@pytest.mark.skipif(not AMD_AVAILABLE, reason="AMD ROCm device required")
def test_hip_runtime_noncontiguous_input_alias_preserves_mutation(compiled_hip_module):
    path, partition = compiled_hip_module
    handler = runtime_segment_handlers(path, partition, target_device="hip")[
        "segment_0"
    ]
    graph = _build_graph(handler, 2)
    source = torch.arange(24, device="cuda", dtype=torch.float32).reshape(4, 6).t()
    result, _, alias, _ = graph(source, 5)
    torch.testing.assert_close(result, source + 1)
    alias.fill_(9)
    torch.testing.assert_close(source, torch.full_like(source, 9))


@pytest.mark.skipif(not AMD_AVAILABLE, reason="AMD ROCm device required")
def test_hip_runtime_requires_device_for_scalar_only_segment(compiled_hip_module):
    path, partition = compiled_hip_module
    handler = runtime_segment_handlers(path, partition, target_device="hip")[
        "segment_1"
    ]
    graph = _build_graph(handler, 1)
    with pytest.raises(RuntimeError, match="device"):
        graph(12)
    partition = replace(partition, device=f"cuda:{torch.cuda.current_device()}")
    handler = runtime_segment_handlers(path, partition, target_device="hip")[
        "segment_1"
    ]
    assert _build_graph(handler, 1)(12) == (12,)
