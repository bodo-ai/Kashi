"""Exercise device-neutral GPU launch ABI validation without GPU hardware.

Fake tensors stand in for GPU storage; the real CUDA/HIP tests verify actual
kernel execution. The driver callback here records ABI values, not results.
"""

from contextlib import nullcontext

import pytest
import torch
from kashi import iree_gpu_dispatch
from kashi.iree_gpu_dispatch import create_gpu_kernel
from torch._subclasses.fake_tensor import FakeTensorMode

_READ = "%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>"
_WRITE = "%arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>"
_GRID = (
    "%one = arith.constant 1 : index\nhal.return %one, %one, %one : index, index, index"
)
_ZERO_GRID = "%zero = arith.constant 0 : index\n%one = arith.constant 1 : index\nhal.return %zero, %one, %one : index, index, index"


@pytest.fixture
def fake_gpu(monkeypatch):
    """Use fake CUDA tensors while recording stream use and driver arguments."""
    recorded = []
    stream = object()
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # Lifetime is exercised with real streams separately; this test isolates
    # launch argument packing without constructing CUDA events for fake tensors.
    monkeypatch.setattr(
        iree_gpu_dispatch,
        "_launch_retaining_bindings",
        lambda _stream, _bindings, launch: launch(),
    )
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda _tensor: 128)
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, value: recorded.append((tensor, value)),
    )
    with FakeTensorMode():
        yield recorded, stream


def _build(
    flow, binding_count, constant_count, launch, *, grid=_GRID, offsets=(), alignment=16
):
    metadata = (
        flow,
        binding_count,
        constant_count,
        set(offsets),
        ["%device: !hal.device"],
        grid,
        {},
        (64, 1, 1),
        0,
    )
    return create_gpu_kernel(
        "example", "entry", metadata, [alignment] * binding_count, launch, vendor="hip"
    )


def _invoke(kernel, *args, shape=(4,), dtype=torch.float32, tied=-1, dims=()):
    return kernel(
        *args,
        workload=[],
        argument_dims=dims,
        result_shapes=[shape],
        result_dtypes=[dtype],
        tied_operands=[tied],
    )


def test_gpu_dispatch_packs_scalar_constants_and_records_stream(fake_gpu):
    """The driver receives binding order and little-endian 64-bit scalar words."""
    records, stream = fake_gpu
    launches = []
    kernel = _build(
        [_READ, "%arg1: i64", _WRITE], 2, 2, lambda *args: launches.append(args)
    )
    source = torch.empty(4, device="cuda:0")
    result = _invoke(kernel, source, -2)
    device, launched_stream, bindings, constants, grid, block, shared = launches[0]
    assert device == source.device and launched_stream is stream
    assert bindings == [source, result]
    assert constants == [0xFFFFFFFE, 0xFFFFFFFF]
    assert grid == (1, 1, 1) and block == (64, 1, 1) and shared == 0
    assert records == [(source, stream), (result, stream)]
    assert result.shape == source.shape and result.dtype == source.dtype


def test_gpu_dispatch_zero_grid_preserves_tied_output(fake_gpu):
    """No kernel is launched for an empty grid; tied results retain their input."""
    records, _ = fake_gpu
    kernel = _build(
        [_READ], 1, 0, lambda *args: pytest.fail("unexpected launch"), grid=_ZERO_GRID
    )
    source = torch.empty(4, device="cuda:0")
    assert _invoke(kernel, source, tied=0) is source
    assert not records


def test_gpu_dispatch_rejects_invalid_tied_outputs(fake_gpu):
    """A tied result cannot change its source's shape, dtype, or operand index."""
    kernel = _build(
        [_READ], 1, 0, lambda *args: pytest.fail("unexpected launch"), grid=_ZERO_GRID
    )
    source = torch.empty(4, device="cuda:0")
    with pytest.raises(NotImplementedError, match="tied operand"):
        _invoke(kernel, source, tied=3)
    with pytest.raises(NotImplementedError, match="tied operand"):
        _invoke(kernel, source, tied=0, shape=(2, 2))
    with pytest.raises(NotImplementedError, match="tied operand"):
        _invoke(kernel, source, tied=0, dtype=torch.int32)


def test_gpu_dispatch_rejects_mismatched_metadata(fake_gpu):
    """Incorrect dimensions, arity, shape and binding count fail before launch."""
    kernel = _build(
        [_READ, _WRITE],
        2,
        0,
        lambda *args: pytest.fail("unexpected launch"),
        grid=_ZERO_GRID,
    )
    source = torch.empty(4, device="cuda:0")
    with pytest.raises(ValueError, match="argument dimensions"):
        _invoke(kernel, source, dims=(4,))
    with pytest.raises(ValueError, match="negative output shape"):
        _invoke(kernel, source, shape=(-1,))
    with pytest.raises(ValueError, match="inconsistent result metadata"):
        kernel(
            source,
            workload=[],
            argument_dims=[],
            result_shapes=[(4,)],
            result_dtypes=[],
            tied_operands=[],
        )
    bad_binding = _build([_READ, _WRITE], 1, 0, lambda *args: None, grid=_ZERO_GRID)
    with pytest.raises(NotImplementedError, match="binding count"):
        _invoke(bad_binding, source)


def test_gpu_dispatch_rejects_bad_tensor_bindings(fake_gpu):
    """CPU, noncontiguous and wrong-dtype inputs cannot become GPU pointers."""
    kernel = _build([_READ, _WRITE], 2, 0, lambda *args: None, grid=_ZERO_GRID)
    with pytest.raises(NotImplementedError, match="tensor device"):
        _invoke(kernel, torch.empty(4))
    with pytest.raises(NotImplementedError, match="non-contiguous"):
        _invoke(kernel, torch.empty_strided((4,), (2,), device="cuda:0"))
    with pytest.raises(ValueError, match="tensor does not match"):
        _invoke(kernel, torch.empty(4, dtype=torch.int32, device="cuda:0"))


def test_gpu_dispatch_rejects_missing_and_extra_scalar_constants(fake_gpu):
    """Flow scalar words must match the executable's declared constant ABI."""
    source = torch.empty(4, device="cuda:0")
    missing = _build(
        [_READ, "%arg1: i64", _WRITE], 2, 3, lambda *args: None, grid=_ZERO_GRID
    )
    with pytest.raises(NotImplementedError, match="missing flow scalar"):
        _invoke(missing, source, 7)
    extra = _build(
        [_READ, "%arg1: i64", _WRITE], 2, 1, lambda *args: None, grid=_ZERO_GRID
    )
    with pytest.raises(NotImplementedError, match="unmapped flow scalar"):
        _invoke(extra, source, 7)


def test_stream_offset_constants_trace_reconstructed_push_constants():
    """Coalesced output offsets become zero without losing size constants."""
    from kashi.iree_gpu_dispatch import _offset_constants

    body = """
      %c32 = arith.constant 32 : i64
      %zero = arith.constant 0 : index
      %low = arith.extui %arg2 : i32 to i64
      %high = arith.extui %arg3 : i32 to i64
      %shift = arith.shli %high, %c32 : i64
      %combined = arith.ori %low, %shift : i64
      %index = arith.index_castui %combined : i64 to index
      %checked:2 = util.assume.int %index<udiv = 16>,
          %zero<umin = 0> : index, index
      %input = stream.binding.subspan %arg0[%zero] : !stream.binding
      %output = stream.binding.subspan %arg1[%checked#0] : !stream.binding
      return
    """
    assert _offset_constants(body, 2, 4, "entry") == {0, 1}


@pytest.mark.parametrize(
    "body, message",
    [
        (
            "%offset = arith.constant 4 : index\n%out = stream.binding.subspan %arg0[%offset] : !stream.binding",
            "nonzero fixed",
        ),
        (
            "%out = stream.binding.subspan %arg1[%arg2] : !stream.binding",
            "binding ordinal",
        ),
        (
            "%out = stream.binding.subspan %arg0[%arg1] : !stream.binding\n%used = arith.muli %arg1, %arg1 : i32",
            "also used",
        ),
        (
            "%out = stream.binding.subspan %arg0[%unknown] : !stream.binding",
            "offset value",
        ),
    ],
)
def test_stream_offset_rejects_unsafe_binding_metadata(body, message):
    """Malformed or computational offsets must not silently shift GPU pointers."""
    from kashi.iree_gpu_dispatch import _offset_constants

    with pytest.raises(NotImplementedError, match=message):
        _offset_constants(body, 1, 1, "entry")


@pytest.mark.parametrize(
    ("family", "library", "symbol"),
    [
        ("cuda", "libcuda.so.1", "cuStreamGetCaptureInfo"),
        ("hip", "/opt/wheel/libamdhip64.so.7", "hipStreamGetCaptureInfo"),
    ],
)
def test_capture_identity_uses_matching_driver(monkeypatch, family, library, symbol):
    """A reused stream handle cannot be mistaken for an earlier capture."""
    import ctypes
    from types import SimpleNamespace

    from kashi import iree_runtime
    from kashi.iree_gpu_dispatch import _capture_identity

    class Query:
        def __call__(self, stream, status, identity):
            assert stream.value == 123
            ctypes.cast(status, ctypes.POINTER(ctypes.c_int))[0] = 1
            ctypes.cast(identity, ctypes.POINTER(ctypes.c_uint64))[0] = 456
            return 0

    query = Query()

    def open_driver(name):
        assert name == library
        return SimpleNamespace(**{symbol: query})

    monkeypatch.setattr(iree_gpu_dispatch, "_CAPTURE_QUERY", None)
    monkeypatch.setattr(iree_gpu_dispatch.ctypes, "CDLL", open_driver)
    monkeypatch.setattr(torch.version, "cuda", "13.0" if family == "cuda" else None)
    monkeypatch.setattr(torch.version, "hip", "7.0" if family == "hip" else None)
    if family == "hip":
        monkeypatch.setattr(iree_runtime, "_mapped_hip_libraries", lambda: {library})
    assert _capture_identity(SimpleNamespace(cuda_stream=123)) == 456


@pytest.mark.parametrize("result, status", [(0, 0), (1, 1)])
def test_capture_identity_rejects_driver_errors(monkeypatch, result, status):
    """A failed or inactive capture query cannot authorize pointer retention."""
    import ctypes
    from types import SimpleNamespace

    def query(_stream, status_ptr, _identity):
        ctypes.cast(status_ptr, ctypes.POINTER(ctypes.c_int))[0] = status
        return result

    monkeypatch.setattr(iree_gpu_dispatch, "_CAPTURE_QUERY", (object(), query))
    with pytest.raises(RuntimeError, match="capture"):
        iree_gpu_dispatch._capture_identity(SimpleNamespace(cuda_stream=123))


def test_capture_identity_rejects_missing_driver(monkeypatch):
    """Missing native capture support cannot silently authorize graph replay."""
    from types import SimpleNamespace

    def missing(_library):
        raise OSError("missing driver")

    monkeypatch.setattr(iree_gpu_dispatch, "_CAPTURE_QUERY", None)
    monkeypatch.setattr(iree_gpu_dispatch.ctypes, "CDLL", missing)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        iree_gpu_dispatch._capture_identity(SimpleNamespace(cuda_stream=123))


def test_gpu_scalar_push_constants_preserve_bit_patterns():
    """Push constants preserve floating and signed integer representations."""
    from kashi.iree_gpu_dispatch import _pack_scalar

    assert _pack_scalar(3.5, "f32", "entry") == [0x40600000]
    assert _pack_scalar(-1, "i32", "entry") == [0xFFFFFFFF]
    assert _pack_scalar(0x100000002, "i64", "entry") == [2, 1]
    with pytest.raises(NotImplementedError, match="outside"):
        _pack_scalar(0x100000000, "i32", "entry")
    with pytest.raises(NotImplementedError, match="scalar"):
        _pack_scalar(1.5, "i32", "entry")


def test_capture_identity_rejects_ambiguous_build(monkeypatch):
    """Capture IDs cannot be read from an unrelated device driver."""
    from types import SimpleNamespace

    monkeypatch.setattr(iree_gpu_dispatch, "_CAPTURE_QUERY", None)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="PyTorch build"):
        iree_gpu_dispatch._capture_identity(SimpleNamespace(cuda_stream=123))
