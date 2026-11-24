"""Offline HSACO ABI validation and direct HIP execution tests."""

import gc
import struct

import msgpack
import pytest
import torch
from kashi.iree_hip import _hsaco_metadata, create_hip_kernel


def _image(metadata, flags=0x54C):
    """Construct an ELF64 AMDGPU ET_DYN with an AMDGPU metadata PT_NOTE."""
    desc = msgpack.packb(metadata, use_bin_type=True)
    name = b"AMDGPU\0"
    note = struct.pack("<III", len(name), len(desc), 32)
    note += name + b"\0" * (-len(name) % 4)
    note += desc + b"\0" * (-len(desc) % 4)
    names = b"\0.shstrtab\0.dynstr\0.dynsym\0"
    strings = b"\0entry.kd\0"
    symbols = b"\0" * 24 + struct.pack("<IBBHQQ", 1, 0x11, 0, 1, 0, 64)
    names_offset = 120 + len(note)
    strings_offset = names_offset + len(names)
    symbols_offset = (strings_offset + len(strings) + 7) & ~7
    section_offset = symbols_offset + len(symbols)
    ident = b"\x7fELF\x02\x01\x01\x40\x03" + b"\0" * 7
    header = ident + struct.pack(
        "<HHIQQQIHHHHHH", 3, 224, 1, 0, 64, section_offset, flags, 64, 56, 1, 64, 4, 1
    )
    phdr = struct.pack("<IIQQQQQQ", 4, 4, 120, 0, 0, len(note), len(note), 4)
    section = lambda name, typ, off, size, link, align, entsize: struct.pack(
        "<IIQQQQIIQQ", name, typ, 0, 0, off, size, link, 0, align, entsize
    )
    sections = b"\0" * 64
    sections += section(1, 3, names_offset, len(names), 0, 1, 0)
    sections += section(11, 3, strings_offset, len(strings), 0, 1, 0)
    sections += section(19, 11, symbols_offset, len(symbols), 2, 8, 24)
    padding = b"\0" * (symbols_offset - strings_offset - len(strings))
    return header + phdr + note + names + strings + padding + symbols + sections


def _metadata():
    return {
        "amdhsa.version": [1, 2],
        "amdhsa.target": "amdgcn-amd-amdhsa--gfx942",
        "amdhsa.kernels": [
            {
                ".name": "entry",
                ".symbol": "entry.kd",
                ".kernarg_segment_size": 24,
                ".kernarg_segment_align": 8,
                ".group_segment_fixed_size": 1024,
                ".args": [
                    {
                        ".offset": 0,
                        ".size": 8,
                        ".value_kind": "global_buffer",
                        ".address_space": "global",
                        ".align": 16,
                    },
                    {
                        ".offset": 8,
                        ".size": 8,
                        ".value_kind": "global_buffer",
                        ".address_space": "global",
                        ".align": 8,
                    },
                    {".offset": 16, ".size": 4, ".value_kind": "by_value"},
                    {".offset": 20, ".size": 4, ".value_kind": "by_value"},
                ],
            }
        ],
    }


def test_hsaco_metadata_accepts_exact_binding_and_constant_layout():
    target, alignments = _hsaco_metadata(_image(_metadata()), "entry", 2, 2, (64, 1, 1))
    assert target == "gfx942"
    assert alignments == [16, 8]


def test_hsaco_accepts_iree_target_triple_and_validates_elf_flags():
    metadata = _metadata()
    metadata["amdhsa.target"] = "amdgpu9.42-amd-amdhsa-unknown-gfx942"
    image = bytearray(_image(metadata))
    target, _ = _hsaco_metadata(bytes(image), "entry", 2, 2, (64, 1, 1))
    assert target == "gfx942"
    image[48:52] = struct.pack("<I", 0x53F)  # gfx90a, not gfx942
    with pytest.raises(ValueError, match="target"):
        _hsaco_metadata(bytes(image), "entry", 2, 2, (64, 1, 1))
    image[48:52] = struct.pack("<I", 0xA4C)  # xnack- and sramecc-
    target, _ = _hsaco_metadata(bytes(image), "entry", 2, 2, (64, 1, 1))
    assert target == "gfx942:sramecc-:xnack-"


def test_hsaco_gfx90a_target_triple_matches_elf():
    metadata = _metadata()
    metadata["amdhsa.target"] = "amdgpu9.0a-amd-amdhsa-unknown-gfx90a"
    target, _ = _hsaco_metadata(_image(metadata, 0x53F), "entry", 2, 2, (64, 1, 1))
    assert target == "gfx90a"
    metadata["amdhsa.target"] = "amdgpu9.0b-amd-amdhsa-unknown-gfx90a"
    with pytest.raises(ValueError, match="target"):
        _hsaco_metadata(_image(metadata, 0x53F), "entry", 2, 2, (64, 1, 1))


def test_stream_binding_alignment_uses_per_binding_compiler_assumptions():
    from kashi.iree_hip import _binding_alignments

    args = [
        "%arg0: !stream.binding {stream.alignment = 64 : index}",
        "%arg1: !stream.binding",
        "%arg2: !stream.binding {stream.alignment = 128 : index}",
    ]
    assert _binding_alignments(args, [1, 32, 1], "entry") == [64, 32, 128]
    with pytest.raises(ValueError, match="alignment"):
        _binding_alignments(
            ["%arg0: !stream.binding {stream.alignment = 3 : index}"], [1], "entry"
        )


def test_gpu_dispatch_rejects_lazy_negative_or_conjugate_view():
    from kashi.iree_gpu_dispatch import _validate_tensor

    neg = torch._neg_view(torch.ones(4))
    assert neg.is_contiguous() and neg.is_neg()
    with pytest.raises(NotImplementedError, match="negative|conjugate"):
        _validate_tensor(
            neg,
            "%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>",
            "entry",
        )
    conjugate = torch.ones(4, dtype=torch.complex64).conj()
    assert conjugate.is_contiguous() and conjugate.is_conj()
    with pytest.raises(NotImplementedError, match="negative|conjugate"):
        _validate_tensor(
            conjugate,
            "%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>",
            "entry",
        )


def test_gpu_launch_retains_external_owner_until_stream_completion(monkeypatch):
    import threading
    import weakref

    from kashi import iree_gpu_dispatch as dispatch

    waiting = threading.Event()
    complete = threading.Event()

    class Event:
        def record(self, stream):
            assert stream is stream_token

        def query(self):
            waiting.set()
            return complete.is_set()

    class Owner:
        pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    stream_token = object()
    owner = Owner()
    ref = weakref.ref(owner)
    dispatch._launch_retaining_bindings(stream_token, [owner], lambda: None)
    del owner
    assert waiting.wait(5)
    gc.collect()
    assert ref() is not None
    complete.set()
    for _ in range(100):
        gc.collect()
        if ref() is None:
            break
        threading.Event().wait(0.01)
    assert ref() is None


def test_gpu_launch_keeps_owners_when_event_recording_fails(monkeypatch):
    import weakref

    from kashi import iree_gpu_dispatch as dispatch

    class Event:
        def record(self, stream):
            raise RuntimeError("record failed")

    class Owner:
        pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    retained = []
    monkeypatch.setattr(dispatch, "_UNRELEASED", retained)
    owner = Owner()
    ref = weakref.ref(owner)
    with pytest.raises(RuntimeError, match="record failed"):
        dispatch._launch_retaining_bindings(object(), [owner], lambda: None)
    del owner
    gc.collect()
    assert ref() is not None
    retained.clear()
    gc.collect()
    assert ref() is None


def test_capture_requires_matching_live_graph_and_retains_bindings(monkeypatch):
    import weakref

    from kashi import iree_gpu_dispatch as dispatch

    class Stream:
        device = torch.device("cuda:0")
        cuda_stream = 123

    class Graph:
        def __init__(self):
            self.retained = []

        def retain_object(self, obj, *, synchronize_before_release):
            assert synchronize_before_release
            self.retained.append(obj)

    class Owner:
        pass

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Stream())
    monkeypatch.setattr(dispatch, "_capture_identity", lambda stream: 77)
    graph = Graph()
    owner = Owner()
    ref = weakref.ref(owner)
    dispatch._capture_started(graph)
    dispatch._retain_captured_bindings(Stream(), [owner])
    del owner
    gc.collect()
    assert ref() is not None
    dispatch._capture_ended(graph)
    with pytest.raises(RuntimeError, match="capture"):
        dispatch._retain_captured_bindings(Stream(), [])
    graph.retained.clear()
    gc.collect()
    assert ref() is None


def test_capture_does_not_confuse_different_streams(monkeypatch):
    from kashi import iree_gpu_dispatch as dispatch

    class Stream:
        device = torch.device("cuda:0")
        cuda_stream = 123

    class Other(Stream):
        cuda_stream = 456

    class Graph:
        def retain_object(self, obj, *, synchronize_before_release):
            raise AssertionError("wrong graph must not be used")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Stream())
    monkeypatch.setattr(dispatch, "_capture_identity", lambda stream: 77)
    graph = Graph()
    dispatch._capture_started(graph)
    try:
        with pytest.raises(RuntimeError, match="capture"):
            dispatch._retain_captured_bindings(Other(), [])
    finally:
        dispatch._capture_ended(graph)


def test_capture_rejects_stale_stream_capture_identity(monkeypatch):
    from kashi import iree_gpu_dispatch as dispatch

    class Stream:
        device = torch.device("cuda:0")
        cuda_stream = 123

    class Graph:
        def retain_object(self, obj, *, synchronize_before_release):
            raise AssertionError("stale graph must not be used")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Stream())
    monkeypatch.setattr(dispatch, "_capture_identity", lambda stream: 1)
    graph = Graph()
    dispatch._capture_started(graph)
    try:
        monkeypatch.setattr(dispatch, "_capture_identity", lambda stream: 2)
        with pytest.raises(RuntimeError, match="capture"):
            dispatch._retain_captured_bindings(Stream(), [])
    finally:
        dispatch._capture_ended(graph)


def test_hsaco_rejects_note_and_elf_feature_disagreement():
    metadata = _metadata()
    metadata["amdhsa.target"] = "amdgpu9.42-amd-amdhsa-unknown-gfx942:xnack+"
    image = bytearray(_image(metadata))
    image[48:52] = struct.pack("<I", 0x64C)  # xnack-, sramecc any
    with pytest.raises(ValueError, match="target"):
        _hsaco_metadata(bytes(image), "entry", 2, 2, (64, 1, 1))


@pytest.mark.parametrize(
    "change",
    [
        "missing_export",
        "wrong_symbol",
        "unknown_kind",
        "bad_pointer_size",
        "bad_constant_size",
        "overlap",
        "outside_segment",
        "missing_target",
        "unknown_hidden_arg",
        "wrong_block",
        "dynamic_lds",
        "bad_address_space",
        "bad_alignment",
        "bad_version",
        "duplicate_feature",
    ],
)
def test_hsaco_rejects_bad_metadata(change):
    metadata = _metadata()
    kernel = metadata["amdhsa.kernels"][0]
    args = kernel[".args"]
    if change == "missing_export":
        kernel[".name"] = "other"
    elif change == "wrong_symbol":
        kernel[".symbol"] = "other.kd"
    elif change == "unknown_kind":
        args[0][".value_kind"] = "made_up"
    elif change == "bad_pointer_size":
        args[0][".size"] = 4
    elif change == "bad_constant_size":
        args[2][".size"] = 8
    elif change == "overlap":
        args[1][".offset"] = 4
    elif change == "outside_segment":
        kernel[".kernarg_segment_size"] = 20
    elif change == "missing_target":
        del metadata["amdhsa.target"]
    elif change == "unknown_hidden_arg":
        args.append(
            {".offset": 24, ".size": 8, ".value_kind": "hidden_global_offset_x"}
        )
        kernel[".kernarg_segment_size"] = 32
    elif change == "wrong_block":
        kernel[".reqd_workgroup_size"] = [32, 1, 1]
    elif change == "dynamic_lds":
        args.append(
            {".offset": 24, ".size": 8, ".value_kind": "dynamic_shared_pointer"}
        )
        kernel[".kernarg_segment_size"] = 32
    elif change == "bad_address_space":
        args[0][".address_space"] = "local"
    elif change == "bad_alignment":
        args[0][".align"] = 3
    elif change == "bad_version":
        metadata["amdhsa.version"] = [2, 0]
    elif change == "duplicate_feature":
        metadata["amdhsa.target"] += ":xnack+:xnack-"
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        _hsaco_metadata(_image(metadata), "entry", 2, 2, (64, 1, 1))


def test_hsaco_rejects_malformed_elf_and_wrong_argument_counts():
    with pytest.raises((ValueError, RuntimeError)):
        _hsaco_metadata(b"not ELF", "entry", 2, 2, (64, 1, 1))
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        _hsaco_metadata(_image(_metadata()), "entry", 1, 2, (64, 1, 1))


def test_hsaco_rejects_bad_note_missing_descriptor_and_wrong_machine():
    image = bytearray(_image(_metadata()))
    image[140] = 0xC1  # Reserved MessagePack byte at start of AMDGPU note.
    with pytest.raises(ValueError):
        _hsaco_metadata(bytes(image), "entry", 2, 2, (64, 1, 1))
    image = _image(_metadata()).replace(b"\0entry.kd\0", b"\0other.kd\0")
    with pytest.raises(ValueError, match="descriptor"):
        _hsaco_metadata(image, "entry", 2, 2, (64, 1, 1))
    image = bytearray(_image(_metadata()))
    image[18:20] = struct.pack("<H", 62)  # EM_X86_64
    with pytest.raises(ValueError, match="AMDGPU"):
        _hsaco_metadata(bytes(image), "entry", 2, 2, (64, 1, 1))
    image = bytearray(_image(_metadata()))
    image[7] = 0  # ELFOSABI_NONE, not AMDGPU HSA
    with pytest.raises(ValueError, match="AMDGPU"):
        _hsaco_metadata(bytes(image), "entry", 2, 2, (64, 1, 1))


def test_hip_factory_rejects_missing_artifacts(tmp_path):
    with pytest.raises(RuntimeError, match="phase"):
        create_hip_kernel("missing", "entry", artifact_dir=tmp_path)


def test_hip_hal_layout_must_match_stream_abi():
    from kashi.iree_hip import _validate_hal_layout

    text = """
#layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, None>]>
hal.executable private @example {
  hal.executable.export public @entry ordinal(0) layout(#layout) count(%arg0: !hal.device) { }
}
"""
    _validate_hal_layout(text, "example", "entry", 2, 2)
    with pytest.raises(ValueError, match="layout"):
        _validate_hal_layout(text, "example", "entry", 2, 1)
    with pytest.raises(ValueError, match="layout"):
        _validate_hal_layout(text, "example", "entry", 1, 2)


def test_hip_missing_function_propagates_driver_error():
    import ctypes

    from kashi import iree_hip

    class Driver:
        def hipModuleLoadData(self, result, image):
            ctypes.cast(result, ctypes.POINTER(ctypes.c_void_p))[0] = 123
            return 0

        def hipModuleGetFunction(self, result, module, name):
            return 9

        def hipGetErrorString(self, status):
            return b"symbol not found"

    with pytest.raises(RuntimeError, match="hipModuleGetFunction.*symbol not found"):
        iree_hip._function_handle(Driver(), 100, 200, b"unique image", "missing")


def test_target_features_require_matching_device():
    from kashi.iree_hip import _check_target

    _check_target("gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack-")
    with pytest.raises(RuntimeError, match="target"):
        _check_target("gfx942", "gfx90a")
    with pytest.raises(RuntimeError, match="target"):
        _check_target("gfx942:xnack-", "gfx942:xnack+")
    with pytest.raises(RuntimeError, match="target"):
        _check_target("gfx942:xnack-", "gfx942")


def test_hip_driver_reports_missing_library(monkeypatch):
    from kashi import iree_hip, iree_runtime

    def missing(_name):
        raise OSError("missing")

    monkeypatch.setattr(iree_hip, "_DRIVER", None)
    monkeypatch.setattr(
        iree_runtime, "_mapped_hip_libraries", lambda: {"/missing/libamdhip64.so.7"}
    )
    monkeypatch.setattr(iree_hip.ctypes, "CDLL", missing)
    with pytest.raises(RuntimeError, match="HIP|hip"):
        iree_hip._driver()


def test_hip_driver_reuses_pytorch_versioned_runtime(monkeypatch):
    """Direct launches must use PyTorch's loaded HIP library, including `.so.7`."""
    from kashi import iree_hip, iree_runtime

    class Function:
        def __call__(self, *args):
            return 0

    class Library:
        def __getattr__(self, name):
            return Function()

    path = "/opt/wheel/libamdhip64.so.7"
    library = Library()

    def load(name):
        assert name == path
        return library

    monkeypatch.setattr(iree_hip, "_DRIVER", None)
    monkeypatch.setattr(iree_runtime, "_mapped_hip_libraries", lambda: {path})
    monkeypatch.setattr(iree_hip.ctypes, "CDLL", load)
    assert iree_hip._driver() is library


@pytest.mark.parametrize(
    "paths", [set(), {"/one/libamdhip64.so.7", "/two/libamdhip64.so.7"}]
)
def test_hip_driver_rejects_unidentifiable_runtime(monkeypatch, paths):
    """An absent or ambiguous HIP runtime must not launch against the wrong context."""
    from kashi import iree_hip, iree_runtime

    monkeypatch.setattr(iree_hip, "_DRIVER", None)
    monkeypatch.setattr(iree_runtime, "_mapped_hip_libraries", lambda: paths)
    with pytest.raises(RuntimeError, match="uniquely identify"):
        iree_hip._driver()


def test_hip_error_is_not_suppressed():
    from kashi.iree_hip import _check

    class Driver:
        def hipGetErrorString(self, status):
            return b"test error"

    with pytest.raises(RuntimeError, match="hipModuleLoadData.*test error"):
        _check(Driver(), "hipModuleLoadData", 7)


def test_hip_rejects_non_primary_context(monkeypatch):
    import ctypes

    from kashi import iree_hip

    class Driver:
        def hipDevicePrimaryCtxRetain(self, result, device):
            ctypes.cast(result, ctypes.POINTER(ctypes.c_void_p))[0] = 123
            return 0

    monkeypatch.setattr(iree_hip, "_PRIMARY_CONTEXTS", {})
    iree_hip._check_context(Driver(), 0, 123)
    with pytest.raises(RuntimeError, match="context"):
        iree_hip._check_context(Driver(), 0, 456)


def _graph(op, device="cpu"):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    value = graph.call_function(op, (x, y))
    graph.output(value)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        inp = torch.empty(16, 16, device=device)
        x.meta["val"] = y.meta["val"] = value.meta["val"] = inp
    return torch.fx.GraphModule(torch.nn.Module(), graph)


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
def test_offline_rocm_compilation_produces_validated_hsaco(tmp_path, monkeypatch, arch):
    """The actual compiler artifact must be accepted without initializing HIP."""
    from kashi.iree_backend import compile_iree_fx
    from kashi.iree_reassembly import IreeKernel

    monkeypatch.setenv("KASHI_IREE_EXECUTION", "direct")
    compiled = compile_iree_fx(
        _graph(torch.ops.aten.add.Tensor),
        output_dir=tmp_path,
        target_device="hip",
        rocm_target=arch,
    )
    assert list((compiled.artifact_dir / "kernel_files").glob("*.hsaco"))
    for phase in ("flow", "stream", "executable-targets"):
        assert list((compiled.artifact_dir / "phases").glob(f"*.{phase}.mlir"))
    kernels = [
        n.target for n in compiled.graph.graph.nodes if isinstance(n.target, IreeKernel)
    ]
    assert kernels
    for kernel in kernels:
        create_hip_kernel(
            kernel.executable, kernel.export, artifact_dir=compiled.artifact_dir
        )


@pytest.mark.skipif(
    not (torch.version.hip and torch.cuda.is_available()),
    reason="AMD HIP device required",
)
@pytest.mark.parametrize("op", [torch.ops.aten.add.Tensor, torch.ops.aten.mm.default])
def test_direct_hip_launches_on_current_stream_and_keeps_allocations(
    tmp_path, monkeypatch, op
):
    from kashi import iree_hip
    from kashi.iree_backend import compile_iree_fx
    from kashi.iree_reassembly import IreeKernel

    launches = []
    check = iree_hip._check

    def observe(lib, name, status):
        check(lib, name, status)
        if name == "hipModuleLaunchKernel":
            launches.append(name)

    monkeypatch.setattr(iree_hip, "_check", observe)
    monkeypatch.setenv("KASHI_IREE_EXECUTION", "direct")
    compiled = compile_iree_fx(
        _graph(op, "cuda"), output_dir=tmp_path, target_device="hip"
    )
    assert any(isinstance(n.target, IreeKernel) for n in compiled.graph.graph.nodes)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(256, device="cuda", dtype=torch.float32).reshape(16, 16)
        y = torch.ones_like(x)
        expected = op(x, y)
        result = compiled.graph(x, y)
        del x, y
        gc.collect()
        result = result + 1
    stream.synchronize()
    assert launches
    torch.testing.assert_close(result, expected + 1)


@pytest.mark.skipif(
    not (torch.version.hip and torch.cuda.is_available()),
    reason="AMD HIP device required",
)
def test_direct_hip_zero_grid_and_binding_validation(tmp_path, monkeypatch):
    from kashi import iree_hip

    target = torch.cuda.get_device_properties(0).gcnArchName
    arch = target.split(":")[0]
    phases = tmp_path / "phases"
    files = tmp_path / "kernel_files"
    phases.mkdir()
    files.mkdir()
    (phases / "x.flow.mlir").write_text("""
flow.executable private @example {
  func.func @entry(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>) { return }
}
""")
    (phases / "x.stream.mlir").write_text("""
stream.executable private @example {
  func.func @entry(%arg0: !stream.binding) { return }
}
""")
    (phases / "x.executable-targets.mlir").write_text(f'''
#target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {{arch = "{arch}"}}>
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>]>
hal.executable private @example {{
  hal.executable.export public @entry layout(#layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {{
    %c1 = arith.constant 1 : index
    hal.return %arg1, %c1, %c1 : index, index, index
  }} <workgroup_size = [64 : index, 1 : index, 1 : index]>
}}
''')
    metadata = _metadata()
    metadata["amdhsa.target"] = "amdgcn-amd-amdhsa--" + arch
    kernel_metadata = metadata["amdhsa.kernels"][0]
    kernel_metadata[".args"] = kernel_metadata[".args"][:1]
    kernel_metadata[".kernarg_segment_size"] = 8
    machine = next(
        code for code, name in iree_hip._ELF_MACHINES.items() if name == arch
    )
    (files / "module_example_rocm_hsaco_fb.hsaco").write_bytes(
        _image(metadata, 0x500 | machine)
    )
    kernel = create_hip_kernel("example", "entry", artifact_dir=tmp_path)

    def unexpected_driver():
        raise AssertionError("zero grid must not launch")

    monkeypatch.setattr(iree_hip, "_driver", unexpected_driver)
    kwargs = {
        "workload": [0],
        "argument_dims": [],
        "result_shapes": [],
        "result_dtypes": [],
        "tied_operands": [],
    }
    x = torch.empty(4, 4, device="cuda")
    assert kernel(x, **kwargs) == ()
    with pytest.raises(NotImplementedError, match="non-contiguous"):
        kernel(x.T, **kwargs)
    with pytest.raises(NotImplementedError, match="device"):
        kernel(torch.empty(4, 4), **kwargs)
    misaligned = torch.empty(17, device="cuda")[1:].view(4, 4)
    assert misaligned.data_ptr() % 16
    with pytest.raises(NotImplementedError, match="alignment"):
        kernel(misaligned, **{**kwargs, "workload": [1]})
