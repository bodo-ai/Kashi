"""Exercise device and invocation selection independently of installed GPUs."""

import pytest
import torch
from kashi.iree_backend import compile_iree_fx
from torch._subclasses.fake_tensor import FakeTensorMode


def _graph(*, identity=False):
    """Build a CPU FX graph with the fake metadata required by the importer."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    value = x if identity else graph.call_function(torch.ops.aten.neg.default, (x,))
    graph.output(value)
    with FakeTensorMode() as fake_mode:
        fake = fake_mode.from_tensor(torch.empty(4, dtype=torch.float32))
        x.meta["val"] = fake
        if not identity:
            value.meta["val"] = -fake
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_cpu_graph_executes_with_default_mode_even_when_cuda_target_is_set(
    tmp_path, monkeypatch
):
    """CPU input must use the local IREE runtime regardless of GPU environment."""
    monkeypatch.delenv("KASHI_IREE_EXECUTION", raising=False)
    monkeypatch.setenv("IREE_CUDA_TARGET", "sm_80")
    compiled = compile_iree_fx(_graph(), output_dir=tmp_path)
    source = torch.tensor([1.0, -2.0, 3.0, -4.0])
    torch.testing.assert_close(compiled.graph(source), -source)
    assert compiled.execution_mode == "iree"
    assert compiled.vmfb_path is not None and compiled.vmfb_path.is_file()
    assert not compiled.ptx_files


def test_cpu_direct_mode_rejects_even_empty_graph(tmp_path, monkeypatch):
    """An explicit unsupported CPU mode is never accepted as an empty-driver shortcut."""
    monkeypatch.setenv("KASHI_IREE_EXECUTION", "direct")
    with pytest.raises(ValueError, match="direct.*CPU|CPU.*direct"):
        compile_iree_fx(_graph(identity=True), output_dir=tmp_path, target_device="cpu")


def test_legacy_cuda_mode_is_rejected(tmp_path, monkeypatch):
    """The mode setting has a single device-independent spelling for direct calls."""
    monkeypatch.setenv("KASHI_IREE_EXECUTION", "cuda")
    with pytest.raises(ValueError, match="KASHI_IREE_EXECUTION"):
        compile_iree_fx(_graph(identity=True), output_dir=tmp_path)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ("gfx942", ("gfx942", "")),
        ("gfx942:sramecc+:xnack-", ("gfx942", "+sramecc,-xnack")),
        ("gfx90a:xnack+", ("gfx90a", "+xnack")),
    ],
)
def test_rocm_target_features_are_translated(metadata, expected):
    """The compiler receives architecture and feature flags separately."""
    from kashi.iree_device import parse_rocm_target

    assert parse_rocm_target(metadata) == expected


@pytest.mark.parametrize(
    "metadata",
    ["sm_80", "gfx942:unknown+", "gfx942:xnack+:xnack-", "gfx942:xnack"],
)
def test_invalid_rocm_target_metadata_is_rejected(metadata):
    """Unsupported target features cannot be silently lost or contradicted."""
    from kashi.iree_device import parse_rocm_target

    with pytest.raises(ValueError, match="ROCm"):
        parse_rocm_target(metadata)


def test_device_family_follows_build_not_accelerator_visibility(monkeypatch):
    """CPU stays CPU; CUDA-shaped PyTorch devices follow the actual build."""
    from kashi.iree_device import select_target

    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    assert select_target("cpu").family == "cpu"
    assert select_target("cuda:1").family == "cuda"
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "7.0")
    assert select_target("cuda:1").family == "hip"
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    with pytest.raises(ValueError, match="ambiguous"):
        select_target("cuda:1")


def test_explicit_target_precedence_and_conflicts(monkeypatch):
    """Explicit architecture enables cross-compilation, but never overrides family."""
    from kashi.iree_device import select_target

    monkeypatch.setenv("IREE_CUDA_TARGET", "sm_80")
    assert select_target("cpu").family == "cpu"
    assert select_target("cpu", cuda_target="sm_90").family == "cuda"
    assert select_target("cpu", rocm_target="gfx942").family == "hip"
    with pytest.raises(ValueError, match="target"):
        select_target("cpu", cuda_target="sm_80", rocm_target="gfx942")
    with pytest.raises(ValueError, match="target"):
        select_target("cpu", target_device="cpu", cuda_target="sm_80")
    with pytest.raises(ValueError, match="device"):
        select_target(None)


def test_offline_target_flags_do_not_probe_gpu(monkeypatch):
    """Explicit architectures compile offline and preserve ROCm feature signs."""
    from kashi.iree_device import select_target

    def unexpected_probe(*args, **kwargs):
        raise AssertionError("offline compilation probed a GPU")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_probe)
    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected_probe)
    assert select_target("cpu", cuda_target="sm_90").compiler_flags() == [
        "--iree-hal-target-device=cuda",
        "--iree-cuda-target=sm_90",
    ]
    assert select_target(
        "cpu", rocm_target="gfx942:sramecc+:xnack-"
    ).compiler_flags() == [
        "--iree-hal-target-device=hip",
        "--iree-rocm-target=gfx942",
        "--iree-rocm-target-features=+sramecc,-xnack",
    ]


def test_detected_rocm_architecture_uses_captured_device(monkeypatch):
    """A ROCm device's own architecture and features determine its artifact."""
    from types import SimpleNamespace

    from kashi.iree_device import select_target

    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "7.0")

    def properties(device):
        assert device == torch.device("cuda:1")
        return SimpleNamespace(gcnArchName="gfx90a:xnack+")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    assert select_target("cuda:1").compiler_flags() == [
        "--iree-hal-target-device=hip",
        "--iree-rocm-target=gfx90a",
        "--iree-rocm-target-features=+xnack",
    ]


def test_missing_gpu_architecture_fails_with_actionable_error(monkeypatch):
    """No unrelated installed accelerator is selected when target discovery fails."""
    from kashi.iree_device import select_target

    def unavailable(*args, **kwargs):
        raise RuntimeError("no device")

    monkeypatch.delenv("IREE_CUDA_TARGET", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable)
    monkeypatch.setattr(torch.cuda, "get_device_properties", unavailable)
    with pytest.raises(ValueError, match="cuda_target"):
        select_target("cpu", target_device="cuda").compiler_flags()
    with pytest.raises(ValueError, match="rocm_target"):
        select_target("cpu", target_device="hip").compiler_flags()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_device": "metal"},
        {"cuda_target": ""},
        {"target_device": "cuda", "rocm_target": "gfx942"},
    ],
)
def test_invalid_target_overrides_fail(kwargs):
    """Invalid overrides are rejected rather than ignored or reinterpreted."""
    from kashi.iree_device import select_target

    with pytest.raises(ValueError, match="target"):
        select_target("cpu", **kwargs)


def test_unsupported_source_device_and_mode_are_rejected(monkeypatch):
    """Unsupported source families and legacy mode values must fail clearly."""
    from kashi.iree_device import execution_mode, select_target

    with pytest.raises(ValueError, match="source device"):
        select_target("meta")
    monkeypatch.setenv("KASHI_IREE_EXECUTION", "cuda")
    with pytest.raises(ValueError, match="KASHI_IREE_EXECUTION"):
        execution_mode(select_target("cpu"))


def _cuda_metadata_graph():
    """Capture a CUDA-shaped fake graph without allocating device memory."""
    graph = _graph()
    with FakeTensorMode():
        value = torch.empty(4, device="cuda:0")
        for node in graph.graph.nodes:
            if node.op != "output":
                node.meta["val"] = value
    return graph


def test_unindexed_cuda_device_matches_current_tensor_device(monkeypatch):
    """A CUDA allocation and CUDA tensor on the same GPU are not mixed devices."""
    from kashi.iree_backend import _source_device

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert _source_device(_cuda_metadata_graph(), "cuda") == "cuda:0"


def test_cpu_cross_compilation_accepts_cuda_fake_metadata(tmp_path, monkeypatch):
    """CPU cross-compilation is possible without changing captured fake metadata."""
    monkeypatch.setenv("KASHI_IREE_EXECUTION", "iree")
    compiled = compile_iree_fx(
        _cuda_metadata_graph(), output_dir=tmp_path, target_device="cpu"
    )
    source = torch.tensor([1.0, -2.0, 3.0, -4.0])
    torch.testing.assert_close(compiled.graph(source), -source)


def test_tensor_output_metadata_selects_cpu_without_tensor_inputs(
    tmp_path, monkeypatch
):
    """A tensor-producing graph can identify its device from its output metadata."""
    graph = torch.fx.Graph()
    result = graph.call_function(torch.ops.aten.zeros.default, ([4],))
    graph.output(result)
    with FakeTensorMode() as fake_mode:
        result.meta["val"] = fake_mode.from_tensor(torch.zeros(4))
    monkeypatch.delenv("KASHI_IREE_EXECUTION", raising=False)
    compiled = compile_iree_fx(
        torch.fx.GraphModule(torch.nn.Module(), graph), output_dir=tmp_path
    )
    torch.testing.assert_close(compiled.graph(), torch.zeros(4))
    assert compiled.target_device == "cpu"
