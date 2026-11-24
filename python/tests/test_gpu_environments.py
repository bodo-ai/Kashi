"""Ensure each physically present GPU family gets its matching test process."""

import os
import subprocess
from pathlib import Path

import pytest
import torch
from gpu_test_environment import physical_gpu_families, torch_family


def test_pci_gpu_family_detection(tmp_path):
    """Detect display and compute accelerators, not unrelated PCI devices."""
    for name, vendor, pci_class in (
        ("nvidia", "0x10de", "0x030200"),
        ("amd", "0x1002", "0x030000"),
        ("audio", "0x1002", "0x040300"),
        ("other", "0x8086", "0x030000"),
    ):
        device = tmp_path / name
        device.mkdir()
        (device / "vendor").write_text(vendor)
        (device / "class").write_text(pci_class)
    assert physical_gpu_families(tmp_path) == {"cuda", "hip"}


def test_wsl_gpu_family_detection_without_pci_passthrough(tmp_path, monkeypatch):
    """WSL exposes the host GPU through DXG, not Linux PCI sysfs."""
    import platform
    import shutil

    monkeypatch.setattr(
        platform, "release", lambda: "6.18.58.1-microsoft-standard-WSL2"
    )
    monkeypatch.setattr(shutil, "which", lambda _: "powershell.exe")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="PCI\\VEN_1002&DEV_73BF\nPCI\\VEN_8086&DEV_A780\n"
        ),
    )
    assert physical_gpu_families(tmp_path) == {"hip"}


def test_wsl_gpu_discovery_failure_is_not_treated_as_no_gpu(tmp_path, monkeypatch):
    """Broken Windows interop must not silently skip a present accelerator."""
    import platform

    monkeypatch.setattr(platform, "release", lambda: "microsoft-standard-WSL2")

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("Windows interop unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="Windows GPU adapters"):
        physical_gpu_families(tmp_path)


def test_gpu_child_uses_detected_family():
    """An isolated child must not pass by skipping all GPU executions."""
    expected = os.environ.get("KASHI_GPU_TEST_CHILD")
    if expected is not None:
        assert torch_family() == expected
        assert torch.cuda.is_available(), (
            f"Detected {expected} GPU is unavailable to PyTorch"
        )


def test_other_available_gpu_families_in_matching_environment():
    """Run each additional physical vendor with its own locked PyTorch wheel.

    This is also run from CPU-only test environments: a GPU host must not lose
    accelerator coverage merely because the active wheel lacks GPU support.
    Child processes use an explicit vLLM platform plugin on mixed-vendor hosts.
    """
    if os.environ.get("KASHI_GPU_TEST_CHILD"):
        return
    families = physical_gpu_families()
    current = torch_family()
    if current in families and not torch.cuda.is_available():
        pytest.fail(
            f"Detected a {current} GPU, but its PyTorch runtime cannot use it; "
            "install/configure its driver or remove it from test visibility"
        )
    repo_root = Path(__file__).resolve().parents[2]
    for family in sorted(families - {current}):
        env = os.environ.copy()
        env["KASHI_GPU_TEST_CHILD"] = family
        env.pop("VLLM_TARGET_DEVICE", None)
        env.pop("VLLM_PLUGINS", None)
        # CPU activation selects Triton's CPU backend and interpreter. Neither
        # setting may leak into the matching CUDA or ROCm test environment.
        if current == "cpu":
            env.pop("TRITON_INTERPRET", None)
            env.pop("TRITON_DEFAULT_BACKEND", None)
        command = [
            "pixi",
            "run",
            "--locked",
            "--no-install",
            "-e",
            f"{family if family == 'cuda' else 'rocm'}-test",
            "python",
            "-m",
            "pytest",
            "-q",
            "python/tests/test_gpu_environments.py::test_gpu_child_uses_detected_family",
            "python/tests/test_iree_execution.py",
            "python/tests/test_iree_qwen3_integration.py::test_iree_qwen3_matches_uncompiled_generation",
            f"python/tests/test_iree_{'cuda' if family == 'cuda' else 'hip'}.py",
            "python/tests/test_iree_runtime.py",
        ]
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=1800,
            check=False,
        )
        assert result.returncode == 0, (
            f"Detected {family} hardware, but its isolated tests failed. Install "
            f"the matching locked environment with `pixi install --locked -e "
            f"{'cuda' if family == 'cuda' else 'rocm'}-test` and configure its driver.\n"
            f"{result.stdout}\n{result.stderr}"
        )
