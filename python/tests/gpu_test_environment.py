"""Detect accelerator families and explicitly select vLLM's test platform."""

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_VENDOR_FAMILIES = {"0x10de": "cuda", "0x1002": "hip"}
_PLATFORM_CLASSES = {
    "cpu": "vllm.platforms.cpu.CpuPlatform",
    "cuda": "vllm.platforms.cuda.CudaPlatform",
    "hip": "vllm.platforms.rocm.RocmPlatform",
}


def physical_gpu_families(devices=Path("/sys/bus/pci/devices")) -> set[str]:
    """Return NVIDIA/AMD GPU families, including Windows adapters exposed to WSL.

    Linux PCI discovery catches missing drivers or mismatched PyTorch wheels.
    WSL exposes GPUs via DXG rather than PCI sysfs, so use the Windows adapter
    inventory there. Detection does not claim the GPU can execute a kernel.
    """
    families = set()
    if devices.is_dir():
        for device in devices.iterdir():
            try:
                vendor = (device / "vendor").read_text().strip().lower()
                pci_class = int((device / "class").read_text().strip(), 16) >> 16
            except (OSError, ValueError):
                continue
            if pci_class in (0x03, 0x12) and vendor in _VENDOR_FAMILIES:
                families.add(_VENDOR_FAMILIES[vendor])
    if "microsoft" in platform.release().lower():
        # SSH-launched WSL sessions need not include Windows executables on PATH.
        powershell = shutil.which("powershell.exe") or (
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        try:
            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    (
                        "Get-CimInstance Win32_VideoController -ErrorAction Stop | "
                        "ForEach-Object { $_.PNPDeviceID }"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                "Cannot discover Windows GPU adapters from WSL"
            ) from error
        for vendor in re.findall(r"(?i)PCI\\VEN_([0-9a-f]{4})", result.stdout):
            family = _VENDOR_FAMILIES.get(f"0x{vendor.lower()}")
            if family is not None:
                families.add(family)
    return families


def torch_family() -> str:
    """Return the active PyTorch wheel's CPU, CUDA, or ROCm family."""
    cuda, hip = bool(torch.version.cuda), bool(torch.version.hip)
    if cuda and hip:
        raise RuntimeError("The PyTorch build ambiguously claims both CUDA and HIP")
    return "cuda" if cuda else "hip" if hip else "cpu"


def select_vllm_platform_for_tests():
    """Select this wheel's platform through vLLM's public platform-plugin API.

    vLLM probes NVML and AMD SMI independently of the PyTorch wheel, so both
    built-in plugins can activate on a mixed-vendor host. A process-local
    explicit plugin selects the actual wheel's platform without hiding devices
    or changing vLLM's implementation. Keep the temporary distribution alive
    until process shutdown because discovery happens lazily.
    """
    devices = physical_gpu_families()
    family = torch_family()
    if len(devices) < 2 and not (family == "cpu" and devices):
        return None
    workspace = tempfile.TemporaryDirectory(prefix="kashi-vllm-platform-")
    root = Path(workspace.name)
    distribution = root / "kashi_test_platform-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: kashi-test-platform\nVersion: 1.0\n")
    (distribution / "entry_points.txt").write_text(
        "[vllm.platform_plugins]\nkashi_test_platform = kashi_test_platform:select\n"
    )
    (root / "kashi_test_platform.py").write_text(
        f"def select():\n    return {_PLATFORM_CLASSES[family]!r}\n"
    )
    sys.path.insert(0, str(root))
    os.environ["VLLM_PLUGINS"] = "kashi_test_platform"
    return workspace
