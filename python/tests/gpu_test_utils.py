"""Select a usable GPU and run vLLM integration tests in a fresh process."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# vLLM validates this fraction of total memory at startup even when the KV
# cache has an explicit size. Allow room for the child's separate CUDA context.
GPU_MEMORY_UTILIZATION = 0.8
_CONTEXT_HEADROOM = 256 * 1024**2


def select_gpu(*, min_free_bytes, min_cuda_major=0):
    """Return a visible CUDA device identifier with enough free memory.

    Args:
        min_free_bytes: Minimum free memory needed by the model itself.
        min_cuda_major: Minimum CUDA compute-capability major version.

    Returns:
        An identifier valid in the parent visibility mask, including MIG IDs.
        Skips the test if no visible CUDA device can satisfy vLLM's budget.
    """
    if not torch.version.cuda or not torch.cuda.is_available():
        pytest.skip("an available NVIDIA CUDA device is required")
    candidates = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        properties = torch.cuda.get_device_properties(index)
        required = max(
            min_free_bytes,
            math.ceil(total * GPU_MEMORY_UTILIZATION) + _CONTEXT_HEADROOM,
        )
        if free >= required and properties.major >= min_cuda_major:
            candidates.append((free, index))
    if not candidates:
        pytest.skip(
            "no visible CUDA device has enough free memory for the model and vLLM budget"
        )
    _, index = max(candidates, key=lambda candidate: candidate[0])
    # Child CUDA ordinals refer to the host, not to the parent's renumbered
    # devices. Preserve an inherited ordinal, UUID, or MIG identifier verbatim.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",")[index].strip() if visible is not None else str(index)


def run_gpu_child(nodeid, *, child_marker, min_free_bytes, min_cuda_major=0):
    """Run one pytest node on a selected GPU before CUDA is initialized.

    Output is inherited instead of captured through pipes: vLLM or PyTorch may
    leave descendants holding those pipes, which otherwise prevents EOF after
    the test process exits. `pytest -s` in the parent also shows child progress.

    Args:
        nodeid: Exact pytest node to execute in the child process.
        child_marker: Environment variable preventing recursive launches.
        min_free_bytes: Minimum free GPU memory required by the model.
        min_cuda_major: Minimum CUDA compute-capability major version.
    """
    device = select_gpu(min_free_bytes=min_free_bytes, min_cuda_major=min_cuda_major)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=device, **{child_marker: "1"})
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", nodeid],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, (
        f"GPU test {nodeid} failed on {device} (exit {result.returncode})"
    )
