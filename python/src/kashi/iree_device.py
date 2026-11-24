"""Select an IREE compilation target from graph devices and explicit overrides."""

import os
import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IreeTarget:
    """Compilation family and architecture overrides for one captured graph.

    ``source_device`` is the PyTorch device recorded in the source graph. It
    may differ from ``family`` during offline cross-compilation; the runtime
    must still verify that actual tensors belong to the selected family.
    """

    family: str
    source_device: str | None
    cuda_target: str | None
    rocm_target: str | None

    def compiler_flags(self) -> list[str]:
        """Return IREE target options, resolving device architecture if needed."""
        if self.family == "cpu":
            # A local HAL device does not imply an executable backend. Request
            # LLVM explicitly and specialize native code for this host.
            return [
                "--iree-hal-target-device=local",
                "--iree-hal-local-target-device-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host",
            ]
        if self.family == "cuda":
            return [
                "--iree-hal-target-device=cuda",
                f"--iree-cuda-target={_cuda_target(self.cuda_target, self.source_device)}",
            ]
        metadata = self.rocm_target or _device_rocm_target(self.source_device)
        architecture, features = parse_rocm_target(metadata)
        flags = ["--iree-hal-target-device=hip", f"--iree-rocm-target={architecture}"]
        if features:
            flags.append(f"--iree-rocm-target-features={features}")
        return flags


def _cuda_target(cuda_target: str | None, source_device: str | None = None) -> str:
    """Resolve an explicit CUDA architecture or the captured CUDA device's target."""
    target = cuda_target or os.environ.get("IREE_CUDA_TARGET")
    if target:
        return target
    try:
        device = torch.device(source_device) if source_device else None
        if device is not None and device.type != "cuda":
            device = None
        major, minor = torch.cuda.get_device_capability(device)
    except Exception as error:
        raise ValueError(
            "Set cuda_target or IREE_CUDA_TARGET when no CUDA device is available"
        ) from error
    return f"sm_{major}{minor}"


def parse_rocm_target(metadata: str) -> tuple[str, str]:
    """Split a HIP ``gfx`` device name into IREE architecture and feature flags.

    PyTorch reports features as colon-separated names with a trailing sign,
    whereas the IREE LLVM option expects leading signs separated by commas.
    Unknown or repeated features are rejected rather than silently discarded.
    """
    parts = metadata.split(":") if isinstance(metadata, str) else []
    if not parts or not re.fullmatch(r"gfx[0-9a-f]+", parts[0]):
        raise ValueError(f"Invalid ROCm target metadata: {metadata!r}")
    features = []
    seen = set()
    for feature in parts[1:]:
        match = re.fullmatch(r"(sramecc|xnack)([+-])", feature)
        if match is None or match.group(1) in seen:
            raise ValueError(
                f"Invalid or contradictory ROCm target metadata: {metadata!r}"
            )
        seen.add(match.group(1))
        features.append(f"{match.group(2)}{match.group(1)}")
    return parts[0], ",".join(features)


def _device_rocm_target(source_device: str | None) -> str:
    """Read the selected PyTorch HIP device's architecture and feature metadata."""
    try:
        device = torch.device(source_device) if source_device else None
        if device is not None and device.type != "cuda":
            device = None
        properties = torch.cuda.get_device_properties(device)
        metadata = properties.gcnArchName
    except Exception as error:
        raise ValueError(
            "Set rocm_target when no PyTorch HIP device is available"
        ) from error
    # Validate immediately; a bad property must not turn into a guessed target.
    parse_rocm_target(metadata)
    return metadata


def select_target(
    source_device: str | None,
    *,
    target_device: str | None = None,
    cuda_target: str | None = None,
    rocm_target: str | None = None,
) -> IreeTarget:
    """Choose CPU, CUDA, or HIP without probing unrelated physical devices.

    Explicit family/architecture arguments permit offline cross-compilation.
    Otherwise the graph device determines the family, and CUDA-shaped devices
    are disambiguated by the PyTorch build. ``IREE_CUDA_TARGET`` supplies an
    architecture only after CUDA has already been selected.
    """
    if target_device not in (None, "cpu", "cuda", "hip"):
        raise ValueError(f"Invalid target_device: {target_device!r}")
    if cuda_target is not None and not cuda_target:
        raise ValueError("cuda_target must be a non-empty target")
    if rocm_target is not None:
        parse_rocm_target(rocm_target)
    if cuda_target is not None and rocm_target is not None:
        raise ValueError("cuda_target and rocm_target cannot both be supplied")
    if target_device is not None:
        if cuda_target is not None and target_device != "cuda":
            raise ValueError("cuda_target conflicts with target_device")
        if rocm_target is not None and target_device != "hip":
            raise ValueError("rocm_target conflicts with target_device")
        family = target_device
    elif cuda_target is not None:
        family = "cuda"
    elif rocm_target is not None:
        family = "hip"
    else:
        if source_device is None:
            raise ValueError("Cannot select IREE target without tensor device metadata")
        device = torch.device(source_device)
        if device.type == "cpu":
            family = "cpu"
        elif device.type == "cuda":
            cuda_build = bool(torch.version.cuda)
            hip_build = bool(torch.version.hip)
            if cuda_build == hip_build:
                raise ValueError(
                    "CUDA-shaped device has ambiguous PyTorch CUDA/HIP build"
                )
            family = "cuda" if cuda_build else "hip"
        else:
            raise ValueError(f"Unsupported IREE source device: {source_device}")
    return IreeTarget(family, source_device, cuda_target, rocm_target)


def execution_mode(target: IreeTarget) -> str:
    """Resolve ``KASHI_IREE_EXECUTION`` and reject unsupported CPU direct calls."""
    mode = os.environ.get("KASHI_IREE_EXECUTION")
    if mode is None:
        mode = "iree" if target.family == "cpu" else "direct"
    if mode not in ("direct", "iree"):
        raise ValueError(
            f"KASHI_IREE_EXECUTION must be 'direct' or 'iree', got {mode!r}"
        )
    if mode == "direct" and target.family == "cpu":
        raise ValueError("KASHI_IREE_EXECUTION=direct is unsupported for CPU targets")
    return mode
