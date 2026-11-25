"""Shared fixtures for the Python test suite."""

import json
import sys
from pathlib import Path

import pytest
from vllm_test_utils import QWEN_CONFIG, SIMPLE_VLLM_CONFIG, vllm_test_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_MODELS_DIR = Path(__file__).resolve().parent / "models"


@pytest.fixture
def repo_root():
    """Root of the repository checkout."""
    return REPO_ROOT


@pytest.fixture
def simple_vllm_model_dir(tmp_path, monkeypatch):
    """Register the local SimpleVllm model and provide its isolated configuration."""
    with vllm_test_environment(monkeypatch, cpu_cache_space="1"):
        sys.path.insert(0, str(TEST_MODELS_DIR))
        try:
            from simple_vllm_model import SimpleVllmForCausalLM
        finally:
            sys.path.pop(0)
        from vllm import ModelRegistry

        ModelRegistry.register_model("SimpleVllmForCausalLM", SimpleVllmForCausalLM)
        (tmp_path / "config.json").write_text(json.dumps(SIMPLE_VLLM_CONFIG))
        yield tmp_path


@pytest.fixture
def qwen_model_dir(tmp_path, monkeypatch):
    """Provide an offline Qwen3 configuration and isolate vLLM test settings."""
    with vllm_test_environment(
        monkeypatch, cpu_cache_space=None, offline=True, clear_after=True
    ):
        (tmp_path / "config.json").write_text(json.dumps(QWEN_CONFIG))
        yield tmp_path


@pytest.fixture
def qwen38_model_dir(tmp_path, monkeypatch, request):
    """Provide the pinned 27B configuration with only its first two layers.

    The BF16 variant is the unquantized base model for CPU hosts without AMX
    FP8 support; GPU parity uses the original block-FP8 configuration.
    """
    config = json.loads((TEST_MODELS_DIR / "qwen38_27b_config.json").read_text())
    variant = getattr(request, "param", "fp8")
    if variant == "bf16":
        config.pop("quantization_config")
    elif variant != "fp8":
        raise ValueError(f"Unsupported Qwen3.8 test variant: {variant}")
    with vllm_test_environment(
        monkeypatch, cpu_cache_space=None, offline=True, clear_after=True
    ):
        (tmp_path / "config.json").write_text(json.dumps(config))
        yield tmp_path
