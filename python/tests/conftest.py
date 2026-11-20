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
