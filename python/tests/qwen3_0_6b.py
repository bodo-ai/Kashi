"""Explicit, resource-intensive Qwen3-0.6B architecture parity test.

This file is intentionally outside ordinary pytest discovery. It uses dummy
weights and fixed token IDs, not a pretrained checkpoint or tokenizer.
"""

import json
import os

import torch
from test_iree_qwen3_integration import _compare_qwen
from vllm_test_utils import vllm_test_environment

QWEN_0_6B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
}


def test_qwen3_0_6b_matches_uncompiled_generation(tmp_path, monkeypatch):
    """Compile the complete 28-layer model and compare deterministic generation."""
    if torch.version.cuda or torch.version.hip:
        if not torch.cuda.is_available():
            raise RuntimeError("The selected GPU environment has no accessible GPU")
        family = "hip" if torch.version.hip else "cuda"
    else:
        family = "cpu"
    mode = os.environ.get(
        "KASHI_IREE_EXECUTION", "iree" if family == "cpu" else "direct"
    )
    with vllm_test_environment(
        monkeypatch, cpu_cache_space=None, offline=True, clear_after=True
    ):
        (tmp_path / "config.json").write_text(json.dumps(QWEN_0_6B_CONFIG))
        # A 151,936-token vocabulary can have nearly tied top candidates;
        # exact token and numerical log-probability parity are still required.
        _compare_qwen(
            tmp_path,
            tmp_path,
            monkeypatch,
            mode,
            family,
            gpu_memory_utilization=0.8,
            require_margin=False,
        )
