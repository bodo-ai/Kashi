"""End-to-end: SimpleVllm through the real vLLM engine with and without the
FX -> MLIR -> FX backend, using identical seeded weights."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm_backend_spy
from vllm_test_utils import assert_outputs_match, generate, vllm_test_environment

# Weights are filled after load: vLLM's dummy weights (+-1e-3) make every
# logprob identical in float32, so a comparison could not detect broken attention.
WEIGHT_SCALE = 0.3
WEIGHT_SEED = 0

CONFIG = {
    "architectures": ["SimpleVllmForCausalLM"],
    "model_type": "llama",
    "hidden_size": 128,
    "num_attention_heads": 2,  # head size 64; CPU attention rejects 16
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "intermediate_size": 256,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "torch_dtype": "float32",
}

PROMPTS = [
    {"prompt_token_ids": [1, 2, 3, 4, 5]},
    {"prompt_token_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18]},
]


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Register the local model and provide its isolated vLLM configuration."""
    with vllm_test_environment(monkeypatch, cpu_cache_space="1"):
        sys.path.insert(0, str(Path(__file__).parent / "models"))
        try:
            from simple_vllm_model import SimpleVllmForCausalLM
        finally:
            sys.path.pop(0)
        from vllm import ModelRegistry

        ModelRegistry.register_model("SimpleVllmForCausalLM", SimpleVllmForCausalLM)
        (tmp_path / "config.json").write_text(json.dumps(CONFIG))
        yield tmp_path


def _vllm_targets(gm):
    """Return the vLLM attention targets in an FX graph."""
    ops = torch.ops.vllm
    attention = {
        ops.unified_kv_cache_update,
        ops.unified_attention_with_output,
        ops.unified_kv_cache_update.default,
        ops.unified_attention_with_output.default,
    }
    return [
        n.target
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target in attention
    ]


def test_vllm_matches_uncompiled_vllm(model_dir):
    from vllm.config import CompilationConfig, CompilationMode

    options = {
        "weight_seed": WEIGHT_SEED,
        "weight_scale": WEIGHT_SCALE,
        "engine_options": {
            "max_model_len": 128,
            "gpu_memory_utilization": 0.5,
            "seed": 0,
        },
        "sampling_options": {"max_tokens": 4, "temperature": 0, "logprobs": 5},
    }
    compiled = generate(
        model_dir,
        CompilationConfig(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            backend="vllm_backend_spy.spy_compile_fx",
        ),
        PROMPTS,
        **options,
    )
    reference = generate(
        model_dir, CompilationConfig(mode=CompilationMode.NONE), PROMPTS, **options
    )

    # One capture serves batched prefill and decode, and is really rebuilt.
    assert len(vllm_backend_spy.CALLS) == 1
    gm, fn = vllm_backend_spy.CALLS[0]
    assert fn is not gm.forward
    # ...and the rebuilt callable actually ran for batched prefill (5 + 9
    # tokens) and for decode (one token per prompt).
    assert {14, 2} <= set(vllm_backend_spy.TOKEN_COUNTS)

    # The imported MLIR represents attention functionally.
    (text,) = vllm_backend_spy.MODULES
    assert 'torch.operator "torch.kashi.unified_attention"' in text
    assert "unified_attention_with_output" not in text

    vllm_ops = torch.ops.vllm
    assert _vllm_targets(gm) == [
        vllm_ops.unified_kv_cache_update,
        vllm_ops.unified_attention_with_output,
    ]
    assert _vllm_targets(fn.__self__) == [
        vllm_ops.unified_kv_cache_update.default,
        vllm_ops.unified_attention_with_output.default,
    ]

    assert_outputs_match(
        compiled,
        reference,
        near_tie_message="near tie: comparison cannot discriminate",
    )
