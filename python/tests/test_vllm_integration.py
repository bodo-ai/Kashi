"""End-to-end: SimpleVllm through the real vLLM engine with and without the
FX -> MLIR -> FX backend, using identical seeded weights."""

import pytest
import torch
from backend_spy import VLLM_BACKEND, VLLM_SPY
from fx_test_utils import call_targets
from vllm_test_utils import assert_outputs_match, generate

pytestmark = pytest.mark.integration

# Weights are filled after load: vLLM's dummy weights (+-1e-3) make every
# logprob identical in float32, so a comparison could not detect broken attention.
WEIGHT_SCALE = 0.3
WEIGHT_SEED = 0

PROMPTS = [
    {"prompt_token_ids": [1, 2, 3, 4, 5]},
    {"prompt_token_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18]},
]


def _vllm_attention_targets(gm):
    """Return the vLLM attention targets in an FX graph."""
    ops = torch.ops.vllm
    return call_targets(
        gm,
        only={
            ops.unified_kv_cache_update,
            ops.unified_attention_with_output,
            ops.unified_kv_cache_update.default,
            ops.unified_attention_with_output.default,
        },
    )


def test_vllm_matches_uncompiled_vllm(simple_vllm_model_dir):
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
        simple_vllm_model_dir,
        CompilationConfig(mode=CompilationMode.DYNAMO_TRACE_ONCE, backend=VLLM_BACKEND),
        PROMPTS,
        **options,
    )
    reference = generate(
        simple_vllm_model_dir,
        CompilationConfig(mode=CompilationMode.NONE),
        PROMPTS,
        **options,
    )

    # One capture serves batched prefill and decode, and is really rebuilt.
    assert len(VLLM_SPY.calls) == 1
    gm, fn = VLLM_SPY.calls[0]
    assert fn is not gm.forward
    # ...and the rebuilt callable actually ran for batched prefill (5 + 9
    # tokens) and for decode (one token per prompt).
    assert {14, 2} <= set(VLLM_SPY.token_counts)

    # The imported MLIR represents attention functionally.
    (text,) = VLLM_SPY.modules
    assert 'torch.operator "torch.kashi.unified_attention"' in text
    assert "unified_attention_with_output" not in text

    vllm_ops = torch.ops.vllm
    assert _vllm_attention_targets(gm) == [
        vllm_ops.unified_kv_cache_update,
        vllm_ops.unified_attention_with_output,
    ]
    assert _vllm_attention_targets(fn.__self__) == [
        vllm_ops.unified_kv_cache_update.default,
        vllm_ops.unified_attention_with_output.default,
    ]

    assert_outputs_match(
        compiled,
        reference,
        near_tie_message="near tie: comparison cannot discriminate",
    )
