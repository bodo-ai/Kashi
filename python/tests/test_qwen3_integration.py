"""Exercise real Qwen3 prefill and decode through the kashi vLLM backend."""

import pytest
from backend_spy import VLLM_BACKEND, VLLM_SPY
from vllm_test_utils import (
    QWEN_MAX_TOKENS,
    QWEN_PROMPTS,
    assert_outputs_match,
    generate_qwen,
)

pytestmark = pytest.mark.integration


def test_qwen3_generation_matches_uncompiled_vllm(qwen_model_dir):
    """Compare compiled and eager Qwen3 token IDs and selected log-probabilities."""
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    compiled = generate_qwen(
        qwen_model_dir,
        CompilationConfig(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            backend=VLLM_BACKEND,
            # Replaying a CUDA graph would bypass the Python spy on decode.
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        record_generation=True,
    )
    reference = generate_qwen(
        qwen_model_dir,
        CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )

    assert len(VLLM_SPY.calls) == 1
    gm, rebuilt = VLLM_SPY.calls[0]
    assert rebuilt is not gm.forward
    # Both prompts are prefilled in one batch; subsequent calls decode both.
    assert {14, 2} <= set(VLLM_SPY.token_counts)

    assert_outputs_match(
        compiled,
        reference,
        num_prompts=len(QWEN_PROMPTS),
        max_tokens=QWEN_MAX_TOKENS,
        check_logprob_length=True,
        check_finite=True,
        rtol=1e-4,
        atol=1e-5,
        near_tie_message="near tie: token comparison cannot discriminate",
        require_two_logprobs=True,
    )
