"""Exercise real Qwen3 prefill and decode through the kashi vLLM backend."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm_backend_spy
from vllm_test_utils import (
    QWEN_MAX_TOKENS,
    QWEN_PROMPTS,
    assert_outputs_match,
    generate_qwen,
    model_dir, #noqa
)


def test_qwen3_generation_matches_uncompiled_vllm(model_dir):
    """Compare compiled and eager Qwen3 token IDs and selected log-probabilities."""
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    compiled = generate_qwen(
        model_dir,
        CompilationConfig(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            backend="vllm_backend_spy.spy_compile_fx",
            # Replaying a CUDA graph would bypass the Python spy on decode.
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        record_generation=True,
    )
    reference = generate_qwen(
        model_dir,
        CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )

    assert len(vllm_backend_spy.CALLS) == 1
    gm, rebuilt = vllm_backend_spy.CALLS[0]
    assert rebuilt is not gm.forward
    # Both prompts are prefetched in one batch; subsequent calls decode both.
    assert {14, 2} <= set(vllm_backend_spy.TOKEN_COUNTS)

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
