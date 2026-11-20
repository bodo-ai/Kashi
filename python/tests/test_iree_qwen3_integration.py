"""Compare executable IREE backends against real vLLM Qwen3 inference."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm_backend_spy
from vllm_test_utils import (
    QWEN_MAX_TOKENS,
    QWEN_PROMPTS,
    assert_outputs_match,
    generate_qwen,
    model_dir,
)

# Pytest discovers the imported fixture by name; expose that use to Ruff.
__all__ = ["model_dir"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["cuda", "iree"])
def test_iree_qwen3_matches_uncompiled_generation(
    model_dir, tmp_path, monkeypatch, mode
):
    """Both modes preserve prefill, decode, token IDs and selected log-probabilities."""
    from kashi.iree_backend import compile_iree_fx
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    compiled_graphs = []

    def compile_iree(gm, example_inputs):
        """Use the real IREE pipeline while retaining the vLLM invocation spy."""
        result = compile_iree_fx(gm, output_dir=tmp_path)
        compiled_graphs.append(result)
        return result.graph.forward

    monkeypatch.setattr(vllm_backend_spy, "compile_fx", compile_iree)
    compiled = generate_qwen(
        model_dir,
        CompilationConfig(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            backend="vllm_backend_spy.spy_compile_fx",
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        record_generation=True,
    )
    reference = generate_qwen(
        model_dir,
        CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE),
    )

    assert len(compiled_graphs) == 1 and compiled_graphs[0].execution_mode == mode
    assert {14, 2} <= set(vllm_backend_spy.TOKEN_COUNTS)
    assert_outputs_match(
        compiled,
        reference,
        num_prompts=len(QWEN_PROMPTS),
        max_tokens=QWEN_MAX_TOKENS,
        check_finite=True,
        rtol=1e-4,
        atol=1e-5,
    )
