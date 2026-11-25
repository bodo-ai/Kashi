"""Compare executable IREE backends against real vLLM Qwen3 inference."""

import backend_spy
import pytest
from backend_spy import VLLM_BACKEND, VLLM_SPY
from vllm_test_utils import (
    QWEN_MAX_TOKENS,
    QWEN_PROMPTS,
    assert_outputs_match,
    generate_qwen,
)

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["cuda", "iree"])
def test_iree_qwen3_matches_uncompiled_generation(
    qwen_model_dir, tmp_path, monkeypatch, mode
):
    """Both modes preserve prefill, decode, token IDs and selected log-probabilities."""
    from kashi.iree_backend import compile_iree_fx
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    compiled_graphs = []
    executions = []

    if mode == "cuda":
        from kashi import iree_cuda

        check = iree_cuda._check

        def record_launch(lib, name, status):
            """Record successful driver launches without replacing the launch."""
            check(lib, name, status)
            if name == "cuLaunchKernel":
                executions.append(name)

        monkeypatch.setattr(iree_cuda, "_check", record_launch)
    else:
        from kashi.iree_runtime import _SegmentTarget

        invoke = _SegmentTarget.__call__

        def record_segment(target, *args):
            """Record successful VMFB calls while executing the real segment."""
            result = invoke(target, *args)
            executions.append(target.name)
            return result

        monkeypatch.setattr(_SegmentTarget, "__call__", record_segment)

    def compile_iree(gm, example_inputs):
        """Compile with IREE and require real execution on every graph invocation."""
        result = compile_iree_fx(gm, output_dir=tmp_path)
        compiled_graphs.append(result)

        def run(*args):
            before = len(executions)
            outputs = result.graph(*args)
            # vLLM_SPY separately records actual request invocations after warmup.
            assert len(executions) > before, f"no {mode} execution during graph call"
            return outputs

        return run

    monkeypatch.setattr(backend_spy, "compile_fx", compile_iree)
    compiled = generate_qwen(
        qwen_model_dir,
        CompilationConfig(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            backend=VLLM_BACKEND,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        record_generation=True,
    )
    reference = generate_qwen(
        qwen_model_dir,
        CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE),
    )

    assert len(compiled_graphs) == 1 and compiled_graphs[0].execution_mode == mode
    assert compiled_graphs[0].ptx_files and compiled_graphs[0].vmfb_path.is_file()
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
