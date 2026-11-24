"""Compare executable IREE backends against real vLLM Qwen3 inference."""

import os
import subprocess
from pathlib import Path

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

pytestmark = pytest.mark.integration


def _compare_qwen(
    qwen_model_dir,
    tmp_path,
    monkeypatch,
    mode,
    family,
    *,
    gpu_memory_utilization=0.1,
    require_margin=True,
):
    """Compare real compiled prefill/decode and token scores for one device."""
    from kashi.iree_backend import compile_iree_fx
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    compiled_graphs = []
    executions = []

    if mode == "direct":
        if family == "cuda":
            from kashi import iree_cuda as launcher

            launch_name = "cuLaunchKernel"
        else:
            from kashi import iree_hip as launcher

            launch_name = "hipModuleLaunchKernel"

        check = launcher._check

        def record_launch(lib, name, status):
            """Record successful driver launches without replacing the launch."""
            check(lib, name, status)
            if name == launch_name:
                executions.append(name)

        monkeypatch.setattr(launcher, "_check", record_launch)
    else:
        if family == "cpu":
            from kashi.iree_cpu_runtime import _SegmentTarget
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
        gpu_memory_utilization=gpu_memory_utilization,
    )
    reference = generate_qwen(
        qwen_model_dir,
        CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE),
        gpu_memory_utilization=gpu_memory_utilization,
    )

    assert len(compiled_graphs) == 1 and compiled_graphs[0].execution_mode == mode
    result = compiled_graphs[0]
    assert result.target_device == family and result.vmfb_path.is_file()
    if family == "cuda":
        assert result.ptx_files
    elif family == "hip":
        assert result.hsaco_files
    else:
        assert result.native_files
    # vLLM may batch both prompts together or schedule them separately, in
    # particular for the full-size model. Require actual prefill and decode
    # calls in either schedule rather than requiring one specific batch size.
    counts = set(VLLM_SPY.token_counts)
    prompt_lengths = [len(prompt["prompt_token_ids"]) for prompt in QWEN_PROMPTS]
    assert sum(prompt_lengths) in counts or set(prompt_lengths) <= counts
    assert any(size in counts for size in range(1, len(QWEN_PROMPTS) + 1))
    assert_outputs_match(
        compiled,
        reference,
        num_prompts=len(QWEN_PROMPTS),
        max_tokens=QWEN_MAX_TOKENS,
        check_logprob_length=True,
        check_finite=True,
        rtol=1e-4,
        atol=1e-5,
        near_tie_message=(
            "near tie: token comparison cannot discriminate" if require_margin else None
        ),
        require_two_logprobs=True,
    )


@pytest.mark.skipif(
    not ((torch.version.cuda or torch.version.hip) and torch.cuda.is_available()),
    reason="CUDA or HIP device required",
)
@pytest.mark.parametrize("mode", ["direct", "iree"])
def test_iree_qwen3_matches_uncompiled_generation(
    qwen_model_dir, tmp_path, monkeypatch, mode
):
    """Both GPU modes match the uncompiled model on the active vendor's device."""
    family = "hip" if torch.version.hip else "cuda"
    _compare_qwen(qwen_model_dir, tmp_path, monkeypatch, mode, family)


def test_iree_cpu_qwen3_matches_uncompiled_generation(request, tmp_path, monkeypatch):
    """Run CPU Qwen even when the main test process uses an accelerator wheel."""
    if torch.version.cuda or torch.version.hip:
        # A GPU vLLM wheel lacks the CPU native extensions. Keep the CPU wheel
        # and platform probe in a separate locked environment and process.
        env = os.environ.copy()
        env.update(VLLM_TARGET_DEVICE="cpu", TRITON_INTERPRET="1")
        command = [
            "pixi",
            "run",
            "--locked",
            "--no-install",
            "-e",
            "cpu-test",
            "python",
            "-m",
            "pytest",
            "-q",
            f"{__file__}::test_iree_cpu_qwen3_matches_uncompiled_generation",
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        assert completed.returncode == 0, (
            "CPU Qwen subprocess failed; install the locked cpu-test environment "
            f"with `pixi install --locked -e cpu-test`:\n{completed.stdout}\n{completed.stderr}"
        )
        return

    assert os.environ.get("VLLM_TARGET_DEVICE") == "cpu"
    qwen_model_dir = request.getfixturevalue("qwen_model_dir")
    _compare_qwen(qwen_model_dir, tmp_path, monkeypatch, "iree", "cpu")
