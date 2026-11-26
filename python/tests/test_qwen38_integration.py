"""Exercise the first two full-width Qwen3.8-27B text layers end to end."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gpu_test_utils
import pytest
import torch
from backend_spy import VLLM_BACKEND, VLLM_SPY
from vllm_test_utils import (
    QWEN_MAX_TOKENS,
    QWEN_PROMPTS,
    assert_outputs_match,
    generate_qwen38,
)

pytestmark = pytest.mark.integration
_CHILD = "KASHI_QWEN38_TEST_CHILD"
_MIN_GPU_BYTES = 7 * 1024**3
_MIN_CPU_BYTES = 10 * 1024**3


def test_gpu_selection_preserves_visible_mig_identifier_on_ties(monkeypatch):
    """The selected child inherits the actual visible MIG instance, even on ties."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.version, "cuda", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _: (_MIN_GPU_BYTES, _MIN_GPU_BYTES)
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=8, uuid=object()),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-first,MIG-second")
    assert (
        gpu_test_utils.select_gpu(min_free_bytes=_MIN_GPU_BYTES, min_cuda_major=8)
        == "MIG-first"
    )


def test_gpu_selection_respects_vllm_memory_budget(monkeypatch):
    """A larger free byte count is unusable if vLLM's 90% budget exceeds it."""
    from types import SimpleNamespace

    gib = 1024**3
    memory = [(10 * gib, 16 * gib), (8 * gib, 8 * gib)]
    monkeypatch.setattr(torch.version, "cuda", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: memory[index])
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(major=8)
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert (
        gpu_test_utils.select_gpu(min_free_bytes=_MIN_GPU_BYTES, min_cuda_major=8)
        == "1"
    )


def test_gpu_child_exits_without_waiting_for_descendant_output(tmp_path, monkeypatch):
    """A descendant holding inherited output must not keep the parent waiting."""
    executable = tmp_path / "pytest-child"
    release = tmp_path / "release"
    grandchild = (
        "import pathlib,time; "
        f"release=pathlib.Path({str(release)!r}); "
        "deadline=time.monotonic()+15; "
        "exec('while not release.exists() and time.monotonic() < deadline:\\n time.sleep(0.01)')"
    )
    executable.write_text(
        f"#!{sys.executable}\n"
        "import subprocess,sys\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
    )
    executable.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(gpu_test_utils, "select_gpu", lambda **kwargs: "0")
    with ThreadPoolExecutor(max_workers=1) as pool:
        child = pool.submit(
            gpu_test_utils.run_gpu_child,
            "unused-node-id",
            child_marker=_CHILD,
            min_free_bytes=_MIN_GPU_BYTES,
            min_cuda_major=8,
        )
        try:
            child.result(timeout=5)
        finally:
            release.touch()


def _assert_two_gdn_layers(*, expect_fp8):
    """Check both recurrent layers and, when requested, real FP8 weights."""
    assert len(VLLM_SPY.calls) == 1
    graph, rebuilt = VLLM_SPY.calls[0]
    assert rebuilt is not graph.forward
    gdn = [
        node for node in graph.graph.nodes if "gdn_attention_core" in str(node.target)
    ]
    assert len(gdn) == 2, f"expected two GDN layers, found {len(gdn)}"
    layer_names = [node.kwargs.get("layer_name", node.args[-1]) for node in gdn]
    assert layer_names[0] != layer_names[1]
    if expect_fp8:
        from vllm.scalar_type import scalar_types

        nodes = list(graph.graph.nodes)
        fp8_tensor = any(
            isinstance(
                value := node.meta.get("val", node.meta.get("example_value")),
                torch.Tensor,
            )
            and value.dtype == torch.float8_e4m3fn
            for node in nodes
        )
        # Marlin stores FP8 codes packed into int32 tensors on GPUs without
        # native FP8, so the operator's scalar type identifies that path.
        packed_fp8 = any(
            "marlin_gemm" in str(node.target)
            and node.kwargs.get("b_q_type_id") == scalar_types.float8_e4m3fn.id
            for node in nodes
        )
        assert fp8_tensor or packed_fp8, "expected FP8 weights or an FP8 Marlin GEMM"
    counts = set(VLLM_SPY.token_counts)
    lengths = [len(prompt["prompt_token_ids"]) for prompt in QWEN_PROMPTS]
    assert sum(lengths) in counts or set(lengths) <= counts
    assert any(count in counts for count in range(1, len(QWEN_PROMPTS) + 1))


def _compare(model_dir, compilation_config, *, expect_fp8, rtol=1e-4):
    """Compare separate engines using the same two prompts and dummy weights."""
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    compiled = generate_qwen38(model_dir, compilation_config, record_generation=True)
    reference = generate_qwen38(
        model_dir,
        CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE),
    )
    _assert_two_gdn_layers(expect_fp8=expect_fp8)
    assert_outputs_match(
        compiled,
        reference,
        num_prompts=len(QWEN_PROMPTS),
        max_tokens=QWEN_MAX_TOKENS,
        check_logprob_length=True,
        check_finite=True,
        rtol=rtol,
        atol=1e-5,
        near_tie_message="near tie: token comparison cannot discriminate",
        require_two_logprobs=True,
    )


@pytest.mark.parametrize(
    "reduced_precision", [False, True], ids=["fp32-accum", "default-bf16"]
)
@pytest.mark.parametrize("mode", ["fx", "cuda", "iree"])
def test_qwen38_fp8_gpu_matches_uncompiled(
    request, monkeypatch, tmp_path, mode, reduced_precision
):
    """Match 0.6B's FX, direct CUDA, and IREE-runtime parity tests using FP8."""
    if os.environ.get(_CHILD) != "1":
        gpu_test_utils.run_gpu_child(
            request.node.nodeid,
            child_marker=_CHILD,
            min_free_bytes=_MIN_GPU_BYTES,
            min_cuda_major=8,
        )
        return

    import backend_spy
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    model_dir = request.getfixturevalue("qwen38_model_dir")
    executions = []
    if mode != "fx":
        from kashi.iree_backend import compile_iree_fx

        monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
        if mode == "cuda":
            from kashi import iree_cuda

            check = iree_cuda._check

            def record_launch(lib, name, status):
                check(lib, name, status)
                if name == "cuLaunchKernel":
                    executions.append(name)

            monkeypatch.setattr(iree_cuda, "_check", record_launch)
        else:
            from kashi.iree_runtime import _SegmentTarget

            invoke = _SegmentTarget.__call__

            def record_segment(target, *args):
                result = invoke(target, *args)
                executions.append(target.name)
                return result

            monkeypatch.setattr(_SegmentTarget, "__call__", record_segment)

        def compile_iree(gm, inputs):
            result = compile_iree_fx(gm, output_dir=tmp_path)
            assert (
                result.execution_mode == mode
                and result.ptx_files
                and result.vmfb_path.is_file()
            )

            def run(*args):
                before = len(executions)
                outputs = result.graph(*args)
                assert len(executions) > before
                return outputs

            return run

        monkeypatch.setattr(backend_spy, "compile_fx", compile_iree)

    # IREE accumulates BF16 linear operations in FP32. Test both PyTorch's
    # default reduced-precision reduction and full FP32 accumulation; restore
    # the process-wide setting even when compilation or inference fails.
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        reduced_precision
    )
    try:
        _compare(
            model_dir,
            CompilationConfig(
                mode=CompilationMode.DYNAMO_TRACE_ONCE,
                backend=VLLM_BACKEND,
                cudagraph_mode=CUDAGraphMode.NONE,
            ),
            expect_fp8=True,
            # FP8 can amplify BF16 rounding across layers. Use one BF16 epsilon
            # for default reduced-precision IREE comparisons; the full-FP32 and
            # FX round-trip comparisons retain the 0.6B test's strict tolerance.
            rtol=torch.finfo(torch.bfloat16).eps
            if reduced_precision and mode != "fx"
            else 1e-4,
        )
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


@pytest.mark.parametrize("qwen38_model_dir", ["bf16", "fp8"], indirect=True)
@pytest.mark.skipif(
    bool(torch.version.cuda or torch.version.hip), reason="CPU vLLM wheel required"
)
def test_qwen38_cpu_matches_uncompiled(qwen38_model_dir, request):
    """Run BF16 and, on supported CPUs, FP8 two-layer model parity."""
    import psutil
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    variant = request.node.callspec.params["qwen38_model_dir"]
    if variant == "fp8":
        from vllm import _custom_ops as ops

        has_amx = getattr(torch.cpu, "_is_amx_tile_supported", lambda: False)()
        if not has_amx or not getattr(ops, "_supports_cpu_fp8_w8a16", False):
            pytest.skip(
                "vLLM block-FP8 CPU kernels require AMX and the compiled FP8 op"
            )

    available = psutil.virtual_memory().available
    limit = Path("/sys/fs/cgroup/memory.max")
    usage = Path("/sys/fs/cgroup/memory.current")
    if limit.is_file() and usage.is_file():
        raw = limit.read_text().strip()
        if raw != "max":
            available = min(available, int(raw) - int(usage.read_text()))
    if available < _MIN_CPU_BYTES:
        pytest.skip(
            "two-layer Qwen3.8 BF16 requires at least 10 GiB of available CPU memory"
        )
    _compare(
        qwen38_model_dir,
        CompilationConfig(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            backend=VLLM_BACKEND,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        expect_fp8=variant == "fp8",
    )
