"""Exercise real Qwen3 prefill and decode through the kashi vLLM backend."""

import gc
import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm_backend_spy


# A local config and token IDs avoid downloading a model or tokenizer. A head
# size of 64 is supported by both vLLM's CPU and GPU attention backends.
CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 128,
    "head_dim": 64,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "num_hidden_layers": 1,
    "intermediate_size": 256,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "torch_dtype": "float32",
}
PROMPTS = [
    {"prompt_token_ids": [1, 2, 3, 4, 5]},
    {"prompt_token_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18]},
]
MAX_TOKENS = 4


def _fill_weights(model):
    """Replace dummy weights with deterministic, distinguishable model weights."""
    # Dummy-load values are too small to make attention and logits distinguishable.
    # Use the same CPU-generated values for both engines, including on GPU.
    generator = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            values = torch.randn(param.shape, generator=generator)
            if "norm.weight" in name:
                values = 1.0 + 0.1 * values
            else:
                values = 0.15 * values
            param.copy_(values)


def _generate(model_dir, compilation_config, *, record_generation=False):
    """Generate token completions with the requested vLLM compilation config.

    When ``record_generation`` is true, discard startup calls from the backend
    spy so its token counts describe only the requests being generated.
    """
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    llm = None
    try:
        llm = LLM(
            model=str(model_dir),
            skip_tokenizer_init=True,
            load_format="dummy",
            dtype="float32",
            max_model_len=64,
            max_num_batched_tokens=64,
            max_num_seqs=2,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            async_scheduling=False,
            kv_cache_memory_bytes=64 * 1024 * 1024,
            seed=42,
            compilation_config=compilation_config,
            # Greedy decoding does not need the CPU sampler's Triton JIT warmup.
            kernel_config={"enable_jit_warmup": False},
        )
        llm.apply_model(_fill_weights)
        if record_generation:
            # Exclude vLLM's startup/profile calls: these counts must come from
            # the actual requests, not warmup of the compiled graph.
            vllm_backend_spy.TOKEN_COUNTS.clear()
        outputs = llm.generate(
            PROMPTS,
            SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=0,
                seed=42,
                logprobs=5,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        return [output.outputs[0] for output in outputs]
    finally:
        del llm
        gc.collect()
        cleanup_dist_env_and_memory()


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Provide an offline Qwen3 configuration and isolate vLLM test settings."""
    from vllm import envs
    import torch._inductor.config

    for name, value in {
        "VLLM_USE_AOT_COMPILE": "0",  # the spy returns a Python callable
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",  # keep the spy in-process
        "VLLM_DISABLE_COMPILE_CACHE": "1",  # require a fresh backend invocation
        "HF_HUB_OFFLINE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    # An inherited legacy setting otherwise overrides the bounded cache size.
    monkeypatch.delenv("VLLM_CPU_KVCACHE_SPACE", raising=False)
    # A previous in-process engine may have cached different environment values.
    envs.disable_envs_cache()
    # PyTorch's config module needs its own patch context: pytest's setattr
    # undo deletes the proxied setting and breaks later in-process tests.
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    vllm_backend_spy.CALLS.clear()
    vllm_backend_spy.TOKEN_COUNTS.clear()
    with torch._inductor.config.patch(cpp_cache_precompile_headers=False):
        yield tmp_path
    vllm_backend_spy.CALLS.clear()
    vllm_backend_spy.TOKEN_COUNTS.clear()


def test_qwen3_generation_matches_uncompiled_vllm(model_dir):
    """Compare compiled and eager Qwen3 token IDs and selected log-probabilities."""
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    compiled = _generate(
        model_dir,
        CompilationConfig(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            backend="vllm_backend_spy.spy_compile_fx",
            # Replaying a CUDA graph would bypass the Python spy on decode.
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        record_generation=True,
    )
    reference = _generate(
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

    assert len(compiled) == len(reference) == len(PROMPTS)
    for got, want in zip(compiled, reference, strict=True):
        assert len(want.token_ids) == len(got.token_ids) == MAX_TOKENS
        assert list(got.token_ids) == list(want.token_ids)
        assert got.logprobs is not None and want.logprobs is not None
        assert len(got.logprobs) == len(want.logprobs) == MAX_TOKENS
        for token, got_step, want_step in zip(
            want.token_ids, got.logprobs, want.logprobs, strict=True
        ):
            got_logprob = got_step[token].logprob
            want_logprob = want_step[token].logprob
            assert math.isfinite(got_logprob) and math.isfinite(want_logprob)
            torch.testing.assert_close(
                torch.tensor(got_logprob),
                torch.tensor(want_logprob),
                rtol=1e-4,
                atol=1e-5,
            )
            top = sorted((lp.logprob for lp in want_step.values()), reverse=True)
            assert len(top) >= 2 and top[0] - top[1] > 1e-4, (
                "near tie: token comparison cannot discriminate"
            )
