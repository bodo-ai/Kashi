"""Shared setup, generation and parity checks for the vLLM integration tests."""

import gc
import json
import math
from contextlib import contextmanager
from functools import partial

import pytest
import torch
import vllm_backend_spy


# A local config and token IDs avoid downloading a model or tokenizer. A head
# size of 64 is supported by both vLLM's CPU and GPU attention backends.
QWEN_CONFIG = {
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
QWEN_PROMPTS = [
    {"prompt_token_ids": [1, 2, 3, 4, 5]},
    {"prompt_token_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18]},
]
QWEN_MAX_TOKENS = 4


@contextmanager
def vllm_test_environment(monkeypatch, *, cpu_cache_space, offline=False, clear_after=False):
    """Isolate vLLM settings and the backend spy, restoring Inductor safely."""
    import torch._inductor.config

    for name, value in {
        "VLLM_USE_AOT_COMPILE": "0",  # the spy returns a Python callable
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",  # keep the spy in-process
        "VLLM_DISABLE_COMPILE_CACHE": "1",  # require a fresh backend invocation
    }.items():
        monkeypatch.setenv(name, value)
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    if cpu_cache_space is None:
        # An inherited legacy setting otherwise overrides the bounded cache size.
        monkeypatch.delenv("VLLM_CPU_KVCACHE_SPACE", raising=False)
        # A previous in-process engine may have cached different environment values.
        from vllm import envs

        envs.disable_envs_cache()
    else:
        monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", cpu_cache_space)

    vllm_backend_spy.CALLS.clear()
    vllm_backend_spy.TOKEN_COUNTS.clear()
    # PyTorch's config module needs its own patch context: pytest's setattr
    # undo deletes the proxied setting and breaks later in-process tests.
    # vLLM's CPU sampler's precompiled-header build can fail with macOS clang.
    try:
        with torch._inductor.config.patch(cpp_cache_precompile_headers=False):
            yield
    finally:
        if clear_after:
            vllm_backend_spy.CALLS.clear()
            vllm_backend_spy.TOKEN_COUNTS.clear()


def fill_weights(model, *, seed, scale, norm_scale=None):
    """Fill sorted parameters with deterministic CPU values, optionally scaling norms around one."""
    # Dummy-load values are too small to make attention and logits distinguishable.
    # Use the same CPU-generated values for both engines, including on GPU.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            values = torch.randn(param.shape, generator=generator)
            if norm_scale is not None and "norm.weight" in name:
                values = 1.0 + norm_scale * values
            else:
                values = scale * values
            param.copy_(values)


def generate(
    model_dir,
    compilation_config,
    prompts,
    *,
    weight_seed,
    weight_scale,
    norm_scale=None,
    engine_options,
    sampling_options,
    generation_options=None,
    record_generation=False,
):
    """Generate completions from a seeded dummy model and always clean up vLLM."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    llm = None
    try:
        llm = LLM(
            model=str(model_dir),
            skip_tokenizer_init=True,
            load_format="dummy",
            dtype="float32",
            kv_cache_memory_bytes=64 * 1024 * 1024,
            compilation_config=compilation_config,
            # The startup warmup compiles a random sampler that can abort in Triton
            # on Linux CPU. These tests exercise greedy decoding, not sampler warmup.
            kernel_config={"enable_jit_warmup": False},
            **engine_options,
        )
        llm.apply_model(
            partial(fill_weights, seed=weight_seed, scale=weight_scale, norm_scale=norm_scale)
        )
        if record_generation:
            # Exclude vLLM's startup/profile calls: these counts must come from
            # the actual requests, not warmup of the compiled graph.
            vllm_backend_spy.TOKEN_COUNTS.clear()
        outputs = llm.generate(
            prompts, SamplingParams(**sampling_options), **(generation_options or {})
        )
        return [output.outputs[0] for output in outputs]
    finally:
        del llm
        gc.collect()
        cleanup_dist_env_and_memory()


def generate_qwen(model_dir, compilation_config, *, record_generation=False):
    """Run the offline Qwen configuration used by both vLLM and IREE tests."""
    return generate(
        model_dir,
        compilation_config,
        QWEN_PROMPTS,
        weight_seed=42,
        weight_scale=0.15,
        norm_scale=0.1,
        engine_options={
            "max_model_len": 64,
            "max_num_batched_tokens": 64,
            "max_num_seqs": 2,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "seed": 42,
        },
        sampling_options={
            "max_tokens": QWEN_MAX_TOKENS,
            "temperature": 0,
            "seed": 42,
            "logprobs": 5,
            "ignore_eos": True,
        },
        generation_options={"use_tqdm": False},
        record_generation=record_generation,
    )


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Provide an offline Qwen3 configuration and isolate vLLM test settings."""
    with vllm_test_environment(monkeypatch, cpu_cache_space=None, offline=True, clear_after=True):
        (tmp_path / "config.json").write_text(json.dumps(QWEN_CONFIG))
        yield tmp_path


def assert_outputs_match(
    actual,
    expected,
    *,
    num_prompts=None,
    max_tokens=None,
    check_logprob_length=False,
    check_finite=False,
    rtol=None,
    atol=None,
    near_tie_message=None,
    require_two_logprobs=False,
):
    """Compare token IDs and selected log-probabilities with optional model-specific checks."""
    if num_prompts is not None:
        assert len(actual) == len(expected) == num_prompts
    close_options = {}
    if rtol is not None:
        close_options["rtol"] = rtol
    if atol is not None:
        close_options["atol"] = atol
    for got, want in zip(actual, expected, strict=True):
        if max_tokens is not None:
            assert len(got.token_ids) == len(want.token_ids) == max_tokens
        assert list(got.token_ids) == list(want.token_ids)
        if check_logprob_length:
            assert got.logprobs is not None and want.logprobs is not None
            assert len(got.logprobs) == len(want.logprobs) == max_tokens
        for token, got_step, want_step in zip(
            want.token_ids, got.logprobs, want.logprobs, strict=True
        ):
            got_logprob = got_step[token].logprob
            want_logprob = want_step[token].logprob
            if check_finite:
                assert math.isfinite(got_logprob) and math.isfinite(want_logprob)
            torch.testing.assert_close(
                torch.tensor(got_logprob), torch.tensor(want_logprob), **close_options
            )
            if near_tie_message is not None:
                top = sorted((lp.logprob for lp in want_step.values()), reverse=True)
                if require_two_logprobs:
                    assert len(top) >= 2 and top[0] - top[1] > 1e-4, near_tie_message
                else:
                    assert top[0] - top[1] > 1e-4, near_tie_message
