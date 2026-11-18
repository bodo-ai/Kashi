"""End-to-end: SimpleVllm through the real vLLM engine with and without the
FX -> MLIR -> FX backend, using identical seeded weights."""

import gc
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm_backend_spy

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


def _fill_weights(model):
    generator = torch.Generator().manual_seed(WEIGHT_SEED)
    with torch.no_grad():
        for _, param in sorted(model.named_parameters()):
            param.copy_(torch.randn(param.shape, generator=generator) * WEIGHT_SCALE)


def _generate(model_dir, compilation_config):
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    llm = LLM(
        model=str(model_dir),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="float32",
        max_model_len=128,
        seed=0,
        compilation_config=compilation_config,
    )
    try:
        llm.apply_model(_fill_weights)
        outputs = llm.generate(
            PROMPTS, SamplingParams(max_tokens=4, temperature=0, logprobs=5)
        )
        return [o.outputs[0] for o in outputs]
    finally:
        del llm
        gc.collect()
        cleanup_dist_env_and_memory()


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    for name, value in {
        "VLLM_USE_AOT_COMPILE": "0",  # needs a serializable backend result
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",  # keep the spy in-process
        "VLLM_CPU_KVCACHE_SPACE": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",  # a cached graph would skip the backend
    }.items():
        monkeypatch.setenv(name, value)
    # vLLM's sampler compiles a logprob helper with Inductor on CPU; Inductor's
    # precompiled-header build fails with this macOS clang. Unrelated to kashi.
    import torch._inductor.config

    monkeypatch.setattr(torch._inductor.config, "cpp_cache_precompile_headers", False)

    sys.path.insert(0, str(Path(__file__).parent / "models"))
    try:
        from simple_vllm_model import SimpleVllmForCausalLM
    finally:
        sys.path.pop(0)
    from vllm import ModelRegistry

    ModelRegistry.register_model("SimpleVllmForCausalLM", SimpleVllmForCausalLM)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    vllm_backend_spy.CALLS.clear()
    vllm_backend_spy.TOKEN_COUNTS.clear()
    return tmp_path


def _vllm_targets(gm):
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

    compiled = _generate(
        model_dir,
        CompilationConfig(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            backend="vllm_backend_spy.spy_compile_fx",
        ),
    )
    reference = _generate(model_dir, CompilationConfig(mode=CompilationMode.NONE))

    # One capture serves batched prefill and decode, and is really rebuilt.
    assert len(vllm_backend_spy.CALLS) == 1
    gm, fn = vllm_backend_spy.CALLS[0]
    assert fn is not gm.forward
    # ...and the rebuilt callable actually ran for batched prefill (5 + 9
    # tokens) and for decode (one token per prompt).
    assert {14, 2} <= set(vllm_backend_spy.TOKEN_COUNTS)

    vllm_ops = torch.ops.vllm
    assert _vllm_targets(gm) == [
        vllm_ops.unified_kv_cache_update,
        vllm_ops.unified_attention_with_output,
    ]
    assert _vllm_targets(fn.__self__) == [
        vllm_ops.unified_kv_cache_update.default,
        vllm_ops.unified_attention_with_output.default,
    ]

    for got, want in zip(compiled, reference, strict=True):
        assert list(got.token_ids) == list(want.token_ids)
        for token, got_step, want_step in zip(
            want.token_ids, got.logprobs, want.logprobs, strict=True
        ):
            torch.testing.assert_close(
                torch.tensor(got_step[token].logprob),
                torch.tensor(want_step[token].logprob),
            )
            top = sorted((lp.logprob for lp in want_step.values()), reverse=True)
            assert top[0] - top[1] > 1e-4, "near tie: comparison cannot discriminate"
