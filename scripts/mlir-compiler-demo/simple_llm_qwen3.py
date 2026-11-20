import argparse
import json
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

import torch
import simple_llm as compiler
from vllm import LLM, SamplingParams


def create_llm(backend: str = "mlir_roundtrip_backend") -> LLM:
    """Create a bounded Qwen3 instance using the selected compilation backend.

    Dummy weights and fixed seeds make this a repeatable compiler-parity check,
    rather than a comparison of downloaded model weights or stochastic outputs.
    """
    return LLM(
        model="Qwen/Qwen3-0.6B",
        load_format="dummy",
        seed=42,
        max_model_len=128,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        gpu_memory_utilization=0.5,
        compilation_config={
            "mode": 2,
            "backend": backend,
        },
    )


def run_generation(backend: str) -> dict[str, list[int] | list[float]]:
    """Generate a short deterministic sample and extract each chosen token's score.

    vLLM stores log-probabilities by token ID; requiring the generated ID in each
    entry ensures the returned scores correspond to the emitted sequence.
    """
    qwen3 = create_llm(backend)
    outputs = qwen3.generate(
        ["Hello, my name is"],
        SamplingParams(
            max_tokens=4,
            temperature=0,
            seed=42,
            logprobs=5,
        ),
    )
    # Check alignment before pairing each generated token with its own score.
    completion = outputs[0].outputs[0]
    if completion.logprobs is None:
        raise RuntimeError("vLLM did not return token log-probabilities")
    if len(completion.logprobs) != len(completion.token_ids):
        raise RuntimeError("vLLM returned misaligned token log-probabilities")

    selected_token_logprobs = []
    for token_id, token_logprobs in zip(
        completion.token_ids,
        completion.logprobs,
    ):
        token_logprob = token_logprobs.get(token_id)
        if token_logprob is None:
            raise RuntimeError(f"Missing log-probability for generated token {token_id}")
        selected_token_logprobs.append(float(token_logprob.logprob))

    return {
        "token_ids": list(completion.token_ids),
        "selected_token_logprobs": selected_token_logprobs,
    }


def main(argv=None):
    """Run generation for a selected backend, optionally emitting parity JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("mlir_roundtrip_backend", "normalized_fx_backend"),
        default="mlir_roundtrip_backend",
    )
    parser.add_argument("--json-result", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(42)
    result = run_generation(args.backend)
    if args.json_result:
        print("QWEN3_PARITY_RESULT=" + json.dumps(result, sort_keys=True))
    else:
        print(result["token_ids"])


if __name__ == "__main__":
    main()


