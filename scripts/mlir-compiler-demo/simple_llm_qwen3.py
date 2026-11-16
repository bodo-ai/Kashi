import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

import torch
import simple_llm as compiler
from vllm import LLM, SamplingParams


def create_llm() -> LLM:
    return LLM(
        model="Qwen/Qwen3-0.6B",
        load_format="dummy",
        max_model_len=128,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        gpu_memory_utilization=0.5,
        compilation_config={
            "mode": 2,
            "backend": compiler.mlir_roundtrip_backend.__name__,
        },
    )


def main():
    torch.manual_seed(42)
    qwen3 = create_llm()
    outputs = qwen3.generate(
        ["Hello, my name is"],
        SamplingParams(max_tokens=4),
    )
    print(outputs[0].outputs[0].token_ids)


if __name__ == "__main__":
    main()
