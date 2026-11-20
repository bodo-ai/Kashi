# Qwen3 backend output-parity check

## Intent

Add a controlled end-to-end check that the normalized-only backend and the FX→MLIR→FX round-trip backend produce equivalent Qwen3 0.6B generation outputs without checkpoint weights.

## Goals and non-goals

- Add a vLLM backend that applies the same FX and `torch.Size` normalization as `mlir_roundtrip_backend`, then returns the normalized FX forward function.
- Run each backend in a separate, sequential process with a fresh vLLM engine and KV cache.
- Use the same Qwen3 model/config, explicit model seed, prompt, and greedy sampling settings for both runs.
- Compare generated token IDs exactly and each generated token's log-probability within absolute tolerance `1e-3`.
- This is an end-to-end parity check, not exact equality of FX nodes or every intermediate activation.
- Keep dummy weight loading; model metadata may still be resolved.

## Approaches

1. **Separate sequential processes — selected.** Each run receives the same model seed/config/prompt and starts with an empty KV cache. Process isolation avoids shared vLLM distributed state, and sequential execution limits GPU memory use. Compare token IDs and selected-token log-probabilities.
2. **Two LLM instances in one process.** Rejected: vLLM distributed state and CUDA allocations are process-global enough to make isolation less reliable.
3. **Run both FX graphs inside one backend invocation.** Rejected: Qwen3 custom operators mutate KV-cache state; cloning/restoring that hidden state depends on private vLLM internals and is a larger, less reliable change.

## Behavior and constraints

Both runs use `Qwen/Qwen3-0.6B`, `load_format="dummy"`, `LLM(seed=42)`, the same prompt (`"Hello, my name is"`), and `SamplingParams(max_tokens=4, temperature=0, seed=42, logprobs=5)`. The comparison worker selects either `normalized_fx_backend` or `mlir_roundtrip_backend`; the harness runs them one at a time and fails on token-ID differences or selected-token log-probability differences outside tolerance. GPU selection remains an environment setting.

## Verification

Run the comparison harness in the provided vLLM/PyTorch environment on one selected GPU. Confirm the normalized-only and MLIR round-trip runs report dummy weight loading, produce identical token IDs, and have selected-token log-probabilities within `1e-3` absolute tolerance.
