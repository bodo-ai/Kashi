# Qwen3 0.6B MLIR demo without checkpoint weights

## Intent

Add a sibling to `scripts/mlir-compiler-demo/simple_llm.py` that sends the complete, native vLLM Qwen3 0.6B model through the existing FX-to-MLIR round-trip backend, without loading pretrained checkpoint weights.

## Goals and non-goals

- Keep `simple_llm.py`'s existing OPT demo behavior while extending its FX normalization and MLIR-to-FX reconstruction to cover every operation in the Qwen3 compile graph that the current converter lacks.
- Add the Qwen3 sibling script using the native vLLM model and dummy weight loading.
- Do not load downloaded checkpoint weights or reimplement Qwen3 layers.
- Configuration and tokenizer metadata may still be resolved separately from weights.

## Approaches

1. **Extend the existing two-way converter for the operations observed in Qwen3 — selected.** Keeps the MLIR round trip real, makes conversion gaps explicit, and avoids unrelated converter generalization.
2. **Build a generic converter for arbitrary ATen operations.** Rejected: broader than this model requires and increases the converter's unverified surface.
3. **Avoid unsupported model operations or bypass the round trip.** Rejected: would no longer exercise the full Qwen3 path requested.

## Observed compatibility gap

An initial run with dummy weights resolved `Qwen3ForCausalLM` and traced the model, then failed in the forward FX-to-MLIR importer because `Tensor.long()` was not normalized. The traced graph also contains `size`, `split`, and vLLM operations for RMSNorm, fused add+RMSNorm, rotary embedding, SiLU-and-multiply, attention, and KV-cache update. The reverse importer currently handles only a small subset of ATen operations and the two attention/cache custom operators. Extend both directions for the operation kinds actually exercised by this Qwen3 graph, including any additional gaps uncovered by the round-trip run.

## Behavior and constraints

The new script will select `Qwen/Qwen3-0.6B`, enable vLLM's dummy weight loader, and use the registered `mlir_roundtrip_backend`. It will retain the short prompt and generation shape of the existing demo. The converter changes must preserve the existing simple demo and round-trip semantics; they are scoped to FX/MLIR operation kinds needed by the traced Qwen3 graph, not arbitrary future models.

## Verification

Add focused coverage for the newly supported FX/MLIR operation kinds in both directions. Then run the Qwen3 script in `/home/taanders/uv/torchvllm_local/`'s environment and confirm the complete model graph round-trips and generates token IDs without loading checkpoint weights. Account for the GPU memory limit observed in the initial experiment by keeping the vLLM memory utilization conservative.
