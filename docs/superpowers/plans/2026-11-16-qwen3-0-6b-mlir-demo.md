# Qwen3 0.6B MLIR Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a no-checkpoint-weights Qwen3 0.6B MLIR demo and make the existing FX↔MLIR converter handle the complete Qwen3 compile graph.

**Architecture:** Extend the existing FX normalization and MLIR-to-FX reconstruction only for operation kinds observed in the Qwen3 graph. Add a sibling script using vLLM's built-in `Qwen3ForCausalLM` and `load_format="dummy"`, reusing the registered round-trip backend.

**Tech Stack:** Python 3.12, PyTorch, vLLM, torch-mlir, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-16-qwen3-0-6b-mlir-demo-design.md`

## Global Constraints

- Model identifier: `Qwen/Qwen3-0.6B`.
- Use vLLM `load_format="dummy"`; do not load checkpoint weights.
- Support the actual Qwen3 graph's missing operation kinds in both FX→MLIR and MLIR→FX.
- Preserve the existing OPT-based `simple_llm.py` demo.
- Model config and tokenizer metadata may still be resolved separately from weights.
- Use conservative `gpu_memory_utilization=0.5` in the local GPU smoke test.

## Review Focus

- `Tensor.long()` must preserve int64 dtype across the round trip — `test_qwen3_tensor_ops_round_trip`.
- `split()`'s multi-result outputs must preserve order and shapes — `test_qwen3_tensor_ops_round_trip`.
- Shape construction and tuple access in the observed graph must survive normalization — `test_normalize_qwen3_fx_ops`.
- vLLM custom operators must preserve their target and result arity in both directions — `test_qwen3_custom_ops_round_trip` and full Qwen3 smoke.
- The Qwen3 entry point must explicitly request dummy weights — `test_qwen3_demo_uses_dummy_weights`.
- Existing OPT conversion must still work — run `simple_llm.py` as a regression smoke test.

---

### Task 1: Normalize Qwen3 FX shapes and methods

**Files:**
- Modify: `scripts/mlir-compiler-demo/simple_llm.py`
- Create: `scripts/mlir-compiler-demo/test_simple_llm.py`

**Interfaces:**
- Preserve `normalize_fx(gm)` and `normalize_torch_size(gm)` as the FX normalization entry points.
- Tests must exercise real FX graph nodes, not source text.

- [ ] **Step 1: Write `test_normalize_qwen3_fx_ops`**

Build an FX graph with the observed `long`, `size`, `split`, `view`, `torch.Size`, and `operator.getitem` patterns. Assert the normalized graph uses the intended ATen operations and preserves argument order.

- [ ] **Step 2: Run the normalization test and verify the expected assertion failure**

Run: `python -m unittest discover -s scripts/mlir-compiler-demo -p 'test_*.py' -v`
Expected: an assertion shows `long`/`split` (and any other unsupported observed method) remains unnormalized; current full-model trace fails on `Tensor.long()`.

- [ ] **Step 3: Implement only the observed FX normalizations**

Normalize Qwen3's tensor methods and shape/tuple nodes to the appropriate ATen representation. Keep existing normalizations and explicit errors; do not broaden to unrelated model operations.

- [ ] **Step 4: Run `test_normalize_qwen3_fx_ops`**

Run the unittest command above. Expected: the normalized-target and argument assertions pass.

### Task 2: Reconstruct Qwen3 MLIR operations back to FX

**Files:**
- Modify: `scripts/mlir-compiler-demo/simple_llm.py`
- Modify: `scripts/mlir-compiler-demo/test_simple_llm.py`

**Interfaces:**
- Preserve `mlir_to_fx(module) -> fx.GraphModule` and test through `stateless_fx_import` with `VllmFxImporter`.

- [ ] **Step 1: Write `test_qwen3_tensor_ops_round_trip` and `test_qwen3_custom_ops_round_trip`**

Round-trip the normalized tensor-method graph and a graph containing the observed RMSNorm, fused add+RMSNorm, rotary embedding, SiLU-and-multiply, unified attention, and KV-cache operators. Assert tensor values/dtypes/shapes, split ordering, custom targets, and result arity. Convert expected unsupported-operation exceptions into test assertion failures with the offending op name.

- [ ] **Step 2: Run the focused tests and verify the expected assertion failures**

Run: `python -m unittest discover -s scripts/mlir-compiler-demo -p 'test_*.py' -v`
Expected: the test captures the first unsupported Qwen3 operation from FX import or `mlir_to_fx` as an assertion failure.

- [ ] **Step 3: Implement reconstruction for the observed ATen and vLLM operations**

Resolve ATen namespace/overload names from registered PyTorch operators and schemas; reconstruct dtype constants, keyword-only arguments, and tuple/list results. Add explicit reverse mappings for the observed vLLM custom operators and preserve unsupported-operation errors.

- [ ] **Step 4: Run all converter tests**

Run the unittest command above. Expected: normalization and bidirectional round-trip tests pass for each observed operation family.

### Task 3: Add the Qwen3 no-weights demo

**Files:**
- Create: `scripts/mlir-compiler-demo/simple_llm_qwen3.py`
- Modify: `scripts/mlir-compiler-demo/README.md`
- Modify: `scripts/mlir-compiler-demo/test_simple_llm.py`

**Interfaces:**
- Add `create_llm() -> LLM` to construct the model engine; `main()` generates with the existing short prompt and the registered `mlir_roundtrip_backend`.

- [ ] **Step 1: Write `test_qwen3_demo_uses_dummy_weights`**

Assert the sibling script is discoverable before importing it, then patch the expensive `LLM` constructor and assert `create_llm()` requests `Qwen/Qwen3-0.6B`, `load_format="dummy"`, and backend `mlir_roundtrip_backend`.

- [ ] **Step 2: Run the test and verify the expected assertion failure**

Run: `python -m unittest discover -s scripts/mlir-compiler-demo -p 'test_*.py' -v`
Expected: an assertion reports that the Qwen3 sibling script is missing, rather than a test import error.

- [ ] **Step 3: Implement the sibling script and document the run command**

Import `mlir_roundtrip_backend` from sibling `simple_llm.py` to register the backend; instantiate the native Qwen3 model with dummy loading, max length 128, 256 MiB KV cache, compile mode 2, and GPU utilization 0.5. Update the README with the command and note that config/tokenizer metadata may still be resolved.

- [ ] **Step 4: Run unit tests and the existing OPT demo**

Run the unittest command above, then run `simple_llm.py` with the provided environment. Expected: converter tests pass and the original demo still generates token IDs.

- [ ] **Step 5: Run the full Qwen3 smoke test**

Use `/home/taanders/uv/torchvllm_local/torchvllm_local/bin/python` with `/home/taanders/uv/torchvllm_local/env.sh`, `CUDA_VISIBLE_DEVICES=1`, and the script's 0.5 GPU utilization. Run `scripts/mlir-compiler-demo/simple_llm_qwen3.py`.

Expected: startup reports `load_format=dummy`; the complete Qwen3 FX graph converts to MLIR and back; generation prints token IDs. No checkpoint weight files are loaded.
