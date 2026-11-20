# Qwen3 Backend Output-Parity Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compare Qwen3 0.6B generation through the normalized FX backend and the FX→MLIR→FX backend under controlled identical initial conditions.

**Architecture:** Add a registered backend that normalizes FX and returns its forward function. Make the Qwen3 worker select either backend and emit token IDs plus generated-token log-probabilities. A harness runs the worker twice in sequential subprocesses, giving each engine an isolated process and fresh KV cache, then checks output parity.

**Tech Stack:** Python 3.12, PyTorch, vLLM, `unittest`, subprocesses.

**Spec:** `docs/superpowers/specs/2026-11-16-qwen3-output-parity-design.md`

## Global Constraints

- Model identifier: `Qwen/Qwen3-0.6B`.
- Use vLLM `load_format="dummy"`; do not load checkpoint weights.
- Both workers use `LLM(seed=42)`, the same prompt `"Hello, my name is"`, and `SamplingParams(max_tokens=4, temperature=0, seed=42, logprobs=5)`.
- Run the two workers sequentially in separate processes; each must start with a fresh KV cache.
- Require exact token-ID equality and selected-token log-probability equality within absolute tolerance `1e-3`.
- GPU selection remains caller-controlled via the environment.

## Review Focus

- The baseline backend must apply the same FX and torch.Size normalization as the round-trip backend — `test_normalized_fx_backend_matches_normalized_reference`.
- Backend selection must preserve model seed, prompt, and greedy sampling settings — `test_qwen3_worker_uses_comparison_settings`.
- A token mismatch must fail the parity check — `test_compare_qwen3_rejects_token_mismatch`.
- Log-probability drift greater than `1e-3` must fail the parity check — `test_compare_qwen3_rejects_logprob_drift`.
- Missing or misaligned per-token log-probability data must fail clearly — `test_compare_qwen3_rejects_invalid_logprobs`.

---

### Task 1: Add the normalized-only backend

**Files:**
- Modify: `scripts/mlir-compiler-demo/simple_llm.py`
- Modify: `scripts/mlir-compiler-demo/test_simple_llm.py`

**Interfaces:**
- Add registered backend `normalized_fx_backend(gm: fx.GraphModule, example_inputs) -> Callable`.
- It applies `normalize_fx` then `normalize_torch_size` and returns `gm.forward`.

- [ ] **Step 1: Write `test_normalized_fx_backend_matches_normalized_reference`**

Use the existing small tensor-method graph and a real tensor input. Assert the backend result returns the same tensor values, dtype, shape, and detached status as the normalized reference graph.

- [ ] **Step 2: Run the test and verify the expected failure**

Run: `python -m unittest discover -s scripts/mlir-compiler-demo -p 'test_*.py' -v`
Expected: an assertion reports that `normalized_fx_backend` is unavailable.

- [ ] **Step 3: Implement and register `normalized_fx_backend`**

Reuse the existing two normalizers; do not add a second normalization implementation.

- [ ] **Step 4: Run the focused unit suite**

Run the unittest command above. Expected: the normalized-only backend behavior test passes.

### Task 2: Make the Qwen3 worker deterministic and selectable

**Files:**
- Modify: `scripts/mlir-compiler-demo/simple_llm_qwen3.py`
- Modify: `scripts/mlir-compiler-demo/test_simple_llm.py`

**Interfaces:**
- Change `create_llm()` to `create_llm(backend: str = "mlir_roundtrip_backend") -> LLM`.
- Add `run_generation(backend: str) -> dict[str, list[int] | list[float]]` returning `token_ids` and `selected_token_logprobs`.
- Add `--backend` and `--json-result` worker arguments; no-argument behavior remains the existing MLIR demo.

- [ ] **Step 1: Write `test_qwen3_worker_uses_comparison_settings`**

Use a specific fake `LLM` boundary result. Assert `create_llm()` requests the model, dummy loading, and seed 42; assert `run_generation()` passes the exact prompt and sampling parameters and returns the hand-specified token/log-probability result.

- [ ] **Step 2: Run the worker test and verify the expected failure**

Run: `python -m unittest discover -s scripts/mlir-compiler-demo -p 'test_*.py' -v`
Expected: an assertion reports that the worker lacks backend selection, explicit seed, or structured generation output.

- [ ] **Step 3: Implement deterministic worker options and result output**

Use the design's exact prompt, `max_tokens=4`, `temperature=0`, `seed=42`, and `logprobs=5`. In JSON-result mode emit generated token IDs and each generated token's log-probability under a unique output marker.

- [ ] **Step 4: Run the unit suite**

Run the unittest command above. Expected: the worker configuration and serialization tests pass.

### Task 3: Add the sequential parity harness

**Files:**
- Create: `scripts/mlir-compiler-demo/compare_simple_llm_qwen3.py`
- Modify: `scripts/mlir-compiler-demo/test_simple_llm.py`
- Modify: `scripts/mlir-compiler-demo/README.md`

**Interfaces:**
- Add `compare_outputs(reference: dict, candidate: dict) -> None`, raising `AssertionError` on mismatch.
- The harness runs `simple_llm_qwen3.py` once with each backend in separate sequential subprocesses, parses the marker output, and calls `compare_outputs`.

- [ ] **Step 1: Write comparator tests**

Add tests for matching outputs, token-ID mismatch, log-probability drift above `1e-3`, and missing/misaligned log-probability entries.

- [ ] **Step 2: Run tests and verify expected failures**

Run: `python -m unittest discover -s scripts/mlir-compiler-demo -p 'test_*.py' -v`
Expected: the comparator is missing and each new test fails with its specified parity condition.

- [ ] **Step 3: Implement the sequential subprocess harness**

Run the normalized-only backend first and the MLIR round-trip backend second. Pass through the environment unchanged so both use the same caller-selected GPU; separate processes guarantee independent vLLM state and fresh KV caches. Compare exact token IDs and selected-token log-probabilities with `rel_tol=0` and `abs_tol=1e-3`.

- [ ] **Step 4: Document and run the Qwen3 parity command**

Document `compare_simple_llm_qwen3.py` in the README. Run it in the supplied environment with a single selected GPU (for example `CUDA_VISIBLE_DEVICES=1`). Expected: both startup logs report dummy loading, token IDs match exactly, and all selected-token log-probabilities are within tolerance.

- [ ] **Step 5: Run the focused unit suite**

Run the unittest command above. Expected: every backend, worker, and comparator test passes.
