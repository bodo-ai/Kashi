# IREE Kernel Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute IREE-generated CUDA kernels from the reconstructed FX graph, using either direct CUDA launches or the IREE runtime, and validate both paths against uncompiled execution.

**Architecture:** Keep `import_fx`, partitioning, and driver reconstruction as the frontend. Extend flow reconstruction with executable dispatch targets; implement a CUDA-driver launcher and a VMFB runtime bridge behind the same driver graph. Compose them in `compile_iree_fx`, then expose a `torch.compile`/vLLM backend that owns its temporary artifacts.

**Tech Stack:** Python 3.12 and Pixi, PyTorch 2.14, IREE compiler/runtime 3.14, vLLM 0.33, pytest, CUDA driver API, and DLPack.

**Spec:** `specs/iree-kernel-execution.md`

## Global Constraints

- Support `KASHI_IREE_EXECUTION=cuda` (default) and `iree`; reject every other value. The existing PyTorch-only `compile_fx` remains unchanged.
- Use IREE as an external dependency; do not fork or vendor it. Compiled segment execution is CUDA-only.
- Preserve driver argument order, output pytree, custom-operator ordering, and data dependencies. Never substitute the original graph or a no-op kernel on failure.
- Reject unsupported ABI forms, layouts, devices, and missing metadata rather than guessing. Maintain the caller's CUDA stream dependencies and tensor storage lifetimes.
- A CPU-only run may validate parsers, partitioning, and compilation artifacts; it cannot establish that CUDA execution works. GPU tests must be reported as skipped when CUDA is unavailable.
- General symbolic-shape reuse, expanded alias/mutation semantics, unsigned-dtype support, CUDA graph replay, and CI coverage-policy changes are not requirements of this plan.

## File Structure

- Modify `python/src/kashi/iree_reassembly.py`: attach executable targets to flow dispatches while retaining inspection-only behavior.
- Create `python/src/kashi/iree_cuda.py`: interpret IREE's launch metadata and execute PTX through the CUDA driver.
- Create `python/src/kashi/iree_runtime.py`: bind VMFB segments to IREE's CUDA runtime and safely exchange tensors with PyTorch.
- Modify `python/src/kashi/iree_backend.py`: emit artifacts, select the execution mode, rebuild the driver, and expose the owning backend callable.
- Extend `python/tests/test_iree_reassembly.py` and `python/tests/test_iree_backend.py`; create `python/tests/test_iree_cuda.py`, `python/tests/test_iree_runtime.py`, `python/tests/test_iree_execution.py`, and `python/tests/test_iree_qwen3_integration.py`.
- Update `README.md` with mode selection, prerequisites, artifact inspection, and test commands.

## Review Focus

- A missing or malformed flow/stream/target dump, PTX file, or export must fail before an incorrect kernel can launch. **Task 2.**
- Mixed-device, non-contiguous, misaligned, or inconsistent tensor bindings must fail instead of passing incorrect pointers to CUDA. **Task 2.**
- An IREE result or imported tensor must not outlive its storage or race a producer/consumer on another stream. **Task 3.**
- An invalid mode or compiler failure must not return the original FX graph or leak an owned temporary workspace. **Task 4.**
- A Qwen parity check must exercise actual compiled requests and distinguish a genuine match from non-finite or near-tied outputs. **Task 5.**

---

### Task 1: Make flow dispatches executable

**Files:**
- Modify: `python/src/kashi/iree_reassembly.py`
- Test: `python/tests/test_iree_reassembly.py`

**Interfaces:**
- `IreeKernel(executable, export, executor=None)` remains an FX-callable target. When present, `executor(*args, **kwargs)` performs the dispatch.
- `flow_segment_handlers(flow_ir: str, *, device: str | None, kernel_factory=None)` returns segment-name → `(graph, args) -> list` handlers. `kernel_factory(executable, export)` returns an executor.

- [ ] **Step 1: Add CPU-side dispatch tests.** Verify that an attached executor receives the flow operands and workload and returns its computed result; a target without an executor must raise. For example, using a real PyTorch computation as the test executor:

  ```python
  def test_kernel_executor_runs_and_inspection_only_rejects():
      kernel = IreeKernel("exe", "entry", executor=lambda x, **kw: x + 2)
      torch.testing.assert_close(kernel(torch.tensor([1]), workload=[]), torch.tensor([3]))
      with pytest.raises(NotImplementedError, match="inspection-only"):
          IreeKernel("exe", "entry")(torch.tensor([1]), workload=[])
  ```

  Extend the flow fixture to check operand ordering, result unpacking, and the workload/result-shape/dtype metadata passed to a supplied factory. Keep the existing inspection-only and unsupported-flow checks.

- [ ] **Step 2: Run the tests and confirm the new executable-target test fails.**

  ```sh
  pixi run --locked test-python -q python/tests/test_iree_reassembly.py
  ```

- [ ] **Step 3: Implement the factory and executor contract.** Share one target per `(executable, export)` in a parsed flow module; pass workload, argument dimensions, result shapes, dtypes, and tied operands to executable targets. Preserve no-executor inspection behavior and reject unsupported dispatch forms.

- [ ] **Step 4: Re-run the command from Step 2.** Expect all reassembly tests to pass without requiring a GPU.

### Task 2: Launch PTX through the CUDA driver

**Files:**
- Create: `python/src/kashi/iree_cuda.py`
- Create: `python/tests/test_iree_cuda.py`

**Interfaces:**
- `create_cuda_kernel(executable: str, export: str, *, artifact_dir: Path)` returns an FX-compatible callable accepting flow arguments and the metadata supplied in Task 1.
- `_grid(count_args, body, maps, workload, export)` evaluates the supported IREE workgroup-count expressions and returns three launch dimensions.

- [ ] **Step 1: Add launch and ABI tests.** Use representative flow, stream, executable-target, and PTX fixtures. On CUDA, launch a kernel on a non-default PyTorch stream and compare its output with the expected tensor, including the independent-output-offset case. On CPU, cover arithmetic and affine grid expressions, rounding, and invalid divisors. Add a missing-artifact regression:

  ```python
  def test_missing_flow_phase_is_rejected(tmp_path):
      with pytest.raises(RuntimeError, match="flow phase"):
          create_cuda_kernel("missing", "entry", artifact_dir=tmp_path)
  ```

  Add deterministic rejection cases for incompatible parameter counts/types, non-contiguous or mixed-device tensor bindings, incorrect shape/dtype, and invalid alignment; none may issue a CUDA launch.

- [ ] **Step 2: Run the tests and confirm they fail before the launcher exists.**

  ```sh
  pixi run --locked test-python -q python/tests/test_iree_cuda.py
  ```

  The launch case requires a CUDA device; report it as skipped, not passed, on a CPU-only host.

- [ ] **Step 3: Implement the launcher.** Parse the unique required phase dumps and PTX entry, validate the binding/scalar ABI, derive the grid, block and shared-memory parameters, and pack scalars in the expected 32-bit words. Allocate PyTorch-owned untied outputs; reuse a tied input only when its metadata is supported and consistent. Use the current CUDA context and stream, keep buffers alive with `record_stream`, cache loaded modules/functions per context, and propagate driver errors. Do not launch for an empty grid.

- [ ] **Step 4: Re-run Step 2 on CUDA.** Expect the numerical, stream, and failure-path tests to pass; retain CPU parser coverage for hosts without CUDA.

### Task 3: Execute segments through the IREE runtime

**Files:**
- Create: `python/src/kashi/iree_runtime.py`
- Create: `python/tests/test_iree_runtime.py`

**Interfaces:**
- `runtime_segment_handlers(vmfb_path: Path, partition: PartitionResult) -> dict[str, Callable]` produces segment-name → `(graph, args) -> list` handlers for compiled segment functions.
- Each FX target invokes the matching VMFB export and returns its ordered flat results.

- [ ] **Step 1: Add runtime tests.** Compile a small real CUDA VMFB with a tensor input, an integer input, and multiple outputs. Build an FX graph from a handler and compare its tensor output with `source + 1`; check the scalar output and repeated execution. Produce the input and consume the output on a non-default stream, then release the graph/runtime references and verify the returned PyTorch tensor remains valid. Add rejection checks for mixed CUDA devices, missing device information for a scalar-only segment, and unsupported result metadata.

  The scalar-only failure should exercise the public handler rather than a synthetic runtime:

  ```python
  def test_runtime_requires_device_for_scalar_only_segment(compiled_module):
      path, partition = compiled_module
      graph = _build_graph(runtime_segment_handlers(path, partition)["segment_1"], 1)
      with pytest.raises(RuntimeError, match="device"):
          graph(12)
  ```

- [ ] **Step 2: Run the runtime tests and confirm the new behavior fails before implementation.**

  ```sh
  pixi run --locked test-python -q python/tests/test_iree_runtime.py -rs
  ```

  A CUDA runner is required for these cases; a skipped run is not verification.

- [ ] **Step 3: Implement the runtime bridge.** Match PyTorch and IREE CUDA devices by physical UUID, cache VM contexts/exports per device, and import supported arguments via DLPack. Synchronize before the runtime consumes PyTorch data. Export supported results into safe PyTorch-owned CUDA storage, using device-to-device copies and completion waits as needed; do not stage tensor contents through host memory. Propagate unsupported representations and runtime failures.

- [ ] **Step 4: Re-run Step 2 on CUDA.** Expect correct values, device, stream ordering, and retained-output lifetime.

### Task 4: Compose compilation, mode selection, and the public backend

**Files:**
- Modify: `python/src/kashi/iree_backend.py`
- Extend: `python/tests/test_iree_backend.py`
- Create: `python/tests/test_iree_execution.py`

**Interfaces:**
- `compile_iree_fx(gm, *, output_dir, cuda_target=None, compiler=None) -> IreeCompilation` returns the executable graph and `artifact_dir`, `flow_ir`, `ptx_files`, `kernels_ptx`, `vmfb_path`, and `execution_mode`.
- `compile_iree_backend(gm, example_inputs)` returns an executable callable owning the temporary workspace.

- [ ] **Step 1: Add composition and failure tests.** Cover default `cuda`, explicit `iree`, invalid mode, graph-device architecture selection and explicit override, missing compiler, and an empty-segment driver that does not invoke IREE. Verify a compilation failure releases a backend-owned temporary directory; verify the callable and artifacts survive repeated use. For the invalid mode:

  ```python
  def test_invalid_iree_execution_mode_is_rejected(tmp_path, monkeypatch):
      graph = torch.fx.Graph()
      x = graph.placeholder("x")
      graph.output(x)
      gm = torch.fx.GraphModule(torch.nn.Module(), graph)
      monkeypatch.setenv("KASHI_IREE_EXECUTION", "invalid")
      with pytest.raises(ValueError, match="KASHI_IREE_EXECUTION"):
          compile_iree_fx(gm, output_dir=tmp_path, cuda_target="sm_80")
  ```

  On CUDA, run both modes against eager PyTorch for a small tensor graph, a matrix multiplication, and a multi-segment graph with an existing custom-operator boundary. Assert output structure, values, and boundary ordering; ensure the generated execution path is actually invoked.

- [ ] **Step 2: Run the focused tests and confirm the newly required mode/execution behavior fails before composition.**

  ```sh
  pixi run --locked test-python -q python/tests/test_iree_backend.py python/tests/test_iree_execution.py -rs
  ```

- [ ] **Step 3: Implement composition.** Compile the partitioned module to a VMFB while retaining the flow, stream, executable-target, and PTX dumps. Select the CUDA kernel factory from Task 2 or the VMFB segment handlers from Task 3, then rebuild the driver with the original output spec. Use the graph's CUDA device capability unless an explicit target or `IREE_CUDA_TARGET` is supplied. Retain the `compile_iree_backend` workspace with the callable and clean it up on failure or release.

- [ ] **Step 4: Re-run Step 2 on CUDA.** Expect both modes to match eager results and propagate errors; a CPU-only run must not be presented as GPU verification.

### Task 5: Validate real vLLM Qwen3 inference and document use

**Files:**
- Create: `python/tests/test_iree_qwen3_integration.py`
- Reuse: `python/tests/vllm_test_utils.py`, `python/tests/backend_spy.py`, and the `qwen_model_dir` fixture in `python/tests/conftest.py`
- Modify: `README.md`

**Interfaces:**
- `kashi.iree_backend.compile_iree_backend` is usable as the dotted vLLM backend with `CompilationMode.DYNAMO_TRACE_ONCE`.
- `generate_qwen` runs the offline, one-layer dummy model with seed 42, two fixed token-ID prompts, greedy decoding, and four output tokens.

- [ ] **Step 1: Add real Qwen3 parity tests for both modes.** Run compiled and `CompilationMode.NONE` vLLM engines with identical weights and requests. Use the `backend_spy.VLLM_BACKEND` dotted wrapper with a test-local adapter around `compile_iree_fx`; track successful direct CUDA launches or VMFB segment calls without replacing their execution. Disable CUDA graph replay and verify actual prefill and decode requests invoke IREE after warmup. Assert two outputs, four token IDs each, exact token-ID equality, finite selected-token log-probabilities close at `rtol=1e-4`, `atol=1e-5`, and at least two reference candidates with a top-two gap greater than `1e-4` at every step. Separately exercise `compile_iree_backend` through `torch.compile` and document its dotted vLLM name.

- [ ] **Step 2: Run the integration tests.** Expect both modes to pass on CUDA after Tasks 1–4; diagnose failures without replacing the generated execution path.

  ```sh
  pixi run --locked test-python -v python/tests/test_iree_qwen3_integration.py -rs
  ```

- [ ] **Step 3: Document the entry points and tests.** Explain `KASHI_IREE_EXECUTION`, CUDA requirements, direct launch versus IREE runtime synchronization, explicit artifact inspection through `compile_iree_fx`, binding constraints, and the commands above. State that the Qwen test uses a small seeded dummy model and checks selected-token log-probabilities rather than every model output.

- [ ] **Step 4: Run the complete affected checks.**

  ```sh
  pixi run --locked test-python -q python/tests/test_iree_reassembly.py python/tests/test_iree_cuda.py python/tests/test_iree_runtime.py python/tests/test_iree_backend.py python/tests/test_iree_execution.py python/tests/test_iree_qwen3_integration.py -rs
  pixi run --locked test-python -q python/tests/test_round_trip.py python/tests/test_qwen3_integration.py -rs
  pixi run --locked lint
  ```

  Expect the affected tests and lint to pass. Run on a CUDA-capable host to satisfy the execution and Qwen criteria; report any skipped GPU cases explicitly rather than treating CPU-only checks as equivalent.
