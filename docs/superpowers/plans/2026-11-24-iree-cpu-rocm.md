# IREE CPU and ROCm Execution Implementation Plan

> **For agentic workers:** Use the repository's spec-first workflow and the applicable Superpowers execution and test-development skills. Check off each step only after its stated verification.

**Goal:** Run the existing IREE Qwen3 and FX pipeline on CPU and AMD GPUs, retaining NVIDIA support and using `KASHI_IREE_EXECUTION=direct|iree`.

**Architecture:** Resolve the target from graph tensor metadata and PyTorch's build, then compile the existing partitioned module through IREE to CPU, CUDA, or HIP artifacts. Reuse shared GPU dispatch metadata parsing for direct CUDA/HIP launches, add a CPU-specific VMFB runtime bridge, and extend the existing GPU VMFB bridge to HIP. Run CPU Qwen parity in a CPU environment even when the main process uses a GPU wheel.

**Tech Stack:** Python 3.12, PyTorch, IREE compiler/runtime 3.14, vLLM 0.33, CUDA/HIP driver APIs, DLPack, pytest, Pixi, ELF and MessagePack metadata.

**Spec:** `specs/iree-cpu-rocm-execution.md`

## Global Constraints

- CPU execution is IREE-runtime-only; `direct` on CPU fails, even for an empty-segment graph. An unset mode defaults to `iree` for CPU and `direct` for GPU targets.
- `direct` and `iree` are the only accepted values of `KASHI_IREE_EXECUTION`; migrate existing `cuda` callers without an alias.
- Select CPU from CPU inputs even when a GPU exists. Distinguish CUDA and HIP builds despite both using PyTorch `cuda` devices. Preserve explicit `cuda_target`, add `target_device` and `rocm_target`, and reject conflicting overrides.
- Compile through the same complete IREE pipeline as CUDA and retain the VMFB and phase dumps. Preserve CUDA PTX access; expose CPU native artifacts and HIP HSACO under the artifact directory.
- No eager, CPU/GPU, CUDA/HIP, or direct/runtime fallback on error. Keep tensor device, lifetime, mutation and stream semantics intact, or reject unsupported cases.
- The small offline Qwen3 CPU parity case runs in every normal test run, including GPU-equipped hosts. The full 0.6B architecture is an explicit separate test. GPU tests are gated by the correct vendor and compatible software.
- CPU code remains covered by the CPU changed-line gate. Only the AMD-specific direct-launch module may be added to the existing GPU-only exemption.
- Preserve the existing CUDA implementation, tests, and documented public behavior except the explicitly approved execution-mode rename. Do not fork or vendor IREE.

## File Ownership and Interfaces

- `python/src/kashi/iree_device.py`: target-family/architecture detection and execution-mode selection; shared by the backend and tests.
- `python/src/kashi/iree_gpu_dispatch.py`: shared flow/stream/HAL dispatch metadata parsing used by direct GPU launchers.
- `python/src/kashi/iree_cpu_runtime.py`: CPU VMFB segment handlers using a local IREE device and CPU tensors.
- `python/src/kashi/iree_hip.py`: HSACO ABI validation and direct HIP module/kernel calls.
- `python/src/kashi/iree_cuda.py`: existing direct NVIDIA launcher, refactored to consume shared dispatch parsing without changing its CUDA ABI checks.
- `python/src/kashi/iree_runtime.py`: existing GPU VMFB bridge, extended from CUDA to HIP with correct device identity and DLPack device codes.
- `python/src/kashi/iree_backend.py`: public target arguments, IREE compiler flags/artifacts, mode selection, and segment-handler routing.
- `python/tests/`: CPU tests, shared hardware detection, vendor-specific GPU tests, Qwen subprocess/parity coverage, and an explicitly invoked 0.6B test.
- `pyproject.toml` / `pixi.lock`: isolated CPU test environment and any explicit dependencies needed for HSACO ELF/MessagePack metadata.
- `scripts/check_diff_coverage.py`: add only the AMD direct-launch module to the pre-existing opt-in GPU-only exemption.
- `README.md`: target/mode use, CPU and AMD artifacts, test prerequisites, commands, and hardware limitations.

## Review Focus

1. A CPU graph on a GPU host must remain on CPU; explicit cross-compilation overrides and conflicting modes must not select a wrong runtime. **Task 2.**
2. A malformed HSACO, wrong export, or mismatch in pointer/constant ABI must fail before HIP launch. **Tasks 3 and 5.**
3. CPU, CUDA, and HIP buffers must use the correct device identity, DLPack code, synchronization, and lifetime. **Tasks 4 and 6.**
4. A CUDA PyTorch/vLLM wheel must not cause the always-required CPU Qwen test to skip or use GPU. **Tasks 1 and 7.**
5. A missing AMD GPU must skip only AMD execution tests; offline HSACO compilation/validation must not be reported as AMD execution. **Tasks 3, 5, and 8.**

---

### Task 1: Isolate test environments and dependencies

**Files:** `pyproject.toml`, `pixi.lock`, `python/tests/conftest.py`, and test utilities.

**Interfaces:** A locked CPU test environment must be selectable without replacing the active CUDA/ROCm environment; use `VLLM_TARGET_DEVICE=cpu` before vLLM import in the CPU process.

- [x] Add an explicitly CPU-only Pixi feature/environment for Linux CPU and macOS ARM CPU, using the existing platform-specific PyTorch/vLLM pins. Keep the default and `cpp-coverage` environments unchanged.
- [x] Add explicit maintained ELF/MessagePack parsing dependencies for HSACO metadata, and update the lock for every supported platform. Do not copy IREE sources or embed a second IREE compiler.
- [x] Add a test utility that invokes the CPU Qwen test in a separate process/environment when the active PyTorch wheel is CUDA or ROCm; on a CPU wheel use an isolated CPU-selected process directly. Set CPU platform selection before importing vLLM. A missing CPU environment or child failure must fail the parent test, not skip it.
- [x] Validate locked installation and a harmless CPU subprocess on a GPU host and a CPU environment. CPU tests must not depend on GPU visibility. Do not start a GPU workload during this setup.

### Task 2: Resolve target families and execution modes

**Files:** create `python/src/kashi/iree_device.py`; modify `python/src/kashi/iree_backend.py`; add `python/tests/test_iree_device.py` and update existing mode tests.

**Interfaces:** Resolve a target descriptor containing family (`cpu`, `cuda`, `hip`), graph device, architecture, and ROCm features. Extend `compile_iree_fx(..., *, cuda_target=None, target_device=None, rocm_target=None, ...)`; preserve `cuda_target` behavior for offline CUDA compilation.

- [x] Add failing tests for CPU tensors on a GPU host, CUDA and HIP build discrimination, explicit target precedence, `IREE_CUDA_TARGET` not forcing a CPU graph onto CUDA, conflicting overrides, empty CPU graphs with explicit `direct`, old `cuda` mode rejection, and the per-family default mode.
- [x] Test ROCm target strings such as `gfx942:sramecc+:xnack-` mapping to architecture `gfx942` and features `+sramecc,-xnack`; malformed and contradictory features must fail.
- [x] Implement selection without consulting GPU presence when graph tensors are on CPU. Keep actual execution device validation separate from offline cross-compilation.
- [x] Run the focused device tests and existing CUDA target/invalid-mode tests. Update existing tests and documentation from the old `cuda` mode value to `direct`.

### Task 3: Share GPU dispatch metadata and validate HSACO offline

**Files:** create `python/src/kashi/iree_gpu_dispatch.py`; modify `python/src/kashi/iree_cuda.py`; create `python/tests/test_iree_hip.py`; update `python/tests/test_iree_cuda.py` imports.

**Interfaces:** Shared parsing returns the flow argument descriptions, stream binding and constant counts, offset-constant mapping, workgroup count region, affine maps, and block/target metadata. CUDA retains PTX-specific symbol and parameter checks. HIP additionally validates AMDGPU ELF note metadata and exact kernel export names.

- [x] Add tests that compile a representative `gfx942` module with IREE offline, confirm it produces an HSACO and the usual phase dumps, and parse real code-object metadata. Verify pointer and 32-bit constant parameter ordering, sizes and offsets against the stream/target ABI. Reject a missing export, malformed ELF/metadata, unsupported parameter kind, count mismatch, and incompatible target.
- [x] Extract the genuinely backend-neutral flow/stream/HAL parsing, grid evaluation, scalar packing and tensor validation from the CUDA launcher. Migrate its callers and CPU-side grid tests; do not retain duplicate implementations or compatibility aliases.
- [x] Add HIP-specific HSACO parsing using the declared ELF/MessagePack dependencies. Do not infer the HIP symbol from CUDA PTX name rewriting or accept unvalidated binaries.
- [x] Run offline HIP compilation/metadata tests and the existing CUDA parser/launch tests. On systems without the compiler's HIP target, report the compilation limitation; on systems without an AMD GPU, offline success is not a GPU execution result.

### Task 4: Execute CPU VMFB segments

**Files:** create `python/src/kashi/iree_cpu_runtime.py`; create `python/tests/test_iree_cpu_runtime.py`; integrate the CPU handler in `python/src/kashi/iree_backend.py`.

**Interfaces:** `runtime_segment_handlers(vmfb_path: Path, partition: PartitionResult)` returns segment-name → `(graph, args) -> list` handlers, using an IREE local device and CPU tensors.

- [x] Add a failing CPU integration test that constructs a metadata-bearing CPU FX graph, compiles it with `compile_iree_fx`, calls the returned graph on CPU, and compares real outputs with eager PyTorch. Run it in the main environment even when CUDA/HIP is present.
- [x] Cover multi-result and scalar arguments, output lifetime after releasing temporary VM objects, non-contiguous supported inputs, missing/invalid artifacts, explicit `direct` rejection, and the empty-segment path. Use real CPU VMFBs and tensors, not simulated runtime objects.
- [x] Implement local-device VMFB loading and caching, CPU DLPack/host-memory import, dtype/shape and ownership validation, and safe output conversion. Avoid CUDA/HIP initialization and avoid copies when ownership and layout are safe.
- [x] Run the CPU runtime tests on the default environment and on a CPU-only environment. Verify that CPU execution is not skipped or redirected on a GPU-equipped machine.

### Task 5: Launch AMD kernels directly through HIP

**Files:** create `python/src/kashi/iree_hip.py`; extend `python/tests/test_iree_hip.py`; add the new GPU-only module to `scripts/check_diff_coverage.py`'s existing opt-in exemption.

**Interfaces:** `create_hip_kernel(executable: str, export: str, *, artifact_dir: Path)` returns an FX-compatible callable with the same dispatch metadata contract as `create_cuda_kernel`.

- [ ] On AMD hardware, add real kernel tests for a small tensor graph and matrix multiplication. Check exact target selection, same-device bindings, current PyTorch HIP stream, result lifetime, zero grids, and rejection of unsupported layout/ABI. Require actual HIP module launch, not merely an HSACO file.
  AMD hardware was unavailable on this host; hardware-gated tests were skipped.
- [x] Implement `hipModuleLoadData`, exact-symbol `hipModuleGetFunction`, and `hipModuleLaunchKernel` using the active PyTorch HIP context and stream. Pack binding pointers and 32-bit constants as validated in Task 3; apply IREE's HIP static-shared-memory convention and reject unsupported dynamic shared memory. Cache modules/functions per actual device/context and keep launch resources alive.
- [x] Test missing HIP library, driver errors, missing kernels and invalid metadata without suppressing failures. Preserve the existing CUDA implementation's behavior.
- [x] Run offline tests on every host and actual HIP execution tests only on a matching AMD GPU. Report hardware absence explicitly.

### Task 6: Execute AMD segments through IREE's HIP runtime

**Files:** modify `python/src/kashi/iree_runtime.py`; extend `python/tests/test_iree_runtime.py`; route the selected GPU runtime from `python/src/kashi/iree_backend.py`.

**Interfaces:** `runtime_segment_handlers(vmfb_path, partition)` continues to provide FX segment handlers, selecting the matching IREE `cuda` or `hip` driver from the resolved target.

- [x] Add NVIDIA and AMD runtime tests for correct physical device matching, DLPack device types (CUDA `2`, ROCm `10`), real VMFB execution, non-default streams, lifetime, and error propagation. Do not use enumeration order as device identity.
- [x] Extend the runtime bridge to select the correct driver and physical UUID/context, import/export on the correct device, and synchronize PyTorch and IREE streams. Keep outputs safe for the PyTorch allocator and preserve supported aliases; reject unrepresentable or ambiguous cases.
- [x] Run existing NVIDIA runtime regressions and AMD runtime tests when the corresponding GPU is available. Do not mark an offline compile as runtime execution.

### Task 7: Complete Qwen3 parity and vendor-aware test gating

**Files:** `python/tests/test_iree_qwen3_integration.py`, `python/tests/vllm_test_utils.py`, `python/tests/conftest.py`, GPU-specific test modules, and an explicitly invoked `python/tests/qwen3_0_6b.py`.

**Interfaces:** The normal test command always runs the small CPU Qwen parity case; accelerator cases run in matching CUDA/ROCm environments. The separate 0.6B command uses the real 28-layer, hidden-size-1024 architecture with deterministic dummy weights.

- [x] Add CPU small-Qwen parity using the real vLLM engine and IREE CPU runtime, with exact token IDs and finite selected-token log-probabilities compared at `rtol=1e-4`, `atol=1e-5`. Require actual compiled prefill and decode invocations. Run it even when the parent environment detects a GPU.
- [x] Replace generic `torch.cuda.is_available()` gates with vendor-aware NVIDIA/HIP checks; run both GPU modes on their own family. If both families are present, isolate vLLM platform selection in matching processes. A detected GPU with broken required software fails rather than being misreported as absent.
- [x] Add the explicit full-size Qwen3-0.6B test with its real architecture, deterministic dummy weights, fixed token IDs and an uncompiled reference. Keep it outside normal pytest discovery; document an explicit CPU or matching-GPU invocation. Never silently substitute the one-layer model.
- [x] Run CPU parity in both CPU and accelerator environments, the NVIDIA parity tests on NVIDIA hardware, and AMD parity tests on actual AMD hardware when available.

### Task 8: Document and verify end to end

**Files:** `README.md`; tests from Tasks 1–7.

- [x] Document CPU native object/embedded-ELF artifacts and the runtime-only decision; PTX versus HSACO; `direct|iree` migration; target overrides; device gating and separate CPU environment; and the explicit full-size Qwen3 command.
- [x] Run the focused CPU/device/metadata tests, the complete Python suite, and `pixi run --locked lint`. Run NVIDIA tests on a real NVIDIA GPU and AMD direct/runtime tests on a real AMD GPU when available. Run the explicit full-size CPU case separately.
- [ ] Verify the CPU changed-line gate still includes CPU code and the GPU-only exceptions are named in its summary. State exactly which GPU tests ran or skipped; do not claim AMD parity from offline compilation alone.
  Not run: the changed-line script invokes Git, which the user prohibited. CPU-gated module line coverage was 92%.
