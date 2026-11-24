# IREE Kernel Execution on CPU and AMD GPUs

**Status:** Approved
**Extends:** [Executing IREE-Generated CUDA Kernels](iree-kernel-execution.md). This specification replaces that document's CPU and non-CUDA accelerator non-goals and its `KASHI_IREE_EXECUTION` value contract; its remaining correctness and no-fallback requirements still apply.

## Intent

Extend the existing Qwen3 and IREE execution path to run compiled segments on CPU and AMD GPUs, while preserving NVIDIA CUDA execution. On AMD GPUs support both direct HIP kernel launches and IREE runtime execution. On CPUs use the IREE runtime. Always run CPU correctness tests, including on machines that also have NVIDIA or AMD GPUs; run GPU execution tests only for the matching available GPU kind.

## CPU kernel and invocation decision

IREE's `llvm-cpu` target lowers dispatches to native host machine code. On Linux x86-64, a compilation can emit LLVM IR/bitcode, native `.o` and assembly, and an embedded-ELF executable library stored in the VMFB; other host architectures have corresponding native code. These are not PTX kernels.

Use IREE's local runtime (`local-task` or `local-sync`) to load the VMFB and invoke its segment exports. Directly calling the executable library would require implementing IREE's version-dependent dispatch ABI, workgroup scheduler, bindings, imports, and lifetime management; it is not a second supported CPU mode for this change. An explicitly requested direct CPU execution must fail clearly rather than silently switching modes.

## Goals

- Compile and execute CPU segments with IREE's LLVM-CPU target and local runtime, on CPU tensors even when GPUs are present.
- Compile AMD GPU segments with IREE's HIP/ROCm target; execute the generated AMD code either through HIP's direct module/kernel launch API or through IREE's HIP runtime.
- Preserve the existing NVIDIA direct CUDA and IREE CUDA-runtime paths and their numerical behavior.
- Preserve the model graph's input/output structure, device, supported tensor semantics, custom-operator ordering, and execution dependencies. Unsupported or unsafe cases fail; they never fall back to another backend or return the original FX graph.
- Use the existing `KASHI_IREE_EXECUTION` environment variable uniformly: `direct` for a device's direct GPU launcher and `iree` for the runtime. Exercise a small, offline Qwen3 model on CPU in every normal test run; provide a separate explicit full-size Qwen3-0.6B test.
- Keep CPU tests independent of accelerator presence and gate NVIDIA and AMD execution tests by the appropriate available hardware.

## Decisions and alternatives

| Concern | Decision | Tradeoff |
| --- | --- | --- |
| CPU calls | IREE VM/HAL runtime only. | Avoids duplicating IREE's non-stable native workgroup ABI; there is no direct CPU benchmark mode. |
| AMD calls | Support both direct HIP and IREE HIP runtime. | Allows comparison with the existing CUDA approaches; direct mode must validate a different ABI and code-object format. |
| Invocation setting | `KASHI_IREE_EXECUTION=direct` or `iree`. | One device-independent choice; the old `cuda` value is removed and existing callers and documentation must migrate. |
| CPU Qwen test size | Small seeded, offline Qwen3 runs every time; full-size 0.6B runs through a separate explicit command. | Routine CPU tests remain bounded; full-size coverage is opt-in. |
| CPU vLLM on accelerator hosts | Run CPU vLLM in an isolated CPU PyTorch/vLLM environment when the active environment is a CUDA or ROCm build. | A separate locked environment and subprocess are needed; a GPU vLLM wheel cannot supply the CPU vLLM native operations. |

## Non-goals and anti-goals

- A direct CPU executable-library caller; training/backward support; additional accelerator backends; forking or vendoring IREE.
- Running the full-size Qwen3-0.6B case in every normal test run or requiring a network download for the always-run CPU tests. The full-size command uses the Qwen3-0.6B architecture and deterministic dummy weights, matching the existing dummy-weight usage; it must not claim to validate pretrained checkpoint accuracy.
- A fallback from `direct` to `iree`, from AMD to CUDA, from CPU to a GPU, or from IREE to eager PyTorch when execution fails.
- Claiming AMD execution was verified merely because an HSACO compiled, a HIP library was present, or GPU tests were skipped.

## Compilation and artifacts

### Target selection

Absent an explicit cross-compilation target, select the compilation target from the graph's tensor device and the PyTorch build, not from accelerator presence alone. CPU tensors select CPU even when a GPU is available. PyTorch exposes both NVIDIA and ROCm tensors as `cuda` devices; distinguish the active backend with `torch.version.cuda` versus `torch.version.hip`. Reject mixed devices, ambiguous accelerator builds, or missing device information rather than choosing an unrelated device.

Retain the existing `cuda_target` keyword and `IREE_CUDA_TARGET`. Add `target_device` (`cpu`, `cuda`, or `hip`) for explicit compilation and a `rocm_target` architecture override for builds without an AMD device. An explicit `target_device` selects the family; without it, an explicit `cuda_target` or `rocm_target` selects its corresponding family for offline compilation, including graphs with CPU fake metadata. Supplying both architecture keywords, or an architecture keyword for the wrong explicit family, is an error. `IREE_CUDA_TARGET` affects the CUDA architecture only after CUDA is selected; it never changes a CPU or HIP target. An actual GPU execution must still validate that the loaded artifact matches its tensor device.

For AMD execution, derive the `gfx*` target from the selected PyTorch HIP device when no override is given. Preserve and translate reported target features such as `sramecc` and `xnack` into IREE's separate ROCm target-features option; reject unrecognized or contradictory target metadata instead of guessing. For CPU, target the host's supported architecture or an explicitly configured compatible CPU target; do not claim a host-specialized artifact is portable to other CPUs.

### Common pipeline and retained artifacts

Continue through the full IREE compilation used for CUDA, not merely an intermediate phase. Retain the same flow, stream, and executable-target phase dumps, the compiler log, partition and driver MLIR, and a VMFB. Preserve CUDA PTX artifacts and existing CUDA artifact access. CPU compilation must expose its native object/library artifacts in the artifact directory. AMD compilation must retain its HSACO ELF code object and target metadata, as well as a VMFB using IREE's HIP executable container.

Use `--iree-hal-target-device=local` with the `llvm-cpu` local target backend for CPU; `--iree-hal-target-device=hip` with the selected ROCm target for AMD; and the existing CUDA target for NVIDIA. If there are no compilable segments, execute the driver without invoking IREE on an empty module, preserving the existing empty-artifact behavior, after validating target and mode. An explicit CPU `direct` request remains an error even for an empty-segment graph.

## Invocation modes

### Mode selection

- `KASHI_IREE_EXECUTION=direct`: execute NVIDIA kernels using the CUDA driver or AMD kernels using HIP. CPU direct execution is unsupported and raises a diagnostic before execution.
- `KASHI_IREE_EXECUTION=iree`: execute CPU, NVIDIA, or AMD segments through the matching IREE runtime driver.
- When the variable is unset, use direct execution for CUDA or HIP targets and IREE runtime execution for CPU. Selection depends on the selected compilation target, not on whether a GPU is present. Offline GPU compilation may use the GPU's direct mode without initializing a device; executing that graph still requires a matching GPU.
- The previous `cuda` value is no longer accepted. Update existing tests, examples, and documentation to use `direct`; do not retain an alias.

### CPU runtime

Use an explicit local IREE device and the compiled CPU VMFB. Exchange supported CPU tensors through DLPack or another documented IREE host-memory interface, maintaining dtype, shape, and ownership. The returned values must remain valid after the VM invocation and after temporary intermediates are released. Preserve supported input/output aliases and avoid unnecessary copies; when safe zero-copy ownership or layout cannot be established, use a correctly synchronized copy or reject the case. CPU execution must not initialize a CUDA/HIP driver or depend on GPU visibility.

### AMD direct HIP launch

Resolve the matching HSACO code object and the exact HIP kernel export. Construct and validate the binding-pointer and 32-bit push-constant ABI, argument ordering and sizes, workgroup counts, block dimensions, and code-object metadata from the IREE compilation artifacts. Do not assume CUDA PTX parameter declarations or CUDA-specific symbol naming apply to HSACO. Reject unrecognized metadata, unsupported ABI forms, incompatible code objects, and missing symbols before launch.

Load modules and resolve functions through the HIP module API in the correct device/context. Launch on PyTorch's current HIP stream and retain modules, buffers, and stream usage until asynchronous work finishes. Match IREE's HIP convention of passing zero dynamic shared-memory bytes for code objects with supported static workgroup memory; reject a form that requires unhandled dynamic shared memory. Handle zero-sized grids safely and propagate HIP errors. Inputs and outputs must satisfy supported device, dtype, shape, layout, offset, and alignment constraints.

### IREE GPU runtime

Use the IREE `cuda` driver for NVIDIA and `hip` for AMD. Match the physical PyTorch and IREE devices, preferably by verified UUID, before importing a DLPack capsule; a capsule alone does not guarantee device identity. Handle the correct DLPack device type for CUDA versus ROCm. Synchronize PyTorch producers and IREE consumers as required, retain inputs, and return safe PyTorch-owned outputs without routing GPU tensor data through host memory. Preserve supported aliases or reject an unrepresentable case. Missing drivers, unsupported buffer export, or device/context mismatches must fail rather than switch execution paths.

## Qwen3 coverage

### Always-run small model

Run the existing seeded, offline, one-layer Qwen3 configuration on CPU through the IREE CPU runtime and compare it with uncompiled CPU vLLM for the same prompts and weights. Check generated token IDs exactly, compare finite selected-token log-probabilities with `rtol=1e-4` and `atol=1e-5`, and exercise prefill and decode. This CPU case must run in the default test workflow regardless of NVIDIA or AMD GPU availability.

Because vLLM platform selection and native extensions are process-wide, use a separately locked CPU environment/process when the main test environment contains an accelerator vLLM/PyTorch wheel. Set `VLLM_TARGET_DEVICE=cpu` before importing vLLM in that process; on a CPU environment, execute the same case directly with CPU selected before import. Missing CPU dependencies or failure to run the CPU case is an error, not a skip based on accelerator availability. Preserve existing small Qwen3 tests for NVIDIA and add matching AMD direct/runtime parity cases gated by their GPU kind.

### Explicit full-size test

Provide a documented explicit test command for the real Qwen3-0.6B architecture: 28 layers, hidden size 1024, intermediate size 3072, 16 attention heads, 8 KV heads, head size 128, vocabulary size 151936, and tied embeddings. Use deterministic dummy weights and fixed token IDs; do not substitute the one-layer model under the 0.6B name. Run the selected IREE path and an uncompiled reference, and compare generated results. The command must support CPU and the available matching GPU backend; it is not part of the ordinary always-run CPU suite.

## Test and verification requirements

1. CPU tests run with CPU inputs and IREE's local runtime on CPU-only, NVIDIA-equipped, and AMD-equipped hosts. Cover compilation, real segment invocation, output numerical equality, source device selection, output lifetime, supported scalar/buffer arguments, and error propagation.
2. NVIDIA tests require an available NVIDIA GPU and a CUDA PyTorch build. AMD tests require an available AMD GPU and a ROCm PyTorch build. Neither may run against the other vendor merely because `torch.cuda.is_available()` is true. When both accelerator kinds are present, run each in its compatible environment and isolate the other vendor from vLLM platform probing or use a supported explicit platform-selection mechanism; wheel selection alone is insufficient. If required isolation or software is unavailable, fail with an actionable diagnostic rather than silently skipping a detected GPU.
3. For each GPU family, compare both `direct` and `iree` executions against eager PyTorch and real small-Qwen vLLM inference. Assert that the selected generated launch or VMFB segment actually executes, including prefill and decode. Use deterministic inputs and compare outputs and log-probabilities with documented tolerances.
4. Exercise non-default GPU streams, allocation lifetime, input/output binding validation, unavailable devices, bad modes, incompatible targets, malformed/missing artifacts, and the empty-segment path. Reject unsupported cases rather than hiding errors.
5. Independently validate ROCm target and HSACO/metadata parsing without AMD hardware, and compile representative HIP-targeted artifacts offline when the IREE compiler supports it. These checks are not a substitute for AMD hardware execution.
6. Run CPU and accelerator-specific tests without assuming one PyTorch wheel supports all platforms in one process. Keep dependency environments locked, avoid network/model downloads in routine tests, and report hardware-specific skips explicitly.
7. Keep CPU implementation code and CPU tests inside the existing CPU CI changed-line coverage gate. Do not place new CPU behavior in a module excluded as GPU-only. Preserve the existing CUDA/runtime exemptions and add only the AMD-specific direct-launch module to that GPU-only exemption, explicitly naming it in the CI coverage summary; do not extend the exemption to CPU or device-selection logic.

## Documentation

Document the CPU machine-code artifacts and runtime-only decision; NVIDIA PTX and AMD HSACO artifacts; target and execution-mode selection; migration from the old `cuda` setting; prerequisites; how CPU tests run on accelerator hosts; commands for GPU tests and the explicit full-size Qwen3-0.6B test; and what evidence is unavailable when a matching GPU is absent.
