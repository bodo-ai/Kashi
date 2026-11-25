# Executing IREE-Generated CUDA Kernels

**Status:** Proposed; pending review
**Builds on:** the existing FX → Torch MLIR → partitioned IREE compilation and FX reconstruction pipeline, and the [standalone TorchDynamo backend](vllm-mlir-compilation-plugin/dynamo-backend.md).

## Intent

Complete the IREE compilation loop: the FX graph reconstructed from a compiled model must be callable and must execute the generated kernels. Support both direct CUDA launches of IREE-generated PTX and execution through the IREE runtime. Both paths must consume the same partitioned model and preserve its observable results for the supported inputs.

## Scope

### Goals

- Turn the existing inspection-only IREE graph into an executable graph without replacing its compiled computations with eager PyTorch or silently falling back to the original graph.
- Provide two explicitly selectable execution modes: `cuda` for direct CUDA launches and `iree` for IREE runtime execution. Default to `cuda`; reject unknown modes.
- Preserve the graph's argument order, output structure, data dependencies, and execution order around existing PyTorch custom-operator boundaries.
- Run both modes on CUDA tensors and compare their results with the corresponding uncompiled PyTorch computation. Keep the existing executable PyTorch-only backend separate and unchanged.
- Expose a standard `torch.compile` backend and a vLLM-loadable backend entry point that use the selected IREE execution mode.
- Validate both modes with real vLLM Qwen3 inference against uncompiled vLLM.
- Report unsupported IR, incompatible tensor bindings, missing artifacts, and compilation or launch failures instead of producing an incorrect result.

### Non-goals

- Forking or vendoring IREE; support other accelerator backends, CPU execution of compiled segments, or training/backward compilation.
- Guaranteeing support for every IREE ABI, operation, tensor layout, or dtype. Reject an unsupported case rather than guessing how to execute it.
- Changing the semantics of the separate FX → MLIR → FX PyTorch backend.

### Anti-goals

- Replacing generated kernels with no-op targets, substituting the original FX graph on error, or testing only that compilation produced files without executing the resulting computation.
- Returning an IREE-owned allocation as a normal PyTorch result when its lifetime or stream dependencies cannot be safely tracked.
- Treating CPU-only test success as evidence that CUDA kernels ran.

## Interface and compilation

Keep the existing explicit entry point:

```python
compile_iree_fx(
    gm: torch.fx.GraphModule,
    *,
    output_dir: str | os.PathLike,
    cuda_target: str | None = None,
    compiler: str | os.PathLike | None = None,
) -> IreeCompilation
```

The returned `IreeCompilation.graph` must be a callable `torch.fx.GraphModule`, not an inspection-only placeholder. It retains the output pytree of the input graph. `KASHI_IREE_EXECUTION` selects the mode at compilation time: `cuda` by default, or `iree`. The result records the chosen mode and paths to the unique compilation directory, flow IR, VMFB, individual PTX files, and optional combined PTX listing.

Continue to import and verify the graph, partition it into compiled segments and a PyTorch driver, and compile segments with IREE's CUDA target. Retain the flow, stream, and executable-target phase dumps needed to identify dispatches and their launch ABI. Emit a VMFB as well as the existing PTX artifacts. An explicit CUDA target takes priority over `IREE_CUDA_TARGET`; otherwise use the captured CUDA tensor's device capability. An explicit compiler path overrides tool discovery.

If the graph has no compilable segments, rebuild and execute the driver without invoking IREE on an empty module; represent absent VMFB, flow IR, and PTX artifacts as absent. Compiler errors, unavailable target information, unsupported partition interfaces, and missing or inconsistent phase artifacts must propagate as failures.

### TorchDynamo and vLLM entry point

Expose `kashi.iree_backend.compile_iree_backend(gm, example_inputs)` as a plain `torch.compile` backend. It compiles through `compile_iree_fx` using the selected mode and returns a callable that executes the resulting graph. The backend reads the necessary graph metadata from `gm`; the `example_inputs` parameter remains part of the standard backend interface.

vLLM must be able to select the same backend with the dotted name `kashi.iree_backend.compile_iree_backend` and `CompilationMode.DYNAMO_TRACE_ONCE`. The returned callable owns a temporary artifact directory for as long as it is needed, cleans it up if compilation fails, and does not silently delegate to the PyTorch-only backend.

## Execution design

### Direct CUDA mode

Bind each reconstructed flow dispatch to an executable callable for its IREE executable and export. Preserve its workload, argument dimensions, result shapes and dtypes, and any supported tied-output mapping. Resolve the matching PTX entry and derive its binding/scalar ABI and launch dimensions from IREE's compilation artifacts; do not assume a fixed grid, block size, or parameter layout for all kernels.

Load and launch the PTX through the CUDA driver in PyTorch's active CUDA context, on the caller's current CUDA stream. Allocate untied results with PyTorch on the source device. Record the stream for participating PyTorch buffers so their storage remains valid until asynchronous work finishes. Handle an empty launch grid without issuing an invalid CUDA launch. Any supported tied result must refer to the correct input storage; otherwise reject it.

Initially support dense, contiguous tensor bindings on one CUDA device with compatible shape, dtype, and alignment, together with the scalar forms supported by the generated ABI. Validate these constraints and reject unsupported or inconsistent metadata before launching. Missing CUDA driver/context, missing PTX or export, and CUDA driver errors must be reported to the caller.

### IREE runtime mode

Bind each compiled segment to its VMFB export and invoke it through the IREE CUDA runtime. Select the CUDA device corresponding to the actual PyTorch input tensors; do not assume that IREE and PyTorch enumerate devices in the same order. Reject mixed devices and require an explicit CUDA device for a segment that has no tensor input.

Exchange tensor buffers through DLPack where supported; handle supported scalar and container arguments without changing their meaning. Respect stream dependencies before IREE consumes PyTorch data. Ensure fresh returned tensors have safe PyTorch-owned CUDA storage and that device-to-device copies complete before foreign storage can be released. Do not copy tensor data through host memory. Reject an input or output representation whose semantics or ownership cannot be preserved.

### Reconstruction and lifetime

The driver must invoke real dispatch targets in `cuda` mode or real VMFB segment targets in `iree` mode, with operands and results mapped in order. Preserve existing inspection behavior when a flow graph is deliberately reconstructed without an executor: calling such a target must raise rather than pretend to execute.

Compilation artifacts and loaded CUDA or IREE runtime resources must remain available for the lifetime of the callable that needs them, including repeated calls. Artifacts written to the caller's `output_dir` remain available for inspection; any internally owned temporary workspace must be cleaned up on compilation failure or after its callable is released.

## Acceptance criteria

1. On a CUDA-capable system, compile and execute a small tensor graph in each mode and compare the results with eager PyTorch. Assert that the IREE-generated execution path actually runs, not merely that the original graph returns the right answer.
2. Exercise a non-default PyTorch CUDA stream and verify that both execution modes respect producer/consumer ordering and output lifetime.
3. Exercise at least one matrix multiplication and a multi-segment graph with an existing custom-operator boundary. Verify values, output structure, and ordering against eager execution.
4. Check that the selected CUDA architecture follows the graph's device or an explicit override; reject inconsistent devices and invalid modes.
5. Cover missing compiler, missing or malformed artifacts, unsupported bindings, and driver/runtime failures. Confirm that none silently falls back to the original graph.
6. Run independent, deterministic parser and partition tests on CPU where possible. CUDA execution tests must run only when CUDA is available and must not be reported as executed when skipped.
7. Exercise each mode through the real vLLM engine using the offline, seeded, one-layer dummy Qwen3 configuration. Generate four tokens for each of two prompts using greedy sampling. Compare token IDs exactly with uncompiled vLLM and compare selected-token log-probabilities with `rtol=1e-4` and `atol=1e-5`. Require finite log-probabilities and at least two reference candidates whose top-two log-probabilities differ by more than `1e-4`, so token comparisons are meaningful. Exercise both prefill and decode and assert that real IREE execution occurs for the requests. Disable CUDA graph replay for the parity comparison so Python-side execution tracking is not bypassed.

## Documentation

Document mode selection, CUDA requirements, the distinction between direct launches and IREE runtime execution, artifact inspection, supported binding constraints, and commands for running the execution and Qwen3/vLLM parity tests. Document the `torch.compile` and vLLM backend entry points.
