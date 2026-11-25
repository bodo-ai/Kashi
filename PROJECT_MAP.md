# Project Map

Kashi is an experimental PyTorch FX → MLIR → FX compiler with vLLM
integration. The standalone backend executes the reconstructed graph in
PyTorch; the opt-in IREE backend compiles eligible segments for CUDA.

## Execution paths

- **FX round trip:** `backend.compile_fx` calls `importer.import_fx`, verifies
  the resulting Torch-dialect MLIR, and uses `mlir_to_fx.rebuild_fx` to return
  an executable PyTorch graph. `decompose` optionally rewrites supported vLLM
  custom operations before import.
- **IREE CUDA:** `iree_backend.compile_iree_fx` imports and partitions the
  graph into ordinary IREE segments and a PyTorch driver preserving custom and
  alias-sensitive boundaries. It invokes IREE, retains phase dumps and a VMFB,
  then rebuilds the driver with either `iree_reassembly`/`iree_cuda` direct
  PTX launches (`KASHI_IREE_EXECUTION=cuda`, the default) or `iree_runtime`
  VMFB calls (`iree`). Empty-segment graphs execute the driver without invoking
  IREE. This execution path currently targets CUDA, not CPU or HIP.
- **Integration and operations:** vLLM tests use a shared backend spy and seeded
  model utilities. Runpod scripts separate pod orchestration, durable recovery,
  and the remote benchmark worker.

## Repository layout

This inventory covers maintained source, configuration, and documentation.
It excludes local environments, caches, Git internals, and generated build or
benchmark artifacts.

- `CLAUDE.md` — Shared project instructions loaded by Claude Code and omp.
- `PROJECT_MAP.md` — This architecture and responsibility inventory.
- `README.md` — Setup, backend usage, testing, and operational instructions.
- `pyproject.toml` — Project metadata, Python tooling, Pixi environments, pinned
  platform dependencies, and test/build/lint tasks.
- `pixi.lock` — Resolved Pixi dependency graph for supported environments.
- `CMakeLists.txt` — Builds the nanobind extension and optional native tests,
  mutation instrumentation, lint, and coverage targets.
- `.gitignore` — Excludes local environments, builds, caches, and benchmark artifacts.
- `.clang-format` and `.clang-tidy` — Native formatting and lint configuration.
- `.pre-commit-config.yaml` — Local pre-commit hook configuration.
- `.roborev.toml` — Local agentic review configuration.
- `mull.yml` — Restricts native mutation testing to project-owned library code.
- `.claude/`
  - `.claude/settings.json` — Enables the Claude Code Superpowers plugin.
- `python/src/kashi/` — Python package implementing both compiler paths.
  - `python/src/kashi/__init__.py` — Marks the package; defines no public entry points.
  - `python/src/kashi/backend.py` — TorchDynamo backend for the eager FX→MLIR→FX round trip.
  - `python/src/kashi/importer.py` — Copies and normalizes FX graphs and metadata, handles supported attention functionalization, and imports Torch MLIR with its output pytree specification.
  - `python/src/kashi/decompose.py` — Rewrites supported vLLM custom operations to ATen while checking mutation and alias safety and preserving provenance.
  - `python/src/kashi/mlir_to_fx.py` — Reconstructs supported Torch MLIR operations, constants, custom calls, segment calls, and output pytrees as FX.
  - `python/src/kashi/iree_backend.py` — Orchestrates IREE CUDA compilation, artifact retention, partitioning, and executable driver reconstruction; exposes the Dynamo backend.
  - `python/src/kashi/iree_partition.py` — Splits Torch MLIR around custom, mutation, alias, and shape boundaries and records segment interfaces and source device.
  - `python/src/kashi/iree_reassembly.py` — Converts supported IREE flow segment operations and dispatches into FX handlers, binding optional kernel executors.
  - `python/src/kashi/iree_cuda.py` — Validates flow/stream/target/PTX dispatch metadata and launches CUDA kernels on PyTorch's current stream.
  - `python/src/kashi/iree_runtime.py` — Executes CUDA VMFB segments with IREE, DLPack imports, synchronization, PyTorch-owned outputs, and alias handling.
- `python/tests/` — Python unit, integration, and operational regression tests.
  - `python/tests/.gitkeep` — Preserves the test directory when it is empty.
  - `python/tests/conftest.py` — Supplies isolated local model configurations and shared fixtures.
  - `python/tests/backend_spy.py` — Records compiled graphs and token counts through a vLLM-loadable backend wrapper.
  - `python/tests/fx_test_utils.py` — Supplies fake-metadata recording, round-trip, target-inspection, and pytree comparison helpers.
  - `python/tests/kashi_test_ops.py` — Defines test-only custom/opaque operations and capture helpers for mutation and alias scenarios.
  - `python/tests/vllm_test_utils.py` — Creates seeded vLLM runs, isolates test settings, and compares generated tokens and log probabilities.
  - `python/tests/test_import.py` — Smoke-tests package and native extension imports.
  - `python/tests/test_import_fx.py` — Covers FX import, metadata, custom operations, and vLLM attention functionalization.
  - `python/tests/test_mlir_to_fx.py` — Covers reconstruction of Torch MLIR, custom calls, output structures, and invalid IR.
  - `python/tests/test_round_trip.py` — Checks FX→MLIR→FX numerical and structural parity across operations and metadata styles.
  - `python/tests/test_backend.py` — Checks the standalone Dynamo backend against eager execution and graph reuse.
  - `python/tests/test_decompose.py` — Covers vLLM decomposition, provenance, shape and alias safety, and GPU parity.
  - `python/tests/test_iree_backend.py` — Tests IREE compilation, driver reconstruction, artifact lifetime, and error handling.
  - `python/tests/test_iree_partition.py` — Tests partition boundaries, value flow, device information, and unsupported crossings.
  - `python/tests/test_iree_reassembly.py` — Tests converting IREE flow IR to FX, including dynamic shapes and tied results.
  - `python/tests/test_iree_cuda.py` — Tests direct PTX launch ABI, stream behavior, and workgroup grid evaluation.
  - `python/tests/test_iree_runtime.py` — Tests CUDA VMFB calls, device selection, output lifetime, aliasing, and capture rejection.
  - `python/tests/test_iree_execution.py` — Exercises both CUDA execution modes with real operations, mutations, graph capture, and dtypes.
  - `python/tests/test_iree_qwen3_integration.py` — Compares real vLLM Qwen3 inference through both IREE modes with uncompiled generation.
  - `python/tests/test_qwen3_components_roundtrip.py` — Checks Qwen3-style normalization, gating, splits, and argument handling through the round trip.
  - `python/tests/test_qwen3_integration.py` — Compares compiled and uncompiled vLLM Qwen3 prefill/decode tokens and scores.
  - `python/tests/test_vllm_integration.py` — Compares compiled and uncompiled seeded SimpleVllm generation.
  - `python/tests/test_runpod_controller.py` — Covers Runpod provisioning, artifact download, request handling, and cleanup.
  - `python/tests/test_runpod_recovery.py` — Covers journal transitions, locks, API retries, and conservative pod reconciliation.
  - `python/tests/test_runpod_worker.py` — Covers worker configuration, benchmarking, profiling, artifact access, watchdogs, and failure handling.
  - `python/tests/test_mutation_workflow.py` — Tests native mutation toolchain requirements and changed-file policy.
  - `python/tests/models/`
    - `python/tests/models/simple_vllm_model.py` — Minimal compile-enabled vLLM attention model and causal-LM wrapper for integration tests.
- `cpp/` — Native extension and reserved locations for project-owned native code.
  - `cpp/bindings/module.cpp` — Defines the currently empty nanobind `_native` module.
  - `cpp/lib/.gitkeep`, `cpp/include/.gitkeep`, and `cpp/tests/.gitkeep` — Preserve currently empty locations for native sources, headers, and tests.
- `scripts/` — Development runners and remote benchmark tooling.
  - `scripts/check_diff_coverage.py` — Enforces changed-production-line coverage using Python and native coverage reports.
  - `scripts/test_cpp.py` — Configures, builds, and runs Catch2 tests and optionally exports native coverage.
  - `scripts/test_mutation.py` — Runs native tests with Mull and applies the changed-production-file mutation policy.
  - `scripts/runpod_benchmark.py` — Provisions benchmark pods, communicates with the worker, verifies artifacts, and confirms cleanup.
  - `scripts/runpod_recovery.py` — Maintains durable journals and conservatively reconciles or cleans up potentially billable pods.
  - `scripts/runpod_worker.py` — Runs the remote vLLM benchmark, optional profiling, authenticated artifact service, and deletion watchdog.
- `.github/workflows/` — CI and review automation.
  - `.github/workflows/test.yml` — Runs lint, Python/native tests, packaging, and coverage gates.
  - `.github/workflows/mutation.yml` — Installs the pinned mutation toolchain and runs native mutation tests.
  - `.github/workflows/pr-review.yml` — Coordinates requested specification, conformance, and pull-request reviews.
- `.github/agents/` — Read-only review personas used by the review workflow.
  - `.github/agents/spec-ambiguity.agent.md` — Reviews changed specifications for divergent interpretations.
  - `.github/agents/spec-security.agent.md` — Reviews changed specifications for security and trust-boundary gaps.
  - `.github/agents/spec-operability.agent.md` — Reviews changed specifications for operability and verification gaps.
  - `.github/agents/conformance.agent.md` — Compares changed implementations against changed specification criteria.
  - `.github/agents/pr-review.agent.md` — Reviews selected pull-request changes for defects and regressions.
- `specs/` — Versioned design intent; see `specs/README.md` for the specification workflow.
  - `specs/iree-kernel-execution.md` — IREE-generated CUDA execution design.
  - `specs/runpod-mi300x-vllm-benchmark.md` — Runpod MI300X benchmark design.
  - `specs/compiler-repository-skeleton.md` — Initial compiler repository structure.
  - `specs/rename-to-kashi.md` — Repository and package rename design.
  - `specs/testing-and-mutation-testing.md` — Python/native testing and mutation policy.
  - `specs/agent-development-review-workflow.md` — Agentic development and review workflow design.
  - `specs/vllm-mlir-compilation-plugin/` — Designs for the FX/MLIR/vLLM compiler pipeline.
    - `specs/vllm-mlir-compilation-plugin/vllm-mlir-compilation.md` — Overall FX–MLIR–FX compilation plugin design.
    - `specs/vllm-mlir-compilation-plugin/dynamo-backend.md` — Standalone TorchDynamo backend contract.
    - `specs/vllm-mlir-compilation-plugin/mlir-fx-reconstruction.md` — Torch MLIR to FX reconstruction design.
    - `specs/vllm-mlir-compilation-plugin/vllm-integration.md` — vLLM integration, symbolic shapes, opaque inputs, and custom operators.
    - `specs/vllm-mlir-compilation-plugin/functional-attention.md` — Functional attention import and reconstruction design.
    - `specs/vllm-mlir-compilation-plugin/optional-custom-op-decomposition-design.md` — Optional vLLM custom-op decomposition design.
- `docs/` — Development guides and implementation plans.
  - `docs/development/agentic-workflow.md` — Local agent setup and review workflow.
  - `docs/superpowers/plans/2026-11-18-functional-attention.md` — Functional attention implementation plan.
  - `docs/superpowers/plans/2026-11-18-iree-kernel-execution.md` — IREE CUDA kernel execution implementation plan.
