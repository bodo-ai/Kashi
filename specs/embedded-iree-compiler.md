# Embedded IREE compiler and native Kashi passes

## Goal and scope

Stack PR #23 on PR #20. Run typed Kashi C++ passes at IREE's named MLIR phase boundaries in process. Build against pinned, unmodified IREE and keep the stock IREE wheel for Python bindings.

Keep FX import and reassembly in Python; move MLIR partitioning to C++. Preserve custom-op order, mutation semantics, segment signatures, and the output pytree. Kernel targets remain opaque; default TorchDynamo and vLLM behavior is unchanged.

## Approaches considered

- **Selected: private IREE statically linked into `kashi._native`.** Build Kashi's plugin and the embedding API from the same IREE source graph. The public IREE wheel provides Python MLIR objects; the Kashi extension embeds its own compiler. This duplicates compiler code on disk and in process but keeps upstream package ownership and avoids loading another copy of `libIREECompiler` by the upstream library's name.
- **Replacement IREE Python wheel.** A Kashi-built wheel could share a compiler with the importer and reduce duplication, but a rebuild under `iree-base-compiler` would require a separately hosted artifact or upstream publishing authorization. Giving a replacement a different distribution name while installing the same `iree/compiler` paths does not allow safe coexistence. Neither fits the selected PyPI install contract.
- **Private `iree-compile` executable.** Isolates compiler state and preserves #20's CLI behavior, but requires a subprocess and cannot provide the requested in-process binding. Do not retain it as an override or fallback.

## Branch integration

Rebase the #23 topic onto the actual #20 head and set #20's branch as the stacked PR base. Preserve main's already-landed changes and drop duplicate RunPod edits rather than replaying them. Replace `specs/native-mlir-passes.md` from #23 with this specification; #20 has no corresponding spec to remove. Carry forward #23's native build/test conventions, platform export isolation, and typed pass coverage where applicable. Remove the standalone SDK provisioning and cache, `run_passes` Python and C++ interfaces, arbitrary textual pass pipeline, and native-to-Python transformed-module bridge; there is no compatibility shim. Replace #20's compiler path override, tool lookup, subprocess and subprocess-only failure tests with the embedded path. Do not modify or retarget #20.

## Compilation contract

1. Python imports and verifies FX, then sends one serialized Torch module to `_native.compile_iree(assembly, artifact_dir, *, target, cuda_target=None)`. Preserve locations and resources. Keep the output pytree and FX metadata in Python.
2. Native code parses once in the IREE session context. Port #20's partitioner using SSA use lists and `IRMapping`, preserving ordered live-ins/live-outs, constant and list cloning, custom boundaries, diagnostics, and device normalization on compiled segments only. Save `partitioned.mlir` and `driver.mlir`; return the detected source device for Python's existing device reconciliation. Compile only the segment module, in the same context. Remove the Python partitioner.
3. Use a fresh session and invocation per call, release the GIL during native work, and capture diagnostics in `iree-compile.log`. Errors identify the failing stage and retain available artifacts. No subprocess, global CLI configuration, or fallback. CUDA emits PTX without `ptxas`; VMVX remains available for portable smoke tests.
4. Python reads the driver, authoritative Flow snapshot, and kernel manifest and rebuilds FX with the original pytree. Preserve existing artifact fields and the readable PTX listing.

### Pass placement without patches

Statically register Kashi C++ passes and dialects using IREE's out-of-tree plugin mechanism. Run IREE's standard pipeline up to each requested checkpoint with `ireeCompilerInvocationSetCompileToPhase`, run the registered passes on the same invocation, then resume with `ireeCompilerInvocationSetCompileFromPhase` set to that checkpoint. Use implementation-owned pass pipelines through the invocation API, not user-supplied strings. Split only where passes or artifact inspection require it. No serialization or new context between phases. Save the authoritative Flow snapshot after any Flow checkpoint passes.

Support named checkpoints from input through VM, including `executable-configurations` and `executable-targets`. Passes can target nested operations and use any linked, registered dialect present at the checkpoint. `executable-targets` exposes LLVM/NVVM MLIR before PTX serialization, but not intermediate IR inside target translation. For a real pass that needs an inaccessible point, seek an upstream hook rather than patching or copying IREE's pipeline. IR returned to Python must be parseable by the stock bindings.

### Linked CUDA artifacts

Keep default executable linking. Emit a manifest keyed by the original Flow executable/export pair, with the final PTX file and entry symbol as values. Resolve identities from native executable/export metadata across lowering and linking; if necessary, carry stable identities or assign collision-free symbols before linking with all references updated. Validate final entries against serialized CUDA export metadata and emitted PTX. Reject missing or ambiguous mappings; do not rely on artifact order, unchanged symbols, or a one-file-per-executable convention. Passes after the Flow snapshot must preserve the dispatch ABI and mapping information.

At the pinned revision an in-memory probe resumed from Flow through executable targets to end. Executable targets contained NVVM and no PTX; the split and uninterrupted runs emitted byte-identical linked PTX for the two-segment fixture. This validates phase resumption, not the manifest implementation.

## Build and packaging contract

- Pin unmodified IREE at `5332c68297a68fafd596277d7686f91009ebf1cb` with its LLVM/MLIR and Torch-MLIR submodules. Use IREE's native CMake targets and static implementation with compatible ABI settings, PIC, Torch input support, and required target backends. No vendored source, patch stack, standalone SDK, or dynamic-plugin ABI.
- Every `kashi._native` build requires this compiler. Normal editable and wheel builds resolve the pinned sources and reuse a keyed native cache. Allow an explicit local pinned source/build input for offline work; incompatible inputs fail configuration rather than selecting a system LLVM/MLIR or wheel library. Build wheel artifacts in CI for Linux x86-64 and macOS arm64, the current host architectures. Linux CPU/CUDA/ROCm Pixi variants use the same host wheel; CUDA PTX generation is verified on Linux without a GPU. macOS must build/import and exercise the embedded compiler with VMVX; CUDA PTX on macOS is not part of this contract.
- Ship the private compiler inside the `kashi` platform wheel as part of `_native`, with third-party notices and any permitted non-compiler native dependencies handled by platform wheel tooling. No bundled IREE Python package, `iree-compile`, SDK archives, or shared `libIREECompiler`. Declare `iree-base-compiler==3.14.0` in Kashi's wheel metadata and lock the same dependency in Pixi. Keep the existing `iree-base-runtime` pin independent; this path does not execute IREE runtime kernels. No alternate index or direct URL is needed to install the compiler dependency. Check the actual release wheel against PyPI's upload limits; if necessary, obtain a limit increase for the `kashi` project before publishing rather than omitting compiler functionality.
- Export only the Python initializer from `_native`. Compiler and MLIR symbols bind privately. Never share native handles with the stock wheel. Input, driver and Flow cross as serialized IR; PTX and its manifest cross as artifacts. Preserve Linux loader/TLS compatibility and verify importer compatibility on pin updates.

## Acceptance

1. Port #20's partition fixtures to verify ordered SSA interfaces, mutation boundaries, dynamic live-ins/live-outs, unsupported crossings, and device normalization. On Linux CPU compile the mutating FX fixture for explicit `sm_80` without a CLI or GPU. Verify FX dependencies and pytree, and check each kernel against its manifest PTX entry, including multiple exports per file and renamed or colliding symbols. Kernels remain opaque.
2. Verify typed Torch (before input), Flow or Stream, NVVM at `executable-targets`, and VM passes on one invocation. Exercise an internal custom dialect. Compare split and uninterrupted output for a fixture without transformations. A failed pass reports its checkpoint and diagnostic, produces no graph, and leaves the next compilation unaffected.
3. Build and install repaired platform wheels into clean environments with the upstream IREE dependency and no source tree/SDK. In fresh Python processes try stock IREE first and Kashi first; import FX, run embedded compilation, and parse the Flow dump with the stock bindings. Inspect ELF/Mach-O dependencies and exports to establish that the released `_native` neither references nor exports the stock compiler's native ABI, and audit the wheel file list for overlapping `iree/` paths. Check platform tags, packaged notices, and upload size. Exercise a clean editable build and a cache hit on the same pin. Retain the existing default FX and vLLM behavior tests.

References: [IREE embedding API at the pinned revision](https://github.com/iree-org/iree/blob/5332c68297a68fafd596277d7686f91009ebf1cb/compiler/bindings/c/iree/compiler/embedding_api.h), [IREE plugin API](https://github.com/iree-org/iree/blob/5332c68297a68fafd596277d7686f91009ebf1cb/compiler/src/iree/compiler/PluginAPI/Client.h), [HAL link option](https://github.com/iree-org/iree/blob/5332c68297a68fafd596277d7686f91009ebf1cb/compiler/src/iree/compiler/Dialect/HAL/Transforms/Passes.cpp), [PyPI storage limits](https://docs.pypi.org/project-management/storage-limits/).

