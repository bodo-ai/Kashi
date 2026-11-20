# IREE → FX mapping: what had to be converted back

`mlir_roundtrip_backend` can (opt-in, `IREE_PHASE_DUMP_DIR`) split the Torch-MLIR module at the vLLM custom ops, compile the ordinary-op segments with `iree-compile`, and rebuild **one FX graph** from the custom ops plus the compiled segments. This document lists every dialect/operation for which conversion code had to be written, and what the result looks like for Qwen3-0.6B. The backend still returns the plain round-trip graph; the reassembled graph is only printed (tabular + graph form) and saved to `reassembled_fx.txt`.

Numbers below are from one run: Qwen3-0.6B, dummy weights, `--iree-cuda-target=sm_120`, 226 segments (197 non-empty, 29 empty), 225 custom ops.

## Pipeline

```
Torch-MLIR @main ──partition──► segments.mlir (ordinary ops)   ──iree-compile──► flow-phase IR + PTX kernels
        │                       driver.mlir  (@main: calls + custom ops)              │
        │                                          │                                   │
        └──────────────────────────────────────────┴──► mlir_to_fx(driver, segment_handlers) ◄── flow → FX
                                                                   ▼
                                                    one FX graph: custom ops + kernel calls
```

* The **driver** (`driver.mlir`) is the original `@main` with each segment collapsed into `call @segment_N` and every custom op and constant kept in its original position. It is what preserves the wiring between segments and custom ops. The custom ops in it are converted by the *existing* `mlir_to_fx` path, so they come back exactly as in the unpartitioned round trip.
* Each segment is read from IREE's **flow phase** (`*.6.flow.mlir`). That is the last phase that is still tensor-level SSA with the original function boundaries, and it is where IREE has decided what becomes a kernel (`flow.dispatch`). Later phases (stream/hal) are buffers and command buffers with no FX counterpart.
* Kernels are dumped at the end of the same `iree-compile` run (`--iree-hal-link-executables=false --iree-hal-dump-executable-files-to`): `kernels.ptx` (all kernels, one banner per executable) and `kernel_files/` (raw IREE dumps: PTX, LLVM IR, per-executable MLIR).

## Operations with new conversion code

Counts are operations across the 197 compiled segment functions.

### Torch side (driver)

| Dialect / op | Count | FX result | Notes |
|---|---:|---|---|
| `func.call @segment_N` | 197 | The segment's FX nodes are spliced in at the call site (`segment_handlers` hook of `mlir_to_fx`) | Without a handler the call is still "unsupported", so the plain round trip is unchanged. |
| `func.func private @segment_N` (declaration) | 197 | Ignored | Only there so `driver.mlir` verifies standalone. |

### IREE flow phase (inside each `segment_N$async`)

| Dialect / op | Count | FX conversion | Notes |
|---|---:|---|---|
| `flow.dispatch @exe::@export[workload](args)` | 197 | `call_function` on an opaque `IreeKernel` target; `args` = kernel arguments, `kwargs={"workload": [...]}` | **The kernel call.** 8 distinct kernels. Result types are kept in `node.meta["iree_result_types"]`. Several results → `operator.getitem` nodes. Tied (in-place) operands raise `NotImplementedError`. |
| `flow.tensor.reshape` | 224 | `aten.view.default(x, [dims…])` | No-copy reshape. Static dims are ints, `?` dims come from the dimension values. |
| `flow.tensor.empty` | 113 | `aten.empty.memory_format([dims…], dtype=…, device=…)` | IREE tensors have no device; the source graph's device (`cuda:0`) is restored from the partition. |
| `hal.buffer_view.dim<%bv>[i]` | 281 | `aten.sym_size.int(tensor, i)` | How IREE reads a dynamic dimension off an incoming buffer. |
| `hal.tensor.import` | 394 | identity | buffer_view → tensor; the wait fence and dim operands are metadata. |
| `hal.tensor.export` | 478 | identity | tensor → buffer_view. |
| `hal.tensor.barrier join(…) => fence` | 197 | identity on the joined values | Only orders against the signal fence. |
| `arith.index_cast` (i64 → index) | 56 | identity | Python ints are unbounded; this is how the symbolic-int live-in (`arg0`) enters dimension math. |
| `util.return` | 197 | The segment's live-out FX values | Fence results are dropped. |
| `arith.constant` (integer/index) | 0 in Qwen3 | Python int | Appears when IREE flattens a dynamic-shaped elementwise op (found while testing a small graph). |
| `arith.muli` | 0 in Qwen3 | Python int if both operands are ints, else `operator.mul` node | Flattened sizes such as `dim0 * 8`. |

Types handled implicitly: `!hal.buffer_view` (a tensor) and `!hal.fence` (synchronization only; the last two arguments of every `$async` function are dropped). IREE's `coarse-fences` ABI exports each segment as `<name>$async`; the plain `<name>` wrapper that creates the fences is ignored.

### Present in the IR but not converted (not code)

* `flow.executable` / `flow.executable.export` (8): the kernel containers. Only the export name is used, as the `IreeKernel` identity and the label of its PTX.
* `util.func @segment_N` sync wrappers (`util.call`, `hal.fence.create`, `hal.fence.await`): runtime plumbing around the `$async` function.
* `util.global` (1): device handle.

### Explicit failures

Any other op in a segment raises `NotImplementedError("No FX conversion for IREE operation '<name>' in segment_N$async")`. Tested with `arith.addi`. Other likely ones for other models: more `arith.*`/`tensor.*` dimension math, `flow.tensor.slice/update/clone/splat`, tied-operand dispatches.

## Result for Qwen3-0.6B

Original (unpartitioned round-trip) graph: 1042 `call_function` nodes. Reassembled: 1040.

| Original FX | Count | Reassembled | Count |
|---|---:|---|---:|
| `aten.embedding` + `aten.to.dtype` | 1 + 1 | `IreeKernel(…elementwise_broadcast_Dx1024_i32xbf16)` | 1 |
| `aten.linear` | 112 | `IreeKernel(…matmul_D…_bf16xbf16xf32)` (4 shapes) | 112 |
| `aten.split_with_sizes` + `getitem` | 28 + 84 | `IreeKernel(…slow_memcpy)` ×3 per layer | 84 |
| `aten.detach` | 113 | *(removed by IREE: no-op)* | 0 |
| `aten.size.default` | 113 | `aten.sym_size.int` (IREE re-reads dims at each segment entry) | 281 |
| `aten.view` | 252 | `aten.view` (some folded by IREE) | 224 |
| `aten.empty.memory_format` | 113 | `aten.empty.memory_format` | 113 |
| vLLM custom ops (`rms_norm` 57, `fused_add_rms_norm` 56, `rotary_embedding` 28, `silu_and_mul` 28, `unified_kv_cache_update` 28, `unified_attention_with_output` 28) | 225 | same ops, same call form | 225 |

197 kernel call sites, 8 distinct kernels: 1 embedding gather, 4 matmul shapes, 3 `slow_memcpy` (the QKV split). The ordering and operands of the custom ops match the original (checked on the first layer: kernel → `rms_norm` into an `empty` buffer → QKV matmul → copies → `view` → rotary / KV-cache / attention → output-proj matmul → `fused_add_rms_norm` → …).

## Semantic differences to keep in mind (the graph is not executable yet)

* **Kernels are opaque.** Calling an `IreeKernel` raises `NotImplementedError`; launching them needs IREE's runtime.
* **Views became copies where IREE copies.** `Tensor.split` returns views in torch; in the reassembled graph the pieces are separate buffers written by `slow_memcpy` kernels. Fine while nothing mutates the source afterwards; the in-place custom ops (`rotary_embedding` on q/k) now mutate the copies.
* **Views that IREE kept are still views** (`flow.tensor.reshape` → `aten.view`), so aliasing between an `empty` buffer, the custom op that fills it, and later views is preserved.
* **Device.** Segments were compiled with `cuda:0` device constants rewritten to `cpu` (the frontend cannot legalize `aten.empty` with a CUDA device); the reassembly restores `cuda:0` on `empty`. The compile target itself is CUDA.
* **Dtype mapping.** IREE integers are signless; `i8` maps to `int8`.
* **No driver** is built (and reassembly is skipped with a message) if a value crossing a segment boundary cannot be returned from a segment (`unsupported live-outs`, e.g. a scalar float). Qwen3 has none.
