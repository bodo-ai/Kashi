# Optional custom-op decomposition for the IREE backend

## Intent

`scripts/compile_qwen3_iree.py` sends vLLM's Dynamo graph through
`kashi.iree_backend.compile_iree_fx`. Each vLLM custom op is imported as an
opaque `torch.operator`, and `partition_at_custom_ops` starts a new IREE
segment at each one. This change adds an opt-in mode that rewrites the simple
custom ops into ATen ops before MLIR import, so IREE can see and compile their
math.

Every MLIR op produced from one decomposed call carries metadata naming the
original custom op and the specific call it came from. Nothing reads this
metadata yet. It exists so that later IREE passes can still tell "these ops
used to be one `_C.rms_norm`".

## Goals

- An opt-in switch, off by default:
  - `compile_iree_fx(..., decompose_custom_ops=False)`
- Decompose exactly these ops, as they appear in Qwen3-0.6B:

  | Custom op | Schema (mutated args marked `!`) | Replacement |
  |---|---|---|
  | `_C.rms_norm` | `(result!, input, weight?, epsilon)` | `aten.rms_norm(input, [input.shape[-1]], weight, epsilon)` |
  | `_C.fused_add_rms_norm` | `(input!, residual!, weight?, epsilon)` | `residual' = input + residual`; `input' = aten.rms_norm(residual', ...)` |
  | `_C.rotary_embedding` | `(positions, query!, key!?, head_size, cos_sin_cache, is_neox, rope_dim_offset=0, inverse=False)` | ATen gather/chunk/mul/add/cat (NeoX-style rotation) |
  | `_C.silu_and_mul` | `(result!, input)` | `silu(input[..., :d]) * input[..., d:]` |

- Tag every op from a decomposition with its origin, so the tag survives
  into IREE's later IR.

## Non-goals

- `vllm.unified_kv_cache_update` and `vllm.unified_attention_with_output`.
  These stay as custom ops, and so stay as partition boundaries.
- Any IREE pass that reads the tags.
- A general or pluggable decomposition registry. A small dict keyed on the
  four `OpOverload`s is enough.
- `_C.rotary_embedding` variants Qwen3 doesn't use: `is_neox=False`,
  `rope_dim_offset != 0`, `inverse=True`, `key=None`, or partial rotary
  (`cos_sin_cache.shape[-1] != head_size`). These raise `NotImplementedError`.

## Anti-goals

- Silent miscompilation. These ops write into their arguments, and the
  decomposition handles that by rerouting later reads to new values (see
  below). If a write would be visible through some other alias, the rewrite
  raises `NotImplementedError` instead of guessing. Examples of such aliases:
  a graph input, or the base or a sibling view of the mutated tensor read after
  the op.
- Changing behavior when the flag is off. Graphs and MLIR must be
  byte-identical to today's.

## Approaches considered

1. **Rewrite in FX during `importer._normalize`, and tag via MLIR locations
   (chosen).** `_normalize` already rewrites Dynamo nodes into ATen, so FX's
   helpers handle the rewriting work: node replacement, `meta["val"]`
   through FakeTensor propagation, and use lists. Tagged nodes get a
   `NameLoc` from the existing `_OpaqueContextCache` by overriding
   `get_node_location`.
2. **Rewrite `torch.operator` in MLIR after import.** This would put
   the decomposition next to the partitioner. But it means hand-building
   Torch-dialect ops and their result types in Python bindings, which costs
   more code for the same result.
3. **Retrace with vLLM's `forward_native` implementations.** This reuses
   vLLM's reference math, but needs a second trace of the whole model. That
   changes graph structure beyond these four ops and couples kashi to vLLM
   layer internals.

Tag carrier: a discardable attribute such as `kashi.decomposed_from` was
rejected. Torch-to-linalg conversion drops attributes on ops it rewrites.
Locations are carried through conversion, and IREE merges them into
`FusedLoc`s on dispatch regions, so a location is the carrier with the best
chance of reaching later passes. This is best effort, and nothing tests that
the tag survives into flow IR.

## Design

### Switch

`compile_iree_fx(gm, *, ..., decompose_custom_ops: bool = False)` passes the
flag to `import_fx(gm, *, decompose_custom_ops=False)`, which passes it to
`_normalize`.

### Rewrite (FX, inside `_normalize`)

When the flag is on, a decomposition step runs over the copied graph *after*
the existing per-node `_normalize_meta`/`_rewrite` loop. By then every input
is an ATen node with `meta["val"]`, and custom-op packets have been resolved
to `.default`. The decomposition emits only ATen `OpOverload`s, each with its
own `meta["val"]`, so no second normalization pass is needed.

For each matching `.default` node:

1. Emit the replacement ATen nodes immediately before it, with `meta["val"]`
   filled in by running each op on the FakeTensor inputs.
2. **Mutation:** for each mutated argument `m` with new value `m'`, replace
   every use of `m` that comes *after* the custom op with `m'`. Earlier uses
   are left alone. This matches the Qwen3 graph, where the mutated tensors are
   fresh `empty` buffers or views that are only read through that same node
   afterwards. For example, `rotary_embedding` mutates `q_1`/`k_1`, and later
   reads are `q_1.view(...)` and `k_1.view(...)`.
3. **Alias guard:** follow `m`'s chain of `view`/`reshape` producers back to
   its base. If the base is a placeholder, or the base or any other node
   derived from it is read after the custom op, raise `NotImplementedError`.
4. Erase the custom op node. These ops return `()`, so it has no users.

`rotary_embedding` math, NeoX style with `rotary_dim == head_size`:
- `cos, sin = cos_sin_cache[positions].chunk(2, -1)`, each unsqueezed to
  broadcast over heads.
- `q` is viewed as `[tokens, heads, head_size]` and split into halves
  `x1, x2`, giving `cat(x1*cos - x2*sin, x2*cos + x1*sin)`. It is then viewed
  back to the original shape.
- The same steps apply to `key`.

### Tag

Each decomposed call gets an instance id: a counter per op name, in graph
order. Every emitted node gets
`node.meta["kashi_decomposed_from"] = "<op>#<n>"`, for example
`"_C.rms_norm#2"` or `"_C.fused_add_rms_norm#0"`.

`_OpaqueContextCache.get_node_location` is overridden to wrap the base
location. When that key is present it returns
`Location.name("kashi.decomposed:<op>#<n>", childLoc=<original loc or unknown>)`,
so the stack-trace location is kept inside the tag.

`partitioned.mlir` is the file `iree-compile` reads. When the flag is on,
`compile_iree_fx` writes it, and `driver.mlir`, with
`get_asm(enable_debug_info=True)`; otherwise the tags would be stripped before
IREE sees them. With the flag off, both files are still written with plain
`str()`.

Any constants FxImporter makes for a tagged node may or may not get that
node's location. That's fine; only real compute ops need to carry the tag.

## Testing

- Unit tests in `python/tests/test_import_fx.py`, CPU only. vLLM's `_C`
  schemas load without a GPU. For each of the four ops, build a small FX
  graph shaped like the Qwen3 trace (an `empty` buffer, the custom op, then a
  later read), and check:
  - The normalized graph contains no `_C.*` node and runs eagerly to match a
    plain-PyTorch reference implementation.
  - The imported MLIR, printed with debug info, contains
    `kashi.decomposed:_C.<op>#<n>` locations, with distinct ids for two calls
    to the same op.
- Alias guard: a graph that reads the base of a mutated view after the op
  raises `NotImplementedError`.
- Flag off: existing tests pass unchanged.

## Builder's calls (not decided with the user)

- Where the decomposition code lives: a new `kashi/decompose.py`, or a section
  of `importer.py`.
- Exact `NameLoc` string format, as long as it holds the op name and the
  instance id.

Verified while planning: an imported `aten.rms_norm` compiles to flow with
`iree-compile --iree-hal-target-device=cuda`, so no manual fallback is
needed. vLLM's `_C` ops have FakeTensor kernels but no CPU kernels, so the
CPU tests compare against small reference formulas, and only a CUDA-gated
test compares against the real kernels.
