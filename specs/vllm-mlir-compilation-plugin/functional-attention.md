# Functional Attention in Imported MLIR

**Status:** Approved
**Builds on:** [vLLM Integration](vllm-integration.md) (lifts its "nothing models the write" non-goal for attention only)

## Intent

`import_fx` emits `torch.vllm.unified_attention_with_output` as a zero-result `torch.operator` that writes into its `output` operand. The operand is a `!torch.vtensor`, so the IR is wrong: as far as value semantics go, the later reads of `output` see the `empty` it was allocated from. The goal is for the imported Torch-dialect MLIR to say what actually happens. Attention produces a new tensor, and later code reads that tensor. Execution must not change: `rebuild_fx` turns the functional form back into the real vLLM op.

## Goals

- In the imported MLIR, every `vllm::unified_attention_with_output` call becomes a call to a functional dummy op, `torch.operator "torch.kashi.unified_attention"`, that returns the attention result. Every later use of the `output` buffer reads that result instead.
- `rebuild_fx` expands the dummy back into a contiguous `empty_like(query)` + `torch.ops.vllm.unified_attention_with_output.default(...)` and uses the buffer as the result. vLLM's real kernel runs, and the end-to-end test still matches uncompiled vLLM.

## Non-goals

- Functionalizing any other mutating op, including `unified_kv_cache_update` (it already returns its dependency tensor, and its KV-cache side effect is not on a graph value).
- Supporting a non-`None` `output_scale` or `output_block_scale` (quantized output).
- A post-import MLIR rewrite pass.
- A real kernel for the dummy op.

## Anti-goals (failure even if tests pass)

- Dropping, stubbing or reordering the real attention call at runtime.
- Silently rewriting when something else could observe the write: rejecting the rewrite is correct, a wrong IR is not.
- Requiring `kashi` to import vLLM (the importer matches the op by its qualified name).
- Patching vLLM, IREE or PyTorch globally. Defining new ops in kashi's own `kashi` library namespace is allowed.

## Approaches

| Approach | Trade-off | Decision |
| --- | --- | --- |
| **Rewrite during FX normalization (`_normalize`), before IREE imports.** | Follows the existing rewrite pattern on the structural copy. The result type comes from fake-tensor metadata. The resulting MLIR is the same as with B. | **Chosen by the user.** |
| Rewrite the imported MLIR after import. | Operates on the "imported torch-mlir" directly, but needs hand-built result types and ordering-aware use replacement through the Python bindings. No post-import pass exists today. | Rejected. |

## Design

Everything is specialized to `vllm::unified_attention_with_output` (vLLM 0.33 schema); a generic mechanism for other destination-passing ops is deferred until a second one appears.

**Dummy op** (`kashi.importer`). `kashi::unified_attention(Tensor query, Tensor key, Tensor value, <opaque> layer_name, Tensor? kv_cache_dummy_dep=None) -> Tensor`, defined when the module loads. Opaque arguments appear as `PyObject` in a schema and cannot be parsed back, so the schema names a kashi-owned opaque type (registered `typ="reference"`); the dispatcher type is still `PyObject`, as in vLLM's op. `output_scale`/`output_block_scale` are omitted because only `None` is supported. The fake impl returns a contiguous `empty_like(query)`; there is no real kernel.

**Importer (`_normalize`).** A call whose `OpOverload` schema name is `vllm::unified_attention_with_output` (matched by name, so `kashi` does not import vLLM):
1. Is rejected (`NotImplementedError` naming the node) when:
   - `output_scale` or `output_block_scale` is not `None`;
   - `output`'s fake value is not contiguous, or differs from `query`'s in shape or dtype (strides of `query` are not compared: in the vLLM graph it is a strided view of a `chunk`);
   - `output` is not fresh: following `aten.view`s back from it must reach `aten.empty` (a graph input, `chunk`/`getitem`, `slice` or any other producer is rejected);
   - any other node computed from the buffer before the call is used after it.
2. Is replaced by `kashi.unified_attention(query, key, value, layer_name, kv_cache_dummy_dep=...)`, with `output`'s fake metadata.
3. Uses of `output` after the call are redirected to the new node; uses before it are unchanged. The now-dead `empty`/`view` stay in place: no DCE, per the existing spec.

**Rebuild (`rebuild_fx`).** `torch.operator "torch.kashi.unified_attention"` (exactly 5 operands and 1 result) becomes `aten.empty_like(query, memory_format=torch.contiguous_format)` followed by `torch.ops.vllm.unified_attention_with_output.default(query, key, value, buffer, layer_name, kv_cache_dummy_dep=dep)`; the MLIR result maps to the buffer. All other `torch.operator`s are unchanged.

**Spec update.** In `vllm-integration.md`, the non-goal "Functionalization, mutation or aliasing semantics of vLLM ops" gets an exception pointing here. The observed-MLIR example stays as the historical probe.

## Testing

- **Importer** (`test_import_fx.py`): graphs calling the real vLLM ops are captured with Dynamo without running them (real attention needs vLLM's forward context).
  - The output buffer's later uses read the dummy's result, no `unified_attention_with_output` remains, and the module verifies.
  - Chained layers, and `output` passed as `kv_cache_dummy_dep`, import correctly.
  - Each reject case raises.
- **Rebuild** (`test_mlir_to_fx.py`, hand-written MLIR): the dummy expands to `empty_like` + the vLLM op, once, with the right binding; the result is the buffer; a dummy without a result is rejected.
- **End-to-end** (`test_vllm_integration.py`): the imported MLIR contains `torch.kashi.unified_attention` and no `unified_attention_with_output`; execution still matches uncompiled vLLM.

## Left to the builder

- Helper names and diagnostic wording.
