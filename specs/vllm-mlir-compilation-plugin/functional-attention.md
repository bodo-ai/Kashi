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

**Rewrite table** (my call). There is a private table in `kashi.importer` keyed by the qualified op name: `"vllm::unified_attention_with_output" -> (dummy name "unified_attention", output arg "output", like arg "query")`. Both directions read it, so component tests can add a test-only entry (for example, for `kashi_test::write`) with `monkeypatch`, without vLLM.

**Dummy op.** It is registered lazily in a `torch.library.Library("kashi", "FRAGMENT")`, the first time a matching call is normalized. Its schema is derived from the original op's schema:
- drop the output argument;
- strip alias/mutation annotations;
- return `Tensor`.

Opaque arguments appear as `PyObject` in a schema and cannot be parsed back, so the derived schema names a kashi-owned opaque type (registered `typ="reference"`) in their place; the dispatcher type is still `PyObject`. This keeps `layer_name`'s opaque type (and every other argument type) identical without naming vLLM types. The fake impl returns a contiguous `empty_like(<like arg>)`. There is no real impl.

**Importer (`_normalize`).** For a call whose resolved `OpOverload` matches the table:
1. Reject (`NotImplementedError` naming the node) when:
   - any other mutated argument is not `None`;
   - the output's fake value is not contiguous, or differs from the like argument's fake value in shape or dtype. Strides of the like argument are not compared: in the vLLM graph `query` is a strided view of a `chunk`;
   - the call has users;
   - the output buffer is not fresh: following `view`s back from it must reach an `aten.empty` (a graph input, `chunk`/`getitem`, `slice` or any other producer is rejected);
   - any node that aliases the output buffer, other than the output node itself, is used after the call. "Aliases" means the buffer's base and views of it, found by walking `view`/`aten.view` producers and their users.
2. Replace the call with the dummy op, passing every argument except the output. Copy fake metadata from the output node.
3. Redirect uses of the output node that come after the call to the new node. Uses before the call are unchanged. The now-dead `empty`/`view` stay in place: no DCE, per the existing spec.

**Rebuild (`rebuild_fx`).** For a `torch.operator` whose name is a dummy in the table:
1. Resolve the original op.
2. Emit `aten.empty_like(<like operand>, memory_format=torch.contiguous_format)`.
3. Bind the operands to the original schema with the new buffer inserted at the output position.
4. Emit the call once, and map the MLIR result to the buffer node.

All other `torch.operator`s are unchanged.

**Spec update.** In `vllm-integration.md`, the non-goal "Functionalization, mutation or aliasing semantics of vLLM ops" gets an exception pointing here. The observed-MLIR example stays as the historical probe.

## Testing

- **Importer** (`test_import_fx.py`, test op via `monkeypatch`, no vLLM):
  - the output buffer's later uses read the dummy's result;
  - no zero-result `torch.operator` remains;
  - the module verifies.
  - Each reject case raises.
- **Rebuild** (`test_rebuild_fx.py`, hand-written MLIR):
  - the dummy expands to `empty_like` + the original op, once, with the right positional and keyword binding;
  - the result is the buffer.
- **Backend** (`test_backend.py`): the `OpsModel` round trip with the test entry still matches eager.
- **End-to-end** (`test_vllm_integration.py`):
  - the imported MLIR contains `torch.kashi.unified_attention` and no `unified_attention_with_output`;
  - the existing assertions (rebuilt targets in order, logprob match) still pass.

## Left to the builder

- Helper names.
- How the schema is derived (string manipulation of `str(schema)` vs `FunctionSchema` construction).
- Diagnostic wording.
- How the end-to-end test gets hold of the imported MLIR (for example, by having the spy also record `import_fx`'s result).
