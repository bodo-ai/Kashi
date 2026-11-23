# vLLM Integration: Symbolic Shapes, Opaque Inputs and Custom Operators

**Status:** Approved
**Parent:** [vLLM FX–MLIR–FX Compilation Plugin](vllm-mlir-compilation.md) §2, §4–6, §9.3, milestone M2 (architectural reference only; this document sets the scope)
**Builds on:** [Standalone TorchDynamo Backend](dynamo-backend.md), [MLIR → FX Reconstruction](mlir-fx-reconstruction.md)

## Intent

`kashi.backend.compile_fx` runs FX → MLIR → FX for small static-shape PyTorch models. This component makes it run the reference vLLM model, `python/tests/models/simple_vllm_model.py`, through the real vLLM engine with real vLLM attention. It also adds an end-to-end test showing that inference with the plugin matches inference without it. That requires three capabilities the compiler lacks today: symbolic token counts, vLLM's opaque `LayerName` graph input, and vLLM custom operators in both directions (IREE's importer emits them; `rebuild_fx` must rebuild them).

## Observed vLLM capture (probe run during design)

Environment: Pixi `vllm-mlir` (osx-arm64), PyTorch 2.14.0, IREE 3.14.0, **vLLM 0.33.0** CPU wheel. The parent spec names vLLM 0.34 on Linux x86-64. This spec targets what is installed here.

Running `SimpleVllmForCausalLM` through `vllm.LLM` with `CompilationConfig(mode=CompilationMode.DYNAMO_TRACE_ONCE, backend="<dotted path>")`:

- The backend is called **once**, and that one capture serves prefill and decode. `positions` does not appear as a graph input.
- Placeholders, in order: embed weight, `SymInt s72` (token count), `input_ids` `[s72]` int32, QKV weight, `LayerName` (meta `val` is a `FakeScriptObject`; real `example_inputs` entry is a `vllm.utils.torch_utils.LayerName`), O-proj weight.
- Body:
  - embedding, linear, `chunk`/`getitem`;
  - `torch.Size([s72, 128])` feeding `torch.empty(size, dtype=float32, device=cpu)`;
  - `view`s with `-1`;
  - `torch.ops.vllm.unified_kv_cache_update(key, value, layer_name)`, which returns a dependency tensor;
  - `torch.ops.vllm.unified_attention_with_output(query, key, value, output, layer_name, kv_cache_dummy_dep=dep)`, which returns nothing and writes `output`;
  - a final `view` and linear.
  - Output: `(x,)`.
- Custom-op targets are `OpOverloadPacket`s, not `OpOverload`s.

The design was validated with probe-only patches to the importer:
- symbolic `torch.Size` folded to a list;
- single-overload packets resolved to `.default`;
- `FakeScriptObject` typed as `!torch.any`.

With those patches the module verified, and inference completed when the probe backend returned the original `gm.forward`. IREE then emits:

```mlir
func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.vtensor<[?],si32>,
                %arg3: !torch.vtensor<[384,128],f32>, %arg4: !torch.any, %arg5: !torch.vtensor<[128,128],f32>) -> ...
  %4 = torch.prim.ListConstruct %arg1, %int128 : (!torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.aten.empty.memory_format %4, ...
  %14 = torch.operator "torch.vllm.unified_kv_cache_update"(%11, %13, %arg4) : (...,  !torch.any) -> !torch.vtensor<[0],f32>
  torch.operator "torch.vllm.unified_attention_with_output"(%7, %11, %13, %9, %arg4, %none, %none, %14) : (...) -> ()
```

Environment facts the tests must account for:
- vLLM 0.33 enables ahead-of-time compilation by default on PyTorch ≥ 2.10, and that path rejects a backend result that is not a `SerializableCallable`.
- CPU attention rejects head size 16 (64 works).
- The default KV-cache reservation does not fit this machine (`VLLM_CPU_KVCACHE_SPACE=1` does).
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the engine in-process, so a test-local backend can record its calls.

## Scope

### Goals

- Import and rebuild graphs with `SymInt` graph inputs and symbolic `torch.Size`, and reuse one compiled callable across several values of the symbolic dimension.
- Import and rebuild graphs whose inputs include hoisted opaque objects (`FakeScriptObject` metadata), passing the live object through on every call.
- Import custom-operator calls given as overload packets, and rebuild `torch.operator` ops, including ops with no results, through the PyTorch dispatcher.
- Run `SimpleVllmForCausalLM` through vLLM with `compile_fx` and match vLLM without compilation.

### Non-goals

- Functionalization, mutation or aliasing semantics of vLLM ops. The rebuilt graph keeps SSA references, so `output` written by attention is the same FX node later viewed. Nothing models the write. Exception: `unified_attention_with_output` is represented functionally in the imported MLIR; see [Functional Attention](functional-attention.md).
- Serializable callables / vLLM AOT compile (tests disable it).
- `STOCK_TORCH_COMPILE` and `VLLM_COMPILE` modes; Inductor.
- The `inputs_embeds` branch.
- Shape arithmetic beyond plain `SymInt` inputs (`sym_size`, `operator.mul` on symbols, etc.).
- Multi-result custom ops; overload packets with more than one overload.
- Packaging entry points and vLLM 0.34 / Linux.
- CI (chosen by the user): these tests run locally with `pixi run -e vllm-mlir pytest`. CI's default environment has no torch, IREE or vLLM and skips them, as it does today's torch tests.
- Making `rebuild_fx` independent of IREE's `TORCH_DTYPE_TO_INT`.

### Anti-goals (failure even if tests pass)

- Replacing, skipping or stubbing vLLM attention or KV-cache ops, or returning the original `gm`.
- Specializing a symbolic dimension to its example value (e.g. emitting `!torch.vtensor<[5],…>` or a `torch.constant.int 5` where the graph had `s72`).
- Capturing `example_inputs` objects (the `LayerName`, weights) into the rebuilt graph instead of taking them as runtime inputs.
- Process-global patching of IREE or PyTorch.

## Approaches considered

| Approach | Trade-off | Decision |
| --- | --- | --- |
| **Extend `import_fx` / `rebuild_fx` in place.** New normalization rules, an opaque-type hook and new structural/`torch.operator` handling in `rebuild.py`; `compile_fx` unchanged. | Reuses the schema-driven binder for custom ops. Every change is unit-testable without vLLM. | **Chosen.** |
| Separate vLLM adapter module wrapping the importer and rebuild. | Duplicates schema binding and splits one translation across two places. | Rejected. |

**Opaque input type** (chosen by the user):

| Option | Trade-off | Decision |
| --- | --- | --- |
| **Plain `!torch.any` block argument.** | What verified in the probe; rebuild needs only a placeholder. Loses the class name in the IR. | **Chosen.** |
| `!torch.any` plus an argument attribute naming the class. | Better diagnostics; more code on both sides with no current reader. | Rejected for now. |
| Torch-MLIR TorchScript class types (`!torch.nn.Module<…>` etc.). | TorchScript semantics; IREE's FX importer has no path producing them. | Rejected. |

**vLLM AOT compile** (chosen by the user): tests set `VLLM_USE_AOT_COMPILE=0`. Making the result serializable is deferred. With AOT enabled, vLLM raises an error rather than falling back silently.

## Design

### Importer (`python/src/kashi/importer.py`)

1. **Symbolic `torch.Size`.** `_fold_size` accepts elements that are `int` or FX nodes whose `meta["val"]` is a `torch.SymInt`, and replaces uses with the list as today. Other element kinds still raise `NotImplementedError`.
2. **Overload packets.** In `_rewrite`, a `call_function` whose target is a `torch._ops.OpOverloadPacket` with exactly one overload is retargeted to that `OpOverload` (structural copy only). Zero or several overloads raise `NotImplementedError` naming the packet and its overloads. ATen packets are covered by the same rule.
3. **Opaque objects.** A node whose `meta["val"]` is a `torch._library.fake_class_registry.FakeScriptObject` imports as `!torch.any`. Implement this as a per-import hook on the `FxImporter` instance, for example a `ContextCache` subclass installed on that importer, with any private-attribute access isolated in one small adapter function. Never patch IREE classes or modules globally. Only placeholders are expected to carry opaque values. An opaque value produced by an operation is out of scope, and the importer should not go out of its way to support it.
4. `SymInt` placeholders already import as `!torch.int`; no change beyond not rejecting them.

### Reconstruction (`python/src/kashi/rebuild.py`)

1. **Block arguments.** Accept `!torch.vtensor`, `!torch.int` and `!torch.any`; each becomes a placeholder in order. Other types are rejected as today.
2. **`torch.operator`.** Read the `name` string attribute (`torch.<ns>.<op>[.<overload>]`). Resolve `torch.ops.<ns>.<op>.<overload or "default">`, generalizing `_resolve_aten` to any namespace (ATen keeps its existing path and error messages). Then bind operands through the existing schema binder (`_decode`, positional vs `kwarg_only`).
   - Unknown namespace, op or overload, or an operand-count mismatch: raise the existing `_error` form.
   - Ops with zero results: emit the `call_function` node once and record no result value. The node has no users, and FX must not drop it (no DCE runs).
   - Ops with one result map it as today. More than one result is rejected.
3. Registered `torch.aten.*` ops are unchanged.

### Backend

`kashi.backend.compile_fx` is unchanged. vLLM resolves it from the dotted string `"kashi.backend.compile_fx"`. Tests point vLLM at a test-local wrapper instead (below).

## Testing

All tests run in the `vllm-mlir` Pixi environment. Component tests must not import vLLM.

### Component: importer (`python/tests/test_import_fx.py`)

- Symbolic `torch.Size` feeding `torch.empty` imports as `prim.ListConstruct` of a `!torch.int` block argument and a constant. The result type has `?` in that dimension.
- A single-overload custom op given as a packet imports as `torch.operator "torch.<ns>.<op>"`. A packet with several overloads raises `NotImplementedError` that names it.
- A graph with an opaque placeholder imports with a `!torch.any` argument passed to the custom op.

Fixtures: a small test-only custom op library registered with `torch.library` (e.g. namespace `kashi_test`), with:
- an op taking a tensor and a test opaque class registered the way vLLM registers `LayerName` (`register_opaque_type(..., typ="value", hoist=True)` or its non-deprecated equivalent);
- a zero-result op that writes an output tensor;
- a single-result op feeding it through a keyword-only argument.

Graphs come from `torch.compile` with a capturing backend so they carry real Dynamo metadata.

### Component: reconstruction (`python/tests/test_rebuild_fx.py`)

Hand-written MLIR (not from `import_fx`):
- `!torch.int` and `!torch.any` block arguments become placeholders, and `!torch.int` flows into a `ListConstruct` size.
- `torch.operator` resolves to the test op's `OpOverload`, with positional and keyword-only arguments bound by schema.
- A zero-result `torch.operator` appears exactly once in the graph.
- An unknown namespace/op/overload and an unsupported block-argument type each raise `NotImplementedError` with the op location.

### Round trip / backend (`python/tests/test_backend.py`)

Using the existing `Spy`:
- **Symbolic reuse:** `torch.compile(model, backend=spy)` with `torch._dynamo.mark_dynamic(x, 0)`, called with two different sizes of that dimension. The spy is called exactly once, and both results match eager. The probe confirmed that a model with embedding → linear → chunk → `view(-1, …)` plus `torch.empty(ids.shape[0], 8)` captures as a plain `SymInt` input with no shape arithmetic.
- **Custom ops and opaque input:** a model calling the test ops with a hoisted test opaque object. Spy called once, results match eager, and the zero-result op's side effect on its output buffer is visible in the result.

### End-to-end (`python/tests/test_vllm_integration.py`)

- Skip when `vllm` is not importable.
- Register `SimpleVllmForCausalLM` with `ModelRegistry`.
- Use a local Llama-type `config.json` with `hidden_size=128`, `num_attention_heads=2`, `num_key_value_heads=2`, `vocab_size=128` and `max_position_embeddings=256`. This gives a head size of 64.
- Engine arguments: `skip_tokenizer_init=True`, `load_format="dummy"`, `dtype="float32"`, `max_model_len=128`.
- Environment: `VLLM_USE_AOT_COMPILE=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_CPU_KVCACHE_SPACE=1`.

**Weights** (chosen by the user):
- `load_format="dummy"` fills float weights uniformly in ±1e-3, and vLLM 0.33 does not let the range be configured. Through this model that yields logits of about 1e-10, and all logprobs collapse to the same float32 value (about −ln 128). A comparison could then not detect broken attention.
- So after each engine loads, the test overwrites every model parameter in place with the same seeded values via `LLM.apply_model`. Use a deterministic random fill, not a constant, so attention is non-trivial. Scale it so the logits are clearly non-uniform, e.g. normal with std around 0.1–0.5; the builder picks and asserts it.
- Weights reach the backend as runtime graph inputs (see the parameter-update test in `test_backend.py`), so the compiled graph uses the new values. The model fixture is not modified.
- The test asserts that the reference run's top-1 vs top-2 logprob gap at each step is well above tolerance, so the comparison can actually distinguish outputs.

Two runs with identical config, seed, weights and prompts:
- **Plugin:** `CompilationConfig(mode=CompilationMode.DYNAMO_TRACE_ONCE, backend="<test module>.spy_compile_fx")`, where the test-local wrapper calls `compile_fx` and records calls.
- **Reference:** `CompilationMode.NONE`. On CPU, vLLM routes attention through the same `torch.ops.vllm.*` ops in both runs.

Prompts: at least two token-ID prompts of different lengths (e.g. 5 and 9 tokens), with `SamplingParams(max_tokens=4, temperature=0, logprobs=5)`. This exercises prefill at two sizes and decode.

Assertions:
1. The wrapper was called exactly once, and its returned callable is not the received `gm.forward`.
2. The graph it received contains `torch.ops.vllm.unified_kv_cache_update` and `torch.ops.vllm.unified_attention_with_output` targets. Its rebuilt graph contains the corresponding `OpOverload` targets, in the same order.
3. Generated token IDs are identical, and per-step logprobs for the chosen tokens match within `torch.testing.assert_close` float32 defaults. Both runs execute the same eager kernels in the same order, so a looser tolerance needs a stated reason.

## Left to the builder

- Exact helper names and the mechanics of the opaque-type hook.
- Test op names, shapes and seeds.
- Whether the two vLLM runs share a process or use subprocesses. Choose whatever reliably frees the first engine; if subprocesses are used, the call record must still be observable.
- How the test-local wrapper module is made importable by dotted name.
- Diagnostic wording beyond naming the node/op and location.
