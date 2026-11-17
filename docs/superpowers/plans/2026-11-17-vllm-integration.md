# vLLM Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kashi.backend.compile_fx` run `SimpleVllmForCausalLM` through real vLLM by adding symbolic `SymInt` inputs, opaque (`LayerName`-style) inputs and custom-operator support to `import_fx` and `rebuild_fx`, plus an end-to-end test against uncompiled vLLM.

**Architecture:** Extend the existing importer (normalization rules + a per-importer opaque-type hook) and reconstruction (new block-argument types + `torch.operator`) in place; `compile_fx` is unchanged. A test-only custom-op library mirrors vLLM's op shapes so component tests never import vLLM.

**Tech Stack:** Python 3.14, PyTorch 2.14.0, IREE 3.14.0 Python bindings, vLLM 0.33.0 CPU, pytest; Pixi environment `vllm-mlir`.

**Spec:** `specs/vllm-mlir-compilation-plugin/vllm-integration.md`

## Global Constraints

- All test commands: `pixi run -e vllm-mlir pytest ...`. Local only; no CI changes.
- Component tests (`test_import_fx.py`, `test_rebuild_fx.py`, `test_backend.py`) must not import vLLM.
- Opaque values import as plain `!torch.any`; no argument attributes.
- No process-global patching of IREE or PyTorch; the opaque-type hook is per `FxImporter` instance.
- Never stub/replace vLLM ops, return the original `gm`, specialize a symbolic dimension to its example value, or capture `example_inputs` objects into the rebuilt graph.
- Out of scope: shape arithmetic beyond plain `SymInt` inputs, multi-overload packets (reject), multi-result custom ops (reject), `inputs_embeds`, stock/VLLM_COMPILE modes, serializable callables, mutation semantics.
- vLLM test env: `VLLM_USE_AOT_COMPILE=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_CPU_KVCACHE_SPACE=1`.

## Review Focus

- Two opaque inputs of the same class with different values (vLLM with several layers passes one `LayerName` per layer): each op must receive its own object, not the first one — pinned in Task 2 (`test_custom_ops_with_opaque_inputs` uses two tags with different factors).
- An ATen op reaching the importer as a multi-overload packet (e.g. `torch.ops.aten.add`) must fail naming the packet, not pick `.default` — pinned in Task 1 (`test_import_fx_rejects_multi_overload_packet`).
- A stale vLLM compile cache from an earlier run must not make "backend called exactly once" pass or fail spuriously — pinned in Task 3 (`VLLM_DISABLE_COMPILE_CACHE=1`).
- Prefill of several prompts batched into one step plus single-token decode: one capture must serve all token counts — covered by Task 3's two prompts generated together.
- The opaque-type hook must not leak into later imports in the same process — pinned in Task 1 (`test_import_fx_opaque_hook_is_not_global`).

---

### Task 1: Test custom-op library and importer support

**Files:**
- Create: `python/tests/kashi_test_ops.py`
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`: add `pythonpath = ["python/tests"]`)
- Modify: `python/src/kashi/importer.py` (`import_fx`, `_rewrite`, `_fold_size`; new opaque hook)
- Test: `python/tests/test_import_fx.py`

**Interfaces:**
- Produces (test helper `kashi_test_ops`, imported as `import kashi_test_ops` after `pytest.importorskip("torch")`):
  - `class Tag(torch._opaque_base.OpaqueBase)` with `factor: float`, `__eq__`/`__hash__` on `factor`, `__fx_repr__`; registered via `torch._library.opaque_object.register_custom_class(Tag, typ="constant", hoist=True)`.
  - Ops in namespace `kashi_test`, defined with `torch.library.Library("kashi_test", "FRAGMENT")` + `define` (not `custom_op`, which rejects kwarg-only tensors), each with a CPU/`CompositeExplicitAutograd` impl and `register_fake`:
    - `scale(Tensor x, <Tag> tag) -> Tensor` — returns `x * tag.factor`.
    - `write(Tensor x, Tensor(a!) out, <Tag> tag, *, Tensor dep) -> ()` — `out.copy_(x + tag.factor)`.
    - `<Tag>` is `get_opaque_type_name(Tag)`; the dispatcher schema shows it as `PyObject`.
  - `class OpsModel(nn.Module)`: `__init__(self, a: Tag, b: Tag)`; `forward(x)`: `d = torch.ops.kashi_test.scale(x, self.a)`; `out = torch.empty(torch.Size([x.shape[0], 4]))`; `torch.ops.kashi_test.write(x, out, self.b, dep=d)`; return `(out.view(-1), d)`. Inputs are `[n, 4]` float.
  - `def capture(model, *inputs) -> tuple[torch.fx.GraphModule, list]`: `torch._dynamo.reset()`, `torch.compile(model, backend=..., fullgraph=True)` with a backend that records `(gm, example_inputs)` and returns `gm.forward`; calls it once; returns the record. Callers `mark_dynamic` inputs themselves.
- Produces (importer): `import_fx` additionally accepts `SymInt` placeholders, symbolic `torch.Size`, single-overload `OpOverloadPacket` targets and `FakeScriptObject` placeholders.

- [ ] **Step 1: Write `kashi_test_ops.py` and the pytest `pythonpath` setting** (test scaffolding; no assertions of its own).

- [ ] **Step 2: Write the failing importer tests** in `test_import_fx.py`

Common setup: `x = torch.randn(3, 4); torch._dynamo.mark_dynamic(x, 0); gm, ex = kashi_test_ops.capture(OpsModel(Tag(2.0), Tag(3.0)), x)`.

- `test_import_fx_symbolic_size_feeding_empty`: precondition — `gm` contains a `call_function` with target `torch.Size`. `text = str(import_fx(gm, ex).module)`:
  - `func.func @main` signature contains `!torch.int` and `!torch.vtensor<[?,4],f32>`;
  - a `torch.prim.ListConstruct %argN, %int4` where `%argN` is the `!torch.int` argument;
  - `torch.aten.empty.memory_format` result type `!torch.vtensor<[?,4],f32>`;
  - no `torch.constant.int 3` anywhere (not specialized).
- `test_import_fx_custom_ops_and_opaque_inputs`: signature has exactly two `!torch.any` arguments; text contains `torch.operator "torch.kashi_test.scale"` and `torch.operator "torch.kashi_test.write"`, the latter with `-> ()`; the module verifies.
- `test_import_fx_rejects_multi_overload_packet`: hand-built FX graph (as existing tests do, `meta["val"]` set by running an interpreter) with `call_function` target `torch.ops.aten.add` (packet) on two `[2]` tensors. `pytest.raises(NotImplementedError, match="Ambiguous overload packet.*aten.add")` (today's generic "Unsupported FX node" message already mentions `aten.add`, so the match must include the new wording).
- `test_import_fx_opaque_hook_is_not_global`: record `fx_importer.ContextCache.value_info_to_type` (from `iree.compiler.extras.fx_importer`) before and after `import_fx(gm, ex)`; assert identity unchanged.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pixi run -e vllm-mlir pytest python/tests/test_import_fx.py -v`
Expected: the three new import tests FAIL (`torch.Size with non-constant dimensions`; `Unsupported FX node ... OpOverloadPacket`; message does not match `Ambiguous overload packet`). `..._not_global` already passes; it guards the implementation, not the RED step.

- [ ] **Step 4: Implement in `importer.py`**

- `_fold_size(node)`: accept dims that are `int` or `torch.fx.Node` with `isinstance(n.meta.get("val"), torch.SymInt)`; otherwise keep raising `NotImplementedError`.
- `_rewrite`: before the existing checks, a `call_function` whose target is `torch._ops.OpOverloadPacket`: if `len(target.overloads()) == 1`, set `node.target = getattr(target, target.overloads()[0])` and return; otherwise raise `NotImplementedError(f"Ambiguous overload packet {target} (overloads {target.overloads()}): {node.format_node()}")`.
- Opaque hook: a `ContextCache` subclass whose `value_info_to_type` returns `IrType.parse("!torch.any", self._c)` for `FakeScriptObject` values and defers otherwise; install it on the one `FxImporter` created in `import_fx`, in one adapter function (it sets the private `_cc`, constructed like IREE does: `(importer._c, py_attr_tracker=importer._py_attr_tracker)`). Nothing global.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run -e vllm-mlir pytest python/tests/test_import_fx.py -v`
Expected: all PASS.

- [ ] **Step 6: Full suite and commit**

Run: `pixi run -e vllm-mlir pytest python/tests` — Expected: all pass.

```bash
git add pyproject.toml python/tests/kashi_test_ops.py python/tests/test_import_fx.py python/src/kashi/importer.py
git commit -m "import symbolic sizes, single-overload packets and opaque inputs"
```

### Task 2: Reconstruction of scalar/opaque arguments and `torch.operator`, plus backend round trips

**Files:**
- Modify: `python/src/kashi/rebuild.py` (`rebuild_fx` block-argument check and op dispatch; generalize `_resolve_aten`; binder)
- Test: `python/tests/test_rebuild_fx.py`, `python/tests/test_backend.py`

**Interfaces:**
- Consumes: `kashi_test_ops` (`Tag`, ops, `OpsModel`) from Task 1; `import_fx` support from Task 1.
- Produces: `rebuild_fx` accepts `!torch.int`/`!torch.any` block arguments and `torch.operator` ops. No signature changes.

- [ ] **Step 1: Write the failing reconstruction tests** in `test_rebuild_fx.py` (hand-written MLIR via `_parse`, `import kashi_test_ops` for registration)

Fixture `CUSTOM` MLIR: `@main(%arg0: !torch.int, %arg1: !torch.vtensor<[?,4],f32>, %arg2: !torch.any, %arg3: !torch.any)`; `%0 = torch.operator "torch.kashi_test.scale"(%arg1, %arg2)`; `ListConstruct %arg0, %int4`; `torch.aten.empty.memory_format` (dtype `%int6`, device `"cpu"`, rest none); `torch.operator "torch.kashi_test.write"(%arg1, %empty, %arg3, %0) : (...) -> ()`; return `%empty, %0`.

- `test_rebuild_scalar_and_opaque_arguments`: 4 placeholders in order; the `empty` node's size argument is `[placeholder0, 4]`.
- `test_rebuild_custom_operator`: `_targets(gm) == [kashi_test.scale.default, aten.empty.memory_format, kashi_test.write.default]`; the write node has `args == (x, empty, tag_b)` and `kwargs == {"dep": scale_node}`.
- `test_rebuild_zero_result_operator_runs_once`: exactly one write node; `gm(3, x, Tag(2.0), Tag(3.0))` with `x = randn(3, 4)` returns `(x + 3.0, x * 2.0)`.
- `test_rebuild_rejects` extended (or new parametrized test) with: `torch.operator "torch.kashi_test.missing"`, `"torch.no_such_ns.op"`, `"torch.kashi_test.scale.bogus"` → `NotImplementedError` matching `"Unsupported MLIR op torch.operator"` and the op name; a `!torch.float` block argument → existing "Unsupported block argument" message.

- [ ] **Step 2: Write the failing backend tests** in `test_backend.py` (reuse `Spy`; these tests set dynamism with `mark_dynamic`, not `dynamic=False`)

- `test_symbolic_dimension_reuses_compiled_graph`: model = embedding(10, 8) → linear(8, 24) → `chunk(3, dim=-1)` → `q.view(-1, 2, 4)`, plus `torch.empty(ids.shape[0], 8)`; return `(view, empty)`. `ids = tensor([1, 2, 3])`, `mark_dynamic(ids, 0)`; call with it and with `tensor([1, 2, 3, 4, 5])`. Spy called once; leaf 0 `assert_close` to eager for both sizes; leaf 1 shape/dtype only.
- `test_custom_ops_with_opaque_inputs`: `OpsModel(Tag(2.0), Tag(3.0))`, `x = randn(3, 4)` with `mark_dynamic(x, 0)`, then also `randn(5, 4)`. Spy called once; results equal eager `(x + 3.0).view(-1)` and `x * 2.0`; spy's `gm` has `OpOverloadPacket` targets for both ops and the rebuilt `fn.__self__` has the `.default` `OpOverload`s in the same order.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pixi run -e vllm-mlir pytest python/tests/test_rebuild_fx.py python/tests/test_backend.py -v`
Expected: new tests FAIL with `Unsupported block argument ... !torch.int` (rebuild and backend) and `unsupported operation` for `torch.operator`.

- [ ] **Step 4: Implement in `rebuild.py`**

- Block arguments: accept types whose string starts with `!torch.vtensor` or equals `!torch.int` / `!torch.any`; update the error text's list of supported types.
- Generalize `_resolve_aten(op, name)` to `_resolve(op, name)` parsing `torch.<ns>.<op>[.<overload>]`; look up `getattr(torch.ops, ns)` then the packet (must be an `OpOverloadPacket`; `torch.ops` creates namespaces lazily, so the packet check is what rejects unknown namespaces) and the overload; error `no torch.ops.<ns>.<op>.<overload>` via `_error`.
- Dispatch: `torch.aten.*` unchanged; `torch.operator` reads `ir.StringAttr(op.attributes["name"]).value` and goes through the same binder. Binder: zero results → emit node, return `[]`; one result → `[node]`; more → `_error(op, "expected at most 1 result ...")` (aten keeps requiring exactly 1).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run -e vllm-mlir pytest python/tests/test_rebuild_fx.py python/tests/test_backend.py -v`
Expected: all PASS.

- [ ] **Step 6: Full suite and commit**

Run: `pixi run -e vllm-mlir pytest python/tests` — Expected: all pass.

```bash
git add python/src/kashi/rebuild.py python/tests/test_rebuild_fx.py python/tests/test_backend.py
git commit -m "rebuild torch.operator and scalar/opaque arguments; symbolic and custom-op backend tests"
```

### Task 3: vLLM end-to-end test

**Files:**
- Create: `python/tests/vllm_backend_spy.py`
- Test: `python/tests/test_vllm_integration.py`

**Interfaces:**
- Consumes: `kashi.backend.compile_fx`; `python/tests/models/simple_vllm_model.py` (`SimpleVllmForCausalLM`, unmodified).
- Produces: `vllm_backend_spy.CALLS: list[tuple[GraphModule, Callable]]` and `vllm_backend_spy.spy_compile_fx(gm, example_inputs)` (calls `compile_fx`, appends, returns). vLLM backend string: `"vllm_backend_spy.spy_compile_fx"` (importable via the Task 1 `pythonpath`).

- [ ] **Step 1: Write the test** `test_vllm_matches_uncompiled_vllm`

- `pytest.importorskip("vllm")`. `monkeypatch.setenv` for `VLLM_USE_AOT_COMPILE=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_CPU_KVCACHE_SPACE=1`, `VLLM_DISABLE_COMPILE_CACHE=1` before constructing any engine. Register with `ModelRegistry.register_model("SimpleVllmForCausalLM", SimpleVllmForCausalLM)` (import the fixture via `models.simple_vllm_model` or by path).
- `config.json` in `tmp_path`: `{"architectures": ["SimpleVllmForCausalLM"], "model_type": "llama", "hidden_size": 128, "num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 1, "intermediate_size": 256, "vocab_size": 128, "max_position_embeddings": 256, "torch_dtype": "float32"}`.
- Engine args (both runs): `skip_tokenizer_init=True, load_format="dummy", dtype="float32", max_model_len=128, seed=0`.
- After each engine is built: `llm.apply_model(fill)` where `fill` iterates `sorted(model.named_parameters())` and copies `torch.randn(p.shape, generator=torch.Generator().manual_seed(0)) * 0.3` (one generator across all params) into each under `torch.no_grad()`.
- Prompts `[{"prompt_token_ids": [1, 2, 3, 4, 5]}, {"prompt_token_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18]}]`, `SamplingParams(max_tokens=4, temperature=0, logprobs=5)`, one `generate` call with both.
- Plugin run: `CompilationConfig(mode=CompilationMode.DYNAMO_TRACE_ONCE, backend="vllm_backend_spy.spy_compile_fx")`. Reference run: `CompilationConfig(mode=CompilationMode.NONE)`. Free the first engine before building the second (builder's choice of mechanism; record it as a ruling).

Assertions:
1. `len(CALLS) == 1`; `fn is not gm.forward`.
2. Targets in namespace `vllm`: original `gm` → packets `[unified_kv_cache_update, unified_attention_with_output]`; rebuilt `fn.__self__` → the corresponding `.default` overloads, same order.
3. Per prompt: token IDs identical; per step, the chosen token's logprob `assert_close` (float32 defaults).
4. Per step in the reference run: top-1 minus top-2 logprob `> 1e-4` (the comparison can discriminate). If the seed/scale yields a near tie, change the scale (0.1–0.5) or seed — not the threshold.

- [ ] **Step 2: Run it**

Run: `pixi run -e vllm-mlir pytest python/tests/test_vllm_integration.py -v -s`
Expected: PASS. This is an acceptance test over product code finished in Tasks 1–2, so there is no RED step; assertion 4 is what shows the comparison can fail. A failure here is a product finding: use superpowers:systematic-debugging and fix it in the owning module with a component regression test first.

- [ ] **Step 3: Full suite and commit**

Run: `pixi run -e vllm-mlir pytest python/tests` — Expected: all pass.

```bash
git add python/tests/vllm_backend_spy.py python/tests/test_vllm_integration.py
git commit -m "add vLLM end-to-end test for the FX-MLIR-FX backend"
```
