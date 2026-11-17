# Standalone TorchDynamo Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A plain-callable `torch.compile` backend, `kashi.backend.compile_fx`, that runs FX → MLIR → FX and executes the rebuilt graph eagerly.

**Architecture:** `compile_fx` composes the existing `import_fx` and `rebuild_fx`, verifies the MLIR in between, and returns the rebuilt `GraphModule.forward`. Integration tests compile small `nn.Module`s through a test-local spy around `compile_fx`.

**Tech Stack:** Python, PyTorch 2.14 (TorchDynamo), IREE 3.14 Python bindings, pytest; Pixi environment `vllm-mlir`.

**Spec:** `specs/vllm-mlir-compilation-plugin/dynamo-backend.md`

## Global Constraints

- Independent of vLLM: no `vllm` imports in product code or these tests.
- Plain callable only: no `register_backend`, no entry points, no import-time side effects, no backend options.
- Never return `gm` / `gm.forward` or fall back to eager; errors from `import_fx`, `verify()` or `rebuild_fx` propagate.
- No test hooks or counters in product code.
- Tests compile with `dynamic=False` and call `torch._dynamo.reset()` before each test.
- Run tests with `pixi run -e vllm-mlir pytest ...`.

## Review Focus

- Calling the compiled model again with new values of the same shape reuses the compiled callable (no second backend call) and stays correct — pinned in Task 1 (`test_reuses_compiled_graph_for_new_values`).
- In-place updates to module parameters after compilation are seen, because weights arrive as runtime inputs rather than captured `example_inputs` — pinned in Task 1 (`test_sees_parameter_updates`).
- Single-tensor return vs tuple return keeps Dynamo's calling convention — pinned in Task 1 via the `single` model.
- Integer (index) user inputs mixed with float parameters in the placeholder list — covered by the `embed_chunk_view` model.
- Uninitialized `torch.empty` outputs compared only by shape/dtype — covered by the `empty` model.

---

### Task 1: `compile_fx` backend and `torch.compile` integration tests

**Files:**
- Create: `python/src/kashi/backend.py`
- Test: `python/tests/test_backend.py`

**Interfaces:**
- Consumes: `kashi.importer.import_fx(gm, example_inputs) -> ImportedModule` (fields `module: ir.Module`, `out_spec: TreeSpec`); `kashi.rebuild.rebuild_fx(module, out_spec) -> torch.fx.GraphModule`.
- Produces: `kashi.backend.compile_fx(gm: torch.fx.GraphModule, example_inputs: Sequence[object]) -> Callable[..., Any]`.

- [ ] **Step 1: Write the failing tests** in `python/tests/test_backend.py`

Use `pytest.importorskip("torch")` as the other tests do. Contents:

```python
@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()


class Spy:
    """Wraps compile_fx; records each graph received and callable returned."""
    def __init__(self):
        self.calls = []  # list of (gm, returned_callable)
    def __call__(self, gm, example_inputs):
        fn = compile_fx(gm, example_inputs)
        self.calls.append((gm, fn))
        return fn
```

Models (seed with `torch.manual_seed(0)` before construction):

- `EmbedChunkView`: `nn.Embedding(10, 8)`, `nn.Linear(8, 24)`, `nn.Linear(8, 8)`; `forward(ids)`: `q, k, v = qkv(embed(ids)).chunk(3, dim=-1)`; return `(o(q).view(2, -1), k)`. Input `torch.tensor([1, 2, 3, 4])`.
- `Single`: `nn.Linear(4, 3)`; `forward(x)` returns `linear(x).view(-1)`. Input `torch.randn(2, 4)`.
- `Empty`: `forward(x)` returns `(torch.empty(2, 3), x.view(-1))`. Input `torch.randn(2, 2)`. Compare leaf 0 by `(shape, dtype)` only.

Tests:

- `test_backend_matches_eager[case]`, parametrized over the three models:
  - `len(spy.calls) == 1`
  - `gm, fn = spy.calls[0]`; `fn is not gm.forward`; `isinstance(fn.__self__, torch.fx.GraphModule)` and `fn.__self__ is not gm`
  - `tree_structure(actual) == tree_structure(expected)`; each leaf `torch.testing.assert_close` (or shape/dtype for `Empty` leaf 0)
- `test_reuses_compiled_graph_for_new_values`: compile `Single`, call with `torch.randn(2, 4)` twice (different values); second result matches eager and `len(spy.calls) == 1`.
- `test_sees_parameter_updates`: compile `Single`, call once, then `with torch.no_grad(): model.linear.weight.add_(1.0)`; next call matches eager and `len(spy.calls) == 1`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e vllm-mlir pytest python/tests/test_backend.py -v`
Expected: collection error / FAIL with `ModuleNotFoundError: No module named 'kashi.backend'`.

- [ ] **Step 3: Implement `compile_fx(gm, example_inputs) -> Callable` in `python/src/kashi/backend.py`**

Body exactly as in the spec's Interface section: `import_fx` → `imported.module.operation.verify()` → return `rebuild_fx(imported.module, imported.out_spec).forward`. Module docstring states it is a plain `torch.compile` backend with no registration.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e vllm-mlir pytest python/tests/test_backend.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the full Python suite**

Run: `pixi run -e vllm-mlir pytest python/tests -v`
Expected: all pass (existing import, rebuild and round-trip tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add python/src/kashi/backend.py python/tests/test_backend.py
git commit -m "add standalone TorchDynamo backend composing import_fx and rebuild_fx"
```
