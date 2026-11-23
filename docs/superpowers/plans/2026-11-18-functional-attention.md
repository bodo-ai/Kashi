# Functional Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Imported MLIR represents `vllm::unified_attention_with_output` as a functional dummy op `kashi::unified_attention` that returns the attention result. `rebuild_fx` expands the dummy back into the real op.

**Architecture:** A new module, `kashi.functional`, owns three things: the rewrite table, the lazily registered dummy ops and the reverse lookup. `kashi.importer._normalize` swaps matching calls for the dummy on its structural copy and redirects later uses of the output buffer to the dummy's result. `kashi.rebuild` expands a dummy `torch.operator` into `aten.empty_like` + the original op.

**Tech Stack:** Python 3.12, PyTorch 2.14 (`torch.library`, FX), IREE 3.14 FX importer and MLIR bindings, vLLM 0.33 (end-to-end test only), pytest via Pixi.

**Spec:** `specs/vllm-mlir-compilation-plugin/functional-attention.md`. The parent is `specs/vllm-mlir-compilation-plugin/vllm-integration.md`.

## Global Constraints

- `kashi` must not import vLLM. Ops are matched by qualified name, e.g. `"vllm::unified_attention_with_output"`.
- No global patching of vLLM, IREE or PyTorch. Defining ops in a `torch.library.Library("kashi", "FRAGMENT")` and registering a kashi-owned opaque type are allowed.
- No DCE: the dead `empty`/`view` that fed the buffer stay in the graph.
- Reject rather than guess: unsupported cases raise `NotImplementedError` naming the node (`node.format_node()`).
- Component tests (`test_functional.py`, `test_import_fx.py`, `test_rebuild_fx.py`, `test_backend.py`) must not import vLLM. They add test entries with `monkeypatch.setitem(functional._FUNCTIONAL_OPS, ...)`.
- Run tests with `pixi run python -m pytest <path> -v` from the repo root.

## Review Focus

1. **Output buffer is a graph input.** The write would be visible to the caller, so import must reject it ("graph input"). Pinned in Task 2.
2. **The buffer's base, or another view of it, is read after the call.** The functional form would hide the write, so import must reject it ("aliases"). Pinned in Task 2.
3. **Several attention calls in one graph** (one per layer). Each call is rewritten and a later call can consume an earlier result; the dummy is registered once. Pinned in Task 1 (idempotent registration) and Task 2 (`TwoWritesModel`).
4. **The original op isn't registered when rebuilding** (vLLM not loaded). Rebuild must raise the standard unresolvable-operator error, not a `KeyError`/`AttributeError`. Pinned in Task 3.
5. **`query` is a strided view of `chunk`** (the vLLM case). Import must accept it and rebuild must allocate a contiguous buffer. Pinned in Task 1 (fake impl on a non-contiguous input) and Task 2 (shape/dtype only, not strides).

---

### Task 0: Branch and spec

- [ ] **Step 1:** `git checkout -b feature/functional-attention`
- [ ] **Step 2:** Commit the spec and this plan:

```bash
git add specs/vllm-mlir-compilation-plugin/functional-attention.md docs/superpowers/plans/2026-11-18-functional-attention.md
git commit -m "Add functional attention spec and plan"
```

### Task 1: `kashi.functional` — table, dummy ops, reverse lookup

**Files:**
- Create: `python/src/kashi/functional.py`
- Test: `python/tests/test_functional.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class FunctionalForm: dummy: str; output: str; like: str`. These are the dummy op's name in the `kashi` namespace, the mutated output argument's name, and the argument whose shape/dtype the result takes.
  - `_FUNCTIONAL_OPS: dict[str, FunctionalForm]`, keyed by the original op's qualified name. Initial content, exactly: `{"vllm::unified_attention_with_output": FunctionalForm("unified_attention", output="output", like="query")}`.
  - `functional_form(op: torch._ops.OpOverload) -> FunctionalForm | None`. Matches `op._schema.name` against the table, and only the default overload (`op._schema.overload_name == ""`).
  - `dummy_op(op: torch._ops.OpOverload, form: FunctionalForm) -> torch._ops.OpOverload`. Registers `kashi::<form.dummy>` on first call and returns `torch.ops.kashi.<form.dummy>.default`. It is idempotent.
  - `functional_original(dummy: str) -> tuple[str, FunctionalForm] | None`. For a bare dummy name (e.g. `"unified_attention"`) it returns the original qualified name (e.g. `"vllm::unified_attention_with_output"`) and its form, or `None`. It does not resolve the op.

- [ ] **Step 1: Write the failing tests** in `python/tests/test_functional.py`:

```python
import pytest
torch = pytest.importorskip("torch")

from torch._subclasses.fake_tensor import FakeTensorMode

from kashi import functional

WRITE = functional.FunctionalForm("write_functional", output="out", like="x")


@pytest.fixture
def test_ops(monkeypatch):
    import kashi_test_ops

    monkeypatch.setitem(functional._FUNCTIONAL_OPS, "kashi_test::write", WRITE)
    return kashi_test_ops


def test_vllm_entry():
    assert functional._FUNCTIONAL_OPS["vllm::unified_attention_with_output"] == (
        functional.FunctionalForm("unified_attention", output="output", like="query")
    )


def test_functional_form_matches_by_name(test_ops):
    assert functional.functional_form(torch.ops.kashi_test.write.default) == WRITE
    assert functional.functional_form(torch.ops.kashi_test.scale.default) is None


def test_dummy_schema_drops_output_and_returns_tensor(test_ops):
    write = torch.ops.kashi_test.write.default
    dummy = functional.dummy_op(write, WRITE)

    assert dummy is torch.ops.kashi.write_functional.default
    assert functional.dummy_op(write, WRITE) is dummy
    args = dummy._schema.arguments
    assert [a.name for a in args] == ["x", "tag", "dep"]
    assert [a.kwarg_only for a in args] == [False, False, True]
    assert str(args[1].type) == "PyObject"
    assert all(a.alias_info is None for a in args)
    assert [str(r.type) for r in dummy._schema.returns] == ["Tensor"]


def test_dummy_fake_impl_is_contiguous_like(test_ops):
    dummy = functional.dummy_op(torch.ops.kashi_test.write.default, WRITE)
    with FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn(4, 3)).t()
        y = dummy(x, test_ops.Tag(1.0), dep=x)
    assert (y.shape, y.dtype, y.is_contiguous()) == ((3, 4), torch.float32, True)


def test_functional_original(test_ops):
    assert functional.functional_original("write_functional") == ("kashi_test::write", WRITE)
    assert functional.functional_original("nope") is None
```

- [ ] **Step 2:** Run `pixi run python -m pytest python/tests/test_functional.py -v`. Expected: FAIL with `ModuleNotFoundError: kashi.functional` (`test_vllm_entry` too).
- [ ] **Step 3: Implement `python/src/kashi/functional.py`.** Required points:
  - **Schema string.** Build the dummy's schema from `op._schema` as `<dummy>(<args>) -> Tensor`, where `<args>` is every original argument except `form.output`. Keep each argument's type, name, default and `*` kwarg-only position, and strip alias annotations (`Tensor(a3!)` → `Tensor`, `Tensor(a6!)?` → `Tensor?`).
  - **Opaque arguments.** `PyObject` cannot be parsed back. Replace it with the name of a kashi-owned opaque type: `class _Opaque(OpaqueBase)` registered with `register_opaque_type(_Opaque, typ="reference")`, name from `get_opaque_type_name(_Opaque)`. The resulting schema type is still `PyObject` (verified by a probe during planning).
  - **Fake impl.** Register with `torch.library.register_fake`. It binds by schema and returns `torch.empty_like(<like>, memory_format=torch.contiguous_format)`. There is no real kernel.
  - **Idempotence.** Skip registration when `torch.ops.kashi` already has that name.
- [ ] **Step 4:** Run the Step 2 command. Expected: 5 passed.
- [ ] **Step 5:** Commit: `git add python/src/kashi/functional.py python/tests/test_functional.py && git commit -m "Add functional dummy ops for destination-passing custom ops"`

### Task 2: Importer rewrite

**Files:**
- Modify: `python/src/kashi/importer.py`. Add the call in `_rewrite` (lines 96-101, after packet resolution / for `OpOverload` targets) and a new `_functionalize(graph, node, form)`.
- Modify: `python/tests/kashi_test_ops.py`. Add one op and the test models below.
- Test: `python/tests/test_import_fx.py`

**Interfaces:**
- Consumes: `functional_form`, `dummy_op`, `FunctionalForm` (Task 1).
- Produces:
  - Imported MLIR contains `torch.operator "torch.kashi.<dummy>"(...) -> !torch.vtensor<...>`, and later uses of the buffer read that result.
  - `kashi_test_ops.functional_entries(monkeypatch)`, which installs the `write` and `write_scaled` entries. Tasks 3 and 4 use it.

- [ ] **Step 1: Extend `kashi_test_ops.py`.**
  - Op: `LIB.define(f"write_scaled(Tensor x, Tensor(a!) out, Tensor(b!)? block_scale, *, Tensor dep) -> ()")`. It has an impl that copies `x` into `out` and a fake impl that returns `None`.
  - `functional_entries(monkeypatch)` sets `"kashi_test::write" → FunctionalForm("write_functional", "out", "x")` and `"kashi_test::write_scaled" → FunctionalForm("write_scaled_functional", "out", "x")`.
  - Models, each `nn.Module(tag)` with these exact forward bodies (`W = torch.ops.kashi_test.write`):
    - `InputBufferModel.forward(x, out)`: `W(x, out, self.tag, dep=x); return out.view(-1)`
    - `AliasedBufferModel.forward(x)`: `base = torch.empty(12); out = base.view(3, 4); W(x, out, self.tag, dep=x); return base`
    - `MismatchedBufferModel.forward(x)`: `out = torch.empty(3, 5); W(x, out, self.tag, dep=x); return out`
    - `StridedBufferModel.forward(x)`: `out = torch.ops.aten.t.default(torch.empty(4, 3)); W(x, out, self.tag, dep=x); return out`
    - `ScaledWriteModel.forward(x)`: `out = torch.empty(3, 4); s = torch.empty(1); torch.ops.kashi_test.write_scaled(x, out, s, dep=x); return out`
    - `TwoWritesModel.forward(x)`: `a = torch.empty(3, 4); W(x, a, self.tag, dep=x); b = torch.empty(3, 4); W(a, b, self.tag, dep=a); return b`
- [ ] **Step 2: Write the failing tests** in `test_import_fx.py`. Capture static models with `kashi_test_ops.capture(Model(Tag(3.0)), torch.randn(3, 4))`, plus `torch.empty(3, 4)` for `InputBufferModel`.

```python
@pytest.fixture
def functional_write(monkeypatch):
    import kashi_test_ops
    kashi_test_ops.functional_entries(monkeypatch)
    return kashi_test_ops


def test_import_fx_functionalizes_write(functional_write):
    gm, _ = _captured_ops_model()
    result = import_fx(gm)
    text = str(result.module)

    assert result.module.operation.verify()
    assert '"torch.kashi_test.write"' not in text
    line = next(l for l in text.splitlines() if '"torch.kashi.write_functional"' in l)
    assert line.rstrip().endswith("-> !torch.vtensor<[?,4],f32>")
    result_name = line.split("=")[0].strip()
    assert re.search(rf"torch\.aten\.view {re.escape(result_name)},", text)
    assert gm.graph.find_nodes(op="call_function", target=torch.ops.kashi_test.write)  # input untouched


def test_import_fx_functionalizes_each_call(functional_write):
    # TwoWritesModel: two dummy calls; the second takes the first's result as x and dep;
    # the module returns the second's result.
    ...


@pytest.mark.parametrize("model, match", [
    ("InputBufferModel", "graph input"),
    ("AliasedBufferModel", "aliases"),
    ("MismatchedBufferModel", "shape"),
    ("StridedBufferModel", "contiguous"),
    ("ScaledWriteModel", "block_scale"),
])
def test_import_fx_functionalize_rejects(functional_write, model, match):
    with pytest.raises(NotImplementedError, match=match):
        import_fx(_capture(functional_write, model))
```

  `_capture(ops, name)` is a test helper: `ops.capture(getattr(ops, name)(ops.Tag(3.0)), *inputs)` with `inputs = (torch.randn(3, 4), torch.empty(3, 4))` for `InputBufferModel` and `(torch.randn(3, 4),)` otherwise.
  For `test_import_fx_functionalizes_each_call`, assert exactly what the comment says from the MLIR text: two `torch.kashi.write_functional` lines; the second's first and last operands are the first's result name; `return` names the second's result. The existing `test_import_fx_custom_ops_and_opaque_inputs` (no fixture) must keep its `-> ()` assertion.
- [ ] **Step 3:** Run `pixi run python -m pytest python/tests/test_import_fx.py -v`. Expected: the new tests FAIL (no rewrite yet) and the existing ones pass.
- [ ] **Step 4: Implement `_functionalize(graph, node, form)`.**
  - **Binding.** Bind `node.args`/`node.kwargs` by `node.target._schema` into an ordered name → value mapping, filling defaults.
  - **Reject checks,** in this order:
    1. any other argument with `alias_info.is_write` whose value is not `None` (message names the argument);
    2. `node.users` is non-empty;
    3. the output's `meta["val"]` is not contiguous ("contiguous");
    4. the output's shape or dtype differs from the like argument's meta (`"shape"`; strides of `like` are not compared);
    5. the buffer base is a placeholder ("graph input");
    6. the alias check ("aliases"): take the alias set as the base plus every node transitively derived from it that comes before the call (excluding the call itself). Reject if any alias-set node other than the output node has a user after the call.
  - **Base.** Follow `args[0]` from the output node while the target is `torch.ops.aten.view.default`. Earlier `view` method calls are already rewritten by the time the call is reached.
  - **Replacement.** Insert `dummy_op(target, form)` before the call, with the remaining arguments (positional unless kwarg-only), and `meta = dict(output.meta)`. Replace uses of the output node that come after the call with the new node, then erase the call. Position comes from a fresh `{n: i for i, n in enumerate(graph.nodes)}`.
- [ ] **Step 5:** Run the Step 3 command, then `pixi run python -m pytest python/tests/test_functional.py python/tests/test_import_fx.py -v`. Expected: all pass.
- [ ] **Step 6:** Commit: `git commit -am "Functionalize destination-passing ops during FX import"` (also `git add` the test file).

### Task 3: Rebuild expansion

**Files:**
- Modify: `python/src/kashi/rebuild.py`. Change the `torch.operator` branch (lines ~301-303) and split `_call` (lines ~337-360) into a binder plus emission.
- Test: `python/tests/test_rebuild_fx.py`, `python/tests/test_backend.py`

**Interfaces:**
- Consumes: `functional_original` (Task 1), `kashi_test_ops.functional_entries` (Task 2).
- Produces: `rebuild_fx` maps `torch.operator "torch.kashi.<dummy>"` → `aten.empty_like.default(<like>, memory_format=torch.contiguous_format)` + the original op; the MLIR result maps to the buffer node.

- [ ] **Step 1: Write the failing tests.**

```python
FUNCTIONAL = """
func.func @main(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.any) -> !torch.vtensor<[?,4],f32> {
  %0 = torch.operator "torch.kashi.write_functional"(%arg0, %arg1, %arg0) : (!torch.vtensor<[?,4],f32>, !torch.any, !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32>
  return %0 : !torch.vtensor<[?,4],f32>
}
"""

def test_rebuild_expands_functional_dummy(functional_write):
    gm = rebuild_fx(_parse(FUNCTIONAL))
    x, tag = gm.graph.find_nodes(op="placeholder")
    empty, write = _call_nodes(gm)
    assert (empty.target, empty.args, empty.kwargs) == (
        torch.ops.aten.empty_like.default, (x,), {"memory_format": torch.contiguous_format})
    assert (write.target, write.args, write.kwargs) == (
        torch.ops.kashi_test.write.default, (x, empty, tag), {"dep": x})
    (output,) = gm.graph.find_nodes(op="output")
    assert output.args[0] is empty
    x_val = torch.randn(3, 4)
    torch.testing.assert_close(gm(x_val, functional_write.Tag(3.0)), x_val + 3.0)

def test_rebuild_functional_dummy_rejects_unregistered_original(functional_write, monkeypatch):
    # entry "kashi_test::missing" -> FunctionalForm("missing_functional", "out", "x");
    # MLIR names "torch.kashi.missing_functional" -> raises NotImplementedError
    # matching r"Unsupported MLIR op torch\.operator.*no torch\.ops\.kashi_test\.missing\.default"

def test_rebuild_functional_dummy_requires_one_result(functional_write):
    # FUNCTIONAL with the dummy written as `-> ()` -> NotImplementedError matching "expected 1 result"
```

  In `test_backend.py`, add `test_functionalized_custom_ops_round_trip(monkeypatch)`. It is the same as `test_custom_ops_with_opaque_inputs` but calls `kashi_test_ops.functional_entries(monkeypatch)` first. Assertions:
  - the eager comparisons hold for both sizes, and the spy is called once;
  - the rebuilt targets filtered to `{aten.empty_like.default, kashi_test.scale.default, kashi_test.write.default}` are `[scale.default, empty_like.default, write.default]`.
- [ ] **Step 2:** Run `pixi run python -m pytest python/tests/test_rebuild_fx.py python/tests/test_backend.py -v`. Expected: the new tests FAIL with `no torch.ops.kashi.write_functional.default` (or similar).
- [ ] **Step 3: Implement the expansion in `rebuild.py`.**
  - When the qualified name starts with `torch.kashi.` and `functional_original(<rest>)` is not `None`:
    - resolve the original through the existing `_resolve(op, "torch." + original.replace("::", "."))`, so missing ops get the standard error;
    - check exactly 1 result (existing message form);
    - check the operand count is `len(schema args) - 1`;
    - emit `empty_like` on the operand at the like argument's index in the dummy order;
    - insert it at the output argument's index;
    - bind through the shared schema binder and emit the call once.
  - All other paths are unchanged.
- [ ] **Step 4:** Run `pixi run python -m pytest python/tests -v --ignore=python/tests/test_vllm_integration.py`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -am "Expand functional dummies back to destination-passing ops on rebuild"`

### Task 4: End-to-end and spec update

**Files:**
- Modify: `python/tests/vllm_backend_spy.py`, `python/tests/test_vllm_integration.py`
- Modify: `specs/vllm-mlir-compilation-plugin/vllm-integration.md` (Non-goals, first bullet), `specs/vllm-mlir-compilation-plugin/functional-attention.md` (Status)

- [ ] **Step 1: Write the failing test.**
  - The spy gains `MODULES = []`, and `spy_compile_fx` appends `str(import_fx(gm).module)` before calling `compile_fx`.
  - The fixture clears `MODULES`.
  - In `test_vllm_matches_uncompiled_vllm`, after the call-count assertion:

```python
(text,) = vllm_backend_spy.MODULES
assert 'torch.operator "torch.kashi.unified_attention"' in text
assert "unified_attention_with_output" not in text
```

- [ ] **Step 2:** Run `pixi run python -m pytest python/tests/test_vllm_integration.py -v`. Expected: PASS. Tasks 2–3 already apply to the real vLLM op, so this test confirms them end to end. If it fails, the failure is real; debug it, don't loosen the test.
- [ ] **Step 3: Update the specs.**
  - In `vllm-integration.md`, append to the first non-goal: "Exception: `unified_attention_with_output` is represented functionally in the imported MLIR; see [Functional Attention](functional-attention.md)."
  - Set `functional-attention.md` Status to `Approved`.
- [ ] **Step 4:** Run the full suite: `pixi run python -m pytest python/tests -v`. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -am "Check functional attention end to end; update specs"`
