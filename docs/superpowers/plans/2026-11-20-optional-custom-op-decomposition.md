# Optional Custom-Op Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Behind an opt-in flag, rewrite vLLM's `_C.rms_norm`, `_C.fused_add_rms_norm`, `_C.silu_and_mul` and `_C.rotary_embedding` into ATen ops before MLIR import, and tag every resulting MLIR op with a `NameLoc` naming the original call.

**Architecture:** A new `kashi/decompose.py` rewrites a normalized FX graph in place. It emits ATen nodes with FakeTensor `meta["val"]`, sends later reads of each mutated argument to the new value, and stores `meta["kashi_decomposed_from"]`. `importer._normalize` calls it when the flag is on. `_OpaqueContextCache.get_node_location` turns the meta key into `loc("kashi.decomposed:<op>#<n>"(<original loc>))`. The flag runs from `compile_iree_fx` and the script.

**Tech Stack:** Python, torch.fx, FakeTensor, IREE's FxImporter and MLIR Python bindings, pytest (`pixi run test-python`).

**Spec:** `docs/superpowers/specs/2026-11-20-optional-custom-op-decomposition-design.md`

## Global Constraints

- Off by default: `decompose_custom_ops: bool = False` everywhere. With it off, graphs and MLIR are byte-identical to today's.
- Only these four ops: `_C::rms_norm`, `_C::fused_add_rms_norm`, `_C::silu_and_mul`, `_C::rotary_embedding`. vLLM attention and KV-cache ops are untouched.
- Match ops by `node.target.name()` (e.g. `"_C::rms_norm"`). `kashi/decompose.py` must not import vLLM.
- Tag value: `f"{op_name.replace('::', '.')}#{n}"`, where `n` counts calls to that op in graph order from 0, e.g. `"_C.rms_norm#2"`. Meta key: `"kashi_decomposed_from"`. MLIR location: `Location.name(f"kashi.decomposed:{tag}", childLoc=<base loc or unknown>)`.
- Unsupported cases raise `NotImplementedError`; never silently miscompile.
- `_C::rms_norm` → `aten.rms_norm.default(input, [input.shape[-1]], weight, epsilon)` (verified to lower through IREE for CUDA).
- Rotary support only: `is_neox=True`, `rope_dim_offset=0`, `inverse=False`, `key` not None, `cos_sin_cache.shape[-1] == head_size`.
- Tests that use `_C` ops start with `pytest.importorskip("vllm._custom_ops")`, which registers the ops. There are no CPU kernels, so eager references are written in the test file.

## Review Focus

1. **bfloat16 inputs** (real checkpoints run with `--dtype auto`): emitted nodes keep the input dtype in `meta["val"]` and in eager output. Test in Task 1.
2. **`rms_norm` with `weight=None`** (the schema allows `Tensor?`): decomposes and matches the reference without a weight. Test in Task 1.
3. **Symbolic token dimension** (Dynamo marks the token dim dynamic, e.g. `s72`): rotary views must use `-1` for the token dim, never a baked-in example size. Test in Task 3.
4. **Qwen's chain: `rms_norm` writes a buffer, a view of it feeds `rotary_embedding`, then a view of that is read.** Both rewrites must compose and the alias guard must not fire. Test in Task 3.
5. **Rotary called with defaulted args omitted** (Dynamo passes 6 positional args, not 8): arguments are bound with schema defaults. Test in Task 3.

---

### Task 1: Decomposition core with `_C::rms_norm`

**Files:**
- Create: `python/src/kashi/decompose.py`
- Test: `python/tests/test_decompose.py`

**Interfaces:**
- Produces:
  - `DECOMPOSED_FROM_KEY: str = "kashi_decomposed_from"`
  - `decompose_custom_ops(graph: torch.fx.Graph) -> None`. Rewrites in place. The graph must already be normalized: ATen `OpOverload` targets with `meta["val"]` on every value node.
  - Private registry `_DECOMPOSITIONS: dict[str, Callable[[_Emitter, list], dict[torch.fx.Node, torch.fx.Node]]]`, keyed by `target.name()`. A rule takes an emitter and the schema-bound argument list (defaults filled in) and returns `{mutated_arg_node: new_value_node}`.
  - `_Emitter.__call__(target: OpOverload, *args) -> torch.fx.Node`. Inserts before the custom op, sets `meta["val"]` by running `target` on the args' `meta["val"]` under the args' fake mode (`torch._guards.detect_fake_mode`), and sets `meta[DECOMPOSED_FROM_KEY]` to the current tag.
- Later tasks only add entries to `_DECOMPOSITIONS`.

- [ ] **Step 1: Write the failing tests** in `python/tests/test_decompose.py`. Add a small helper `_fake(gm)` that runs `fx_test_utils.RecordExamples(gm, "val")` under `FakeTensorMode` with fake inputs. Add `_ref_rms(x, w, eps)`, which computes `x * rsqrt(mean(x², -1, keepdim) + eps) * w`, with the weight optional.

```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_weight", [True, False])
def test_rms_norm_result_reaches_later_reads(dtype, with_weight):
    # x, w placeholders; out = aten.empty.memory_format([3, 8], dtype=dtype);
    # _C.rms_norm(out, x, w or None, 1e-6); y = aten.neg(out); output y
    ...
    decompose_custom_ops(gm.graph); gm.recompile()
    assert not [n for n in gm.graph.nodes if _is_c_op(n)]
    assert gm(x, w).dtype == dtype
    torch.testing.assert_close(gm(x, w), -_ref_rms(x, w, 1e-6), **_tol(dtype))

def test_each_call_gets_its_own_tag():
    # two rms_norm calls writing two buffers
    tags = {n.meta.get(DECOMPOSED_FROM_KEY) for n in gm.graph.nodes} - {None}
    assert tags == {"_C.rms_norm#0", "_C.rms_norm#1"}
    assert all(DECOMPOSED_FROM_KEY not in n.meta for n in original_non_custom_nodes)

def test_mutating_a_graph_input_raises():
    # out is a placeholder
    with pytest.raises(NotImplementedError, match="graph input"): ...

def test_reading_base_of_mutated_view_after_call_raises():
    # out = empty([3, 8]); v = aten.view(out, [3, 8]); rms_norm(v, x, w); y = neg(out)
    with pytest.raises(NotImplementedError, match="alias"): ...

def test_graph_without_custom_ops_is_unchanged():
    before = str(gm.graph); decompose_custom_ops(gm.graph)
    assert str(gm.graph) == before
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `pixi run test-python python/tests/test_decompose.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kashi.decompose'`.

- [ ] **Step 3: Implement `decompose_custom_ops` and the `_C::rms_norm` rule in `python/src/kashi/decompose.py`**

For each node in a snapshot of the graph's order, if `target.name()` is in `_DECOMPOSITIONS`:
1. Bind args with `torch.fx.operator_schemas.normalize_function(target, args, kwargs, normalize_to_only_use_kwargs=True)`, which fills in defaults.
2. Run the rule with an emitter tagged `<op>#<n>`.
3. For each `{m: m'}`, replace uses of `m` by nodes positioned after the custom op. Compare positions with a node-index dict computed once before the rewrite, and give new nodes the custom op's index. Use `m.replace_all_uses_with(m', delete_user_cb=...)`.
4. Run the alias guard (below).
5. Erase the custom op.

Alias guard, run after step 3:
- Follow `m`'s `aten.view.default` / `aten.reshape.default` / `aten._unsafe_view.default` producer chain back to its base.
- If the base is a placeholder, raise with the message `"... mutates a graph input ..."`.
- If the base, or any node reachable from the base through those view ops, still has a user positioned after the custom op, raise with `"... reads an alias of ... after ..."`.

- [ ] **Step 4: Run the tests to make sure they pass**

Run: `pixi run test-python python/tests/test_decompose.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/kashi/decompose.py python/tests/test_decompose.py
git commit -m "Add opt-in FX decomposition of _C.rms_norm with origin tags."
```

### Task 2: `_C::fused_add_rms_norm` and `_C::silu_and_mul`

**Files:**
- Modify: `python/src/kashi/decompose.py`
- Test: `python/tests/test_decompose.py`

**Interfaces:**
- Consumes: `_DECOMPOSITIONS`, `_Emitter` and `decompose_custom_ops` from Task 1.
- Produces: two more registry entries.

- [ ] **Step 1: Write the failing tests.** Mutated tensors come from `aten.clone.default` of placeholders, so the graph-input guard does not fire.

```python
def test_fused_add_rms_norm_updates_input_and_residual():
    # i = clone(x); r = clone(res); pre = aten.mul.Tensor(i, 2.0)
    # fused_add_rms_norm(i, r, w, 1e-6); output (pre, i, r)
    pre, i, r = gm(x, res, w)
    torch.testing.assert_close(pre, 2 * x)          # read before the call keeps old value
    torch.testing.assert_close(r, x + res)
    torch.testing.assert_close(i, _ref_rms(x + res, w, 1e-6))

def test_chained_fused_add_rms_norm_threads_the_residual():
    # Qwen shape: two calls share residual r; second call's input is clone(y)
    # expected: r2 = y + (x + res); i2 = rms(r2)

def test_silu_and_mul_writes_result():
    # x [3, 8]; out = empty([3, 4]); silu_and_mul(out, x); output out
    torch.testing.assert_close(gm(x), F.silu(x[:, :4]) * x[:, 4:])
```

- [ ] **Step 2: Run to make sure they fail**

Run: `pixi run test-python python/tests/test_decompose.py -v -k "fused or silu"`
Expected: FAIL. The `_C` node is still present, so eager execution raises `Could not run '_C::...' with arguments from the 'CPU' backend`.

- [ ] **Step 3: Add the two rules**

`fused_add_rms_norm` returns `{input: aten.rms_norm(r', [d], weight, eps), residual: r'}` with `r' = aten.add.Tensor(input, residual)`. `silu_and_mul` returns `{result: aten.mul.Tensor(aten.silu(x[..., :d]), x[..., d:])}`. Here `d = input.shape[-1] // 2`, and the halves are taken with `aten.slice.Tensor(input, -1, start, end)`.

- [ ] **Step 4: Run to make sure they pass**

Run: `pixi run test-python python/tests/test_decompose.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/kashi/decompose.py python/tests/test_decompose.py
git commit -m "Decompose _C.fused_add_rms_norm and _C.silu_and_mul."
```

### Task 3: `_C::rotary_embedding` (NeoX only)

**Files:**
- Modify: `python/src/kashi/decompose.py`
- Test: `python/tests/test_decompose.py`

**Interfaces:**
- Consumes: Task 1's registry, emitter and alias guard.
- Produces: one registry entry.

- [ ] **Step 1: Write the failing tests.** Use `T = 5` tokens, `head_size = 16`, 2 query heads (`q [T, 32]`), 1 KV head (`k [T, 16]`), `cos_sin_cache [10, 16]` and `positions = aten.arange` of int64. `_ref_rope(pos, x, head_size, cache)` gathers `cache[pos]`, splits it into `cos` and `sin`, views `x` as `[T, -1, head_size]`, and computes `cat(x1*cos - x2*sin, x2*cos + x1*sin)`.

```python
def test_rotary_embedding_neox_matches_reference():
    # q = clone(qin); k = clone(kin); rotary_embedding(pos, q, k, 16, cache, True)  # 6 args
    q, k = gm(pos, qin, kin, cache)
    torch.testing.assert_close(q, _ref_rope(pos, qin, 16, cache))
    torch.testing.assert_close(k, _ref_rope(pos, kin, 16, cache))

def test_rotary_views_keep_token_dim_symbolic():
    views = [n for n in gm.graph.nodes if n.target is torch.ops.aten.view.default
             and DECOMPOSED_FROM_KEY in n.meta]
    assert views and all(n.args[1][0] == -1 for n in views)

def test_qwen_rms_norm_then_rotary_chain():
    # out = empty([T, 32]); rms_norm(out, qv, w, eps) where qv = view(qin, [T, 2, 16])
    # q1 = view(out, [T, 32]); rotary_embedding(pos, q1, k, 16, cache, True)
    # query = view(q1, [-1, 2, 16]); output query   (out has qv's shape [T, 2, 16])
    torch.testing.assert_close(query, _ref_rope(pos, _ref_rms(qv, w, eps).view(T, 32), 16, cache).view(-1, 2, 16))

@pytest.mark.parametrize("override", [
    {"is_neox": False}, {"rope_dim_offset": 1}, {"inverse": True},
    {"key": None}, {"cos_sin_cache": "cache_width_8"},
])
def test_unsupported_rotary_variants_raise(override):
    with pytest.raises(NotImplementedError, match="rotary_embedding"): ...
```

- [ ] **Step 2: Run to make sure they fail**

Run: `pixi run test-python python/tests/test_decompose.py -v -k rotary`
Expected: FAIL. Eager execution of the remaining `_C::rotary_embedding` raises, and the unsupported cases do not raise `NotImplementedError`.

- [ ] **Step 3: Add the rule**

Check the supported-variant conditions first, raising `NotImplementedError("unsupported _C.rotary_embedding: ...")`. Then:
- `cos, sin` come from `aten.index.Tensor(cache, [positions])`, sliced in half and unsqueezed at dim −2.
- Each of query and key goes through `aten.view.default(x, [-1, heads, head_size])` → rotate halves → `aten.cat.default` → `aten.view.default(..., [-1, x.shape[-1]])`. `heads = x.shape[-1] // head_size`.

The rule returns `{query: q', key: k'}`.

- [ ] **Step 4: Run to make sure they pass**

Run: `pixi run test-python python/tests/test_decompose.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/kashi/decompose.py python/tests/test_decompose.py
git commit -m "Decompose NeoX _C.rotary_embedding."
```

### Task 4: Wire the flag through import, location tags, backend and script

**Files:**
- Modify: `python/src/kashi/importer.py`, which covers `import_fx`, `_normalize` and `_OpaqueContextCache`
- Modify: `python/src/kashi/iree_backend.py` (`compile_iree_fx`)
- Modify: `scripts/compile_qwen3_iree.py` (`_parse_args`, `iree_backend`, `_STATE`)
- Test: `python/tests/test_decompose.py`

**Interfaces:**
- Consumes: `decompose_custom_ops` and `DECOMPOSED_FROM_KEY` from Task 1.
- Produces:
  - `import_fx(gm, *, decompose_custom_ops: bool = False) -> ImportedModule`
  - `_normalize(gm, *, decompose_custom_ops: bool = False)`. This calls `decompose.decompose_custom_ops(graph)` after the per-node loop and before `graph.lint()`.
  - `compile_iree_fx(..., decompose_custom_ops: bool = False)`, which passes the flag to `import_fx`. When the flag is on, it writes `partitioned.mlir` and `driver.mlir` with `module.operation.get_asm(enable_debug_info=True)` instead of `str(module)`. Without this, `iree-compile` reads a file with no locations and never sees the tags. With the flag off, they are still written with `str()`, so the output is byte-identical.
  - `_OpaqueContextCache.get_node_location(node)`. When `DECOMPOSED_FROM_KEY` is in `node.meta`, it returns `Location.name(f"kashi.decomposed:{tag}", childLoc=super().get_node_location(node) or Location.unknown(self._c), context=self._c)`. Otherwise it returns the base result unchanged.
  - Script flag `--decompose-custom-ops` (store_true), saved as `_STATE["decompose_custom_ops"]` and forwarded by `iree_backend`.

- [ ] **Step 1: Write the failing tests.** The graph is Qwen-like: `rms_norm` ×2 plus `silu_and_mul`, with `meta["val"]` set through `_fake`.

```python
def test_import_tags_decomposed_ops_with_name_locations():
    imported = import_fx(gm, decompose_custom_ops=True)
    assert imported.module.operation.verify()
    text = str(imported.module)
    assert "torch.operator" not in text
    ops = [op.operation for op in _main_block(imported.module).operations]
    locs = {str(op.location) for op in ops if op.name == "torch.aten.rms_norm"}
    assert len(locs) == 2 and all('loc("kashi.decomposed:_C.rms_norm#' in l for l in locs)
    asm = imported.module.operation.get_asm(enable_debug_info=True)
    assert '"kashi.decomposed:_C.silu_and_mul#0"' in asm

def test_import_without_flag_keeps_custom_ops():
    text = imported_default.module.operation.get_asm(enable_debug_info=True)
    assert 'torch.operator "torch._C.rms_norm"' in text
    assert "kashi.decomposed" not in text

def test_decomposed_graph_is_a_single_iree_segment():
    partition = partition_at_custom_ops(import_fx(gm, decompose_custom_ops=True).module)
    assert len(partition.segments) == 1  # one segment per custom op, plus one

def test_compile_iree_fx_forwards_the_flag(monkeypatch, tmp_path):
    # monkeypatch kashi.iree_backend.import_fx to record kwargs then raise a sentinel
    assert seen == {"decompose_custom_ops": True}

@pytest.mark.skipif(sys.platform == "darwin", reason="IREE CUDA compilation")
def test_partitioned_artifact_keeps_tags(tmp_path):
    # compile_iree_fx(gm, output_dir=tmp_path, cuda_target="sm_80", decompose_custom_ops=True)
    assert "kashi.decomposed:_C.rms_norm#0" in (result.artifact_dir / "partitioned.mlir").read_text()
```

- [ ] **Step 2: Run to make sure they fail**

Run: `pixi run test-python python/tests/test_decompose.py -v -k "import or segment or forwards"`
Expected: FAIL with `TypeError: import_fx() got an unexpected keyword argument 'decompose_custom_ops'`.

- [ ] **Step 3: Implement the importer, backend and script changes** listed in Interfaces. In the script, also mention the flag in the module docstring example.

- [ ] **Step 4: Run the full Python suite to confirm the flag-off path is unchanged**

Run: `pixi run test-python`
Expected: all PASS. GPU-gated tests are skipped.

- [ ] **Step 5: Commit**

```bash
git add python/src/kashi/importer.py python/src/kashi/iree_backend.py scripts/compile_qwen3_iree.py python/tests/test_decompose.py
git commit -m "Expose opt-in custom-op decomposition through import_fx, compile_iree_fx and the Qwen3 script."
```

### Task 5: GPU parity and end-to-end check

**Files:**
- Test: `python/tests/test_decompose.py`

**Interfaces:**
- Consumes: Tasks 1–4.

- [ ] **Step 1: Write a CUDA-gated parity test**

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM _C kernels need CUDA")
@pytest.mark.parametrize("case", ["rms_norm", "fused_add_rms_norm", "silu_and_mul", "rotary_embedding"])
def test_decomposition_matches_vllm_kernel_on_cuda(case):
    # build the case's graph on cuda; run an untouched copy (real _C kernels) and the
    # decomposed copy on identical inputs
    torch.testing.assert_close(decomposed_out, kernel_out, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run it on a CUDA machine**

Run: `pixi run test-python python/tests/test_decompose.py -v -k cuda`
Expected: 4 PASS on CUDA, and 4 SKIPPED elsewhere.

- [ ] **Step 3: Run the script end to end on a CUDA machine**

Run: `python scripts/compile_qwen3_iree.py --decompose-custom-ops --output-dir build/qwen3-iree-decomp --generate`
Expected:
- The script exits 0.
- `grep -c 'torch.operator' build/qwen3-iree-decomp/kashi-iree-*/driver.mlir` prints `2`, for `unified_kv_cache_update` and `unified_attention_with_output` in the one-layer config.
- `grep -c 'kashi.decomposed' build/qwen3-iree-decomp/kashi-iree-*/partitioned.mlir` is non-zero.
- `grep -c 'kashi.decomposed' build/qwen3-iree-decomp/kashi-iree-*/phases/*.flow.mlir` is reported in the final summary. This is informational: tags surviving into flow is best effort, not a pass condition.

- [ ] **Step 4: Commit**

```bash
git add python/tests/test_decompose.py
git commit -m "Add CUDA parity test for custom-op decompositions."
```
