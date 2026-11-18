# MLIR → FX Reconstruction

**Status:** Proposed; pending human approval
**Parent:** [vLLM FX–MLIR–FX Compilation Plugin](vllm-mlir-compilation.md) §5 (architectural reference only; this document sets the scope)

## Intent

The FX → MLIR importer (`kashi.importer.import_fx`) produces Torch-dialect MLIR for a small set of static-shape operators. This component closes the loop: it rebuilds an executable `torch.fx.GraphModule` from that MLIR so that an FX graph can go FX → MLIR → FX and produce the same results. It is the second half of milestone M1's basic round trip and the base that later work (symbolic shapes, custom operators, vLLM) will extend.

## Scope

### Goals

- `rebuild_fx` turns a Torch-dialect MLIR module into a `GraphModule` whose `forward` runs in eager PyTorch.
- Support exactly the operations and value types that the importer emits for the cases in `python/tests/test_import_fx.py`: `aten.linear`, `aten.embedding`, `aten.view`, `aten.chunk` (with `prim.ListUnpack`), `aten.empty.memory_format`, plus the constants and list construction feeding them.
- Restore the original graph's output structure (e.g. `y` vs `(y,)` vs `(a, b)`) when the importer's recorded output spec is supplied.
- Test reconstruction independently from hand-written MLIR, and test numerical equivalence of the FX → MLIR → FX round trip.

### Non-goals

Symbolic shapes and `SymInt` inputs, custom operators and `torch.operator`, opaque objects / `!torch.any`, mutation and in-place semantics, live `get_attr` bindings and `vtensor.literal`, multi-result ops, nested output structures, regions/control flow, vLLM integration, Inductor, and source-location maps. Extensibility for these is not designed in beyond what the schema-driven approach gives for free.

### Anti-goals (failure even if tests pass)

- Silently skipping, replacing, or approximating an MLIR operation; returning the original FX graph instead of a rebuilt one.
- A per-operator handler for each ATen op instead of resolving through the dispatcher schema.
- Reconstructing from anything other than the MLIR (plus the output spec): no side channel carrying the original FX nodes.

## Approaches considered

| Approach | Trade-off | Decision |
| --- | --- | --- |
| **Schema-driven walker.** A small table for structural ops (constants, lists, return); every `torch.aten.*` op resolved to its `OpOverload` and bound via its schema. | New ATen ops usually cost only a test. Depends on private schema API, isolated in one adapter. Matches the parent spec §5. | **Chosen.** |
| Hand-written handler per ATen op. | Most readable per op, but every op is new code and it is the per-model/per-op table the parent spec rejects. | Rejected. |
| Schema-driven, but constants and lists become FX nodes too. | No benefit; FX passes Python literals in args natively, and extra nodes make rebuilt graphs harder to compare. | Rejected. |

Output structure: the MLIR alone cannot distinguish `return y` from `return (y,)` because the importer flattens outputs. Chosen: the importer records a pytree `TreeSpec` of the output and `rebuild_fx` takes it as an optional argument. Rejected: inferring structure from the MLIR alone, which breaks the `gm.forward` calling convention a Dynamo backend must preserve.

## Interfaces

### Importer change (`python/src/kashi/importer.py`)

```python
@dataclass(frozen=True)
class ImportedModule:
    module: ir.Module                 # Torch-dialect module, func.func @main
    out_spec: pytree.TreeSpec          # structure of the FX graph output

def import_fx(gm, example_inputs) -> ImportedModule
```

- `out_spec` is `torch.utils._pytree.tree_flatten(output_node.args[0])[1]` taken from the normalized graph that is imported.
- `import_fx` now returns `ImportedModule` instead of the `FxImporter`. The existing tests change `result.module_op.verify()` to `result.module.operation.verify()`; `str(result.module)` is unchanged.

### Reconstruction (`python/src/kashi/rebuild.py`)

```python
def rebuild_fx(module: ir.Module, out_spec: pytree.TreeSpec | None = None) -> torch.fx.GraphModule
```

- Does not modify `module`. Returns `torch.fx.GraphModule(torch.nn.Module(), graph)` after `graph.lint()`.
- With `out_spec=None`: one returned value is output bare; several are output as a flat tuple.
- With `out_spec`: its leaf count must equal the number of `func.return` operands, otherwise fail; the output is `tree_unflatten(returned_values, out_spec)`.

## Translation rules

The module must contain exactly one `func.func` named `main` with a single block; otherwise fail. Each block argument becomes a placeholder (`arg0`, `arg1`, …) in order. Block arguments must be `!torch.vtensor` types; any other argument type fails (symbolic inputs are out of scope).

Operations are visited in block order. Each SSA value maps to either an FX `Node` or a plain Python value.

| MLIR op | Result |
| --- | --- |
| `torch.constant.int` / `.float` / `.bool` | Python `int` / `float` / `bool` from the `value` attribute |
| `torch.constant.none` | `None` |
| `torch.constant.device` | the device string (e.g. `"cpu"`), converted only when bound to a `Device` schema slot |
| `torch.prim.ListConstruct` | Python `list` of the operand values (nodes or literals) |
| `torch.prim.ListUnpack` | for result *i*, an `operator.getitem(src, i)` node; `src` must be an FX node |
| `torch.aten.<name>[.<overload>]` | one `call_function` node targeting the resolved `OpOverload` (below) |
| `func.return` | graph output (see Interfaces) |
| anything else | `NotImplementedError` |

### ATen op resolution and binding

1. Name `torch.aten.<name>` resolves to `torch.ops.aten.<name>.default`; `torch.aten.<name>.<overload>` to `torch.ops.aten.<name>.<overload>`. Missing packet or overload fails.
2. The op must have exactly one result; otherwise fail.
3. The operand count must equal the schema's argument count (the importer emits every argument explicitly); otherwise fail.
4. Operands bind in schema order: non-keyword-only arguments become positional `args`, keyword-only arguments become `kwargs` by name.
5. Values are decoded by the schema argument's **real type** (`Argument.real_type`, since `Argument.type` reports `ScalarType` as `int`):
   - `ScalarType` / `ScalarType?` with an `int` → `torch.dtype`, via the inverse of the importer's `TORCH_DTYPE_TO_INT`.
   - `Device` / `Device?` with a `str` → `torch.device`.
   - `Layout?` / `MemoryFormat?` must be `None`; a non-`None` value fails (out of scope).
   - Everything else passes through unchanged.

Private APIs (`OpOverload._schema`, `Argument.real_type`, `fx_importer.TORCH_DTYPE_TO_INT`) are used only from a small adapter section at the top of `rebuild.py`.

ATen nodes are emitted even if they have no users. Rebuilt nodes carry no `meta["val"]`; they are not needed for execution.

### Diagnostics

Every failure raises `NotImplementedError` (matching the importer's convention) whose message names the MLIR op name, its location, and the reason, e.g. `Unsupported MLIR op torch.aten.foo at loc(...): no torch.ops.aten.foo`, or the expected vs observed counts for a mismatch. Messages never include tensor contents.

## Testing

Follow the conventions in `python/tests/test_import_fx.py`: `pytest.importorskip("torch")`, small parametrized tests, structural assertions rather than full-text comparisons.

### `python/tests/test_rebuild_fx.py`: independent reconstruction

Hand-written MLIR strings parsed with `ir.Module.parse` inside `with ir.Context():` (the IREE context registers the `torch` dialect). None of these use `import_fx`.

- One test per op: linear (with a bias tensor and with `torch.constant.none` bias), embedding, view, chunk + `ListUnpack`, `empty.memory_format` with an f16 dtype int and a `cpu` device. Assert the expected `OpOverload` targets and decoded arguments (e.g. `dtype=torch.float16`, `device=torch.device("cpu")`), then run the rebuilt module on random inputs and compare with calling the ATen op directly (shape and dtype only for `empty`).
- Output structure: `out_spec=None` gives bare/tuple outputs; a supplied spec for `(y,)` returns a 1-tuple; a mismatched leaf count fails.
- Rejections: an unknown ATen name, a non-`torch` op (e.g. `arith.constant`), and a non-tensor block argument (`!torch.int`) each raise `NotImplementedError` naming the op or argument.

### `python/tests/test_round_trip.py`: FX → MLIR → FX numerical correctness

For every graph form in `test_import_fx.py` (ATen and raw call forms of linear, embedding, view, chunk/getitem, empty, and `torch.Size` → empty): build the graph, `import_fx`, `rebuild_fx(imported.module, imported.out_spec)`, run the original and rebuilt modules on the same inputs, and assert equal output structure (types and lengths) and `torch.testing.assert_close` values. For `empty`, compare shape and dtype only.

## Left to the builder

- Internal structure of `rebuild.py` (dispatch dict vs `if` chain for the structural ops) and helper naming.
- Whether round-trip tests share graph builders with `test_import_fx.py` or define their own.
- Exact diagnostic wording, within the content rules above.
