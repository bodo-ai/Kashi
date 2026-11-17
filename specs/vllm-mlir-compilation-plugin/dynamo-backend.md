# Standalone TorchDynamo Backend

**Status:** Proposed; pending human approval
**Parent:** [vLLM FX–MLIR–FX Compilation Plugin](vllm-mlir-compilation.md) §3, §9.1 "Compilation backend", milestone M1 (architectural reference only; this document sets the scope)

## Intent

`import_fx` (FX → MLIR) and `rebuild_fx` (MLIR → FX) work and are tested separately and as a round trip. This component composes them into a standard PyTorch compilation backend so `torch.compile` can run a model through FX → MLIR → FX and execute the rebuilt graph in eager PyTorch. It is the standalone, vLLM-independent backend of milestone M1.

## Scope

### Goals

- `compile_fx(gm, example_inputs)`: a plain callable usable as `torch.compile(model, backend=compile_fx)`.
- Execute the reconstructed FX graph with eager PyTorch.
- Integration tests through `torch.compile` on small models using the existing static-shape operator support, verifying that the backend is invoked and that results are numerically equivalent to eager.

### Non-goals

Symbolic shapes, custom operators, opaque objects, vLLM integration, Inductor delegation, packaging entry points, named backend registration (`backend="kashi"`), backend options, a pass-pipeline hook, and `aot_autograd`/ATen-level capture.

### Anti-goals (failure even if tests pass)

- Returning the original `gm` or `gm.forward`, or otherwise falling back to eager when import or reconstruction fails.
- Test-only hooks or counters in product code.

## Approaches considered

| Approach | Trade-off | Decision |
| --- | --- | --- |
| **Direct composition.** `import_fx` → `verify()` → `rebuild_fx` → return `rebuilt.forward`. | Smallest; reuses existing interfaces unchanged. Dynamo already calls the result positionally with the graph's inputs, and `out_spec` restores the output structure. | **Chosen.** |
| Wrap the returned callable (input checks, call counting). | Extra code with no current need. | Rejected. |
| Track invocation inside the backend. | Puts test hooks in product code; a test-local spy does the same job. | Rejected. |
| Wrap with `aot_autograd` to receive ATen graphs. | Inductor-style, but bypasses `import_fx`'s call-form normalization and introduces decompositions outside our supported set. | Rejected. |

Backend selection: a plain callable (chosen by the user). In-process registration under a name mutates Dynamo's global registry on import and belongs with the out-of-scope packaging work.

## Interface (`python/src/kashi/backend.py`)

```python
def compile_fx(gm: torch.fx.GraphModule, example_inputs) -> Callable:
    imported = import_fx(gm, example_inputs)
    imported.module.operation.verify()
    return rebuild_fx(imported.module, imported.out_spec).forward
```

- No backend options; no side effects at import.
- Errors from `import_fx`, `verify()` or `rebuild_fx` propagate. Dynamo raises them by default (wrapped as `BackendCompilerFailed`); the backend never falls back.

## Observed Dynamo behavior (PyTorch 2.14, probe run during design)

- Module parameters arrive as graph placeholders (inlined built-in `nn.Module`s), interleaved with user inputs; `example_inputs` holds the corresponding real tensors and `Parameter`s. No `get_attr` support is needed.
- Raw call forms (`torch.nn.functional.embedding`, `torch._C._nn.linear`, `Tensor.chunk`, `Tensor.view`, `getitem`) are already handled by `import_fx`.
- A probe model (embedding → linear → `.chunk(3, dim=-1)` → linear → `.view`, returning a tuple) compiled through this composition matched eager results.
- Dynamo's default automatic-dynamic behavior recompiles with symbolic shapes when a compiled model is called with a new shape; that is out of scope, so tests compile with `dynamic=False`.

## Testing (`python/tests/test_backend.py`)

Tests use the `vllm-mlir` Pixi environment, `torch.compile(..., backend=spy, dynamic=False)`, and call `torch._dynamo.reset()` before each test so compilations do not leak between tests.

A test-local spy wraps `compile_fx`, recording each `gm` received and each callable returned. For each supported model:

1. The spy was called exactly once.
2. The returned callable is not the received `gm.forward` (and is the `forward` of a different `GraphModule`).
3. Compiled output equals eager output: same pytree structure, and `torch.testing.assert_close` on each leaf.

Models (small `nn.Module`s with fixed seeds):

- Embedding → linear → `.chunk(3, dim=-1)` → linear → `.view`, returning a tuple of two tensors.
- A single-tensor return, to cover bare vs tuple output.
- A model using `torch.empty(...)` (with constant size), comparing only shape and dtype of the uninitialized output.

Unsupported-op test: a model using an operation `import_fx` does not support (e.g. `torch.relu`) raises when compiled and called, rather than returning eager results.

## Left to the builder

Exact model definitions, sizes and seeds; the specific unsupported op; whether `compile_fx` carries type annotations beyond the above; asserting on the wrapped exception type vs its cause.
