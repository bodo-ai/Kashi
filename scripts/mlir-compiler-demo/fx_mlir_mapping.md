# FX ↔ MLIR mapping in the compiler demo

This table documents the explicit mappings in `simple_llm.py` for the Dynamo FX graph used by the demo. `mlir_roundtrip_backend` applies `normalize_fx` and `normalize_torch_size`, calls Torch-MLIR's `stateless_fx_import`, then reconstructs a `GraphModule` with `mlir_to_fx`.

## FX nodes and operands

| FX node / target | Local normalization | MLIR emitted by `stateless_fx_import` | Reverse reconstruction / notes |
|---|---|---|---|
| `placeholder` | None | `func.func @main` block argument, typed from FX value metadata | Becomes an FX placeholder (`arg0`, `arg1`, …). |
| Graph `output` | None | `func.return` with output operands | Becomes `graph.output(tuple(outputs))`. |
| `torch._C._nn.linear` (`call_function`) | Retargeted to `torch.ops.aten.linear.default` | `torch.aten.linear` | Resolved as `aten.linear.default`; arguments reconstructed from the ATen schema. |
| `torch._C._autograd._get_data_attr` (`Tensor.data`) | Retargeted to `torch.ops.aten.detach.default` | `torch.aten.detach` | Resolved as `aten.detach.default`; preserves the inference graph's detached tensor value. |
| `torch.nn.functional.embedding(input_ids, weight, ...)` | Retargeted to `aten.embedding.default`; uses only the first two arguments and forces `(padding_idx=-1, scale_grad_by_freq=False, sparse=False)` | `torch.aten.embedding` | Resolved as `aten.embedding.default`. Extra `F.embedding` options such as `padding_idx` are not preserved. |
| `torch.empty(size, ...)` | Retargeted to `aten.empty.memory_format`; asserts exactly one positional argument and retains kwargs | `torch.aten.empty.memory_format`, with defaults/kwargs represented as operands/constants | Resolved as `aten.empty.memory_format`; schema reconstructs keyword-only arguments as kwargs. |
| `torch.Size(shape)` (`call_function`) | Removed by `normalize_torch_size`; uses are replaced by a Python list of the shape values | No `torch.Size` operation. Shape lists may become `torch.prim.ListConstruct` / `!torch.list<int>` | `torch.Size` is not reconstructed; the list is the supported shape representation. |
| `operator.getitem` on a split/chunk result | No local retargeting | Typically `torch.prim.ListUnpack` for the list result | `ListUnpack` results become FX `operator.getitem` nodes, one per element. |
| `Tensor.long()` (`call_method`) | Retargeted to `aten.to.dtype(x, torch.int64)` | `torch.aten.to.dtype` plus scalar-type constant operands | The schema maps the encoded integer dtype back through `INT_TO_TORCH_DTYPE`. |
| `Tensor.size()` (`call_method`, no dimension) | Retargeted to `aten.size.default(x)` | `torch.aten.size`, returning a shape list | Reconstructed by resolving the default ATen overload. `torch.Size` metadata is typed as `!torch.list<int>`. |
| `Tensor.size(dim)` (`call_method`, with dimension) | Retargeted to `aten.size.int(x, dim)`; dimension becomes positional | `torch.aten.size.int` | Reconstructed by resolving the `.int` overload and its schema. |
| `Tensor.split(sections, dim)` where `sections` is a list/tuple | Retargeted to `aten.split_with_sizes.default(x, sections, dim)` | `torch.aten.split_with_sizes`, normally returning `!torch.list<vtensor>` | List elements are reconstructed through `torch.prim.ListUnpack` / `operator.getitem`. |
| `Tensor.split(size, dim)` where `size` is an integer | Retargeted to `aten.split.Tensor(x, size, dim)` | `torch.aten.split.Tensor` | Reconstructed by resolving the `.Tensor` overload and its schema. |
| `Tensor.view(shape)` (`call_method`) | Retargeted to `aten.view.default`. Static tuple/list shapes are flattened; a single FX node whose value is `torch.Size` stays a direct shape operand | `torch.aten.view` with a `!torch.list<int>` shape operand | Reconstructed by resolving the default ATen overload. Keeping a dynamic size node direct avoids a nested list. |
| `Tensor.chunk(chunks, dim)` (`call_method`) | Unpacks exactly two positional args and reads `dim` from kwargs only (default `0`); a third positional `dim` raises `ValueError` | `torch.aten.chunk`, returning `!torch.list<vtensor>` | Reconstructed by resolving the default overload; list elements use `ListUnpack`. |
| Registered `torch.ops.aten.*` overload (`call_function`) | No local rewrite unless listed above | Corresponding `torch.aten.<op>[.<overload>]` when supported by the installed Torch-MLIR importer | `resolve_aten_target` resolves the namespace and overload; `reconstruct_args` uses the operator schema. Reverse reconstruction requires operand/schema arity to match and exactly one result. This is pass-through support, not a local catalog of every ATen op. |
| Python literal operands (`None`, bool, int, float, str, device) | No local rewrite | `torch.constant.none`, `.bool`, `.int`, `.float`, `.str`, or `.device` as appropriate | Reconstructed as Python values or `torch.device`. Dtype scalar constants are converted back to `torch.dtype` when the schema argument is `dtype`. |
| Python list/tuple operands | No local rewrite | Commonly `torch.prim.ListConstruct` with `!torch.list<int>` or `!torch.list<vtensor>` | `ListConstruct` becomes a Python list; `ListUnpack` becomes `operator.getitem`. |
| Other `call_function`, `call_method`, or `call_module` nodes | No general local rewrite | Passed to `stateless_fx_import`; succeeds only if that importer recognizes the target | Unsupported MLIR operations and unknown `torch.operator` names raise `NotImplementedError` in the reverse importer. |

## vLLM custom operators

These FX targets are normalized from `OpOverloadPacket` to `.default`; Torch-MLIR imports them as `torch.operator` operations. The names shown are the `name` attributes used by `mlir_to_fx`.

| FX target | MLIR operation | MLIR result arity | Reverse mapping |
|---|---|---:|---|
| `torch.ops._C.rms_norm` | `torch.operator "torch._C.rms_norm"` | 0 (mutates output argument) | `torch.ops._C.rms_norm.default` |
| `torch.ops._C.fused_add_rms_norm` | `torch.operator "torch._C.fused_add_rms_norm"` | 0 (side effect) | `torch.ops._C.fused_add_rms_norm.default` |
| `torch.ops._C.rotary_embedding` | `torch.operator "torch._C.rotary_embedding"` | 0 (mutates query/key) | `torch.ops._C.rotary_embedding.default` |
| `torch.ops._C.silu_and_mul` | `torch.operator "torch._C.silu_and_mul"` | 0 (mutates output argument) | `torch.ops._C.silu_and_mul.default` |
| `torch.ops.vllm.unified_kv_cache_update` | `torch.operator "torch.vllm.unified_kv_cache_update"` | 1 | `torch.ops.vllm.unified_kv_cache_update.default` |
| `torch.ops.vllm.unified_attention_with_output` | `torch.operator "torch.vllm.unified_attention_with_output"` | 0 (side effect) | `torch.ops.vllm.unified_attention_with_output.default` |

## FX metadata to Torch types

| FX value metadata | `VllmContextCache.value_info_to_type` result |
|---|---|
| `FakeScriptObject` | `!torch.any` |
| Tuple containing only `FakeTensor` values | `!torch.list<vtensor>` |
| `torch.Size` | `!torch.list<int>` |
| Other metadata | Delegated to Torch-MLIR's `ContextCache` |

The local table is exhaustive for explicit branches in this demo, not for the full FX/Torch-MLIR ecosystem. Other registered ATen overloads are imported by `stateless_fx_import` when the installed Torch-MLIR version supports them; the reverse path resolves their emitted `torch.aten` names dynamically and uses their schemas.
