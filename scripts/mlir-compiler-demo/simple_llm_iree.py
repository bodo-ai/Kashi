"""vLLM Dynamo FX -> IREE-packaged Torch FX importer -> MLIR -> FX round-trip.

Based on simple_llm.py. Keeps vLLM as the runtime and uses its custom ops.
This exercises IREE's packaged *Torch dialect FX importer*, not IREE native
compilation or Turbine's AOT pipeline.
"""

import os
import operator

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

import torch
import torch.nn as nn
import torch.fx as fx
from torch._dynamo import register_backend
from torch._library.fake_class_registry import FakeScriptObject
from torch.fx.node import map_arg
from torch._subclasses.fake_tensor import FakeTensor

from iree.compiler.ir import IntegerAttr, StringAttr, Type
from iree.compiler.extras.fx_importer import (
    FxImporter, ContextCache, TORCH_DTYPE_TO_INT,
)

from vllm import LLM, ModelRegistry, SamplingParams
from vllm.model_executor.layers.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig


class VllmContextCache(ContextCache):
    def value_info_to_type(
        self, val, *, tensor_meta=None, mutable=False
    ):
        if isinstance(val, FakeScriptObject):
            return Type.parse(
                "!torch.any",
                context=self._c,
            )

        if isinstance(val, tuple) and all(
            isinstance(x, FakeTensor) for x in val
        ):
            return Type.parse(
                "!torch.list<vtensor>",
                context=self._c,
            )

        return super().value_info_to_type(
            val,
            tensor_meta=tensor_meta,
            mutable=mutable,
        )



class VllmFxImporter(FxImporter):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self._cc = VllmContextCache(
            self._c,
            py_attr_tracker=self._py_attr_tracker,
        )


INT_TO_TORCH_DTYPE = {
    value: dtype
    for dtype, value in TORCH_DTYPE_TO_INT.items()
}


def reconstruct_args(target, operands, values):
    """ 
    Torch MLIR uses positional args, but in the reverse direction, 
    some functions might have keyword-only arguments (see aten.empty as an example). 
    """
    schema_args = target._schema.arguments

    if len(operands) != len(schema_args):
        raise ValueError(
            f"{target}: expected {len(schema_args)} "
            f"operands, got {len(operands)}"
        )

    args = []
    kwargs = {}

    for schema_arg, operand in zip(schema_args, operands):
        value = values[operand]

        # Torch-MLIR encodes ScalarType as an integer.
        if schema_arg.name == "dtype" and value is not None:
            value = INT_TO_TORCH_DTYPE[value]

        if schema_arg.kwarg_only:
            kwargs[schema_arg.name] = value
        else:
            args.append(value)

    return tuple(args), kwargs


def mlir_to_fx(module_op) -> fx.GraphModule:
    """Reconstruct Dynamo FX from IREE-packaged Torch-dialect MLIR."""
    graph = fx.Graph()

    # FxImporter.module_op is a builtin.module Operation, not a Module wrapper.
    module_op = getattr(module_op, "operation", module_op)
    module_body = module_op.regions[0].blocks[0]
    main = next(
        op for op in module_body.operations
        if op.operation.name == "func.func"
        and StringAttr(
            op.attributes["sym_name"]
        ).value == "main"
    )
    block = main.regions[0].blocks[0]

    values = {
        arg: graph.placeholder(f"arg{i}")
        for i, arg in enumerate(block.arguments)
    }

    torch_ops = {
        "torch.aten.linear": torch.ops.aten.linear.default,
        "torch.aten.view": torch.ops.aten.view.default,
        "torch.aten.embedding": torch.ops.aten.embedding.default,
        "torch.aten.chunk": torch.ops.aten.chunk.default,
        "torch.aten.empty.memory_format": torch.ops.aten.empty.memory_format
    }

    # Opaque custom vLLM operators.
    custom_ops = {
        "torch.vllm.unified_kv_cache_update":
            torch.ops.vllm.unified_kv_cache_update.default,

        "torch.vllm.unified_attention_with_output":
            torch.ops.vllm.unified_attention_with_output.default,
    }

    for op in block.operations:
        operation = op.operation
        name = operation.name

        if name == "func.return":
            outputs = tuple(
                values[arg] for arg in operation.operands
            )
            graph.output(outputs)

        elif name == "torch.constant.none":
            values[operation.results[0]] = None

        elif name == "torch.constant.bool":
            attr = operation.attributes["value"]
            values[operation.results[0]] = bool(IntegerAttr(attr).value)

        elif name == "torch.constant.device":
            attr = operation.attributes["value"]
            values[operation.results[0]] = torch.device(
                StringAttr(attr).value
            )

        elif name == "torch.constant.int":
            value = IntegerAttr(
                operation.attributes["value"]
            ).value

            values[operation.results[0]] = int(value)

        elif name == "torch.prim.ListConstruct":
            values[operation.results[0]] = [
                values[arg] for arg in operation.operands
            ]

        elif name == "torch.prim.ListUnpack":
            source = values[operation.operands[0]]

            for i, result in enumerate(operation.results):
                values[result] = graph.call_function(
                    operator.getitem,
                    args=(source, i),
                )

        elif name in torch_ops:
            args, kwargs = reconstruct_args(torch_ops[name], operation.operands, values)
            result = graph.call_function(
                torch_ops[name],
                args=args,
                kwargs=kwargs,
            )
            values[operation.results[0]] = result

        elif name == "torch.operator":
            op_name = StringAttr(operation.attributes["name"]).value
            if op_name in custom_ops:
                result = graph.call_function(
                    custom_ops[op_name],
                    args=tuple(values[arg] for arg in operation.operands),
                )
                if len(operation.results) == 1:
                    values[operation.results[0]] = result
                elif len(operation.results) == 0:
                    # Side-effect-only operation.
                    pass
                else:
                    raise NotImplementedError(
                        f"Multiple results for {op_name}"
                    )
            else:
                raise NotImplementedError(
                    f"Unsupported torch.operator: {op_name}"
                )

        else:
            raise NotImplementedError(
                f"Unsupported MLIR operation: {name}"
            )

    graph.lint()
    return fx.GraphModule(nn.Module(), graph)


def normalize_fx(gm: fx.GraphModule):
    """Normalize Dynamo callables to ATen for IREE's packaged FX importer."""

    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target is torch._C._nn.linear:
                node.target = torch.ops.aten.linear.default

            elif node.target is torch.nn.functional.embedding:
                # NOTE: aten.embedding expects arguments in a different order than F.embedding
                # TODO: support other arguments e.g. padding_idx
                input_ids, weight = node.args[:2]
                node.target = torch.ops.aten.embedding.default
                node.args = (weight, input_ids, -1, False, False)
                node.kwargs = {}

            elif node.target is torch.empty:
                assert len(node.args) == 1

                size = node.args[0]
                node.target = torch.ops.aten.empty.memory_format
                node.args = (size,)

            # vLLM custom ops:
            elif node.target is torch.ops.vllm.unified_kv_cache_update:
                node.target = torch.ops.vllm.unified_kv_cache_update.default

            elif node.target is torch.ops.vllm.unified_attention_with_output:
                node.target = torch.ops.vllm.unified_attention_with_output.default

        elif node.op == "call_method":
            if node.target == "view":
                x, *shape = node.args

                if len(shape) == 1 and isinstance(
                    shape[0], (tuple, list)
                ):
                    shape = list(shape[0])

                node.op = "call_function"
                node.target = torch.ops.aten.view.default
                node.args = (x, shape)
                node.kwargs = {}

            elif node.target == "chunk":
                x, chunks = node.args
                dim = node.kwargs.get("dim", 0)

                node.op = "call_function"
                node.target = torch.ops.aten.chunk.default
                node.args = (x, chunks, dim)
                node.kwargs = {}

        if "val" not in node.meta:
            if "example_value" in node.meta:
                node.meta["val"] = node.meta["example_value"]

    gm.graph.lint()
    gm.recompile()
    return gm


def normalize_torch_size(gm):
    """ Remove torch.Size constructor from the graph and replace all uses with a tuple/list """
    graph = gm.graph

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not torch.Size:
            continue

        assert len(node.args) == 1
        assert not node.kwargs
        assert isinstance(node.args[0], (tuple, list))

        shape = list(node.args[0])

        for user in list(node.users):
            user.args = map_arg(
                user.args,
                lambda n: shape if n is node else n,
            )
            user.kwargs = map_arg(
                user.kwargs,
                lambda n: shape if n is node else n,
            )

        graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return gm


@register_backend(name="mlir_roundtrip_backend")
def mlir_roundtrip_backend(gm: fx.GraphModule, example_inputs):
    print("\nOriginal Dynamo FX:")
    gm.graph.print_tabular()
    # Normalize raw Dynamo callables into ATen overloads for the importer.
    gm_norm = normalize_fx(gm)
    gm_norm = normalize_torch_size(gm_norm)

    print("\nNormalized ATen FX:")
    gm_norm.graph.print_tabular()

    # IREE packages the Torch-dialect FX importer. This does not invoke
    # IREE optimization or native code generation.
    importer = VllmFxImporter()
    importer.import_graph_module(gm_norm)
    module_op = importer.module_op

    print("\nGenerated MLIR:")
    print(module_op)

    gm_reconstructed = mlir_to_fx(module_op)
    print("\nReconstructed FX:")
    gm_reconstructed.graph.print_tabular()

    return gm_reconstructed.forward


@support_torch_compile
class MyModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_size = hidden_size // num_heads

        self.embed_tokens = nn.Embedding(
            config.vocab_size, hidden_size
        )

        self.qkv_proj = nn.Linear(
            hidden_size, 3 * hidden_size, bias=False
        )

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            scale=head_size ** -0.5,
            num_kv_heads=num_heads,
            prefix=f"{prefix}.attn",
        )

        self.o_proj = nn.Linear(
            hidden_size, hidden_size, bias=False
        )

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_input_ids(input_ids)

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        x = self.attn(q, k, v)
        x = self.o_proj(x)

        return x


class MyModelForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        model_prefix = f"{prefix}.model" if prefix else "model"

        self.model = MyModel(
            vllm_config=vllm_config,
            prefix=model_prefix,
        )

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)


def main():
    ModelRegistry.register_model(
        "MyModelForCausalLM", MyModelForCausalLM
    )

    torch.manual_seed(42)

    mlir_llm = LLM(
        model="facebook/opt-125m",
        hf_overrides={
            "architectures": ["MyModelForCausalLM"]
        },
        load_format="dummy",
        max_model_len=128,
        kv_cache_memory_bytes=256 * 1024 * 1024,  # 256 MiB
        compilation_config={
        "mode": 2,
        "backend": "mlir_roundtrip_backend",
    },
    )

    outputs = mlir_llm.generate(
        ["Hello, my name is"],
        SamplingParams(max_tokens=4),
    )

    mlir = outputs[0].outputs[0].token_ids
    print(mlir)


if __name__ == "__main__":
    main()
