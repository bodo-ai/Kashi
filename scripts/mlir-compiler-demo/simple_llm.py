
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

import torch
import torch.nn as nn
import torch.fx as fx
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func


from torch_mlir.fx import stateless_fx_import
from torch_mlir.ir import Module, IntegerAttr, StringAttr

from vllm import LLM, ModelRegistry, SamplingParams
from vllm.model_executor.layers.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig


def mlir_to_fx(module: Module) -> fx.GraphModule:
    graph = fx.Graph()

    # Find the main MLIR function.
    main = next(
        op for op in module.body.operations
        if op.operation.name == "func.func"
        and StringAttr(
            op.attributes["sym_name"]
        ).value == "main"
    )

    block = main.regions[0].blocks[0]

    # MLIR SSA value -> FX node or Python constant.
    values = {
        arg: graph.placeholder(f"arg{i}")
        for i, arg in enumerate(block.arguments)
    }

    ops = {
        "torch.aten.linear": torch.ops.aten.linear.default,
        "torch.aten.view": torch.ops.aten.view.default,
    }

    for op in block.operations:
        operation = op.operation
        name = operation.name

        if name == "torch.constant.int":
            value = IntegerAttr(
                operation.attributes["value"]
            ).value

            values[operation.results[0]] = int(value)

        elif name == "torch.prim.ListConstruct":
            values[operation.results[0]] = [
                values[arg] for arg in operation.operands
            ]

        elif name in ops:
            args = tuple(
                values[arg] for arg in operation.operands
            )

            result = graph.call_function(
                ops[name],
                args=args,
            )

            values[operation.results[0]] = result

        elif name == "func.return":
            outputs = tuple(
                values[arg] for arg in operation.operands
            )
            graph.output(outputs)

        else:
            raise NotImplementedError(
                f"Unsupported MLIR operation: {name}"
            )

    graph.lint()
    return fx.GraphModule(nn.Module(), graph)


def normalize_fx(gm: fx.GraphModule):
    """ 
    Convert FX graph into format expected by Torch MLIR.
    Torch MLIR expects FX graphs to use ATen operations.
    """
    breakpoint()

    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target is torch._C._nn.linear:
                node.target = torch.ops.aten.linear.default

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

        if "val" not in node.meta:
            if "example_value" in node.meta:
                node.meta["val"] = node.meta["example_value"]

    gm.graph.lint()
    gm.recompile()
    return gm


@register_backend(name="mlir_roundtrip_backend")
def mlir_roundtrip_backend(gm: fx.GraphModule, example_inputs):
    print("\nOriginal Dynamo FX:")
    gm.graph.print_tabular()

    gm = normalize_fx(gm)

    print("\nNormalized ATen FX:")
    gm.graph.print_tabular()

    module = stateless_fx_import(
        gm,
        output_type="raw",
    )

    print("\nGenerated MLIR:")
    print(module)

    gm = mlir_to_fx(module)

    print("\nReconstructed FX:")
    gm.graph.print_tabular()

    return gm.forward


def debug_backend(graph: torch.fx.GraphModule, example_inputs):
    print("\n========== Dynamo FX Graph ==========", flush=True)
    breakpoint()
    graph.graph.print_tabular()
    print("=====================================\n", flush=True)

    compiled = torch._inductor.compile(graph, example_inputs)

    return make_boxed_func(compiled)


# Runs after Aten normalization/decomposition
debug_backend_aot = aot_autograd(
    fw_compiler=debug_backend,
    inference_compiler=debug_backend,
    decompositions={},
)

register_backend(name="debug_backend")(debug_backend_aot)

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

    # baseline_llm = LLM(
    #     model="facebook/opt-125m",
    #     hf_overrides={
    #         "architectures": ["MyModelForCausalLM"]
    #     },
    #     load_format="dummy",
    #     max_model_len=128,
    #     kv_cache_memory_bytes=256 * 1024 * 1024,  # 256 MiB
    #     compilation_config={
    #     "mode": 2,
    #     "backend": "debug_backend",
    # },
    # )

    # outputs = baseline_llm.generate(
    #     ["Hello, my name is"],
    #     SamplingParams(max_tokens=4),
    # )

    # baseline = outputs[0].outputs[0].token_ids


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
