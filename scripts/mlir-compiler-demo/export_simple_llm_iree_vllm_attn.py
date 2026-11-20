"""Export the toy LLM from simple_llm.py to MLIR with IREE-Turbine, keeping
vLLM's real `Attention` layer.

Variant of export_simple_llm_iree.py. Identical except that the attention layer
is NOT replaced by a plain-torch stand-in: MyModel uses
`vllm.model_executor.layers.attention.Attention` exactly as simple_llm.py does,
so (as with torch-mlir) attention should show up as an opaque vLLM custom-op
call in the exported IR rather than expanded matmul/softmax.

Structure follows export_qwen.py (externalize params -> tensor-in/tensor-out
wrapper -> aot.export -> save_mlir).

ENVIRONMENT: needs BOTH vLLM and iree-turbine importable in one interpreter
(unlike export_simple_llm_iree.py, which only needs iree-turbine). Not
installing anything for you; e.g. in a venv that has vllm:
    PYTHONPATH=<iree-venv>/lib/python3.12/site-packages python ...
(the iree dir is appended AFTER vllm's torch so vllm's torch wins), or
`uv pip install iree-turbine` into the vllm venv.

    python export_simple_llm_iree_vllm_attn.py [--seq-len 128]
        [--dtype float16] [--device cuda] [--dynamic] [--with-lm-head]

=============================================================================
DEVIATIONS FROM simple_llm.py
=============================================================================
Inherited from export_simple_llm_iree.py (see that file for rationale):
 * dims hard-coded to facebook/opt-125m (hidden=768, heads=12, vocab=50272);
   random, externalized weights (parameters become flow.parameter.named).
 * flat 1-D input_ids[num_tokens]; `positions` dropped (unused by MyModel).
 * graph = MyModel.forward (hidden states); --with-lm-head adds lm_head.
 * static shapes by default; --dynamic makes num_tokens symbolic.
 * dtype defaults to float16 (override with --dtype).

New, specific to keeping vLLM's Attention (SUBSTANTIVE; read these):
 1. Attention.__init__ needs a current VllmConfig (it registers itself in
    compilation_config.static_forward_context, selects an attention backend,
    reads cache config). We build a default `VllmConfig()` and construct the
    model inside `set_current_vllm_config(...)`. This config is NOT the one
    simple_llm.py gets from LLM(model="facebook/opt-125m", ...); it only
    exists to make the layer constructible. Backend selection therefore
    follows the local platform (FLASH_ATTN on this CUDA box), which does not
    affect the exported IR, but the layer object is real.
 2. Attention picks its backend dtype from torch.get_default_dtype(), so the
    default dtype is set to --dtype while constructing the model.
 3. The reference (eager) forward pass of export_simple_llm_iree.py is
    DROPPED: running vLLM's attention op eagerly requires a live vLLM
    forward context (attn metadata, KV cache) which does not exist here. The
    expected output shape/dtype is printed instead. No numerical check.
 4. torch.export sees the attention op as torch.ops.vllm.
    unified_attention_with_output (mutating `output`, resolved by layer name
    "model.attn", prefix mirrors simple_llm.py). The exported IR keeps the
    layer-name reference, but the op has no real implementation inside the
    IREE module: this MLIR is for inspection/comparison, NOT compilable by
    iree-compile into something runnable unless that op is later lowered.
 5. VLLM_USE_LAYERNAME=0 is set (before importing vllm). On torch>=2.11 vLLM
    passes the layer name to the attention op as an opaque `LayerName` object,
    which torch.export (dynamo export) cannot handle ("export does not
    support pregraph_bytecode"). With it disabled the layer name is a plain
    str attribute of the op call. simple_llm.py's torch.compile path does not
    need this, so the op's layer-name argument may look different there.
 6. Export is done via torch.export.export + aot.export(ExportedProgram), i.e.
    without iree-turbine's run_decompositions() step (see comment in main()).
    Side effect: the aten ops in the graph are not decomposed/functionalized
    like export_simple_llm_iree.py's. Closer to what torch-mlir sees in
    simple_llm.py (a pre-AOT dynamo graph).
 7. Model and example input are placed on --device (default "cuda"). Without
    this everything is on CPU and the IR records `torch.constant.device "cpu"`
    (from the `torch.empty(..., device=query.device)` in vLLM's Attention),
    whereas simple_llm.py runs on CUDA and records "cuda:0". Needs a visible
    GPU and a little GPU memory for the (random, ~190 MB fp16) weights.
"""

import argparse
import os

# Deviation 5 (see docstring): must be set BEFORE vllm is imported.
os.environ.setdefault("VLLM_USE_LAYERNAME", "0")

import torch
import torch.nn as nn
import iree.turbine.aot as aot

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention

# --- Config copied from facebook/opt-125m (what simple_llm.py's hf_config is)
HIDDEN_SIZE = 768
NUM_HEADS = 12
VOCAB_SIZE = 50272


class MyModel(nn.Module):
    """OPT-125m-shaped body using vLLM attention and matching simple_llm names.

    The fused QKV projection is chunked across its last dimension into three
    ``[T, H*D]`` tensors. The real vLLM layer is retained as an opaque attention
    operator in the export; its runtime cache/context is not provided here.
    """

    def __init__(self, prefix: str = ""):
        """Construct the body and register attention under its stable layer prefix.

        Args:
            prefix: Parent layer name used by vLLM's attention context.
        """
        super().__init__()
        hidden_size = HIDDEN_SIZE
        head_size = hidden_size // NUM_HEADS

        self.embed_tokens = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # Unchanged from simple_llm.py: the real vLLM attention layer.
        self.attn = Attention(
            num_heads=NUM_HEADS,
            head_size=head_size,
            scale=head_size ** -0.5,
            num_kv_heads=NUM_HEADS,
            prefix=f"{prefix}.attn",
        )
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input_ids):
        """Map flat token IDs ``[T]`` through the body to hidden states ``[T, HIDDEN_SIZE]``."""
        x = self.embed_tokens(input_ids)
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        x = self.attn(q, k, v)
        x = self.o_proj(x)
        return x


class MyModelForCausalLM(nn.Module):
    """Model body and optional vocabulary logits projection."""
    def __init__(self, prefix: str = ""):
        """Build the body and logits projection.

        Args:
            prefix: Prefix propagated to the vLLM attention layer name.
        """
        super().__init__()
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = MyModel(prefix=model_prefix)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)


# Expose a simple tensor-in/tensor-out interface.
class SimpleLLMForward(torch.nn.Module):
    """Tensor-in/tensor-out export adapter for hidden states or vocabulary logits.

    Args:
        model: Model whose parameters are externalized for export.
        with_lm_head: Whether to include the vocabulary projection in the graph.
    """
    def __init__(self, model: MyModelForCausalLM, with_lm_head: bool):
        """Select the graph output by enabling or omitting the logits projection."""
        super().__init__()
        self.model = model
        self.with_lm_head = with_lm_head

    def forward(self, input_ids):
        """Return per-token hidden states, or logits when the head is enabled."""
        hidden = self.model.model(input_ids)
        if self.with_lm_head:
            return self.model.lm_head(hidden)
        return hidden


def main():
    """Construct vLLM attention under its required config and save exported MLIR."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128,
                        help="num_tokens (default 128 = simple_llm.py's max_model_len)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dynamic", action="store_true",
                        help="export num_tokens as a dynamic dimension")
    parser.add_argument("--with-lm-head", action="store_true")
    parser.add_argument("--device", default="cuda",
                        help="device for model and example input (default cuda)")
    parser.add_argument("--output", default="artifacts/simple-llm-vllm-attn-frontend.mlir")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(42)

    # Deviations 1 & 2: Attention needs a current VllmConfig and reads the
    # default dtype at construction time.
    vllm_config = VllmConfig()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            # Random weights, as with vLLM's load_format="dummy". Created on
            # --device (deviation 7).
            model = MyModelForCausalLM().to(dtype=dtype, device=args.device)
    finally:
        torch.set_default_dtype(prev_dtype)
    model.eval()
    aot.externalize_module_parameters(model)

    wrapper = SimpleLLMForward(model, args.with_lm_head).eval()

    # Fixed-shape example input (1-D, no batch dim, like vLLM's flat tokens).
    example_input = torch.ones((args.seq_len,), dtype=torch.long, device=args.device)

    # Deviation 3: no eager reference run (vLLM attention needs a forward
    # context). Expected output shape/dtype instead.
    out_features = VOCAB_SIZE if args.with_lm_head else HIDDEN_SIZE
    print("Expected output:", (args.seq_len, out_features), dtype)

    dynamic_shapes = None
    if args.dynamic:
        # Tokens in the example input must be > 1 for a dynamic dim.
        dynamic_shapes = {"input_ids": {0: torch.export.Dim("num_tokens", min=2)}}

    # Deviation 6: call torch.export ourselves and hand aot.export the
    # ExportedProgram. aot.export(nn.Module) would additionally run
    # run_decompositions(), which functionalizes the in-place attention op into
    # `auto_functionalized_v2`; iree's FxImporter does not support that HOP.
    # Skipping it keeps the unfunctionalized graph, where the attention op is a
    # plain mutating call, which imports as an opaque `torch.operator`.
    with torch.no_grad(), set_current_vllm_config(vllm_config):
        program = torch.export.export(
            wrapper, args=(example_input,), dynamic_shapes=dynamic_shapes, strict=True
        )
    exported = aot.export(program)
    exported.save_mlir(args.output)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
