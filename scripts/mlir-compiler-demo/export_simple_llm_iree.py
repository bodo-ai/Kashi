"""Export the toy LLM from simple_llm.py to MLIR with IREE-Turbine.

Counterpart of simple_llm.py (vLLM -> torch.compile -> torch-mlir) so the two
MLIR outputs can be compared. Structure follows export_qwen.py
(externalize params -> tensor-in/tensor-out wrapper -> aot.export -> save_mlir).

Run in the IREE venv (needs torch + iree-turbine; does NOT need vLLM or
torch-mlir):

    python export_simple_llm_iree.py [--seq-len 128] [--dtype float16]
                                     [--attn eager|sdpa] [--dynamic]
                                     [--with-lm-head]

=============================================================================
DEVIATIONS FROM simple_llm.py (all deliberate; needed for a fair comparison)
=============================================================================
1. vLLM is NOT imported. simple_llm.py's MyModel uses vllm's `Attention`
   layer, which is a paged-KV-cache custom op (torch.ops.vllm.*) that needs
   vLLM's forward context and cannot be traced by iree-turbine. It is replaced
   by `SimpleCausalAttention` below, a plain-torch causal self-attention with
   the same head count / head size / scale. THIS IS THE ONE SUBSTANTIVE
   DIFFERENCE: in the torch-mlir output the attention appears as an opaque
   custom-op call (and KV-cache write); here it is expanded into real ops
   (matmul/softmax or aten.scaled_dot_product_attention). Expect the
   attention region to differ; embed / qkv_proj / chunk / o_proj should line up.
2. Model dimensions are hard-coded to facebook/opt-125m's config, which
   simple_llm.py takes from `hf_config` (hidden=768, heads=12, vocab=50272).
   Hard-coded so no network / HF download is required. Only the config is
   borrowed; the weights are random (like vLLM's load_format="dummy") and
   externalized, so they appear as flow.parameter.named, not constants.
3. Input is a flat 1-D `input_ids[num_tokens]`, as vLLM feeds it (no batch
   dim), not (1, seq) like export_qwen.py. vLLM's `positions` argument is
   dropped: MyModel never uses it (no RoPE / position embedding), so it would
   be a dead argument.
4. Exported graph = MyModel.forward (embed -> qkv_proj -> chunk -> attn ->
   o_proj -> hidden states). That is what vLLM compiles; it applies
   `lm_head` outside the compiled graph via compute_logits(). Pass
   --with-lm-head to also include lm_head and emit logits (like export_qwen).
5. Shapes: static by default (like export_qwen). vLLM compiles with a
   symbolic num_tokens, so pass --dynamic to export a dynamic first dim.
6. dtype defaults to float16 (matches export_qwen.py and opt-125m's
   checkpoint dtype). vLLM's actual dtype depends on the platform
   (model_config.dtype="auto"); override with --dtype if torch-mlir's output
   shows something else.
7. No `breakpoint()` and no reference-vs-export numerical check beyond a
   shape/dtype print.
"""

import argparse
import os

import torch
import torch.nn as nn
import iree.turbine.aot as aot

# --- Config copied from facebook/opt-125m (what simple_llm.py's hf_config is)
HIDDEN_SIZE = 768
NUM_HEADS = 12
VOCAB_SIZE = 50272


class SimpleCausalAttention(nn.Module):
    """Plain-torch causal self-attention for one un-cached token sequence.

    Args:
        num_heads: Number of query/key/value heads.
        head_size: Width of each head; input width is ``num_heads * head_size``.
        scale: Multiplier applied to query-key scores before softmax.
        impl: ``"eager"`` for explicit matmul/mask/softmax, or ``"sdpa"``
            for PyTorch's scaled-dot-product attention operator.

    Inputs and output have shape ``[T, num_heads * head_size]``. Internally,
    tensors become ``[T, H, D]`` and then ``[1, H, T, D]``; the result is
    transposed back and flattened. This models causal prefill without a KV
    cache and assumes MHA (all heads are KV heads), not vLLM's paged cache.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, impl: str):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.impl = impl

    def forward(self, q, k, v):
        """Attend over a sequence; q, k, and v each have shape ``[T, H*D]``."""
        T = q.shape[0]
        # [T, H*D] -> [1, H, T, D]
        def split(t):
            return t.view(T, self.num_heads, self.head_size).transpose(0, 1).unsqueeze(0)

        q, k, v = split(q), split(k), split(v)

        if self.impl == "sdpa":
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale
            )
        else:
            # "eager": explicit matmul/softmax, as export_qwen.py does with
            # attn_implementation="eager".
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            mask = torch.full((T, T), float("-inf"), dtype=q.dtype, device=q.device)
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)

        # [1, H, T, D] -> [T, H*D]
        return out.squeeze(0).transpose(0, 1).reshape(T, self.num_heads * self.head_size)


class MyModel(nn.Module):
    """Embedding, fused QKV projection, causal attention, and output projection.

    The fused projection is split along its feature axis into three ``[T, H*D]``
    tensors; attention restores that flattened layout for ``o_proj``. Layer
    names match ``simple_llm.MyModel`` for parameter-name comparisons.
    """

    def __init__(self, attn_impl: str):
        """Construct OPT-125m-shaped layers using eager or SDPA attention."""
        super().__init__()
        hidden_size = HIDDEN_SIZE
        head_size = hidden_size // NUM_HEADS

        self.embed_tokens = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn = SimpleCausalAttention(
            num_heads=NUM_HEADS,
            head_size=head_size,
            scale=head_size ** -0.5,
            impl=attn_impl,
        )
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input_ids):
        """Embed flat token IDs ``[T]`` to hidden states ``[T, HIDDEN_SIZE]``."""
        x = self.embed_tokens(input_ids)
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        x = self.attn(q, k, v)
        x = self.o_proj(x)
        return x


class MyModelForCausalLM(nn.Module):
    """Model body and optional vocabulary logits projection."""
    def __init__(self, attn_impl: str):
        """Create the model body and vocabulary projection.

        Args:
            attn_impl: Attention implementation used by the model body.
        """
        super().__init__()
        self.model = MyModel(attn_impl)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)


# Expose a simple tensor-in/tensor-out interface.
class SimpleLLMForward(torch.nn.Module):
    """Tensor-in/tensor-out adapter selecting hidden states or vocabulary logits.

    Args:
        model: Model whose parameters are exposed to the exported program.
        with_lm_head: Whether to include the vocabulary projection in the graph.
    """
    def __init__(self, model: MyModelForCausalLM, with_lm_head: bool):
        """Select hidden-state or logits output for the exported interface."""
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
    """Build the example graph, export it with IREE-Turbine, and save MLIR."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128,
                        help="num_tokens (default 128 = simple_llm.py's max_model_len)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--dynamic", action="store_true",
                        help="export num_tokens as a dynamic dimension")
    parser.add_argument("--with-lm-head", action="store_true")
    parser.add_argument("--output", default="artifacts/simple-llm-frontend.mlir")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(42)

    # Random weights on CPU, as with vLLM's load_format="dummy".
    model = MyModelForCausalLM(args.attn).to(dtype=getattr(torch, args.dtype))
    model.eval()
    aot.externalize_module_parameters(model)

    wrapper = SimpleLLMForward(model, args.with_lm_head).eval()

    # Fixed-shape example input (1-D, no batch dim, like vLLM's flat tokens).
    example_input = torch.ones((args.seq_len,), dtype=torch.long)

    with torch.no_grad():
        reference = wrapper(example_input)

    print("Reference output:", reference.shape)
    print("Reference dtype:", reference.dtype)

    dynamic_shapes = None
    if args.dynamic:
        # Tokens in the example input must be > 1 for a dynamic dim.
        dynamic_shapes = {"input_ids": {0: torch.export.Dim("num_tokens", min=2)}}

    exported = aot.export(wrapper, example_input, dynamic_shapes=dynamic_shapes)
    exported.save_mlir(args.output)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
