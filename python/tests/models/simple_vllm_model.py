
"""Minimal vLLM model for end-to-end compiler integration tests."""

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention


@support_torch_compile
class SimpleVllm(nn.Module):
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


class SimpleVllmForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        model_prefix = f"{prefix}.model" if prefix else "model"

        self.model = SimpleVllm(
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
