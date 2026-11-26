"""Check deterministic, distinct Qwen3.8 test weights before kernel packing."""

import torch
from vllm_test_utils import fill_qwen38_weights


def test_qwen38_initializes_distinct_fp8_layers():
    """Layer reuse must change results even when corresponding shapes match."""

    def make_model():
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for layer in model.layers:
            layer.weight = torch.nn.Parameter(
                torch.zeros(16, 16, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            layer.weight_scale_inv = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.float32), requires_grad=False
            )
        return model

    first, second = make_model(), make_model()
    fill_qwen38_weights(first)
    fill_qwen38_weights(second)
    assert not torch.equal(
        first.layers[0].weight.float(), first.layers[1].weight.float()
    )
    for actual, expected in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(actual.float(), expected.float())
    for layer in first.layers:
        torch.testing.assert_close(layer.weight_scale_inv, torch.full((1,), 0.01))
