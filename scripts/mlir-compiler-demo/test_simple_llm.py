import importlib.util
import operator
import unittest
from unittest.mock import patch

import torch
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from simple_llm import (
    VllmFxImporter,
    mlir_to_fx,
    normalize_fx,
    normalize_torch_size,
)
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.fx import stateless_fx_import


class _ValueRecorder(fx.Interpreter):
    def run_node(self, node):
        value = super().run_node(node)
        node.meta["val"] = value
        node.meta["example_value"] = value
        return value


def _round_trip(gm):
    context = ir.Context()
    torch_d.register_dialect(context)
    importer = VllmFxImporter(context=context)
    module = stateless_fx_import(
        gm,
        output_type="raw",
        fx_importer=importer,
    )
    return mlir_to_fx(module)


def _tensor_method_graph():
    graph = fx.Graph()
    x = graph.placeholder("x")
    data_value = graph.call_function(
        torch._C._autograd._get_data_attr,
        (x,),
    )
    long_value = graph.call_method("long", (data_value,))
    shape = graph.call_method("size", (long_value,))
    chunks = graph.call_method("split", (long_value, [1, 2, 1]), {"dim": -1})
    first_chunk = graph.call_function(operator.getitem, (chunks, 0))
    viewed = graph.call_method("view", (first_chunk, (1, 1)))
    graph.output((data_value, long_value, shape, first_chunk, viewed))
    return fx.GraphModule(nn.Module(), graph)




class Qwen3FxNormalizationTest(unittest.TestCase):
    def test_normalize_qwen3_fx_ops(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        data_value = graph.call_function(
            torch._C._autograd._get_data_attr,
            (x,),
        )
        long_value = graph.call_method("long", (data_value,))
        shape = graph.call_method("size", (long_value,))
        chunks = graph.call_method("split", (long_value, [1, 2, 1]), {"dim": -1})
        first_chunk = graph.call_function(operator.getitem, (chunks, 0))
        viewed = graph.call_method("view", (first_chunk, (1, 1)))
        graph.output((data_value, long_value, shape, first_chunk, viewed))
        gm = fx.GraphModule(nn.Module(), graph)

        normalize_fx(gm)
        normalize_torch_size(gm)

        function_targets = {
            node.target for node in gm.graph.nodes if node.op == "call_function"
        }
        self.assertIn(torch.ops.aten.to.dtype, function_targets)
        self.assertIn(torch.ops.aten.size.default, function_targets)
        self.assertIn(torch.ops.aten.split_with_sizes.default, function_targets)
        self.assertIn(torch.ops.aten.view.default, function_targets)
        self.assertIn(torch.ops.aten.detach.default, function_targets)
        self.assertNotIn(torch.Size, function_targets)
        self.assertTrue(
            any(node.target is operator.getitem for node in gm.graph.nodes)
        )

        split_node = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.split_with_sizes.default
        )
        self.assertEqual(split_node.args[1], [1, 2, 1])
        self.assertEqual(split_node.args[2], -1)

    def test_view_uses_size_result_as_shape(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        shape = graph.call_method("size", (x,))
        viewed = graph.call_method("view", (x, shape))
        graph.output(viewed)
        gm = fx.GraphModule(nn.Module(), graph)

        with FakeTensorMode() as mode:
            _ValueRecorder(gm).run(mode.from_tensor(torch.empty((1, 1))))
        normalize_fx(gm)

        size_node = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.size.default
        )
        view_node = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.view.default
        )
        self.assertIs(view_node.args[1], size_node)

    def test_qwen3_tensor_ops_round_trip(self):
        gm = _tensor_method_graph()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        expected = gm(x)
        with FakeTensorMode() as mode:
            _ValueRecorder(gm).run(mode.from_tensor(x))
        normalize_fx(gm)
        normalize_torch_size(gm)

        try:
            reconstructed = _round_trip(gm)
        except Exception as error:
            self.fail(f"Qwen3 tensor operations failed to round-trip: {error}")

        actual = reconstructed(x)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertFalse(actual[0].requires_grad)
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(actual[1].dtype, torch.int64)
        self.assertEqual(tuple(actual[2]), tuple(expected[2]))
        self.assertTrue(torch.equal(actual[3], expected[3]))
        self.assertEqual(tuple(actual[3].shape), (1, 1))
        self.assertTrue(torch.equal(actual[4], expected[4]))

    def test_qwen3_custom_ops_round_trip(self):
        graph = fx.Graph()
        result = graph.placeholder("result")
        inputs = graph.placeholder("inputs")
        weight = graph.placeholder("weight")
        positions = graph.placeholder("positions")
        cos_sin_cache = graph.placeholder("cos_sin_cache")
        query = graph.placeholder("query")
        key = graph.placeholder("key")
        value = graph.placeholder("value")
        output = graph.placeholder("output")

        custom_op_packets = (
            torch.ops._C.rms_norm,
            torch.ops._C.fused_add_rms_norm,
            torch.ops._C.rotary_embedding,
            torch.ops._C.silu_and_mul,
            torch.ops.vllm.unified_attention_with_output,
            torch.ops.vllm.unified_kv_cache_update,
        )
        custom_ops = tuple(packet.default for packet in custom_op_packets)
        graph.call_function(custom_op_packets[0], (result, inputs, weight, 1e-6))
        graph.call_function(custom_op_packets[1], (result, inputs, weight, 1e-6))
        graph.call_function(
            custom_op_packets[2], (positions, query, key, 1, cos_sin_cache, False)
        )
        graph.call_function(custom_op_packets[3], (result, inputs))
        graph.call_function(
            custom_op_packets[4], (query, key, value, output, "layer")
        )
        updated_key = graph.call_function(
            custom_op_packets[5], (key, value, "layer")
        )
        graph.output((result, updated_key))
        gm = fx.GraphModule(nn.Module(), graph)

        mode = FakeTensorMode()
        example = mode.from_tensor(torch.empty((1, 4)))
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = example
                node.meta["example_value"] = example
            elif node.op == "call_function":
                node.meta["val"] = (
                    example if node.target is custom_op_packets[5] else None
                )
                node.meta["example_value"] = node.meta["val"]

        normalize_fx(gm)
        normalized_targets = {
            node.target for node in gm.graph.nodes if node.op == "call_function"
        }
        for custom_op in custom_ops:
            self.assertIn(custom_op, normalized_targets)
        normalize_torch_size(gm)
        try:
            reconstructed = _round_trip(gm)
        except Exception as error:
            self.fail(f"Qwen3 custom operations failed to round-trip: {error}")

        reconstructed_targets = {
            node.target
            for node in reconstructed.graph.nodes
            if node.op == "call_function"
        }
        for custom_op in custom_ops:
            self.assertIn(custom_op, reconstructed_targets)
        reconstructed_nodes = {
            node.target: node
            for node in reconstructed.graph.nodes
            if node.op == "call_function"
        }
        output_node = next(
            node for node in reconstructed.graph.nodes if node.op == "output"
        )
        self.assertIn(reconstructed_nodes[custom_ops[-1]], output_node.all_input_nodes)
        for side_effect_op in custom_ops[:-1]:
            self.assertFalse(reconstructed_nodes[side_effect_op].users)
            
    def test_qwen3_demo_uses_dummy_weights(self):
        self.assertIsNotNone(importlib.util.find_spec("simple_llm_qwen3"))

        import simple_llm_qwen3

        with patch("simple_llm_qwen3.LLM") as llm_constructor:
            simple_llm_qwen3.create_llm()

        kwargs = llm_constructor.call_args.kwargs
        self.assertEqual(kwargs["model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(kwargs["load_format"], "dummy")
        self.assertEqual(
            kwargs["compilation_config"]["backend"],
            "mlir_roundtrip_backend",
        )


if __name__ == "__main__":
    unittest.main()
