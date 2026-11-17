import pytest
torch = pytest.importorskip("torch")

import operator

from kashi.importer import import_fx

@pytest.mark.parametrize(
    "linear",
    [torch.ops.aten.linear.default, torch._C._nn.linear],
    ids=["aten.linear.default", "torch._C._nn.linear"],
)
@pytest.mark.parametrize("meta_key", ["val", "example_value"])
def test_import_fx_preserves_aten_linear(linear, meta_key):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    bias = graph.placeholder("bias")
    y = graph.call_function(linear, (x, weight, bias))
    graph.output(y)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [torch.randn(2, 4), torch.randn(3, 4), torch.randn(3)]

    for node, value in zip(graph.find_nodes(op="placeholder"), example_inputs):
        node.meta[meta_key] = value

    y.meta[meta_key] = linear(*example_inputs)

    result = import_fx(gm, example_inputs)

    assert result.module_op.verify()
    assert "torch.aten.linear" in str(result.module)


@pytest.mark.parametrize(
    "embedding, indices_first, kwargs",
    [
        (torch.ops.aten.embedding.default, False, {}),
        (
            torch.nn.functional.embedding,
            True,
            {
                "padding_idx": None,
                "max_norm": None,
                "norm_type": 2.0,
                "scale_grad_by_freq": False,
                "sparse": False,
            },
        ),
    ],
    ids=["aten.embedding.default", "torch.nn.functional.embedding"],
)
def test_import_fx_preserves_aten_embedding(embedding, indices_first, kwargs):
    graph = torch.fx.Graph()
    weight = graph.placeholder("weight")
    indices = graph.placeholder("indices")
    args = (indices, weight) if indices_first else (weight, indices)
    y = graph.call_function(embedding, args, kwargs)
    graph.output(y)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [torch.randn(10, 4), torch.tensor([1, 3, 5])]
    weight.meta["val"], indices.meta["val"] = example_inputs
    y.meta["val"] = torch.ops.aten.embedding.default(*example_inputs)

    result = import_fx(gm, example_inputs)

    assert result.module_op.verify()
    assert "torch.aten.embedding" in str(result.module)


@pytest.mark.parametrize(
    "op, target, shape",
    [
        ("call_function", torch.ops.aten.view.default, ([3, 2],)),
        ("call_method", "view", (3, 2)),
    ],
    ids=["aten.view.default", "Tensor.view"],
)
def test_import_fx_preserves_aten_view(op, target, shape):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.create_node(op, target, (x, *shape))
    graph.output(y)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [torch.randn(2, 3)]
    x.meta["val"] = example_inputs[0]
    y.meta["val"] = example_inputs[0].view(3, 2)

    result = import_fx(gm, example_inputs)

    assert result.module_op.verify()
    assert "torch.aten.view" in str(result.module)


@pytest.mark.parametrize(
    "op, target",
    [
        ("call_function", torch.ops.aten.chunk.default),
        ("call_function", torch.chunk),
        ("call_method", "chunk"),
    ],
    ids=["aten.chunk.default", "torch.chunk", "Tensor.chunk"],
)
def test_import_fx_preserves_chunk_with_getitem(op, target):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    chunks = graph.create_node(op, target, (x, 2), {"dim": -1})
    first = graph.call_function(operator.getitem, (chunks, 0))
    second = graph.call_function(operator.getitem, (chunks, 1))
    graph.output((first, second))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    example_inputs = [torch.randn(2, 6)]
    x.meta["val"] = example_inputs[0]
    chunk_vals = example_inputs[0].chunk(2, dim=-1)
    chunks.meta["val"] = chunk_vals
    first.meta["val"], second.meta["val"] = chunk_vals

    result = import_fx(gm, example_inputs)

    assert result.module_op.verify()
    assert "torch.aten.chunk" in str(result.module)


@pytest.mark.parametrize(
    "empty",
    [torch.ops.aten.empty.memory_format, torch.empty],
    ids=["aten.empty.memory_format", "torch.empty"],
)
def test_import_fx_preserves_aten_empty(empty):
    graph = torch.fx.Graph()
    y = graph.call_function(
        empty, ([2, 3],), {"dtype": torch.float16, "device": torch.device("cpu")}
    )
    graph.output(y)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    y.meta["val"] = torch.empty(2, 3, dtype=torch.float16, device="cpu")

    result = import_fx(gm, [])

    assert result.module_op.verify()
    module = str(result.module)
    assert "torch.aten.empty.memory_format" in module
    assert "f16" in module


def test_import_fx_preserves_torch_size_feeding_empty():
    graph = torch.fx.Graph()
    size = graph.call_function(torch.Size, ([2, 3],))
    y = graph.call_function(
        torch.empty, (size,), {"dtype": torch.float32, "device": torch.device("cpu")}
    )
    graph.output(y)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    size.meta["val"] = torch.Size([2, 3])
    y.meta["val"] = torch.empty(2, 3, dtype=torch.float32, device="cpu")

    result = import_fx(gm, [])

    assert result.module_op.verify()
    assert "torch.aten.empty.memory_format" in str(result.module)
