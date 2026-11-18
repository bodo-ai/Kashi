import pytest
torch = pytest.importorskip("torch")

import operator
import re

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree

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

    with FakeTensorMode():
        example_inputs = [torch.randn(2, 4), torch.randn(3, 4), torch.randn(3)]

        for node, value in zip(graph.find_nodes(op="placeholder"), example_inputs):
            node.meta[meta_key] = value

        y.meta[meta_key] = linear(*example_inputs)

    result = import_fx(gm)

    assert result.module.operation.verify()
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

    with FakeTensorMode():
        example_inputs = [torch.randn(10, 4), torch.tensor([1, 3, 5])]
        weight.meta["val"], indices.meta["val"] = example_inputs
        y.meta["val"] = torch.ops.aten.embedding.default(*example_inputs)

    result = import_fx(gm)

    assert result.module.operation.verify()
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

    with FakeTensorMode():
        example_inputs = [torch.randn(2, 3)]
        x.meta["val"] = example_inputs[0]
        y.meta["val"] = example_inputs[0].view(3, 2)

    result = import_fx(gm)

    assert result.module.operation.verify()
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

    with FakeTensorMode():
        example_inputs = [torch.randn(2, 6)]
        x.meta["val"] = example_inputs[0]
        chunk_vals = example_inputs[0].chunk(2, dim=-1)
        chunks.meta["val"] = chunk_vals
        first.meta["val"], second.meta["val"] = chunk_vals

    result = import_fx(gm)

    assert result.module.operation.verify()
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

    with FakeTensorMode():
        y.meta["val"] = torch.empty(2, 3, dtype=torch.float16, device="cpu")

    result = import_fx(gm)

    assert result.module.operation.verify()
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

    with FakeTensorMode():
        size.meta["val"] = torch.Size([2, 3])
        y.meta["val"] = torch.empty(2, 3, dtype=torch.float32, device="cpu")

    result = import_fx(gm)

    assert result.module.operation.verify()
    assert "torch.aten.empty.memory_format" in str(result.module)


def _view_graph(output):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.ops.aten.view.default, (x, [6]))
    graph.output(output(y))
    with FakeTensorMode():
        x.meta["val"] = torch.randn(2, 3)
        y.meta["val"] = torch.randn(6)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


@pytest.mark.parametrize(
    "output",
    [lambda y: y, lambda y: (y,), lambda y: (y, y)],
    ids=["bare", "one-tuple", "two-tuple"],
)
def test_import_fx_records_out_spec(output):
    result = import_fx(_view_graph(output))

    assert result.out_spec == pytree.tree_flatten(output(0))[1]


# --- Symbolic sizes, custom operators and opaque inputs. --------------------


def _captured_ops_model():
    import kashi_test_ops

    x = torch.randn(3, 4)
    torch._dynamo.mark_dynamic(x, 0)
    model = kashi_test_ops.OpsModel(kashi_test_ops.Tag(2.0), kashi_test_ops.Tag(3.0))
    return kashi_test_ops.capture(model, x)


def _main_signature(text):
    return next(line for line in text.splitlines() if "func.func @main" in line)


def test_import_fx_symbolic_size_feeding_empty():
    gm, _ = _captured_ops_model()
    assert gm.graph.find_nodes(op="call_function", target=torch.Size)

    text = str(import_fx(gm).module)

    signature = _main_signature(text)
    assert "!torch.int" in signature
    assert "!torch.vtensor<[?,4],f32>" in signature
    sym_arg = signature.split("@main(")[1].split(",")
    sym_arg = next(a.split(":")[0].strip() for a in sym_arg if "!torch.int" in a)
    assert re.search(
        rf"torch\.prim\.ListConstruct {re.escape(sym_arg)}, %int4\b", text
    )
    assert re.search(
        r"torch\.aten\.empty\.memory_format .* -> !torch\.vtensor<\[\?,4\],f32>", text
    )
    assert "torch.constant.int 3" not in text


def test_import_fx_custom_ops_and_opaque_inputs():
    gm, _ = _captured_ops_model()

    result = import_fx(gm)
    text = str(result.module)

    assert result.module.operation.verify()
    assert _main_signature(text).count("!torch.any") == 2
    assert 'torch.operator "torch.kashi_test.scale"' in text
    write = next(l for l in text.splitlines() if '"torch.kashi_test.write"' in l)
    assert write.rstrip().endswith("-> ()")


def test_import_fx_rejects_multi_overload_packet():
    graph = torch.fx.Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    y = graph.call_function(torch.ops.aten.add, (a, b))
    graph.output(y)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    with FakeTensorMode():
        for node, value in [(a, torch.randn(2)), (b, torch.randn(2)), (y, torch.randn(2))]:
            node.meta["val"] = value

    with pytest.raises(NotImplementedError, match="Ambiguous overload packet.*aten.add"):
        import_fx(gm)


def test_import_fx_opaque_hook_is_not_global():
    from iree.compiler.extras import fx_importer

    before = fx_importer.ContextCache.value_info_to_type
    gm, _ = _captured_ops_model()
    try:
        import_fx(gm)
    except NotImplementedError:
        pass
    assert fx_importer.ContextCache.value_info_to_type is before
