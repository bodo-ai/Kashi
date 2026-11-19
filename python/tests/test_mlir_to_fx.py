import pytest

torch = pytest.importorskip("torch")

import operator

from iree.compiler import ir
from kashi.mlir_to_fx import rebuild_fx
from torch.utils import _pytree as pytree


def _parse(body: str) -> ir.Module:
    with ir.Context():
        return ir.Module.parse(body)


def _targets(gm):
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


def _call_nodes(gm):
    return [n for n in gm.graph.nodes if n.op == "call_function"]


LINEAR = """
func.func @main(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[3,4],f32>, %arg2: !torch.vtensor<[3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}
"""

LINEAR_NO_BIAS = """
func.func @main(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2,3],f32> {
  %none = torch.constant.none
  %0 = torch.aten.linear %arg0, %arg1, %none : !torch.vtensor<[2,4],f32>, !torch.vtensor<[3,4],f32>, !torch.none -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}
"""

EMBEDDING = """
func.func @main(%arg0: !torch.vtensor<[10,4],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4],f32> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %false_0 = torch.constant.bool false
  %0 = torch.aten.embedding %arg0, %arg1, %int-1, %false, %false_0 : !torch.vtensor<[10,4],f32>, !torch.vtensor<[3],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}
"""

VIEW = """
func.func @main(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int3, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[3,2],f32>
  return %1 : !torch.vtensor<[3,2],f32>
}
"""


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_rebuild_linear(bias):
    gm = rebuild_fx(_parse(LINEAR if bias else LINEAR_NO_BIAS))

    assert _targets(gm) == [torch.ops.aten.linear.default]
    x, w, b = torch.randn(2, 4), torch.randn(3, 4), torch.randn(3)
    if bias:
        torch.testing.assert_close(gm(x, w, b), torch.ops.aten.linear.default(x, w, b))
    else:
        torch.testing.assert_close(gm(x, w), torch.ops.aten.linear.default(x, w, None))


def test_rebuild_embedding():
    gm = rebuild_fx(_parse(EMBEDDING))

    (node,) = _call_nodes(gm)
    assert node.target is torch.ops.aten.embedding.default
    weight, indices = gm.graph.find_nodes(op="placeholder")
    assert node.args == (weight, indices, -1, False, False)
    w = torch.randn(10, 4)
    torch.testing.assert_close(gm(w, torch.tensor([1, 3, 5])), w[[1, 3, 5]])


def test_rebuild_view():
    gm = rebuild_fx(_parse(VIEW))

    (node,) = _call_nodes(gm)
    assert node.target is torch.ops.aten.view.default
    assert node.args[1] == [3, 2]
    assert gm(torch.randn(2, 3)).shape == (3, 2)


def test_rebuild_output_structure():
    module = _parse(VIEW)
    x = torch.randn(2, 3)

    assert isinstance(rebuild_fx(module)(x), torch.Tensor)
    out = rebuild_fx(module, pytree.tree_flatten((0,))[1])(x)
    assert isinstance(out, tuple) and len(out) == 1
    with pytest.raises(NotImplementedError, match="func.return"):
        rebuild_fx(module, pytree.tree_flatten((0, 0))[1])


REPEATED_AND_INPUT = """
func.func @main(%arg0: !torch.vtensor<[2,3],f32>) -> (!torch.vtensor<[6],f32>, !torch.vtensor<[6],f32>, !torch.vtensor<[2,3],f32>) {
  %int6 = torch.constant.int 6
  %0 = torch.prim.ListConstruct %int6 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[6],f32>
  return %1, %1, %arg0 : !torch.vtensor<[6],f32>, !torch.vtensor<[6],f32>, !torch.vtensor<[2,3],f32>
}
"""


def test_rebuild_repeated_and_input_outputs():
    gm = rebuild_fx(_parse(REPEATED_AND_INPUT))
    x = torch.randn(2, 3)

    out = gm(x)
    assert len(out) == 3
    assert out[0] is out[1]
    assert out[2] is x


UNUSED_OP = """
func.func @main(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %int6 = torch.constant.int 6
  %0 = torch.prim.ListConstruct %int6 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[6],f32>
  return %arg0 : !torch.vtensor<[2,3],f32>
}
"""


def test_rebuild_keeps_unused_op():
    gm = rebuild_fx(_parse(UNUSED_OP))

    assert _targets(gm) == [torch.ops.aten.view.default]


MULTI_RESULT = """
func.func @main(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0, %1 = torch.aten.var_mean.correction %arg0, %none, %none, %false : !torch.vtensor<[2,3],f32>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}
"""

CUSTOM_OP = """
func.func @main(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
  %0 = torch.operator "aten.relu"(%arg0) : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}
"""

NON_TORCH_OP = """
func.func @main(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
  %c1 = arith.constant 1 : i64
  return %arg0 : !torch.vtensor<[2],f32>
}
"""

NON_TENSOR_ARG = """
func.func @main(%arg0: !torch.float) -> !torch.float {
  return %arg0 : !torch.float
}
"""


@pytest.mark.parametrize(
    "src, match",
    [
        (MULTI_RESULT, "torch.aten.var_mean.correction"),
        (CUSTOM_OP, "torch.operator"),
        (NON_TORCH_OP, "arith.constant"),
        (NON_TENSOR_ARG, "!torch.float"),
    ],
    ids=["multi-result", "torch.operator", "non-torch-op", "non-tensor-arg"],
)
def test_rebuild_rejects(src, match):
    with pytest.raises(NotImplementedError, match=match):
        rebuild_fx(_parse(src))


CHUNK = """
func.func @main(%arg0: !torch.vtensor<[2,6],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>) {
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %0 = torch.aten.chunk %arg0, %int2, %int-1 : !torch.vtensor<[2,6],f32>, !torch.int, !torch.int -> !torch.list<vtensor>
  %1:2 = torch.prim.ListUnpack %0 : !torch.list<vtensor> -> !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>
  return %1#0, %1#1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>
}
"""


def test_rebuild_chunk():
    gm = rebuild_fx(_parse(CHUNK))

    assert _targets(gm) == [
        torch.ops.aten.chunk.default,
        operator.getitem,
        operator.getitem,
    ]
    x = torch.randn(2, 6)
    a, b = gm(x)
    torch.testing.assert_close((a, b), x.chunk(2, dim=-1))


UNPACK_LITERAL_LIST = """
func.func @main(%arg0: !torch.vtensor<[2],f32>) -> !torch.int {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1:2 = torch.prim.ListUnpack %0 : !torch.list<int> -> !torch.int, !torch.int
  return %1#0 : !torch.int
}
"""


def test_rebuild_rejects_unpack_of_literal_list():
    with pytest.raises(NotImplementedError, match="torch.prim.ListUnpack"):
        rebuild_fx(_parse(UNPACK_LITERAL_LIST))


def _empty(dtype="%int5", layout="%none", memory_format="%none"):
    types = {
        "%int5": "!torch.int",
        "%int99": "!torch.int",
        "%int0": "!torch.int",
        "%none": "!torch.none",
    }
    return f"""
func.func @main() -> !torch.vtensor<[2,3],f16> {{
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %int5 = torch.constant.int 5
  %int99 = torch.constant.int 99
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.empty.memory_format %0, {dtype}, {layout}, %cpu, %none, {memory_format} : !torch.list<int>, {types[dtype]}, {types[layout]}, !torch.Device, !torch.none, {types[memory_format]} -> !torch.vtensor<[2,3],f16>
  return %1 : !torch.vtensor<[2,3],f16>
}}
"""


def test_rebuild_empty_decodes_dtype_and_device():
    gm = rebuild_fx(_parse(_empty()))

    (node,) = _call_nodes(gm)
    assert node.target is torch.ops.aten.empty.memory_format
    assert node.args == ([2, 3],)
    assert node.kwargs["dtype"] is torch.float16
    assert node.kwargs["device"] == torch.device("cpu")
    out = gm()
    assert out.shape == (2, 3) and out.dtype == torch.float16


@pytest.mark.parametrize(
    "operand, match",
    [
        ({"dtype": "%int99"}, "ScalarType"),
        ({"layout": "%int0"}, "layout"),
        ({"memory_format": "%int0"}, "memory_format"),
    ],
    ids=["unknown-dtype", "layout", "memory_format"],
)
def test_rebuild_rejects_bad_enum(operand, match):
    with pytest.raises(NotImplementedError, match=match):
        rebuild_fx(_parse(_empty(**operand)))


IDENTITY = """
func.func @main(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  return %arg0 : !torch.vtensor<[2,3],f32>
}
"""


def _with_unparseable_op(name, num_operands):
    """The torch dialect parser rejects these ops; building them directly is
    the only way to reach rebuild_fx's guards against torch-mlir/PyTorch drift."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(IDENTITY)
        block = module.body.operations[0].regions[0].blocks[0]
        (arg,) = block.arguments
        with ir.InsertionPoint(block.operations[0]):
            ir.Operation.create(name, results=[arg.type], operands=[arg] * num_operands)
    return module


@pytest.mark.parametrize(
    "name, num_operands, match",
    [
        ("torch.aten.view.bogus", 2, "no torch.ops.aten.view.bogus"),
        ("torch.aten.view", 1, "expected 2 operands, found 1"),
        ("torch.aten.name", 1, "no torch.ops.aten.name.default"),
    ],
    ids=["unknown-overload", "operand-count", "namespace-attribute"],
)
def test_rebuild_rejects_unresolvable_aten_op(name, num_operands, match):
    with pytest.raises(NotImplementedError, match=match):
        rebuild_fx(_with_unparseable_op(name, num_operands))


# --- Scalar/opaque arguments and torch.operator. -----------------------------

CUSTOM = """
func.func @main(%arg0: !torch.int, %arg1: !torch.vtensor<[?,4],f32>, %arg2: !torch.any, %arg3: !torch.any) -> (!torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>) {
  %0 = torch.operator "torch.kashi_test.scale"(%arg1, %arg2) : (!torch.vtensor<[?,4],f32>, !torch.any) -> !torch.vtensor<[?,4],f32>
  %int4 = torch.constant.int 4
  %1 = torch.prim.ListConstruct %arg0, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %int6 = torch.constant.int 6
  %none = torch.constant.none
  %cpu = torch.constant.device "cpu"
  %2 = torch.aten.empty.memory_format %1, %int6, %none, %cpu, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,4],f32>
  torch.operator "torch.kashi_test.write"(%arg1, %2, %arg3, %0) : (!torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>, !torch.any, !torch.vtensor<[?,4],f32>) -> ()
  return %2, %0 : !torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32>
}
"""


@pytest.fixture
def test_ops():
    import kashi_test_ops

    return kashi_test_ops


def test_rebuild_scalar_and_opaque_arguments(test_ops):
    gm = rebuild_fx(_parse(CUSTOM))

    placeholders = gm.graph.find_nodes(op="placeholder")
    assert len(placeholders) == 4
    (empty,) = gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.empty.memory_format
    )
    assert empty.args[0] == [placeholders[0], 4]


def test_rebuild_custom_operator(test_ops):
    gm = rebuild_fx(_parse(CUSTOM))

    scale, empty, write = _call_nodes(gm)
    assert [scale.target, empty.target, write.target] == [
        torch.ops.kashi_test.scale.default,
        torch.ops.aten.empty.memory_format,
        torch.ops.kashi_test.write.default,
    ]
    x, tag_a, tag_b = (gm.graph.find_nodes(op="placeholder")[i] for i in (1, 2, 3))
    assert scale.args == (x, tag_a)
    assert write.args == (x, empty, tag_b)
    assert write.kwargs == {"dep": scale}


def test_rebuild_zero_result_operator_runs_once(test_ops):
    gm = rebuild_fx(_parse(CUSTOM))
    x = torch.randn(3, 4)

    out, d = gm(3, x, test_ops.Tag(2.0), test_ops.Tag(3.0))

    assert _targets(gm).count(torch.ops.kashi_test.write.default) == 1
    torch.testing.assert_close(out, x + 3.0)
    torch.testing.assert_close(d, x * 2.0)


def _with_operator(name):
    return _parse(f"""
func.func @main(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.any) -> !torch.vtensor<[2],f32> {{
  %0 = torch.operator "{name}"(%arg0, %arg1) : (!torch.vtensor<[2],f32>, !torch.any) -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}}
""")


@pytest.mark.parametrize(
    "name, match",
    [
        ("torch.kashi_test.missing", "no torch.ops.kashi_test.missing.default"),
        ("torch.no_such_ns.op", "no torch.ops.no_such_ns.op.default"),
        ("torch.kashi_test.scale.bogus", "no torch.ops.kashi_test.scale.bogus"),
    ],
    ids=["unknown-op", "unknown-namespace", "unknown-overload"],
)
def test_rebuild_rejects_unresolvable_operator(test_ops, name, match):
    with pytest.raises(
        NotImplementedError, match=rf"Unsupported MLIR op torch\.operator.*{match}"
    ):
        rebuild_fx(_with_operator(name))
