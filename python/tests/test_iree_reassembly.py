"""Exercise flow-to-FX splicing with small, real IREE IR modules."""

import operator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("iree.compiler")

from kashi.iree_reassembly import IreeKernel, flow_segment_handlers

SHAPES = r"""
module {
  util.func public @segment_0$async(%arg0: !hal.buffer_view, %wait: !hal.fence, %signal: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view) {
    %d = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %t = hal.tensor.import wait(%wait) => %arg0 : !hal.buffer_view -> tensor<?x4xf32>{%d}
    %c = arith.constant 2 : index
    %n = arith.muli %d, %c : index
    %r = flow.tensor.reshape %t : tensor<?x4xf32>{%d} -> tensor<?x2xf32>{%n}
    %e = flow.tensor.empty : tensor<?x2xf32>{%n}
    %b:2 = hal.tensor.barrier join(%r, %e : tensor<?x2xf32>, tensor<?x2xf32>) => %signal : !hal.fence
    %x = hal.tensor.export %b#0 : tensor<?x2xf32>{%n} -> !hal.buffer_view
    %y = hal.tensor.export %b#1 : tensor<?x2xf32>{%n} -> !hal.buffer_view
    util.return %x, %y : !hal.buffer_view, !hal.buffer_view
  }
}
"""


def test_dynamic_shape_reshape_empty_and_barrier():
    """Dynamic dimensions drive both view and allocation after ABI aliases."""
    handler = flow_segment_handlers(SHAPES, device="cpu")["segment_0"]
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    outputs = handler(graph, [x])
    graph.output(tuple(outputs))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    view, empty = gm(source)
    torch.testing.assert_close(view, source.view(6, 2))
    assert empty.shape == (6, 2)
    assert empty.dtype == torch.float32 and empty.device.type == "cpu"


DISPATCH = r"""
module {
  util.func public @segment_1$async(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %w = arith.constant 4 : index
    %r:2 = "flow.dispatch"(%w, %a, %b) {entry_points = [@exe::@kernel], operandSegmentSizes = array<i32: 1, 2, 0, 0>, tied_operands = [-1 : index, -1 : index]} : (index, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    util.return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
  }
}
"""


def test_dispatch_workload_metadata_multiresult_and_stable_kernel():
    """A dispatch stays opaque and both results unpack from the same call."""
    handler = flow_segment_handlers(DISPATCH, device=None)["segment_1"]
    graph = torch.fx.Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    first = handler(graph, [a, b])
    second = handler(graph, [a, b])
    kernels = [n for n in graph.nodes if isinstance(n.target, IreeKernel)]
    assert len(kernels) == 2 and kernels[0].target is kernels[1].target
    assert (kernels[0].target.executable, kernels[0].target.export) == ("exe", "kernel")
    assert kernels[0].args == (a, b) and kernels[0].kwargs["workload"] == [4]
    assert kernels[0].meta["iree_result_types"] == ["tensor<4xf32>", "tensor<4xf32>"]
    assert [n.target for n in first] == [operator.getitem, operator.getitem]
    assert [n.args for n in first] == [(kernels[0], 0), (kernels[0], 1)]
    assert second[0].args == (kernels[1], 0)
    with pytest.raises(NotImplementedError, match="opaque"):
        kernels[0].target(a, b, workload=[4])


def test_tied_dispatch_records_aliasing_on_an_opaque_call():
    """In-place dispatches (e.g. IREE's concatenation copies) stay opaque calls.

    The kernels cannot execute yet, so the result is modeled as the call's
    value and the tie is kept as metadata for a future executable lowering.
    """
    text = DISPATCH.replace("[-1 : index, -1 : index]", "[0 : index, -1 : index]")
    handler = flow_segment_handlers(text, device=None)["segment_1"]
    graph = torch.fx.Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")

    results = handler(graph, [a, b])

    (kernel,) = [n for n in graph.nodes if isinstance(n.target, IreeKernel)]
    assert kernel.args == (a, b)
    assert kernel.meta["iree_tied_operands"] == [0, -1]
    assert [n.args for n in results] == [(kernel, 0), (kernel, 1)]


def test_untied_dispatch_has_no_tied_metadata():
    handler = flow_segment_handlers(DISPATCH, device=None)["segment_1"]
    graph = torch.fx.Graph()
    handler(graph, [graph.placeholder("a"), graph.placeholder("b")])

    (kernel,) = [n for n in graph.nodes if isinstance(n.target, IreeKernel)]
    assert "iree_tied_operands" not in kernel.meta


def test_unsupported_operation_is_named():
    """Unknown flow operations must not disappear during reconstruction."""
    text = r"""module {
      util.func public @segment_2$async(%a: index) -> index {
        %r = arith.addi %a, %a : index
        util.return %r : index
      }
    }"""
    handler = flow_segment_handlers(text, device=None)["segment_2"]
    graph = torch.fx.Graph()
    with pytest.raises(NotImplementedError, match="arith.addi"):
        handler(graph, [graph.placeholder("a")])
