"""Integration tests for the opt-in IREE compilation and graph reassembly path."""

import json
import re
import sys

import pytest

torch = pytest.importorskip("torch")
from iree.compiler import ir
from kashi.mlir_to_fx import rebuild_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree

# This operator marks a real mutable boundary without depending on vLLM's GPU
# extensions; its input and destination must remain correctly ordered in FX.
_LIB = torch.library.Library("kashi_iree_integration", "FRAGMENT")
_LIB.define("write(Tensor(a!) out, Tensor value) -> ()")
_LIB.define("fill(Tensor(a!) out, float value) -> ()")


def _write(out, value):
    """Write the intermediate into the custom operator's output buffer."""
    out.copy_(value)


_LIB.impl("write", _write, "CompositeExplicitAutograd")


def _fill(out, value):
    """Consume a scalar at a custom boundary by mutating the output tensor."""
    out.fill_(value)


_LIB.impl("fill", _fill, "CompositeExplicitAutograd")


def _graph_with_mutating_boundary():
    """Build two tensor computations separated by a mutating custom operator."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    out = graph.placeholder("out")
    relu = graph.call_function(torch.ops.aten.relu.default, (x,))
    write = graph.call_function(
        torch.ops.kashi_iree_integration.write.default, (out, relu)
    )
    neg = graph.call_function(torch.ops.aten.neg.default, (out,))
    graph.output({"result": (neg,)})
    with FakeTensorMode() as mode:
        value = mode.from_tensor(torch.empty(4, 8))
        for node in (x, out, relu, neg):
            node.meta["val"] = value
        write.meta["val"] = None
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _depends_on(node, source):
    """Return whether an FX value transitively depends on another graph node."""
    if node is source:
        return True
    return isinstance(node, torch.fx.Node) and any(
        _depends_on(parent, source) for parent in node.all_input_nodes
    )


@pytest.mark.skipif(
    sys.platform == "darwin", reason="CUDA/PTX integration is not exercised on macOS"
)
def test_compile_iree_fx_reassembles_real_ptx_and_custom_boundary(
    tmp_path, monkeypatch
):
    """Real CUDA compilation produces kernel calls around the mutating operator."""
    from kashi.iree_backend import compile_iree_fx
    from kashi.iree_reassembly import IreeKernel

    monkeypatch.setenv("PATH", "")

    compiled = compile_iree_fx(
        _graph_with_mutating_boundary(), output_dir=tmp_path, cuda_target="sm_80"
    )
    assert compiled.ptx_files
    assert any(".entry" in path.read_text() for path in compiled.ptx_files)
    assert compiled.flow_ir.exists()
    calls = [node for node in compiled.graph.graph.nodes if node.op == "call_function"]
    kernels = [node for node in calls if isinstance(node.target, IreeKernel)]
    (write,) = [
        node
        for node in calls
        if node.target is torch.ops.kashi_iree_integration.write.default
    ]
    before = [node for node in kernels if calls.index(node) < calls.index(write)]
    after = [node for node in kernels if calls.index(node) > calls.index(write)]
    assert any(_depends_on(write.args[1], node) for node in before)
    assert any(_depends_on(node, write.args[0]) for node in after)
    manifest = json.loads(compiled.kernel_manifest.read_text())
    linked_files = set()
    for node in kernels:
        kernel = manifest[node.target.executable][node.target.export]
        path = compiled.artifact_dir / kernel["file"]
        linked_files.add(path)
        assert path in compiled.ptx_files
        assert re.search(
            rf"\.entry\s+{re.escape(kernel['entry'])}\s*\(", path.read_text()
        )
        assert node.meta["iree_ptx_entry"] == kernel["entry"]
        assert kernel["entry"] != node.target.export
    assert len(linked_files) == 1 and len(kernels) >= 2
    output = next(
        node for node in compiled.graph.graph.nodes if node.op == "output"
    ).args[0]
    assert isinstance(output, dict) and isinstance(output["result"], tuple)
    with pytest.raises(NotImplementedError, match="opaque"):
        compiled.graph(torch.ones(4, 8), torch.empty(4, 8))


@pytest.mark.skipif(
    sys.platform == "darwin", reason="The public IREE backend requires CUDA compilation"
)
def test_compile_iree_fx_rejects_float_crossing_custom_boundary(tmp_path):
    """An unsupported segment live-out fails before a driver is emitted."""
    from kashi.iree_backend import compile_iree_fx

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    out = graph.placeholder("out")
    scalar = graph.call_function(torch.ops.aten.Float.Tensor, (x,))
    fill = graph.call_function(
        torch.ops.kashi_iree_integration.fill.default, (out, scalar)
    )
    graph.output(out)
    with FakeTensorMode() as mode:
        x.meta["val"] = mode.from_tensor(torch.empty(()))
        out.meta["val"] = mode.from_tensor(torch.empty(4))
    scalar.meta["val"] = 0.0
    fill.meta["val"] = None
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    with pytest.raises(NotImplementedError, match="partition"):
        compile_iree_fx(gm, output_dir=tmp_path, cuda_target="sm_80")

    (artifact_dir,) = tmp_path.glob("kashi-iree-*")
    log = (artifact_dir / "iree-compile.log").read_text()
    assert "unsupported live-out for segment_0" in log
    assert "!torch.float" in log
    assert not (artifact_dir / "driver.mlir").exists()


DRIVER = """
module {
  func.func private @segment_0(!torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
  func.func private @segment_1(!torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
  func.func @main(%x: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
    %a = func.call @segment_0(%x) : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
    %b = func.call @segment_1(%a) : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32>
    return %b : !torch.vtensor<[2],f32>
  }
}
"""


def test_rebuild_splices_segment_calls_and_restores_output_structure():
    """Segment results feed later segments and retain the original output tree."""
    with ir.Context():
        module = ir.Module.parse(DRIVER)
    handlers = {
        "segment_0": lambda graph, args: [
            graph.call_function(torch.ops.aten.add.Scalar, (args[0], 2))
        ],
        "segment_1": lambda graph, args: [
            graph.call_function(torch.ops.aten.neg.default, (args[0],))
        ],
    }
    out_spec = pytree.tree_flatten({"answer": (0,)})[1]
    rebuilt = rebuild_fx(module, out_spec, segment_handlers=handlers)

    result = rebuilt(torch.tensor([1.0, -4.0]))
    assert isinstance(result, dict) and isinstance(result["answer"], tuple)
    torch.testing.assert_close(result["answer"][0], torch.tensor([-3.0, 2.0]))


def test_rebuild_rejects_missing_segment_handler():
    """An unresolved driver call must never be dropped during reconstruction."""
    with ir.Context():
        module = ir.Module.parse(DRIVER)
    handlers = {"segment_0": lambda graph, args: [args[0]]}
    with pytest.raises(NotImplementedError, match="segment_1"):
        rebuild_fx(module, segment_handlers=handlers)


def test_rebuild_rejects_wrong_segment_result_count():
    """A handler cannot silently substitute a mismatched segment signature."""
    with ir.Context():
        module = ir.Module.parse(DRIVER)
    with pytest.raises(NotImplementedError, match="returned 0 values for 1 results"):
        rebuild_fx(module, segment_handlers={"segment_0": lambda graph, args: []})
