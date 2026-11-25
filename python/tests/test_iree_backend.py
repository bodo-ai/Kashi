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


def _alias_only_graph():
    """Build a graph whose view must retain its input's storage."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    view = graph.call_function(torch.ops.aten.view.default, (x, [2, 2]))
    graph.output(view)
    with FakeTensorMode() as fake_mode:
        value = fake_mode.from_tensor(torch.empty(4))
        x.meta["val"] = value
        view.meta["val"] = value.view(2, 2)
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

    # Compilation must not depend on finding iree-compile or ptxas on PATH.
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

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
        assert node.meta["iree_ptx_file"] == path
        assert node.meta["iree_ptx_entry"] == kernel["entry"]
        assert node.meta["iree_stable_export"] == kernel["stable"]
        assert kernel["entry"] != node.target.export
    assert len(linked_files) == 1 and len(kernels) >= 2
    output = next(
        node for node in compiled.graph.graph.nodes if node.op == "output"
    ).args[0]
    assert isinstance(output, dict) and isinstance(output["result"], tuple)


def test_factory_device_is_used_for_cuda_target(monkeypatch):
    """A factory-only source selects its explicit CUDA device, not the default."""
    from kashi.iree_backend import _cuda_target, _imported_device

    module = ir.Module.parse(
        """
module {
  func.func @main() {
    %device = torch.constant.device "cuda:2"
    return
  }
}
""",
        ir.Context(),
    )
    monkeypatch.delenv("IREE_CUDA_TARGET", raising=False)
    devices = []

    def capability(device):
        devices.append(device)
        return (8, 6)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert _cuda_target(None, _imported_device(module)) == "sm_86"
    assert devices == [torch.device("cuda:2")]


@pytest.mark.parametrize("mode", ["cuda", "iree"])
def test_alias_only_graph_executes_without_iree_segments(tmp_path, monkeypatch, mode):
    """A graph containing only alias-sensitive operations needs no VMFB."""
    from kashi.iree_backend import compile_iree_fx

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    monkeypatch.delenv("IREE_CUDA_TARGET", raising=False)

    def no_cuda(*args):
        raise RuntimeError("no CUDA device available")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_cuda)
    gm = _alias_only_graph()
    compiled = compile_iree_fx(gm, output_dir=tmp_path)
    source = torch.arange(4.0)
    result = compiled.graph(source)
    torch.testing.assert_close(result, source.view(2, 2))
    assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert not compiled.ptx_files and compiled.vmfb_path is None
    assert compiled.flow_ir is None and compiled.kernel_manifest is None


def test_alias_only_driver_accepts_distinct_input_devices(tmp_path):
    """Driver-only graphs do not impose the single-device segment restriction."""
    from kashi.iree_backend import compile_iree_fx

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    view = graph.call_function(torch.ops.aten.view.default, (x, [2, 2]))
    graph.output((view, y))
    with FakeTensorMode() as fake_mode:
        x.meta["val"] = fake_mode.from_tensor(torch.empty(4))
        y.meta["val"] = fake_mode.from_tensor(torch.empty(4, device="meta"))
        view.meta["val"] = x.meta["val"].view(2, 2)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    compiled = compile_iree_fx(gm, output_dir=tmp_path)
    source, other = torch.arange(4.0), torch.empty(4, device="meta")
    result, passthrough = compiled.graph(source, other)
    assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert passthrough is other


def test_nonempty_graph_reports_missing_cuda_target(tmp_path, monkeypatch):
    """Target discovery errors are deferred only for graphs with no segments."""
    from kashi.iree_backend import compile_iree_fx

    monkeypatch.delenv("IREE_CUDA_TARGET", raising=False)

    def no_cuda(*args):
        raise RuntimeError("no CUDA device available")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_cuda)
    with pytest.raises(ValueError, match="cuda_target or IREE_CUDA_TARGET"):
        compile_iree_fx(_graph_with_mutating_boundary(), output_dir=tmp_path)


def test_backend_executes_alias_only_graph_and_retains_artifacts(monkeypatch):
    """The Dynamo backend keeps its workspace while its CPU-only driver runs."""
    import gc

    from kashi.iree_backend import compile_iree_backend

    monkeypatch.setenv("KASHI_IREE_EXECUTION", "cuda")
    compiled = compile_iree_backend(_alias_only_graph(), ())
    artifact_dir = compiled.compilation.artifact_dir
    source = torch.arange(4.0)
    result = compiled(source)
    torch.testing.assert_close(result, source.view(2, 2))
    assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert artifact_dir.is_dir()
    del compiled
    gc.collect()
    assert not artifact_dir.exists()


def test_backend_cleans_workspace_when_compilation_fails(tmp_path, monkeypatch):
    """Failure during compilation does not leave a temporary artifact tree."""
    from kashi import iree_backend

    monkeypatch.setattr(iree_backend.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("KASHI_IREE_EXECUTION", "invalid")
    with pytest.raises(ValueError, match="KASHI_IREE_EXECUTION"):
        iree_backend.compile_iree_backend(_alias_only_graph(), ())
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(
    sys.platform == "darwin", reason="CUDA/PTX compilation is unavailable"
)
def test_iree_runtime_mode_builds_real_segment_calls_without_a_gpu(
    tmp_path, monkeypatch
):
    """The runtime mode emits a VMFB and connects real segment targets."""
    from kashi.iree_backend import compile_iree_fx
    from kashi.iree_runtime import _SegmentTarget

    monkeypatch.setenv("KASHI_IREE_EXECUTION", "iree")
    compiled = compile_iree_fx(
        _graph_with_mutating_boundary(), output_dir=tmp_path, cuda_target="sm_80"
    )
    assert compiled.execution_mode == "iree"
    assert compiled.vmfb_path.is_file() and compiled.ptx_files
    targets = [n.target for n in compiled.graph.graph.nodes if n.op == "call_function"]
    assert any(isinstance(target, _SegmentTarget) for target in targets)


def test_rebuilt_segment_keeps_distinct_outputs_and_unsigned_bits():
    """IREE-deduplicated outputs must regain independent storage and dtype."""
    from kashi.iree_backend import _preserve_segment_results, _segment_functions

    graph = torch.fx.Graph()
    source = graph.placeholder("source")

    def reused_view(graph, args):
        """Represent a compiled segment that returns one view twice."""
        view = graph.call_function(torch.ops.aten.view.default, (args[0], [4]))
        return [view, view, 3]

    module = ir.Module.parse(
        """
module {
  func.func @segment_0(%x: !torch.vtensor<[4],si8>, %u: !torch.vtensor<[4],ui8>) -> (!torch.vtensor<[4],si8>, !torch.vtensor<[4],ui8>, !torch.int) {
    %a = torch.aten.relu %x : !torch.vtensor<[4],si8> -> !torch.vtensor<[4],si8>
    %b = torch.aten.relu %u : !torch.vtensor<[4],ui8> -> !torch.vtensor<[4],ui8>
    %c = torch.constant.int 3
    return %a, %b, %c : !torch.vtensor<[4],si8>, !torch.vtensor<[4],ui8>, !torch.int
  }
}
""",
        ir.Context(),
    )
    handler = _preserve_segment_results(
        {"segment_0": reused_view}, _segment_functions(module)
    )
    graph.output(tuple(handler["segment_0"](graph, [source, source])))
    rebuilt = torch.fx.GraphModule(torch.nn.Module(), graph)
    expected_signed = torch.tensor([-1, 0, 127, -128], dtype=torch.int8)
    expected_unsigned = torch.tensor([255, 0, 127, 128], dtype=torch.uint8)
    data = expected_signed.clone()
    first, second, scalar = rebuilt(data)
    torch.testing.assert_close(first, expected_signed)
    torch.testing.assert_close(second, expected_unsigned)
    assert scalar == 3
    first.fill_(0)
    torch.testing.assert_close(second, expected_unsigned)
    torch.testing.assert_close(data, expected_signed)


def test_tied_flow_dispatch_does_not_mutate_segment_input():
    """An IREE tied output may reuse a buffer without changing a source input."""
    from kashi.iree_reassembly import flow_segment_handlers

    flow = """
module {
  util.func public @segment_0$async(%x: tensor<4xf32>) -> tensor<4xf32> {
    %r = "flow.dispatch"(%x) {entry_points = [@exe::@kernel], operandSegmentSizes = array<i32: 0, 1, 0, 0>, tied_operands = [0 : index]} : (tensor<4xf32>) -> tensor<4xf32>
    util.return %r : tensor<4xf32>
  }
}
"""

    def kernel(x, **kwargs):
        x.add_(1)
        return x

    handler = flow_segment_handlers(
        flow, device="cpu", kernel_factory=lambda executable, export: kernel
    )["segment_0"]
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    graph.output(handler(graph, [source])[0])
    rebuilt = torch.fx.GraphModule(torch.nn.Module(), graph)
    data = torch.arange(4.0)
    result = rebuilt(data)
    torch.testing.assert_close(result, torch.arange(4.0) + 1)
    torch.testing.assert_close(data, torch.arange(4.0))


def test_rebuilt_segment_preserves_source_ssa_aliases():
    """Source input and repeated-result aliases are not cloned by reassembly."""
    from kashi.iree_backend import _preserve_segment_results, _segment_functions

    module = ir.Module.parse(
        """
module {
  func.func @segment_0(%x: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) {
    return %x, %x : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>
  }
}
""",
        ir.Context(),
    )
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    handlers = _preserve_segment_results(
        {"segment_0": lambda graph, args: [args[0], args[0]]},
        _segment_functions(module),
    )
    graph.output(tuple(handlers["segment_0"](graph, [source])))
    rebuilt = torch.fx.GraphModule(torch.nn.Module(), graph)
    data = torch.arange(4.0)
    first, second = rebuilt(data)
    assert first is data and second is data


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
