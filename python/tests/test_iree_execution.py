"""Exercise both IREE execution modes with real CUDA tensors."""

import pytest

torch = pytest.importorskip("torch")

_LIB = torch.library.Library("kashi_iree_execution", "FRAGMENT")
_LIB.define("write(Tensor(a!) out, Tensor value) -> ()")
_LIB.define("increment(Tensor(a!) value) -> ()")


def _write(out, value):
    """Copy a result into the buffer mutated by the real custom operator."""
    out.copy_(value)


_LIB.impl("write", _write, "CompositeExplicitAutograd")


def _increment(value):
    """Mutate a view so callers can observe whether its base aliases it."""
    value.add_(3)


_LIB.impl("increment", _increment, "CompositeExplicitAutograd")


@torch.library.register_fake("kashi_iree_execution::increment", lib=_LIB)
def _fake_increment(value):
    """Describe the in-place operator's no-result signature to Dynamo."""


@torch.library.register_fake("kashi_iree_execution::write", lib=_LIB)
def _fake_write(out, value):
    """Describe the custom operator's no-result signature to Dynamo."""


def test_invalid_iree_execution_mode_is_rejected(tmp_path, monkeypatch):
    """An invalid mode must fail instead of silently choosing a backend."""
    from kashi.iree_backend import compile_iree_fx

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    monkeypatch.setenv("KASHI_IREE_EXECUTION", "invalid")
    with pytest.raises(ValueError, match="KASHI_IREE_EXECUTION"):
        compile_iree_fx(gm, output_dir=tmp_path, cuda_target="sm_80")


def test_cuda_target_uses_the_graph_device(monkeypatch):
    """The generated PTX target follows the captured tensor's GPU architecture."""
    from kashi.iree_backend import _cuda_target

    monkeypatch.delenv("IREE_CUDA_TARGET", raising=False)

    def capability(device=None):
        """Represent two GPUs with different architectures."""
        return (9, 0) if device == torch.device("cuda:1") else (8, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert _cuda_target(None, "cuda:1") == "sm_90"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    "mode", [None, "cuda", "iree"], ids=["default", "cuda", "iree"]
)
def test_iree_graph_executes_dynamic_tensors_on_current_stream(
    tmp_path, monkeypatch, mode
):
    """Both modes must execute dynamic shapes on the caller's CUDA stream."""
    from kashi.iree_backend import compile_iree_fx

    if mode is None:
        monkeypatch.delenv("KASHI_IREE_EXECUTION", raising=False)
    else:
        monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    torch._dynamo.reset()

    def backend(gm, example_inputs):
        """Compile the captured graph and return its executable forward method."""
        return compile_iree_fx(gm, output_dir=tmp_path).graph.forward

    def model(x):
        """Use ATen overloads so the FX importer receives unambiguous operators."""
        return torch.ops.aten.neg.default(torch.ops.aten.relu.default(x))

    compiled = torch.compile(model, backend=backend, dynamic=True, fullgraph=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for rows in (31, 7):
            x = torch.linspace(-2, 2, rows * 13, device="cuda").reshape(rows, 13)
            result = compiled(x)
            torch.testing.assert_close(result, model(x))
    stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["cuda", "iree"])
def test_iree_graph_preserves_mutating_custom_boundary(tmp_path, monkeypatch, mode):
    """Both modes must preserve writes consumed by later compiled segments."""
    from kashi.iree_backend import compile_iree_fx

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    torch._dynamo.reset()

    def backend(gm, example_inputs):
        """Compile the captured graph and return its executable forward method."""
        return compile_iree_fx(gm, output_dir=tmp_path).graph.forward

    def model(x, out):
        """Write a transformed tensor across an opaque custom-op boundary."""
        value = torch.ops.aten.relu.default(x)
        torch.ops.kashi_iree_execution.write.default(out, value)
        return torch.ops.aten.neg.default(out)

    x = torch.linspace(-2, 2, 80, device="cuda").reshape(10, 8)
    out = torch.empty_like(x)
    actual = torch.compile(model, backend=backend, fullgraph=True)(x, out)
    expected = torch.relu(x)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(actual, -expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["cuda", "iree"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_iree_graph_matmul_matches_torch(tmp_path, monkeypatch, mode, dtype):
    """Both launch paths reproduce matrix multiplication for real model dtypes."""
    from kashi.iree_backend import compile_iree_fx

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    torch._dynamo.reset()

    def backend(gm, example_inputs):
        """Compile the captured graph and return its executable forward method."""
        return compile_iree_fx(gm, output_dir=tmp_path).graph.forward

    def model(a, b):
        """Apply a matrix multiply whose launch needs tiled workgroups."""
        return torch.ops.aten.mm.default(a, b)

    torch.manual_seed(42)
    a = torch.randn(31, 64, dtype=dtype, device="cuda")
    b = torch.randn(64, 48, dtype=dtype, device="cuda")
    actual = torch.compile(model, backend=backend, fullgraph=True)(a, b)
    torch.testing.assert_close(actual, model(a, b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["cuda", "iree"])
def test_iree_graph_preserves_split_view_mutation(tmp_path, monkeypatch, mode):
    """A custom mutation of a split view must update its original tensor."""
    from kashi.iree_backend import compile_iree_fx

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    torch._dynamo.reset()

    def backend(gm, example_inputs):
        """Compile the captured graph and return its executable forward method."""
        return compile_iree_fx(gm, output_dir=tmp_path).graph.forward

    def model(x):
        """Mutate one split while returning both aliases."""
        left, right = torch.ops.aten.split_with_sizes.default(x, [2, 2], -1)
        torch.ops.kashi_iree_execution.increment.default(left)
        return left, right

    x = torch.arange(24, dtype=torch.float32, device="cuda").reshape(6, 4)
    expected = x.clone()
    expected[:, :2].add_(3)
    left, right = torch.compile(model, backend=backend, fullgraph=True)(x)
    torch.testing.assert_close(x, expected)
    torch.testing.assert_close(left, expected[:, :2])
    torch.testing.assert_close(right, expected[:, 2:])
    assert left.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert right.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["cuda", "iree"])
def test_iree_graph_keeps_distinct_outputs_independent(tmp_path, monkeypatch, mode):
    """IREE common-subexpression elimination must not create new aliases."""
    from kashi.iree_backend import compile_iree_fx

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    torch._dynamo.reset()

    def backend(gm, example_inputs):
        """Compile the captured graph and return its executable forward method."""
        return compile_iree_fx(gm, output_dir=tmp_path).graph.forward

    def model(x):
        """Produce equal-valued results with distinct PyTorch storage."""
        return torch.ops.aten.relu.default(x), torch.ops.aten.relu.default(x)

    x = torch.linspace(-2, 2, 80, device="cuda")
    first, second = torch.compile(model, backend=backend, fullgraph=True)(x)
    expected = torch.relu(x)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    first.add_(1)
    torch.testing.assert_close(second, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["cuda", "iree"])
def test_torch_compile_iree_backend_keeps_artifacts_alive(monkeypatch, mode):
    """The public backend can execute repeatedly without a caller-owned directory."""
    import gc

    from kashi.iree_backend import compile_iree_backend

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    torch._dynamo.reset()

    def model(x):
        """Run an operation lowered to an IREE dispatch."""
        return torch.ops.aten.neg.default(x)

    compiled = torch.compile(model, backend=compile_iree_backend, fullgraph=True)
    x = torch.linspace(-1, 1, 32, device="cuda")
    torch.testing.assert_close(compiled(x), -x)
    gc.collect()
    torch.testing.assert_close(compiled(x + 2), -(x + 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_direct_cuda_backend_supports_graph_replay(monkeypatch):
    """Direct launches participate in CUDA graph capture and replay."""
    from kashi.iree_backend import compile_iree_backend

    monkeypatch.setenv("KASHI_IREE_EXECUTION", "cuda")
    torch._dynamo.reset()

    def model(x):
        """Run a kernel with stable input and output buffers."""
        return torch.ops.aten.neg.default(x)

    compiled = torch.compile(model, backend=compile_iree_backend, fullgraph=True)
    x = torch.linspace(-1, 1, 32, device="cuda")
    compiled(x)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        output = compiled(x)
    x.fill_(3)
    capture.replay()
    torch.testing.assert_close(output, -x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["cuda", "iree"])
def test_iree_graph_preserves_unsigned_dtype(monkeypatch, mode):
    """IREE's signless integer types must retain PyTorch's unsigned dtype."""
    from kashi.iree_backend import compile_iree_backend

    monkeypatch.setenv("KASHI_IREE_EXECUTION", mode)
    torch._dynamo.reset()

    def model(x):
        """Exercise unsigned arithmetic with values crossing the sign bit."""
        return torch.ops.aten.add.Scalar(x, 1)

    x = torch.tensor([0, 127, 254, 255], dtype=torch.uint8, device="cuda")
    result = torch.compile(model, backend=compile_iree_backend, fullgraph=True)(x)
    torch.testing.assert_close(result, model(x))
