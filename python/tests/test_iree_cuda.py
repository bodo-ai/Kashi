"""Direct CUDA launches honor IREE's lowered binding and scalar ABI."""

import pytest

torch = pytest.importorskip("torch")

from kashi.iree_cuda import create_cuda_kernel
from kashi.iree_gpu_dispatch import _grid


@pytest.mark.skipif(
    not (torch.version.cuda and torch.cuda.is_available()),
    reason="NVIDIA CUDA device required",
)
def test_direct_ptx_uses_independent_output_offset_and_current_stream(tmp_path):
    """An offset inserted by stream allocation must not shift a new output buffer."""
    phases = tmp_path / "phases"
    kernels = tmp_path / "kernel_files"
    phases.mkdir()
    kernels.mkdir()
    (phases / "x.flow.mlir").write_text("""
flow.executable private @example {
  flow.executable.export public @entry workgroups(%arg0: index) -> (index, index, index) {
  }
  builtin.module {
    func.func @entry(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>, %arg1: index, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
      return
    }
  }
}
""")
    (phases / "x.stream.mlir").write_text("""
stream.executable private @example {
  builtin.module {
    func.func @entry(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
      %c0 = arith.constant 0 : index
      %c32_i64 = arith.constant 32 : i64
      %0 = arith.extui %arg2 : i32 to i64
      %1 = arith.extui %arg3 : i32 to i64
      %2 = arith.shli %1, %c32_i64 : i64
      %3 = arith.ori %0, %2 : i64
      %4 = arith.index_castui %3 : i64 to index
      %5 = arith.extui %arg4 : i32 to i64
      %6 = arith.extui %arg5 : i32 to i64
      %7 = arith.shli %6, %c32_i64 : i64
      %8 = arith.ori %5, %7 : i64
      %9 = arith.index_castui %8 : i64 to index
      %10:2 = util.assume.int %4<udiv = 16>, %9<umin = 0> : index, index
      %11 = iree_tensor_ext.dispatch.workload.ordinal %10#1, 0 : index
      %12 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%11}
      %13 = stream.binding.subspan %arg1[%10#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%11}
      return
    }
  }
}
""")
    (phases / "x.executable-targets.mlir").write_text("""
#layout = #hal.pipeline.layout<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, None>]>
hal.executable private @example {
  hal.executable.variant public @cuda_nvptx_fb target(#target) {
    hal.executable.export public @entry ordinal(0) layout(#layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    } <workgroup_size = [128 : index, 1 : index, 1 : index], subgroup_size = 32>
  }
}
""")
    (kernels / "module_example_cuda_nvptx_fb.ptx").write_text("""
.version 7.0
.target sm_70
.address_size 64
.visible .entry entry(
  .param .u64 .ptr .global .align 16 input,
  .param .u64 .ptr .global .align 16 output,
  .param .b32 offset_lo,
  .param .b32 offset_hi,
  .param .b32 size_lo,
  .param .b32 size_hi
) .maxntid 128, 1, 1 {
  .reg .pred %p;
  .reg .b32 %r<4>;
  .reg .b64 %rd<8>;
  .reg .f32 %f<2>;
  ld.param.u64 %rd1, [input];
  ld.param.u64 %rd2, [output];
  ld.param.u32 %r1, [offset_lo];
  ld.param.u32 %r2, [offset_hi];
  ld.param.u32 %r3, [size_lo];
  mov.u32 %r0, %tid.x;
  setp.ge.u32 %p, %r0, %r3;
  @%p bra DONE;
  cvt.u64.u32 %rd3, %r1;
  cvt.u64.u32 %rd4, %r2;
  shl.b64 %rd4, %rd4, 32;
  or.b64 %rd3, %rd3, %rd4;
  mul.wide.u32 %rd5, %r0, 4;
  add.u64 %rd6, %rd1, %rd5;
  add.u64 %rd7, %rd2, %rd3;
  add.u64 %rd7, %rd7, %rd5;
  ld.global.f32 %f0, [%rd6];
  add.f32 %f1, %f0, %f0;
  st.global.f32 [%rd7], %f1;
DONE:
  ret;
}
""")
    kernel = create_cuda_kernel("example", "entry", artifact_dir=tmp_path)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(16, device="cuda", dtype=torch.float32)
        result = kernel(
            x,
            16,
            workload=[16],
            argument_dims=[16],
            result_shapes=[[16]],
            result_dtypes=[torch.float32],
            tied_operands=[-1],
        )
        result = result + 1
    stream.synchronize()
    torch.testing.assert_close(
        result.cpu(), torch.arange(16, dtype=torch.float32) * 2 + 1
    )


@pytest.mark.skipif(
    not (torch.version.cuda and torch.cuda.is_available()),
    reason="NVIDIA CUDA device required",
)
def test_direct_ptx_preserves_tied_operand_identity(tmp_path):
    """A tied result must update and return the original CUDA tensor."""
    phases = tmp_path / "phases"
    kernels = tmp_path / "kernel_files"
    phases.mkdir()
    kernels.mkdir()
    (phases / "x.flow.mlir").write_text("""
flow.executable private @example {
  builtin.module {
    func.func @entry(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16xf32>>) {
      return
    }
  }
}
""")
    (phases / "x.stream.mlir").write_text("""
stream.executable private @example {
  builtin.module {
    func.func @entry(%arg0: !stream.binding {stream.alignment = 64 : index}) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16xf32>>
      return
    }
  }
}
""")
    (phases / "x.executable-targets.mlir").write_text("""
hal.executable private @example {
  hal.executable.variant public @cuda_nvptx_fb target(#target) {
    hal.executable.export public @entry ordinal(0) layout(#layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    } <workgroup_size = [16 : index, 1 : index, 1 : index], subgroup_size = 32>
  }
}
""")
    (kernels / "module_example_cuda_nvptx_fb.ptx").write_text("""
.version 7.0
.target sm_70
.address_size 64
.visible .entry entry(.param .u64 .ptr .global .align 16 buffer)
.maxntid 16, 1, 1 {
  .reg .b32 %r;
  .reg .b64 %rd<3>;
  .reg .f32 %f;
  ld.param.u64 %rd0, [buffer];
  mov.u32 %r, %tid.x;
  mul.wide.u32 %rd1, %r, 4;
  add.u64 %rd2, %rd0, %rd1;
  ld.global.f32 %f, [%rd2];
  add.f32 %f, %f, %f;
  st.global.f32 [%rd2], %f;
  ret;
}
""")
    kernel = create_cuda_kernel("example", "entry", artifact_dir=tmp_path)
    x = torch.arange(16, device="cuda", dtype=torch.float32)
    result = kernel(
        x,
        workload=[],
        argument_dims=[],
        result_shapes=[[16]],
        result_dtypes=[torch.float32],
        tied_operands=[0],
    )
    assert result is x
    torch.testing.assert_close(x.cpu(), torch.arange(16, dtype=torch.float32) * 2)


@pytest.mark.parametrize(
    ("op", "lhs", "rhs", "expected"),
    [
        ("addi", 7, 0, 7),
        ("subi", 7, 3, 4),
        ("muli", 7, 3, 21),
        ("divui", 7, 3, 2),
        ("ceildivui", 7, 3, 3),
        ("minui", 7, 3, 3),
        ("maxui", 7, 3, 7),
    ],
)
def test_workgroup_grid_arithmetic(op, lhs, rhs, expected):
    """Interpret IREE grid arithmetic without evaluating unrelated divisions."""
    body = f"""
      %result = arith.{op} %lhs, %rhs : index
      %one = arith.constant 1 : index
      hal.return %result, %one, %one : index, index, index
    """
    args = ["%device: !hal.device", "%lhs: index", "%rhs: index"]
    assert _grid(args, body, {}, [lhs, rhs], "test") == (expected, 1, 1)


@pytest.mark.parametrize("op", ["divui", "ceildivui"])
def test_workgroup_grid_rejects_zero_divisor(op):
    """An invalid divisor must not produce an incorrect GPU launch grid."""
    body = f"%result = arith.{op} %lhs, %rhs : index"
    args = ["%device: !hal.device", "%lhs: index", "%rhs: index"]
    with pytest.raises(NotImplementedError, match="workgroup divisor 0"):
        _grid(args, body, {}, [7, 0], "test")


@pytest.mark.parametrize(
    ("op", "expressions", "expected"),
    [
        ("apply", ["(d0 + 1) * 2 - s0"], 13),
        ("apply", ["d0 floordiv s0"], 2),
        ("apply", ["d0 ceildiv s0"], 3),
        ("apply", ["d0 mod s0"], 1),
        ("apply", ["-d0 + 10"], 3),
        ("min", ["d0 + 1", "d0 ceildiv s0"], 3),
        ("max", ["d0 + 1", "d0 ceildiv s0"], 8),
    ],
)
def test_workgroup_grid_affine_expressions(op, expressions, expected):
    """Preserve affine precedence and rounding in generated launch sizes."""
    body = f"""
      %result = affine.{op} #grid(%lhs)[%rhs]
      %one = arith.constant 1 : index
      hal.return %result, %one, %one : index, index, index
    """
    args = ["%device: !hal.device", "%lhs: index", "%rhs: index"]
    maps = {"#grid": (["d0"], ["s0"], expressions)}
    assert _grid(args, body, maps, [7, 3], "test") == (expected, 1, 1)


@pytest.mark.parametrize("expression", ["d0 floordiv 0", "d1 + 1", "d0 / 2"])
def test_workgroup_grid_rejects_invalid_affine_expressions(expression):
    """Bad affine metadata cannot silently launch an incorrect grid."""
    body = "%result = affine.apply #grid(%lhs)"
    args = ["%device: !hal.device", "%lhs: index"]
    maps = {"#grid": (["d0"], [], [expression])}
    with pytest.raises(NotImplementedError, match="affine"):
        _grid(args, body, maps, [7], "test")


@pytest.mark.skipif(
    not (torch.version.cuda and torch.cuda.is_available()),
    reason="NVIDIA CUDA device required",
)
def test_direct_graph_capture_retains_dlpack_input_through_replay(
    tmp_path, monkeypatch
):
    import gc
    import weakref

    from kashi.iree_backend import compile_iree_fx
    from torch._subclasses.fake_tensor import FakeTensorMode

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    result = graph.call_function(torch.ops.aten.relu.default, (x,))
    graph.output(result)
    with FakeTensorMode():
        value = torch.empty(16, device="cuda")
        x.meta["val"] = result.meta["val"] = value
    model = torch.fx.GraphModule(torch.nn.Module(), graph)
    monkeypatch.setenv("KASHI_IREE_EXECUTION", "direct")
    compiled = compile_iree_fx(model, output_dir=tmp_path, target_device="cuda")
    source = torch.arange(-8, 8, device="cuda", dtype=torch.float32)
    imported = torch.utils.dlpack.from_dlpack(source)
    reference = weakref.ref(imported)
    compiled.graph(torch.zeros_like(source))  # load the module before capture
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        output = compiled.graph(imported)
    del imported, source
    gc.collect()
    assert reference() is not None
    capture.replay()
    capture.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.arange(-8, 8, dtype=torch.float32).relu()
    )
    capture.reset()
    gc.collect()
    assert reference() is None
