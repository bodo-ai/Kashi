"""Behavioral coverage for partitioning Torch-dialect graphs at custom ops."""

from types import SimpleNamespace

import pytest
from iree.compiler import ir
from kashi import _native

TENSOR = "!torch.vtensor<[4],f32>"


def _parse(text):
    """Parse a standalone module using IREE's registered Torch dialect."""
    with ir.Context():
        return ir.Module.parse(text)


def _partition(source, directory, *, partition_only=False):
    """Compile a fixture and load the native partition and driver artifacts."""
    device = None
    try:
        device = _native.compile_iree(str(source), str(directory), target="vmvx")
    except RuntimeError as error:
        # These fixtures exercise Torch interfaces that the backend may not lower.
        if not partition_only or "compile through" not in str(error):
            raise
    return SimpleNamespace(
        module=_parse((directory / "partitioned.mlir").read_text()),
        driver=_parse((directory / "driver.mlir").read_text()),
        device=device,
    )


def _reject_partition(source, directory, diagnostic):
    """Require a partition-stage diagnostic without a partial driver."""
    with pytest.raises(NotImplementedError, match="partition"):
        _partition(source, directory)
    assert diagnostic in (directory / "iree-compile.log").read_text()
    assert not (directory / "driver.mlir").exists()


def _main(module):
    """Find the driver's main entry block, ignoring segment declarations."""
    return next(
        op.regions[0].blocks[0]
        for op in module.body.operations
        if ir.StringAttr(op.attributes["sym_name"]).value == "main"
    )


def _ops(block):
    """Return operations without their dialect-specific Python wrappers."""
    return [op.operation for op in block.operations]


MUTATION = """
module {
  func.func @main(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %one = torch.constant.int 1
    %four = torch.constant.int 4
    %dtype = torch.constant.int 6
    %none = torch.constant.none
    %device = torch.constant.device "cuda:0"
    %shape = torch.prim.ListConstruct %four : (!torch.int) -> !torch.list<int>
    %value = torch.aten.relu %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    %buf = torch.aten.empty.memory_format %shape, %dtype, %none, %device, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[4],f32>
    torch.operator "torch.test.mutate"(%buf, %value) : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> ()
    %sum = torch.aten.add.Tensor %buf, %value, %one : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    return %sum : !torch.vtensor<[4],f32>
  }
}
"""


def test_mutating_boundary_keeps_wiring_and_source_allocation_device(tmp_path):
    """Custom mutations observe the driver-allocated buffer and compiled value."""
    source = _parse(MUTATION)
    before = str(source)
    result = _partition(source, tmp_path)
    assert str(source) == before
    assert result.module.operation.verify()
    assert result.driver.operation.verify()
    assert result.device == "cuda:0"
    assert 'torch.constant.device "cuda:0"' in str(result.driver)
    assert "torch.operator" not in str(result.module)
    ops = _ops(_main(result.driver))
    calls = [op for op in ops if op.name == "func.call"]
    assert [[str(v.type) for v in call.results] for call in calls] == [
        [TENSOR],
        [TENSOR],
    ]
    custom = next(op for op in ops if op.name == "torch.operator")
    allocation = next(op for op in ops if op.name == "torch.aten.empty.memory_format")
    assert ops.index(calls[0]) < ops.index(custom) < ops.index(calls[1])
    assert list(custom.operands) == [allocation.results[0], calls[0].results[0]]
    assert list(calls[1].operands) == [allocation.results[0], calls[0].results[0]]
    assert list(ops[-1].operands) == [calls[1].results[0]]


def test_split_view_mutation_retains_original_base(tmp_path):
    """A mutating custom op must receive a view of the original input storage."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[6,4],f32>) -> (!torch.vtensor<[6,4],f32>, !torch.vtensor<[6,2],f32>) {
    %two = torch.constant.int 2
    %dim = torch.constant.int -1
    %sections = torch.prim.ListConstruct %two, %two : (!torch.int, !torch.int) -> !torch.list<int>
    %pieces = torch.aten.split_with_sizes %x, %sections, %dim : !torch.vtensor<[6,4],f32>, !torch.list<int>, !torch.int -> !torch.list<vtensor>
    %left, %right = torch.prim.ListUnpack %pieces : !torch.list<vtensor> -> !torch.vtensor<[6,2],f32>, !torch.vtensor<[6,2],f32>
    torch.operator "torch.test.increment"(%left) : (!torch.vtensor<[6,2],f32>) -> ()
    return %x, %left : !torch.vtensor<[6,4],f32>, !torch.vtensor<[6,2],f32>
  }
}
""")
    result = _partition(source, tmp_path, partition_only=True)
    block = _main(result.driver)
    ops = _ops(block)
    split = next(op for op in ops if op.name == "torch.aten.split_with_sizes")
    unpack = next(op for op in ops if op.name == "torch.prim.ListUnpack")
    custom = next(op for op in ops if op.name == "torch.operator")
    assert split.operands[0] == block.arguments[0]
    assert unpack.operands[0] == split.results[0]
    assert custom.operands[0] == unpack.results[0]
    assert list(ops[-1].operands) == [block.arguments[0], unpack.results[0]]


def test_tensor_list_keeps_input_aliases_at_boundary(tmp_path):
    """A returned tensor list must retain its elements' original identities."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[4],f32>, %y: !torch.vtensor<[4],f32>) -> !torch.list<vtensor> {
    %items = torch.prim.ListConstruct %x, %y : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.list<vtensor>
    return %items : !torch.list<vtensor>
  }
}
""")
    result = _partition(source, tmp_path, partition_only=True)
    block = _main(result.driver)
    ops = _ops(block)
    construct = next(op for op in ops if op.name == "torch.prim.ListConstruct")
    assert list(construct.operands) == list(block.arguments)
    assert ops[-1].operands[0] == construct.results[0]


def test_multiple_and_adjacent_boundaries_keep_order_and_value_identity(tmp_path):
    """Consecutive mutations must not be coalesced or reordered."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %a = torch.aten.relu %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    torch.operator "torch.test.first"(%a) : (!torch.vtensor<[4],f32>) -> ()
    torch.operator "torch.test.second"(%a) : (!torch.vtensor<[4],f32>) -> ()
    %b = torch.aten.neg %a : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    torch.operator "torch.test.third"(%b) : (!torch.vtensor<[4],f32>) -> ()
    return %b : !torch.vtensor<[4],f32>
  }
}
""")
    result = _partition(source, tmp_path)
    assert result.module.operation.verify() and result.driver.operation.verify()
    ops = [op for op in _ops(_main(result.driver)) if op.name != "func.return"]
    assert [op.name for op in ops] == [
        "func.call",
        "torch.operator",
        "torch.operator",
        "func.call",
        "torch.operator",
    ]
    assert [ir.StringAttr(ops[i].attributes["name"]).value for i in (1, 2, 4)] == [
        "torch.test.first",
        "torch.test.second",
        "torch.test.third",
    ]
    assert (
        ops[1].operands[0]
        == ops[2].operands[0]
        == ops[3].operands[0]
        == ops[0].results[0]
    )
    assert ops[4].operands[0] == ops[3].results[0]


def test_dynamic_int_and_list_cross_segment_boundaries(tmp_path):
    """Runtime shape values must not be treated as cloneable constants."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[4],f32>, %dim: !torch.int) -> !torch.vtensor<[4],f32> {
    %size = torch.aten.size.int %x, %dim : !torch.vtensor<[4],f32>, !torch.int -> !torch.int
    %shape = torch.prim.ListConstruct %size : (!torch.int) -> !torch.list<int>
    torch.operator "torch.test.barrier"(%size, %shape) : (!torch.int, !torch.list<int>) -> ()
    %out = torch.aten.view %x, %shape : !torch.vtensor<[4],f32>, !torch.list<int> -> !torch.vtensor<[4],f32>
    return %out : !torch.vtensor<[4],f32>
  }
}
""")
    result = _partition(source, tmp_path, partition_only=True)
    assert result.module.operation.verify() and result.driver.operation.verify()
    block = _main(result.driver)
    ops = _ops(block)
    size = next(op for op in ops if op.name == "torch.aten.size.int")
    shape = next(op for op in ops if op.name == "torch.prim.ListConstruct")
    barrier = next(op for op in ops if op.name == "torch.operator")
    view = next(op for op in ops if op.name == "torch.aten.view")
    assert list(size.operands) == list(block.arguments)
    assert shape.operands[0] == size.results[0]
    assert list(barrier.operands) == [size.results[0], shape.results[0]]
    assert view.operands[1] == shape.results[0]
    assert ops[-1].operands[0] == view.results[0]


def test_unsupported_crossing_is_reported_without_invalid_driver(tmp_path):
    """An unsupported float live-out cannot silently disappear from the driver."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
    %f = torch.aten.Float.Tensor %x : !torch.vtensor<[],f32> -> !torch.float
    torch.operator "torch.test.scalar"(%f) : (!torch.float) -> ()
    return %x : !torch.vtensor<[],f32>
  }
}
""")
    _reject_partition(
        source, tmp_path, "unsupported live-out for segment_0: '!torch.float'"
    )


def test_rejects_non_main_entry(tmp_path):
    """A different function is never silently partitioned as main."""
    source = _parse("module { func.func @other() { return } }")
    _reject_partition(source, tmp_path, "func.func @main")


def test_custom_result_becomes_later_segment_input(tmp_path):
    """Custom result SSA values feed the following compiled call unchanged."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %custom = torch.operator "torch.test.produce"(%x) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
    %result = torch.aten.neg %custom : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    return %result : !torch.vtensor<[4],f32>
  }
}
""")
    result = _partition(source, tmp_path)
    assert result.module.operation.verify() and result.driver.operation.verify()
    custom, call, ret = _ops(_main(result.driver))
    assert custom.name == "torch.operator" and call.name == "func.call"
    assert call.operands[0] == custom.results[0]
    assert ret.operands[0] == call.results[0]


def test_unsupported_live_out_is_reported_even_when_later_segment_uses_it(tmp_path):
    """Unsupported SSA values do not disappear when also used by a later run."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[],f32>) -> !torch.list<float> {
    %f = torch.aten.Float.Tensor %x : !torch.vtensor<[],f32> -> !torch.float
    torch.operator "torch.test.barrier"(%x) : (!torch.vtensor<[],f32>) -> ()
    %list = torch.prim.ListConstruct %f : (!torch.float) -> !torch.list<float>
    return %list : !torch.list<float>
  }
}
""")
    _reject_partition(
        source, tmp_path, "unsupported live-out for segment_0: '!torch.float'"
    )


def test_unsupported_live_in_is_rejected(tmp_path):
    """An unsupported external segment input must not reach IREE silently."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[4],f32>, %f: !torch.float) -> !torch.vtensor<[4],f32> {
    %one = torch.constant.int 1
    %out = torch.aten.add.Scalar %x, %f, %one : !torch.vtensor<[4],f32>, !torch.float, !torch.int -> !torch.vtensor<[4],f32>
    return %out : !torch.vtensor<[4],f32>
  }
}
""")
    _reject_partition(
        source, tmp_path, "unsupported live-in for segment_0: '!torch.float'"
    )


def test_cpu_device_is_reported_without_rewriting(tmp_path):
    """The returned device describes CPU graphs too, not just CUDA graphs."""
    result = _partition(_parse(MUTATION.replace("cuda:0", "cpu")), tmp_path)
    assert result.device == "cpu"
    assert result.module.operation.verify() and result.driver.operation.verify()
    assert 'torch.constant.device "cpu"' in str(result.driver)


def test_multiple_source_devices_are_rejected(tmp_path):
    """A single device field cannot faithfully describe a mixed-device graph."""
    source = _parse(
        MUTATION.replace(
            '%device = torch.constant.device "cuda:0"',
            '%device = torch.constant.device "cuda:0"\n    %cpu = torch.constant.device "cpu"',
        )
    )
    _reject_partition(source, tmp_path, "multiple source devices")


def test_dynamic_list_argument_is_forwarded_to_each_user(tmp_path):
    """An input list remains the same SSA value at the custom call and view."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[4],f32>, %shape: !torch.list<int>) -> !torch.vtensor<[4],f32> {
    torch.operator "torch.test.inspect_shape"(%shape) : (!torch.list<int>) -> ()
    %out = torch.aten.view %x, %shape : !torch.vtensor<[4],f32>, !torch.list<int> -> !torch.vtensor<[4],f32>
    return %out : !torch.vtensor<[4],f32>
  }
}
""")
    result = _partition(source, tmp_path, partition_only=True)
    assert result.module.operation.verify() and result.driver.operation.verify()
    block = _main(result.driver)
    custom, view, returned = _ops(block)
    assert custom.name == "torch.operator" and view.name == "torch.aten.view"
    assert custom.operands[0] == block.arguments[1]
    assert list(view.operands) == list(block.arguments)
    assert returned.operands[0] == view.results[0]
