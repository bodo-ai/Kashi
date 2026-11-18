"""Behavioral coverage for partitioning Torch-dialect graphs at custom ops."""

import pytest
from iree.compiler import ir

from kashi.iree_partition import partition_at_custom_ops


TENSOR = "!torch.vtensor<[4],f32>"


def _parse(text):
    """Parse a standalone module using IREE's registered Torch dialect."""
    with ir.Context():
        return ir.Module.parse(text)


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


def test_mutating_boundary_keeps_wiring_and_rewrites_only_compiled_device():
    """Losing either result or passing the pre-mutation value changes the graph."""
    source = _parse(MUTATION)
    before = str(source)
    result = partition_at_custom_ops(source)
    assert str(source) == before
    assert result.module.operation.verify()
    assert result.driver.operation.verify()
    assert result.device == "cuda:0"
    assert 'torch.constant.device "cpu"' in str(result.module)
    assert 'torch.constant.device "cuda:0"' in str(result.driver)
    assert "torch.operator" not in str(result.module)
    assert [s.live_out_types for s in result.segments] == [[TENSOR, TENSOR], [TENSOR]]
    ops = _ops(_main(result.driver))
    calls = [op for op in ops if op.name == "func.call"]
    custom = next(op for op in ops if op.name == "torch.operator")
    assert ops.index(calls[0]) < ops.index(custom) < ops.index(calls[1])
    assert list(custom.operands) == [calls[0].results[1], calls[0].results[0]]
    assert list(calls[1].operands) == [calls[0].results[1], calls[0].results[0]]
    assert list(ops[-1].operands) == [calls[1].results[0]]


def test_multiple_and_adjacent_boundaries_keep_order_and_value_identity():
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
    result = partition_at_custom_ops(source)
    assert result.module.operation.verify() and result.driver.operation.verify()
    ops = [op for op in _ops(_main(result.driver)) if op.name != "func.return"]
    assert [op.name for op in ops] == ["func.call", "torch.operator", "torch.operator", "func.call", "torch.operator"]
    assert [ir.StringAttr(ops[i].attributes["name"]).value for i in (1, 2, 4)] == ["torch.test.first", "torch.test.second", "torch.test.third"]
    assert ops[1].operands[0] == ops[2].operands[0] == ops[3].operands[0] == ops[0].results[0]
    assert ops[4].operands[0] == ops[3].results[0]


def test_dynamic_int_and_list_cross_segment_boundaries():
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
    result = partition_at_custom_ops(source)
    assert result.module.operation.verify() and result.driver.operation.verify()
    assert result.segments[0].live_out_types == ["!torch.int", "!torch.list<int>"]
    ops = _ops(_main(result.driver))
    calls = [op for op in ops if op.name == "func.call"]
    barrier = next(op for op in ops if op.name == "torch.operator")
    assert list(barrier.operands) == list(calls[0].results)
    assert calls[1].operands[1] == calls[0].results[1]


def test_unsupported_crossing_is_reported_without_invalid_driver():
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
    result = partition_at_custom_ops(source)
    assert result.module.operation.verify()
    assert result.driver is None
    assert result.segments[0].unsupported_live_outs == ["!torch.float"]


def test_rejects_non_main_entry():
    """A different function is never silently partitioned as main."""
    source = _parse("module { func.func @other() { return } }")
    with pytest.raises((ValueError, NotImplementedError), match="main"):
        partition_at_custom_ops(source)


def test_custom_result_becomes_later_segment_input():
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
    result = partition_at_custom_ops(source)
    assert result.module.operation.verify() and result.driver.operation.verify()
    custom, call, ret = _ops(_main(result.driver))
    assert custom.name == "torch.operator" and call.name == "func.call"
    assert call.operands[0] == custom.results[0]
    assert ret.operands[0] == call.results[0]


def test_unsupported_live_out_is_reported_even_when_later_segment_uses_it():
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
    result = partition_at_custom_ops(source)
    assert result.module.operation.verify()
    assert result.driver is None
    assert result.segments[0].unsupported_live_outs == ["!torch.float"]


def test_unsupported_live_in_is_rejected():
    """An unsupported external segment input must not reach IREE silently."""
    source = _parse("""
module {
  func.func @main(%f: !torch.float) -> !torch.list<float> {
    %list = torch.prim.ListConstruct %f : (!torch.float) -> !torch.list<float>
    return %list : !torch.list<float>
  }
}
""")
    with pytest.raises(NotImplementedError, match="live-in.*!torch.float"):
        partition_at_custom_ops(source)


def test_cpu_device_is_reported_without_rewriting():
    """The returned device describes CPU graphs too, not just CUDA graphs."""
    result = partition_at_custom_ops(_parse(MUTATION.replace("cuda:0", "cpu")))
    assert result.device == "cpu"
    assert result.module.operation.verify() and result.driver.operation.verify()
    assert 'torch.constant.device "cpu"' in str(result.module)


def test_multiple_source_devices_are_rejected():
    """A single device field cannot faithfully describe a mixed-device graph."""
    source = _parse(MUTATION.replace(
        '%device = torch.constant.device "cuda:0"',
        '%device = torch.constant.device "cuda:0"\n    %cpu = torch.constant.device "cpu"',
    ))
    with pytest.raises(NotImplementedError, match="multiple source devices"):
        partition_at_custom_ops(source)


def test_dynamic_list_argument_is_forwarded_to_each_user():
    """An input list remains the same SSA value at the custom call and segment."""
    source = _parse("""
module {
  func.func @main(%x: !torch.vtensor<[4],f32>, %shape: !torch.list<int>) -> !torch.vtensor<[4],f32> {
    torch.operator "torch.test.inspect_shape"(%shape) : (!torch.list<int>) -> ()
    %out = torch.aten.view %x, %shape : !torch.vtensor<[4],f32>, !torch.list<int> -> !torch.vtensor<[4],f32>
    return %out : !torch.vtensor<[4],f32>
  }
}
""")
    result = partition_at_custom_ops(source)
    assert result.module.operation.verify() and result.driver.operation.verify()
    block = _main(result.driver)
    custom, call, _ = _ops(block)
    assert custom.operands[0] == block.arguments[1]
    assert call.operands[0] == block.arguments[0]
    assert call.operands[1] == block.arguments[1]
