"""Exercise the private compiler without a compiler executable or CUDA device."""

import pytest
from iree.compiler import ir
from kashi import _native


def test_parse_error_does_not_poison_next_invocation(tmp_path):
    """A failed parse leaves the next compiler invocation usable."""
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(RuntimeError, match="parse"):
        _native.compile_iree("module { invalid", str(bad), target="vmvx")
    assert "error" in (bad / "iree-compile.log").read_text()

    good = tmp_path / "good"
    good.mkdir()
    _native.compile_iree(
        """module {
          func.func @main(%x: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
            %y = torch.aten.relu %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
            return %y : !torch.vtensor<[4],f32>
          }
        }""",
        str(good),
        target="vmvx",
    )
    assert "func.func @main" in (good / "driver.mlir").read_text()
    assert (good / "flow.mlir").is_file()


def test_partition_preserves_resource_payload_and_source_location(tmp_path):
    """Both partition artifacts retain resource bytes and source locations."""
    assembly = """
module {
  func.func @main() -> !torch.vtensor<[4],f32> {
    %x = torch.vtensor.literal(dense_resource<weights> : tensor<4xf32>) : !torch.vtensor<[4],f32> loc("fixture.py":7:3)
    %y = torch.aten.relu %x : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    return %y : !torch.vtensor<[4],f32>
  }
}
{-#
  dialect_resources: { builtin: { weights: "0x040000000000803f000080bf00000040000000c0" } }
#-}
"""
    _native.compile_iree(assembly, str(tmp_path), target="vmvx")
    for artifact in ("partitioned.mlir", "driver.mlir"):
        text = (tmp_path / artifact).read_text()
        module = ir.Module.parse(text, ir.Context())
        assert module.operation.verify()
        assert 'loc("fixture.py":7:3)' in text
        assert "0x040000000000803f000080bf00000040000000c0" in text.lower()
