"""Compile an imported Torch graph with both supported load orders."""

import platform
import sys
import tempfile
from importlib import import_module
from pathlib import Path


def main():
    """Compile an FX graph using the installed wheel and requested load order."""
    if sys.argv[1] == "iree-first":
        ir = import_module("iree.compiler.ir")
        _native = import_module("kashi._native")
    else:
        _native = import_module("kashi._native")
        ir = import_module("iree.compiler.ir")

    import torch
    from kashi.importer import import_fx
    from torch._subclasses.fake_tensor import FakeTensorMode

    extension = Path(_native.__file__).resolve()
    if not extension.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"did not import the installed extension: {extension}")

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    result = graph.call_function(torch.ops.aten.relu.default, (value,))
    graph.output(result)
    with FakeTensorMode() as mode:
        value.meta["val"] = mode.from_tensor(torch.empty(128))
        result.meta["val"] = value.meta["val"]
    imported = import_fx(torch.fx.GraphModule(torch.nn.Module(), graph))
    assert imported.module.operation.verify()
    assembly = imported.module.operation.get_asm(enable_debug_info=True)
    with tempfile.TemporaryDirectory() as directory:
        _native.compile_iree(assembly, directory, target="vmvx")
        flow = ir.Module.parse(
            (Path(directory) / "flow.mlir").read_text(), ir.Context()
        )
        assert flow.operation.verify()
        assert "flow.dispatch" in str(flow)
    if platform.system() == "Linux":
        with tempfile.TemporaryDirectory() as directory:
            _native.compile_iree(
                assembly, directory, target="cuda", cuda_target="sm_80"
            )
            assert list(Path(directory).glob("*.ptx"))


if __name__ == "__main__":
    main()
