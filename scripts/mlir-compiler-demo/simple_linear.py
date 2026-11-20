
import torch
import torch.nn as nn
import torch.fx as fx

from torch_mlir.fx import stateless_fx_import
from torch_mlir.ir import Module, IntegerAttr, StringAttr


class MyModel(nn.Module):
    """Two-layer linear network with an intermediate reshape."""
    def __init__(self):
        """Create linear layers sized for the demo's input and reshaped features."""
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(6, 1)

    def forward(self, x):
        """Apply the first projection, flatten features, then project output."""
        x = self.linear1(x)
        x = x.view(1, -1)
        return self.linear2(x)


def mlir_to_fx(module: Module) -> fx.GraphModule:
    """Reconstruct an FX graph from the demo's Torch-dialect MLIR subset.

    The importer handles only integer constants, constructed lists, the
    supported ATen calls, and the function return; other operations fail.
    """
    graph = fx.Graph()

    # Find the main MLIR function.
    main = next(
        op for op in module.body.operations
        if op.operation.name == "func.func"
        and StringAttr(
            op.attributes["sym_name"]
        ).value == "main"
    )

    block = main.regions[0].blocks[0]

    # MLIR SSA value -> FX node or Python constant.
    values = {
        arg: graph.placeholder(f"arg{i}")
        for i, arg in enumerate(block.arguments)
    }

    ops = {
        "torch.aten.linear": torch.ops.aten.linear.default,
        "torch.aten.view": torch.ops.aten.view.default,
    }

    for op in block.operations:
        operation = op.operation
        name = operation.name

        if name == "torch.constant.int":
            value = IntegerAttr(
                operation.attributes["value"]
            ).value

            values[operation.results[0]] = int(value)

        elif name == "torch.prim.ListConstruct":
            values[operation.results[0]] = [
                values[arg] for arg in operation.operands
            ]

        elif name in ops:
            args = tuple(
                values[arg] for arg in operation.operands
            )

            result = graph.call_function(
                ops[name],
                args=args,
            )

            values[operation.results[0]] = result

        elif name == "func.return":
            outputs = tuple(
                values[arg] for arg in operation.operands
            )
            graph.output(outputs)

        else:
            raise NotImplementedError(
                f"Unsupported MLIR operation: {name}"
            )

    graph.lint()
    return fx.GraphModule(nn.Module(), graph)


def normalize_fx(gm: fx.GraphModule):
    """Convert Dynamo callables to the ATen forms expected by Torch MLIR.

    Also copy ``example_value`` metadata to ``val``, which the importer uses
    for fake-value information.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target is torch._C._nn.linear:
                node.target = torch.ops.aten.linear.default

        elif node.op == "call_method":
            if node.target == "view":
                x, *shape = node.args

                if len(shape) == 1 and isinstance(
                    shape[0], (tuple, list)
                ):
                    shape = list(shape[0])

                node.op = "call_function"
                node.target = torch.ops.aten.view.default
                node.args = (x, shape)
                node.kwargs = {}

        if "val" not in node.meta:
            if "example_value" in node.meta:
                node.meta["val"] = node.meta["example_value"]

    gm.graph.lint()
    gm.recompile()
    return gm


def my_backend(gm: fx.GraphModule, example_inputs):
    """Round-trip Dynamo FX through Torch MLIR and return reconstructed FX."""
    print("\nOriginal Dynamo FX:")
    gm.graph.print_tabular()

    gm = normalize_fx(gm)

    print("\nNormalized ATen FX:")
    gm.graph.print_tabular()

    module = stateless_fx_import(
        gm,
        output_type="raw",
    )

    print("\nGenerated MLIR:")
    print(module)

    gm = mlir_to_fx(module)
    
    print("\nReconstructed FX:")
    gm.graph.print_tabular()

    return gm.forward


torch.manual_seed(42)

model = MyModel().eval().requires_grad_(False)
x = torch.randn(2, 4)

compiled_model = torch.compile(
    model,
    backend=my_backend,
    fullgraph=True,
)

with torch.inference_mode():
    expected = model(x)
    actual = compiled_model(x)

torch.testing.assert_close(expected, actual)

print("Success!")
