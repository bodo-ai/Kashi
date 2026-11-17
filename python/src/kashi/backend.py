"""Standalone TorchDynamo backend: FX -> MLIR -> FX, executed by eager PyTorch.

A plain callable for ``torch.compile(model, backend=compile_fx)``; it is not
registered under a name.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from kashi.importer import import_fx
from kashi.rebuild import rebuild_fx


def compile_fx(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> Callable[..., Any]:
    """Import ``gm`` to MLIR, verify it, and return the rebuilt graph's
    ``forward``. Errors propagate; there is no fallback to ``gm``."""
    imported = import_fx(gm, example_inputs)
    imported.module.operation.verify()
    return rebuild_fx(imported.module, imported.out_spec).forward
