"""Test-only custom operators and opaque type mirroring vLLM's attention ops.

``Tag`` is a hoisted opaque object (like vLLM's ``LayerName``), so Dynamo
lifts it to a graph input. ``write`` returns nothing, writes ``out`` and takes
a keyword-only dependency tensor, like ``unified_attention_with_output``.
"""

import torch
from torch import nn
from torch._library.opaque_object import get_opaque_type_name, register_opaque_type
from torch._opaque_base import OpaqueBase


class Tag(OpaqueBase):
    def __init__(self, factor: float):
        self.factor = factor

    def __eq__(self, other):
        return isinstance(other, Tag) and other.factor == self.factor

    def __hash__(self):
        return hash(self.factor)

    def __fx_repr__(self):
        return (f"Tag({self.factor!r})", {"Tag": Tag})


register_opaque_type(Tag, typ="value", hoist=True)

# ``torch.library.custom_op`` rejects kwarg-only tensors, so define schemas
# directly, as vLLM does.
LIB = torch.library.Library("kashi_test", "FRAGMENT")
_TAG = get_opaque_type_name(Tag)

LIB.define(f"scale(Tensor x, {_TAG} tag) -> Tensor")
LIB.define(f"write(Tensor x, Tensor(a!) out, {_TAG} tag, *, Tensor dep) -> ()")


def _scale(x, tag):
    return x * tag.factor


def _write(x, out, tag, *, dep):
    out.copy_(x + tag.factor)


LIB.impl("scale", _scale, "CompositeExplicitAutograd")
LIB.impl("write", _write, "CompositeExplicitAutograd")
torch.library.register_fake("kashi_test::scale", lambda x, tag: torch.empty_like(x), lib=LIB)
torch.library.register_fake("kashi_test::write", lambda x, out, tag, *, dep: None, lib=LIB)


class OpsModel(nn.Module):
    def __init__(self, a: Tag, b: Tag):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        d = torch.ops.kashi_test.scale(x, self.a)
        out = torch.empty(torch.Size([x.shape[0], 4]))
        torch.ops.kashi_test.write(x, out, self.b, dep=d)
        return out.view(-1), d


def capture(model, *inputs):
    """Compile ``model`` once with a recording backend; return ``(gm, example_inputs)``."""
    torch._dynamo.reset()
    record = []

    def backend(gm, example_inputs):
        record.append((gm, list(example_inputs)))
        return gm.forward

    torch.compile(model, backend=backend, fullgraph=True)(*inputs)
    (result,) = record
    return result
