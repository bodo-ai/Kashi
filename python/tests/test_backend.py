"""torch.compile integration tests for the standalone FX -> MLIR -> FX backend."""

import pytest
torch = pytest.importorskip("torch")

from torch import nn
from torch.utils import _pytree as pytree

from kashi.backend import compile_fx


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()


class Spy:
    """Wraps compile_fx; records each graph received and callable returned."""

    def __init__(self):
        self.calls = []

    def __call__(self, gm, example_inputs):
        fn = compile_fx(gm, example_inputs)
        self.calls.append((gm, fn))
        return fn


class EmbedChunkView(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 8)
        self.qkv = nn.Linear(8, 24)
        self.o = nn.Linear(8, 8)

    def forward(self, ids):
        q, k, v = self.qkv(self.embed(ids)).chunk(3, dim=-1)
        return self.o(q).view(2, -1), k


class Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x).view(-1)


class Empty(nn.Module):
    def forward(self, x):
        return torch.empty(2, 3), x.view(-1)


CASES = {
    "embed_chunk_view": (EmbedChunkView, lambda: torch.tensor([1, 2, 3, 4])),
    "single": (Single, lambda: torch.randn(2, 4)),
    "empty": (Empty, lambda: torch.randn(2, 2)),
}

# torch.empty returns uninitialized memory: only shape and dtype are comparable.
UNINITIALIZED_LEAVES = {"empty": {0}}


def _compile(model_cls):
    torch.manual_seed(0)
    model = model_cls()
    spy = Spy()
    return model, spy, torch.compile(model, backend=spy, dynamic=False)


@pytest.mark.parametrize("case", CASES)
def test_backend_matches_eager(case):
    model_cls, make_input = CASES[case]
    model, spy, compiled = _compile(model_cls)
    x = make_input()

    actual, expected = compiled(x), model(x)

    assert len(spy.calls) == 1
    gm, fn = spy.calls[0]
    assert fn is not gm.forward
    assert isinstance(fn.__self__, torch.fx.GraphModule)
    assert fn.__self__ is not gm

    assert pytree.tree_structure(actual) == pytree.tree_structure(expected)
    skip = UNINITIALIZED_LEAVES.get(case, set())
    leaves = zip(pytree.tree_leaves(actual), pytree.tree_leaves(expected))
    for i, (a, e) in enumerate(leaves):
        if i in skip:
            assert (a.shape, a.dtype) == (e.shape, e.dtype)
        else:
            torch.testing.assert_close(a, e)


def test_reuses_compiled_graph_for_new_values():
    model, spy, compiled = _compile(Single)
    compiled(torch.randn(2, 4))
    x = torch.randn(2, 4)

    torch.testing.assert_close(compiled(x), model(x))
    assert len(spy.calls) == 1


def test_sees_parameter_updates():
    model, spy, compiled = _compile(Single)
    x = torch.randn(2, 4)
    compiled(x)
    with torch.no_grad():
        model.linear.weight.add_(1.0)

    torch.testing.assert_close(compiled(x), model(x))
    assert len(spy.calls) == 1
