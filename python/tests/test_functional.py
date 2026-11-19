import pytest

torch = pytest.importorskip("torch")

from kashi import functional
from torch._subclasses.fake_tensor import FakeTensorMode

WRITE = functional.FunctionalForm("write_functional", output="out", like="x")


@pytest.fixture
def test_ops(monkeypatch):
    import kashi_test_ops

    monkeypatch.setitem(functional._FUNCTIONAL_OPS, "kashi_test::write", WRITE)
    return kashi_test_ops


def test_vllm_entry():
    assert functional._FUNCTIONAL_OPS["vllm::unified_attention_with_output"] == (
        functional.FunctionalForm("unified_attention", output="output", like="query")
    )


def test_functional_form_matches_by_name(test_ops):
    assert functional.functional_form(torch.ops.kashi_test.write.default) == WRITE
    assert functional.functional_form(torch.ops.kashi_test.scale.default) is None


def test_dummy_schema_drops_output_and_returns_tensor(test_ops):
    write = torch.ops.kashi_test.write.default
    dummy = functional.dummy_op(write, WRITE)

    assert dummy is torch.ops.kashi.write_functional.default
    assert functional.dummy_op(write, WRITE) is dummy
    args = dummy._schema.arguments
    assert [a.name for a in args] == ["x", "tag", "dep"]
    assert [a.kwarg_only for a in args] == [False, False, True]
    assert str(args[1].type) == "PyObject"
    assert all(a.alias_info is None for a in args)
    assert [str(r.type) for r in dummy._schema.returns] == ["Tensor"]


def test_dummy_fake_impl_is_contiguous_like(test_ops):
    dummy = functional.dummy_op(torch.ops.kashi_test.write.default, WRITE)
    with FakeTensorMode() as mode:
        x = mode.from_tensor(torch.randn(4, 3)).t()
        y = dummy(x, test_ops.Tag(1.0), dep=x)
    assert (y.shape, y.dtype, y.is_contiguous()) == ((3, 4), torch.float32, True)


def test_functional_original(test_ops):
    assert functional.functional_original("write_functional") == (
        "kashi_test::write",
        WRITE,
    )
    assert functional.functional_original("nope") is None
