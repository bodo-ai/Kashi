"""Functional stand-ins for destination-passing custom operators.

An operator such as ``vllm::unified_attention_with_output`` returns nothing
and writes its result into an ``output`` argument. In value-semantic
Torch-dialect MLIR that write is invisible, so the importer replaces such a
call with a dummy ``kashi::<name>`` operator that takes the same arguments
minus the output and returns the result. ``rebuild_fx`` expands the dummy
back into a fresh buffer plus the original operator.
"""

import re
from dataclasses import dataclass

import torch
from torch._library.opaque_object import get_opaque_type_name, register_opaque_type
from torch._opaque_base import OpaqueBase


@dataclass(frozen=True)
class FunctionalForm:
    """``dummy``: the stand-in's name in the ``kashi`` namespace. ``output``:
    the original's mutated output argument. ``like``: the argument whose
    shape and dtype the result has (the result is contiguous)."""

    dummy: str
    output: str
    like: str


# Keyed by the original operator's qualified name.
_FUNCTIONAL_OPS: dict[str, FunctionalForm] = {
    "vllm::unified_attention_with_output": FunctionalForm(
        "unified_attention", output="output", like="query"
    ),
}

_LIB = torch.library.Library("kashi", "FRAGMENT")


class _Opaque(OpaqueBase):
    """Stands in for opaque argument types in dummy schemas. A schema reports
    every opaque type as ``PyObject``, which cannot be parsed back, while any
    registered opaque type name parses to ``PyObject``."""


register_opaque_type(_Opaque, typ="reference")
_OPAQUE_NAME = get_opaque_type_name(_Opaque)


def functional_form(op: torch._ops.OpOverload) -> FunctionalForm | None:
    """The functional form of ``op``, if it is the default overload of a
    destination-passing operator in the table."""
    if op._schema.overload_name:
        return None
    return _FUNCTIONAL_OPS.get(op._schema.name)


def dummy_op(op: torch._ops.OpOverload, form: FunctionalForm) -> torch._ops.OpOverload:
    """``torch.ops.kashi.<form.dummy>.default``, defined on first use from
    ``op``'s schema: no output argument, no alias annotations, returns a
    Tensor. It has a fake implementation only."""
    if not hasattr(torch.ops.kashi, form.dummy):
        _LIB.define(_dummy_schema(op, form))
        torch.library.register_fake(
            f"kashi::{form.dummy}", _fake_impl(form), lib=_LIB
        )
    return getattr(torch.ops.kashi, form.dummy).default


def functional_original(dummy: str) -> tuple[str, FunctionalForm] | None:
    """The original operator's qualified name and form for the dummy named
    ``dummy`` (without namespace), or ``None``."""
    for original, form in _FUNCTIONAL_OPS.items():
        if form.dummy == dummy:
            return original, form
    return None


def _dummy_schema(op: torch._ops.OpOverload, form: FunctionalForm) -> str:
    text = str(op._schema)
    params = text[text.index("(") + 1 : text.rindex(") ->")]
    kept = []
    for param in _split_top_level(params):
        if param != "*" and _param_name(param) == form.output:
            continue
        param = re.sub(r"^Tensor\([^)]*\)", "Tensor", param)
        param = re.sub(r"^PyObject\b", _OPAQUE_NAME, param)
        kept.append(param)
    if kept and kept[-1] == "*":
        kept.pop()
    return f"{form.dummy}({', '.join(kept)}) -> Tensor"


def _split_top_level(params: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(params):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(params[start:i].strip())
            start = i + 1
    if params.strip():
        parts.append(params[start:].strip())
    return parts


def _param_name(param: str) -> str:
    return param.split("=", 1)[0].split()[-1]


def _fake_impl(form: FunctionalForm):
    def fake(*args, **kwargs):
        schema = getattr(torch.ops.kashi, form.dummy).default._schema
        names = [a.name for a in schema.arguments if not a.kwarg_only]
        bound = dict(zip(names, args)) | kwargs
        return torch.empty_like(bound[form.like], memory_format=torch.contiguous_format)

    return fake
