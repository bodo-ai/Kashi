"""Opt-in decomposition of vLLM custom operators into ATen operators.

vLLM's ``_C`` operators write their results into argument buffers instead of
returning them. Each decomposition emits ATen nodes computing the new values,
then redirects reads of a mutated argument that follow the call to its new
value. Writes that would be observable through another alias are rejected.

Every emitted node records which call it came from under
``meta[DECOMPOSED_FROM_KEY]`` (for example ``"_C.rms_norm#2"``); the importer
turns that into an MLIR ``NameLoc`` so later IREE passes can still group them.
"""

import math
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils import _pytree as pytree

DECOMPOSED_FROM_KEY = "kashi_decomposed_from"

aten = torch.ops.aten
_VIEW_OPS = (aten.view.default, aten.reshape.default, aten._unsafe_view.default)


class _Emitter:
    """Insert tagged ATen nodes, with FakeTensor metadata, before a custom op."""

    def __init__(self, graph: torch.fx.Graph, before: torch.fx.Node, tag: str):
        self.graph = graph
        self.before = before
        self.tag = tag

    def __call__(self, target: torch._ops.OpOverload, *args) -> torch.fx.Node:
        with self.graph.inserting_before(self.before):
            node = self.graph.call_function(target, args)
        values = pytree.tree_map_only(torch.fx.Node, lambda n: n.meta["val"], args)
        # Use the mode that made the graph's values; detect_fake_mode would
        # prefer a Dynamo backend's TracingContext mode, which differs.
        modes = [
            v.fake_mode for v in pytree.tree_leaves(values) if isinstance(v, FakeTensor)
        ]
        with modes[0] if modes else nullcontext():
            node.meta["val"] = target(*values)
        node.meta[DECOMPOSED_FROM_KEY] = self.tag
        return node


_Rule = Callable[[_Emitter, dict], dict[torch.fx.Node, torch.fx.Node]]


def _rms_norm(emit: _Emitter, a: dict) -> dict:
    x = a["input"]
    normalized = [x.meta["val"].shape[-1]]
    return {
        a["result"]: emit(
            aten.rms_norm.default, x, normalized, a["weight"], a["epsilon"]
        )
    }


def _fused_add_rms_norm(emit: _Emitter, a: dict) -> dict:
    x, residual = a["input"], a["residual"]
    summed = emit(aten.add.Tensor, x, residual)
    normalized = [x.meta["val"].shape[-1]]
    return {
        x: emit(aten.rms_norm.default, summed, normalized, a["weight"], a["epsilon"]),
        residual: summed,
    }


def _silu_and_mul(emit: _Emitter, a: dict) -> dict:
    x = a["input"]
    half = x.meta["val"].shape[-1] // 2
    gate = emit(aten.slice.Tensor, x, -1, 0, half)
    up = emit(aten.slice.Tensor, x, -1, half, 2 * half)
    return {a["result"]: emit(aten.mul.Tensor, emit(aten.silu.default, gate), up)}


def _rotary_embedding(emit: _Emitter, a: dict) -> dict:
    """NeoX-style rotation over the full head, the only variant Qwen3 uses."""
    head_size, cache = a["head_size"], a["cos_sin_cache"]
    problems = [
        message
        for failed, message in (
            (not a["is_neox"], "is_neox=False"),
            (a["rope_dim_offset"] != 0, f"rope_dim_offset={a['rope_dim_offset']}"),
            (a["inverse"], "inverse=True"),
            (a["key"] is None, "key=None"),
            (
                cache.meta["val"].shape[-1] != head_size,
                "partial rotary (cos_sin_cache width != head_size)",
            ),
        )
        if failed
    ]
    if problems:
        raise NotImplementedError(
            f"unsupported _C.rotary_embedding: {', '.join(problems)}"
        )

    half = head_size // 2
    cos_sin = emit(aten.index.Tensor, cache, [a["positions"]])
    cos = emit(
        aten.unsqueeze.default, emit(aten.slice.Tensor, cos_sin, -1, 0, half), -2
    )
    sin = emit(
        aten.unsqueeze.default,
        emit(aten.slice.Tensor, cos_sin, -1, half, head_size),
        -2,
    )

    def rotate(x):
        # The token dimension stays -1 so a symbolic token count is never baked in.
        shape = list(x.meta["val"].shape[1:])
        heads = math.prod(shape) // head_size
        heads_view = emit(aten.view.default, x, [-1, heads, head_size])
        x1 = emit(aten.slice.Tensor, heads_view, -1, 0, half)
        x2 = emit(aten.slice.Tensor, heads_view, -1, half, head_size)
        first = emit(
            aten.sub.Tensor,
            emit(aten.mul.Tensor, x1, cos),
            emit(aten.mul.Tensor, x2, sin),
        )
        second = emit(
            aten.add.Tensor,
            emit(aten.mul.Tensor, x2, cos),
            emit(aten.mul.Tensor, x1, sin),
        )
        rotated = emit(aten.cat.default, [first, second], -1)
        return emit(aten.view.default, rotated, [-1, *shape])

    return {a["query"]: rotate(a["query"]), a["key"]: rotate(a["key"])}


_DECOMPOSITIONS: dict[str, _Rule] = {
    "_C::rms_norm": _rms_norm,
    "_C::fused_add_rms_norm": _fused_add_rms_norm,
    "_C::silu_and_mul": _silu_and_mul,
    "_C::rotary_embedding": _rotary_embedding,
}


def _bind(node: torch.fx.Node) -> dict:
    """Bind a call's arguments to schema names, filling in defaults."""
    bound = {}
    for i, argument in enumerate(node.target._schema.arguments):
        if i < len(node.args):
            bound[argument.name] = node.args[i]
        elif argument.name in node.kwargs:
            bound[argument.name] = node.kwargs[argument.name]
        else:
            bound[argument.name] = argument.default_value
    return bound


def _check_aliases(mutated, op, tag, position, op_position):
    """Reject writes still observable through a graph input or another alias."""
    base = mutated
    while base.op == "call_function" and base.target in _VIEW_OPS:
        base = base.args[0]
    if base.op == "placeholder":
        raise NotImplementedError(
            f"{tag} mutates a graph input {base.name}: {op.format_node()}"
        )
    aliases, pending = {base}, [base]
    while pending:
        for user in pending.pop().users:
            if user.target in _VIEW_OPS and user not in aliases:
                aliases.add(user)
                pending.append(user)
    for alias in aliases:
        for user in alias.users:
            if user not in aliases and position.get(user, op_position) > op_position:
                raise NotImplementedError(
                    f"{tag} writes {mutated.name}, but {user.name} reads an alias "
                    f"of it ({alias.name}) after the call: {op.format_node()}"
                )


def decompose_custom_ops(graph: torch.fx.Graph) -> None:
    """Rewrite supported vLLM custom ops in a normalized graph, in place.

    The graph must already contain ATen ``OpOverload`` targets with
    ``meta["val"]`` on every value node, as produced by the importer's
    normalization.
    """
    position = {node: i for i, node in enumerate(graph.nodes)}
    counts = Counter()
    for op in list(graph.nodes):
        if op.op != "call_function" or not isinstance(op.target, torch._ops.OpOverload):
            continue
        name = op.target.name()
        rule = _DECOMPOSITIONS.get(name)
        if rule is None:
            continue
        tag = f"{name.replace('::', '.')}#{counts[name]}"
        counts[name] += 1
        op_position = position[op]
        replacements = rule(_Emitter(graph, op, tag), _bind(op))
        for mutated, value in replacements.items():
            old, new = mutated.meta["val"], value.meta["val"]
            if old.shape != new.shape or old.dtype != new.dtype:
                raise NotImplementedError(
                    f"{tag} result {tuple(new.shape)}/{new.dtype} does not match "
                    f"{mutated.name} {tuple(old.shape)}/{old.dtype}"
                )
            mutated.replace_all_uses_with(
                value,
                delete_user_cb=lambda user, at=op_position: position.get(user, at) > at,
            )
            _check_aliases(mutated, op, tag, position, op_position)
        graph.erase_node(op)
