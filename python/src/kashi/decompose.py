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
import operator
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils import _pytree as pytree

DECOMPOSED_FROM_KEY = "kashi_decomposed_from"

aten = torch.ops.aten


def _is_view(node: torch.fx.Node) -> bool:
    """Whether ``node`` shares storage with its first argument.

    Covers every ATen op whose schema marks its result as a read-only alias of
    its input (view, slice, select, unsqueeze, transpose, expand, split, ...),
    ``getitem`` of such a multi-output view, and ``_unsafe_view``.
    """
    if node.op != "call_function" or not node.args:
        return False
    target = node.target
    if target is operator.getitem:
        return isinstance(node.args[0], torch.fx.Node) and _is_view(node.args[0])
    if target is aten._unsafe_view.default:
        return True
    if not isinstance(target, torch._ops.OpOverload):
        return False
    returns = target._schema.returns
    alias = returns[0].alias_info if returns else None
    return alias is not None and not alias.is_write


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


# Rules receive the call's arguments by position, with defaults filled in.
# Argument names are not stable: vLLM's CPU build calls the output buffer
# "out" where the CUDA build calls it "result".
_Rule = Callable[[_Emitter, list], dict[torch.fx.Node, torch.fx.Node]]


def _rms_norm(emit: _Emitter, args: list) -> dict:
    """_C.rms_norm(result, x, weight, eps) replaced with aten rms_norm."""
    result, x, weight, epsilon = args
    normalized = [x.meta["val"].shape[-1]]
    return {result: emit(aten.rms_norm.default, x, normalized, weight, epsilon)}


def _fused_add_rms_norm(emit: _Emitter, args: list) -> dict:
    """_C.fused_add_rms_norm(x, residual, weight, eps) replaced with
    residual = x + residual, then residual = rms_norm(residual, weight, eps).
    """
    x, residual, weight, epsilon = args
    summed = emit(aten.add.Tensor, x, residual)
    normalized = [x.meta["val"].shape[-1]]
    return {
        x: emit(aten.rms_norm.default, summed, normalized, weight, epsilon),
        residual: summed,
    }


def _silu_and_mul(emit: _Emitter, args: list) -> dict:
    """_C.silu_and_mul(result, x) replaced with silu(x[..., :d]) * x[..., d:]."""
    result, x = args
    half = x.meta["val"].shape[-1] // 2
    gate = emit(aten.slice.Tensor, x, -1, 0, half)
    up = emit(aten.slice.Tensor, x, -1, half, 2 * half)
    return {result: emit(aten.mul.Tensor, emit(aten.silu.default, gate), up)}


def _rotary_embedding(emit: _Emitter, args: list) -> dict:
    """_C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, ...)
    replaced with an explicit NeoX rotation of query and key, in place.

    With h = head_size // 2, each row of cos_sin_cache is [cos | sin] (h each):

        cos = cos_sin_cache[positions, :h]       # [tokens, 1, h]
        sin = cos_sin_cache[positions, h:]       # [tokens, 1, h]

    and each head of x (query or key, viewed as [tokens, heads, head_size]) is
    split into its first and second halves, x1 = x[..., :h], x2 = x[..., h:]:

        x = cat(x1 * cos - x2 * sin,  x2 * cos + x1 * sin)

    Only full-head NeoX rotation with a key (what Qwen3 uses) is supported.
    """
    positions, query, key, head_size, cache, is_neox, rope_dim_offset, inverse = args
    problems = [
        message
        for failed, message in (
            (not is_neox, "is_neox=False"),
            (rope_dim_offset != 0, f"rope_dim_offset={rope_dim_offset}"),
            (inverse, "inverse=True"),
            (key is None, "key=None"),
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

    # cos, sin = cos_sin_cache[positions] split into [:h] and [h:], with a heads
    # dimension added so they broadcast over every head.
    half = head_size // 2
    cos_sin = emit(aten.index.Tensor, cache, [positions])
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
        # x as [tokens, heads, head_size], then x1 = x[..., :h], x2 = x[..., h:].
        heads_view = emit(aten.view.default, x, [-1, heads, head_size])
        x1 = emit(aten.slice.Tensor, heads_view, -1, 0, half)
        x2 = emit(aten.slice.Tensor, heads_view, -1, half, head_size)
        # first = x1 * cos - x2 * sin, second = x2 * cos + x1 * sin.
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
        # x = cat(first, second), back in x's original shape.
        rotated = emit(aten.cat.default, [first, second], -1)
        return emit(aten.view.default, rotated, [-1, *shape])

    return {query: rotate(query), key: rotate(key)}


_DECOMPOSITIONS: dict[str, _Rule] = {
    "_C::rms_norm": _rms_norm,
    "_C::fused_add_rms_norm": _fused_add_rms_norm,
    "_C::silu_and_mul": _silu_and_mul,
    "_C::rotary_embedding": _rotary_embedding,
}


def _bind(node: torch.fx.Node) -> list:
    """Return a call's arguments in schema order, filling in defaults."""
    bound = []
    for i, argument in enumerate(node.target._schema.arguments):
        if i < len(node.args):
            bound.append(node.args[i])
        elif argument.name in node.kwargs:
            bound.append(node.kwargs[argument.name])
        else:
            bound.append(argument.default_value)
    return bound


def _check_aliases(mutated, op, tag, position, op_position):
    """Reject writes still observable through a graph input or another alias."""
    base = mutated
    while _is_view(base):
        base = base.args[0]
    if base.op == "placeholder":
        raise NotImplementedError(
            f"{tag} mutates a graph input {base.name}: {op.format_node()}"
        )
    aliases, pending = {base}, [base]
    while pending:
        for user in pending.pop().users:
            if _is_view(user) and user not in aliases:
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

    The graph must already contain ATen ``OpOverload`` targets with ``meta["val"]`` on
    every value node, as produced by the importer's normalization.
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
