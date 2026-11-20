"""A ``torch.compile`` backend that wraps ``compile_fx`` and records what it does.

Each :class:`BackendSpy` records every graph it receives together with the
callable ``compile_fx`` returned. For graphs with an ``input_ids`` input (vLLM
models), it also records the token count of every invocation.

vLLM loads backends by qualified name, so a module-level instance is exposed
through :func:`spy_compile_fx`; pass :data:`VLLM_BACKEND` as the backend name.
"""

from kashi.backend import compile_fx


def _input_ids_index(gm):
    """Return the index of the ``input_ids`` placeholder, or ``None`` if absent."""
    placeholders = gm.graph.find_nodes(op="placeholder")
    matches = [i for i, n in enumerate(placeholders) if "input_ids" in n.name.lower()]
    assert len(matches) <= 1, f"ambiguous input_ids placeholders: {matches}"
    return matches[0] if matches else None


class BackendSpy:
    """Backend recording ``(gm, compiled_fn)`` pairs and per-call token counts."""

    def __init__(self):
        self.calls = []
        self.token_counts = []

    def clear(self):
        self.calls.clear()
        self.token_counts.clear()

    def __call__(self, gm, example_inputs):
        fn = compile_fx(gm, example_inputs)
        self.calls.append((gm, fn))
        input_index = _input_ids_index(gm)
        if input_index is None:
            return fn

        def run(*args):
            self.token_counts.append(args[input_index].shape[0])
            return fn(*args)

        return run


VLLM_SPY = BackendSpy()
VLLM_BACKEND = f"{__name__}.spy_compile_fx"


def spy_compile_fx(gm, example_inputs):
    """vLLM-loadable entry point that records into :data:`VLLM_SPY`."""
    return VLLM_SPY(gm, example_inputs)
