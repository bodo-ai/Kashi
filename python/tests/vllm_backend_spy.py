"""vLLM-loadable backend (``"vllm_backend_spy.spy_compile_fx"``) that records
each graph it receives and the callable ``compile_fx`` returns."""

from kashi.backend import compile_fx

CALLS = []


def spy_compile_fx(gm, example_inputs):
    fn = compile_fx(gm, example_inputs)
    CALLS.append((gm, fn))
    return fn
