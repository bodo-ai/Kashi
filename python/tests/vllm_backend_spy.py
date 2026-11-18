"""vLLM-loadable backend (``"vllm_backend_spy.spy_compile_fx"``) that records
each graph it receives, the callable ``compile_fx`` returns, and the token
count of every invocation of that callable."""

from kashi.backend import compile_fx

CALLS = []
TOKEN_COUNTS = []


def spy_compile_fx(gm, example_inputs):
    fn = compile_fx(gm, example_inputs)
    CALLS.append((gm, fn))

    def run(*args):
        # The captured graph's only int input is the symbolic token count.
        (count,) = [a for a in args if type(a) is int]
        TOKEN_COUNTS.append(count)
        return fn(*args)

    return run
