"""vLLM-loadable backend (``"vllm_backend_spy.spy_compile_fx"``) that records
each graph it receives, the callable ``compile_fx`` returns, and the token
count of every invocation of that callable."""

from kashi.backend import compile_fx

CALLS = []
TOKEN_COUNTS = []


def spy_compile_fx(gm, example_inputs):
    fn = compile_fx(gm, example_inputs)
    # Models can have multiple symbolic integer inputs (tokens and positions).
    # Count the actual input IDs instead of guessing which integer is the batch.
    placeholders = list(gm.graph.find_nodes(op="placeholder"))
    (input_index,) = [
        i for i, node in enumerate(placeholders) if "input_ids" in node.name.lower()
    ]
    CALLS.append((gm, fn))

    def run(*args):
        TOKEN_COUNTS.append(args[input_index].shape[0])
        return fn(*args)

    return run
