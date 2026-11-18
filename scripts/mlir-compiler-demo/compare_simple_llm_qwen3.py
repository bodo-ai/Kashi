import json
import math
import subprocess
import sys
from pathlib import Path


RESULT_MARKER = "QWEN3_PARITY_RESULT="


def compare_outputs(reference: dict, candidate: dict) -> None:
    """Require matching generated tokens and close selected-token log-probs.

    Comparing both catches backend discrepancies that token IDs alone can hide.
    """
    reference_ids = reference.get("token_ids")
    candidate_ids = candidate.get("token_ids")
    if not isinstance(reference_ids, list) or not isinstance(candidate_ids, list):
        raise AssertionError("Missing or invalid token IDs")
    if reference_ids != candidate_ids:
        raise AssertionError(
            f"Qwen3 token IDs differ: {reference_ids} != {candidate_ids}"
        )

    reference_logprobs = reference.get("selected_token_logprobs")
    candidate_logprobs = candidate.get("selected_token_logprobs")
    if not isinstance(reference_logprobs, list) or not isinstance(
        candidate_logprobs, list
    ):
        raise AssertionError("Missing or invalid selected-token log-probabilities")
    if len(reference_logprobs) != len(reference_ids) or len(candidate_logprobs) != len(
        candidate_ids
    ):
        raise AssertionError("Selected-token log-probabilities are misaligned with token IDs")

    # Validate log-probabilities as well as tokens: parity includes model scores,
    # and finite values avoid treating NaNs or infinities as successful matches.
    for index, (reference_value, candidate_value) in enumerate(
        zip(reference_logprobs, candidate_logprobs)
    ):
        try:
            reference_value = float(reference_value)
            candidate_value = float(candidate_value)
        except (TypeError, ValueError) as error:
            raise AssertionError(
                f"Invalid selected-token log-probability at index {index}"
            ) from error
        if not math.isfinite(reference_value) or not math.isfinite(candidate_value):
            raise AssertionError(
                f"Non-finite selected-token log-probability at index {index}"
            )
        if not math.isclose(
            reference_value,
            candidate_value,
            rel_tol=0.0,
            abs_tol=1e-3,
        ):
            raise AssertionError(
                "Selected-token log-probability differs at index "
                f"{index}: {reference_value} != {candidate_value}"
            )


def run_backend(backend: str) -> dict:
    """Run the worker for one compiler backend and decode its marked result."""
    worker = Path(__file__).with_name("simple_llm_qwen3.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--backend",
            backend,
            "--json-result",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result_lines = [
        line[len(RESULT_MARKER):]
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if len(result_lines) != 1:
        raise RuntimeError(
            f"Expected one parity result from {backend}, got {len(result_lines)}"
        )
    result = json.loads(result_lines[0])
    if not isinstance(result, dict):
        raise RuntimeError(f"Invalid parity result from {backend}")
    return result


def main():
    """Compare normalized-FX and MLIR-roundtrip generation results."""
    normalized = run_backend("normalized_fx_backend")
    round_trip = run_backend("mlir_roundtrip_backend")
    compare_outputs(normalized, round_trip)
    print(
        "Qwen3 output parity passed: "
        f"{normalized['token_ids']} with selected-token log-probabilities within 1e-3"
    )


if __name__ == "__main__":
    main()
