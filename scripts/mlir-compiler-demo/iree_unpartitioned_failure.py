"""Demonstrate why vLLM custom ops must be split out before IREE compilation.

Run with ``python iree_unpartitioned_failure.py``. The only external
requirement is ``iree-compile`` (or set ``IREE_COMPILE``); no vLLM model,
checkpoint, Torch Python package, or GPU runtime is needed. The embedded module
keeps one ``torch.operator`` in the same function as ordinary ATen operations.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CUSTOM_OP_NAME = "torch.vllm.unified_attention_with_output"

# The ordinary operations around the opaque vLLM op make this an unpartitioned
# graph: a real partitioner would emit the ATen regions as IREE functions and
# leave the custom op outside those functions.
UNPARTITIONED_MLIR = f'''module {{
  func.func @main(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {{
    %a = torch.aten.relu %arg0 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    torch.operator "{CUSTOM_OP_NAME}"(%a) : (!torch.vtensor<[4],f32>) -> ()
    %b = torch.aten.neg %a : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    return %b : !torch.vtensor<[4],f32>
  }}
}}'''


def demonstrate_unpartitioned_failure(compiler: str, cuda_target: str) -> int:
    """Compile the embedded unpartitioned module and recognize IREE's rejection.

    Args:
        compiler: Path to ``iree-compile``.
        cuda_target: IREE CUDA target such as ``sm_80``. Selecting a target does
            not launch a GPU kernel; this command only compiles to an IR phase.

    Returns:
        Zero when IREE rejects the module specifically because of
        ``torch.operator``; nonzero for unrelated failures or unexpected success.
    """
    with tempfile.TemporaryDirectory(prefix="iree-unpartitioned-") as temp_dir:
        mlir_path = Path(temp_dir) / "unpartitioned.mlir"
        mlir_path.write_text(UNPARTITIONED_MLIR)
        command = [
            compiler,
            str(mlir_path),
            "--iree-hal-target-device=cuda",
            f"--iree-cuda-target={cuda_target}",
            "--compile-to=executable-targets",
            "-o",
            os.devnull,
        ]
        print(f"Running: {shlex.join(command)}")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    expected_error = "failed to legalize operation 'torch.operator'"
    if result.returncode == 0:
        print("Unexpected result: IREE compiled the unpartitioned module.")
        return 1
    if expected_error not in result.stdout or CUSTOM_OP_NAME not in result.stdout:
        print(
            "IREE failed, but its output did not identify the unpartitioned "
            "vLLM torch.operator; this does not demonstrate the expected cause."
        )
        return 1

    print(
        "Observed expected failure: IREE cannot legalize the vLLM custom op "
        "while it remains in the unpartitioned module."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse compiler options and run the standalone failure demonstration.

    ``IREE_COMPILE`` and ``IREE_CUDA_TARGET`` can provide defaults; the target
    defaults to ``sm_80`` so running the demo does not need a visible GPU.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler",
        default=os.environ.get("IREE_COMPILE") or shutil.which("iree-compile"),
        help="iree-compile executable (default: IREE_COMPILE or PATH)",
    )
    parser.add_argument(
        "--cuda-target",
        default=os.environ.get("IREE_CUDA_TARGET", "sm_80"),
        help="IREE CUDA target used for compilation (default: IREE_CUDA_TARGET or sm_80)",
    )
    args = parser.parse_args(argv)

    if not args.compiler:
        print("iree-compile not found; pass --compiler, set IREE_COMPILE, or add it to PATH.")
        return 2
    return demonstrate_unpartitioned_failure(args.compiler, args.cuda_target)


if __name__ == "__main__":
    sys.exit(main())
