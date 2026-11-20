"""Compile a Qwen3 model's vLLM graph to CUDA PTX through the kashi IREE backend.

vLLM traces the model once with Dynamo and hands the whole FX graph to the
backend named in its ``CompilationConfig``. This script installs a backend that
runs :func:`kashi.iree_backend.compile_iree_fx` on that graph, writing the
partitioned/driver MLIR, IREE flow IR and per-kernel PTX under ``--output-dir``.

The reassembled IREE graph is not executable yet (its kernel calls are opaque),
so the backend returns the original ``gm.forward`` to vLLM. That keeps engine
start-up and the optional smoke-test generation working while still exercising
the full IREE compilation on the real traced graph.

By default a tiny one-layer Qwen3 config with dummy weights is used so nothing
is downloaded; pass ``--model Qwen/Qwen3-0.6B`` (or a local path) for a real one.
Requires a CUDA machine with vLLM and the IREE compiler installed; it will not
run on macOS.

Example:
    python scripts/compile_qwen3_iree.py --output-dir build/qwen3-iree
    python scripts/compile_qwen3_iree.py --model Qwen/Qwen3-0.6B --cuda-target sm_90
    python scripts/compile_qwen3_iree.py --decompose-custom-ops  # ATen rms_norm, RoPE, ...
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# vLLM resolves the backend by qualified name inside its engine. These must be
# set before vLLM is imported so the engine runs in this process (where the
# ``__main__`` backend and its recorded results live), returns a plain Python
# callable rather than an AOT artifact, and actually invokes the backend instead
# of reusing a cached compilation from an earlier run.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

ROOT = Path(__file__).resolve().parents[1]
# Allow running from a source checkout without installing the kashi wheel.
sys.path.insert(0, str(ROOT / "python" / "src"))

from kashi.iree_backend import IreeCompilation, compile_iree_fx

# Mirrors the offline Qwen3 config used by the integration tests: small enough
# to compile quickly while still containing attention, RMSNorm, RoPE and MLP.
TINY_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 128,
    "head_dim": 64,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "num_hidden_layers": 1,
    "intermediate_size": 256,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "torch_dtype": "float32",
}

# Populated by the backend; vLLM gives us no other handle on what it compiled.
_STATE: dict = {
    "output_dir": None,
    "cuda_target": None,
    "decompose_custom_ops": False,
    "results": [],
}


def iree_backend(gm, example_inputs):
    """vLLM backend: compile ``gm`` with IREE, then run the original graph eagerly.

    Errors from IREE propagate so a failed compilation aborts engine start-up
    instead of silently producing no artifacts.
    """
    print(f"[iree] compiling graph with {len(gm.graph.nodes)} FX nodes", flush=True)
    compiled = compile_iree_fx(
        gm,
        output_dir=_STATE["output_dir"],
        cuda_target=_STATE["cuda_target"],
        decompose_custom_ops=_STATE["decompose_custom_ops"],
    )
    _STATE["results"].append(compiled)
    print(f"[iree] artifacts written to {compiled.artifact_dir}", flush=True)
    # The IREE-reassembled graph cannot execute yet, so vLLM gets the original.
    return gm.forward


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        help="HF model id or local directory (default: tiny offline Qwen3 config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "qwen3-iree",
        help="Parent directory for the per-compilation artifact directory",
    )
    parser.add_argument(
        "--cuda-target",
        help="IREE CUDA arch such as sm_80 (default: IREE_CUDA_TARGET or current GPU)",
    )
    parser.add_argument(
        "--dtype",
        # "auto" follows the config's torch_dtype: float32 for the tiny test
        # config, bfloat16 for real checkpoints (float32 would double their size).
        default="auto",
        help="Model dtype passed to vLLM (default: the config's torch_dtype)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help=(
            "Override num_hidden_layers, e.g. 2 to compile a large model's real "
            "layer shapes without its full memory footprint (dummy weights only)"
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=64,
        help="Maximum sequence length; also bounds the traced token dimension",
    )
    parser.add_argument(
        "--real-weights",
        action="store_true",
        help="Load checkpoint weights instead of vLLM's dummy weights",
    )
    parser.add_argument(
        "--decompose-custom-ops",
        action="store_true",
        help=(
            "Rewrite vLLM's rms_norm, fused_add_rms_norm, rotary_embedding and "
            "silu_and_mul custom ops into ATen ops before IREE import; MLIR "
            "locations record which call each op came from"
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After compiling, generate a few tokens as an end-to-end smoke test",
    )
    return parser.parse_args()


def _summarize(compiled: IreeCompilation):
    """Print where the interesting artifacts are and how many kernels were made."""
    from kashi.iree_reassembly import IreeKernel

    kernels = [
        node
        for node in compiled.graph.graph.nodes
        if node.op == "call_function" and isinstance(node.target, IreeKernel)
    ]
    print(f"  artifact dir : {compiled.artifact_dir}")
    print(f"  flow IR      : {compiled.flow_ir}")
    print(f"  PTX files    : {len(compiled.ptx_files)}")
    if compiled.kernels_ptx is not None:
        print(f"  combined PTX : {compiled.kernels_ptx}")
    print(f"  kernel calls : {len(kernels)} in reassembled FX graph")
    graph_path = compiled.artifact_dir / "reassembled_graph.py"
    graph_path.write_text(compiled.graph.code)
    print(f"  FX code      : {graph_path}")


def main():
    args = _parse_args()
    if args.num_layers is not None and args.real_weights:
        # The checkpoint still holds weights for the removed layers, which the
        # model's weight loader would reject as unknown parameters.
        raise SystemExit("--num-layers cannot be combined with --real-weights")
    _STATE["output_dir"] = args.output_dir
    _STATE["cuda_target"] = args.cuda_target
    _STATE["decompose_custom_ops"] = args.decompose_custom_ops

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    with tempfile.TemporaryDirectory() as tiny_dir:
        model = args.model
        if model is None:
            # A config.json alone is enough with dummy weights and no tokenizer.
            (Path(tiny_dir) / "config.json").write_text(json.dumps(TINY_QWEN3_CONFIG))
            model = tiny_dir
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        elif not args.real_weights:
            print("[iree] using dummy weights; pass --real-weights to load them")

        llm = LLM(
            model=model,
            # skip_tokenizer_init=True,
            load_format="auto" if args.real_weights else "dummy",
            dtype=args.dtype,
            # Every decoder layer traces to the same ops with the same shapes,
            # so a truncated model still exercises the full model's kernels.
            hf_overrides=(
                {"num_hidden_layers": args.num_layers}
                if args.num_layers is not None
                else None
            ),
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len,
            max_num_seqs=2,
            enforce_eager=False,
            compilation_config=CompilationConfig(
                # Trace once and pass the whole graph (custom ops included) to
                # our backend; partitioning at custom ops is IREE's job here.
                mode=CompilationMode.DYNAMO_TRACE_ONCE,
                backend=f"{__name__}.iree_backend",
                # The backend returns eager code; CUDA graphs add nothing here.
                cudagraph_mode=CUDAGraphMode.NONE,
            ),
        )

        if args.generate:
            outputs = llm.generate(
                [{"prompt_token_ids": [1, 2, 3, 4, 5]}],
                SamplingParams(max_tokens=4, temperature=0, ignore_eos=True),
                use_tqdm=False,
            )
            print(
                f"[iree] generated token ids: {list(outputs[0].outputs[0].token_ids)}"
            )

    results = _STATE["results"]
    if not results:
        raise SystemExit("vLLM never invoked the IREE backend; nothing was compiled")
    print(f"\nCompiled {len(results)} graph(s) with IREE:")
    for compiled in results:
        _summarize(compiled)


if __name__ == "__main__":
    main()
