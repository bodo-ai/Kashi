### Running scripts:

``` shell
# Build vLLM from source
git clone https://github.com/bodo-ai/vllm.git
uv venv
source .venv/bin/activate
cd vllm

# build from source (CUDA enabled environment):
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto

# build from source (CPU)
uv pip install -r requirements/cpu.txt
uv pip install -e .
cd ..

# install torch-mlir + tabulate
uv pip install -r requirements.txt
# Convert simple linear model through FX->MLIR
python simple_linear.py
# Convert partial LLM decode block through FX->MLIR
python simple_llm.py
# Convert full Qwen3 0.6B through FX->MLIR without checkpoint weights.
# Configuration and tokenizer metadata may still be resolved.
python simple_llm_qwen3.py
# Compare normalized-only and FX->MLIR->FX Qwen3 outputs (dummy weights).
python compare_simple_llm_qwen3.py
```

### IREE phase dump and FX reassembly (optional)

Set `IREE_PHASE_DUMP_DIR` to partition a copy of the generated Torch-MLIR
module at the vLLM custom ops (which IREE cannot ingest) and compile the
ordinary-op segments with `iree-compile`, dumping the IR after each phase and
the generated GPU kernels. The segments' kernel calls are then converted back
and spliced together with the custom-op calls into **one FX graph**, which is
printed in tabular and graph form (after the "Reassembled FX" header) so it can
be compared with the original graph printed at the top. The backend still
returns the plain round-trip graph; everything here is diagnostic, and any
failure is printed and swallowed.

``` shell
IREE_COMPILE=/path/to/iree-compile \
IREE_PHASE_DUMP_DIR=/tmp/qwen3_phase_dump \
python simple_llm_qwen3.py
```

- `IREE_COMPILE`: compiler path (default: `iree-compile` on `PATH`).
- `IREE_CUDA_TARGET`: e.g. `sm_120` (default: from the current CUDA device).

Each run writes a numbered directory (`0/`, `1/`, ...) containing:

| File | Contents |
|---|---|
| `partitioned.mlir` | The ordinary-op segments (`segment_<N>`) that IREE compiles. |
| `driver.mlir` | The source `@main` with each segment collapsed to a call and the custom ops kept in place: the wiring needed to reassemble. |
| `kernels.ptx` | Every kernel IREE generated (PTX), one banner per executable. Raw per-kernel dumps (PTX, LLVM IR, MLIR) are in `kernel_files/`. |
| `reassembled_fx.txt` | The reassembled FX graph (tabular + graph form), as printed. |
| `phases/partitioned.<N>.<phase>.mlir` | IR after each IREE compilation phase. |
| `boundaries.txt`, `segments.txt`, `partition_stats.json` | Custom-op boundaries (with mutated operand indices), per-segment shape, and fragmentation / executable / dispatch statistics. |
| `iree-compile.log` | Compiler output. |

`iree_phase_dump/FILES.md` documents every file in a run directory in detail, as a "which question → which file" lookup (kernel sizes, fusion decisions, phase-by-phase diffs, boundary types, etc.).

Kernel calls appear in the reassembled graph as opaque `iree_kernel.<export>`
targets (not executable yet). `iree_to_fx_mapping.md` lists every IREE
dialect/operation that needed conversion back to FX, with counts for Qwen3.

Tests: `python -m unittest discover -s . -p 'test_*.py'` (set `IREE_COMPILE`
to also run the tests that use the real IREE compiler).