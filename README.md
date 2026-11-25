# Kashi

An experimental ML compiler that captures PyTorch FX graphs, translates them to MLIR, and reconstructs executable FX graphs. The project includes a PyTorch compilation backend and integration with vLLM.

## Development setup

Install [Pixi 0.82.0 or newer](https://pixi.sh/latest/installation/). The supported development platforms are Linux x86-64 and macOS on Apple silicon. On macOS, install Xcode Command Line Tools (`xcode-select --install`), including the Apple SDK.

```sh
pixi install --locked
pixi run python -c 'import kashi._native'
```

Pixi installs the project in editable mode. Python source changes are picked up automatically. After changing C++ or CMake files, rebuild and reinstall with:

```sh
pixi run python -m pip install --no-deps -e .
```

The rebuild uses an isolated Python build environment.

Build a wheel with `pixi run python -m build --wheel`. Standard Python build tools also work without Pixi when the required compiler and build tools are available.

## vLLM and MLIR development

The default Pixi environment selects a locked platform based on the host. GPU variants take priority over the CPU fallback:

| Platform | Requirements | Python / PyTorch |
| --- | --- | --- |
| `linux-64-cuda` | NVIDIA driver supporting CUDA 13.0, compute capability 7.5+, glibc 2.39+ | 3.12 / 2.14.0 CUDA 13.0 |
| `linux-64-rocm` | AMD GPU, compatible ROCm driver, glibc 2.39+ | 3.12 / 2.13.0 ROCm 10.0 |
| `linux-64-cpu` | glibc 2.39+ | 3.12 / 2.14.0 CPU |
| `osx-arm64-cpu` | macOS 14+ on Apple silicon | 3.12 / 2.14.0 |

All variants use vLLM 0.33.0 and IREE compiler/runtime 3.14.0. Python 3.12 and the shared Linux glibc requirement avoid a [Pixi lock validation bug](https://github.com/prefix-dev/pixi/issues/7040) with differing rich-platform wheel tags. ROCm uses PyTorch 2.13 to match its upstream vLLM wheel. Pixi detects AMD GPU presence but does not check ROCm driver compatibility or guarantee that the wheel supports the detected GPU; consult the [vLLM GPU requirements](https://docs.vllm.ai/en/v0.33.0/getting_started/installation/gpu/). Upstream macOS CPU support in vLLM is experimental.

Use `pixi info` to inspect the selected platform. To select a specific compatible variant, pass `--platform`, for example `pixi run --locked --platform linux-64-cpu test-python`.

## Formatting and linting

Run all check-only tasks with `pixi run --locked lint`. The Linux CI job
runs the same command on pushes and pull requests.

| Language | Rewrite formatting | Check formatting | Lint |
| --- | --- | --- | --- |
| Python (`python/`, `scripts/`) | `pixi run --locked format-python` | `pixi run --locked check-format-python` | `pixi run --locked lint-python` |
| C++ (`cpp/`) | `pixi run --locked format-cpp` | `pixi run --locked check-format-cpp` | `pixi run --locked lint-cpp` |

Ruff handles Python formatting and linting. C++ uses clang-format with
LLVM style and clang-tidy on the project targets in `build/lint-cpp/`.
Clang-tidy checks headers included by those targets and treats reported
warnings as errors.

## Qwen3 with vLLM

Use the FX-to-MLIR-to-FX backend with vLLM's Qwen3 models. This example uses
dummy weights.

```sh
VLLM_USE_AOT_COMPILE=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 pixi run python - <<'PY'
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    load_format="dummy",
    seed=42,
    max_model_len=128,
    kv_cache_memory_bytes=256 * 1024 * 1024,
    compilation_config=CompilationConfig(
        mode=CompilationMode.DYNAMO_TRACE_ONCE,
        backend="kashi.backend.compile_fx",
    ),
)
print(llm.generate(["Hello, my name is"], SamplingParams(max_tokens=4, temperature=0))[0].outputs[0].text)
PY
```

The backend runs the rebuilt FX graph in PyTorch.
The importer supports dtype-only `Tensor.to(dtype)` conversions, including the
casts captured in CPU Qwen3 graphs; device and tensor overloads are not supported.
CPU RMS normalization and gated-activation operations are also normalized to ATen.

## Experimental IREE execution

`kashi.iree_backend.compile_iree_backend` is a `torch.compile` backend that
partitions an FX graph, compiles ordinary segments with IREE, and executes the
reconstructed graph on CPU, NVIDIA CUDA, or AMD HIP. Custom operations and
alias-sensitive PyTorch operations remain in the driver graph. The existing
`kashi.backend.compile_fx` continues to provide the FX-to-MLIR-to-FX round trip.

Set `KASHI_IREE_EXECUTION` before compilation:

- `direct`: launch generated GPU kernels on PyTorch's current stream. This is
  the default for CUDA and HIP. CPU does not support direct calls.
- `iree`: execute compiled segments through the IREE VM runtime. This is the
  default for CPU and an alternative for either GPU family. GPU runtime calls
  synchronize with PyTorch and return PyTorch-owned device buffers. For vLLM,
  use `cudagraph_mode=CUDAGraphMode.NONE`.

The former `cuda` mode name is no longer accepted; change it to `direct`.
Unknown modes, unsupported operations, incompatible devices, and compiler or
runtime failures raise errors. No mode falls back to another execution path.

```python
import torch
from kashi.iree_backend import compile_iree_backend

model = torch.compile(lambda x: torch.ops.aten.relu.default(x), backend=compile_iree_backend)
result = model(torch.randn(8, device="cpu"))
```

For vLLM, use `backend="kashi.iree_backend.compile_iree_backend"` with
`CompilationMode.DYNAMO_TRACE_ONCE`. The returned callable retains its temporary
compiler artifacts. To retain artifacts explicitly, call
`compile_iree_fx(gm, output_dir=...)`: the result contains the target family,
execution mode, VMFB, flow IR and kernel artifact paths. CPU uses host-specific
LLVM-CPU objects and embedded-ELF libraries, not PTX; CUDA emits PTX; HIP emits
HSACO. The artifact directory also contains partition/driver IR, compiler logs,
and IREE phase dumps. Host-specialized CPU artifacts are not portable across
arbitrary processors.

The graph's tensor device selects the target, independent of other GPUs in the
machine. PyTorch represents both GPU families as `cuda` devices; its build
distinguishes NVIDIA from ROCm. `target_device="cpu"|"cuda"|"hip"` explicitly
selects a family for cross-compilation. `cuda_target="sm_80"` or
`IREE_CUDA_TARGET=sm_80` sets a CUDA architecture; `rocm_target="gfx942"`
selects an AMD architecture without a GPU. ROCm target features reported by
PyTorch, such as `gfx942:sramecc+:xnack-`, are passed to IREE separately.
Explicit architecture arguments conflicting with the selected family are errors.
Executing cross-compiled GPU artifacts still requires a matching GPU and driver.

### IREE tests

The ordinary Python suite always runs a seeded, offline one-layer Qwen3 case
on CPU through the local IREE runtime, including on GPU-equipped machines. On
a CUDA or ROCm PyTorch/vLLM wheel, install the separate locked CPU environment
first; the test runs it in a subprocess with `VLLM_TARGET_DEVICE=cpu`:

```sh
pixi install --locked -e cpu-test
pixi run --locked test-python
```

GPU tests require the matching NVIDIA/CUDA or AMD/ROCm PyTorch build, device,
and driver. The `cuda-test` and `rocm-test` environments support hosts with
both vendors; install each detected vendor's environment before running the
suite. vLLM test processes explicitly select the appropriate platform on a
mixed-vendor host. Missing software for a detected GPU is a test failure.

On WSL, GPU detection queries Windows through PowerShell because GPU PCI devices
are not exposed in Linux sysfs. Windows interop must be available to the test
process. AMD GPU execution additionally requires a compatible Windows driver
and ROCm WSL bridge; check AMD's [WSL compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html).
The pinned IREE 3.14 compiler rejects `gfx1030` (for example, an RX 6800 XT) as
an unknown HIP target, even if PyTorch detects the GPU. It is not substituted
with a different architecture. Linux CPU and ROCm Pixi environments provide
their own NUMA/OpenMP libraries for minimal installations.

```sh
pixi install --locked -e cuda-test  # NVIDIA, when needed
pixi install --locked -e rocm-test  # AMD, when needed
pixi run --locked -e cuda-test test-python -v python/tests/test_iree_qwen3_integration.py
pixi run --locked -e rocm-test test-python -v python/tests/test_iree_qwen3_integration.py
```

The Qwen parity tests exercise real compiled prefill and decode, compare token
IDs exactly, and compare selected-token log-probabilities with `rtol=1e-4` and
`atol=1e-5`. GPU tests run both `direct` and `iree` modes. Offline HIP
compilation and HSACO metadata tests can run without an AMD GPU, but do not
verify AMD execution. By default, the CPU tests use Triton's interpreter for
vLLM's greedy sampler; the compiled IREE CPU kernels still execute native code.

The full-size Qwen3-0.6B architecture is a separate, resource-intensive test;
it uses deterministic dummy weights, not a pretrained checkpoint or tokenizer:

```sh
VLLM_TARGET_DEVICE=cpu TRITON_INTERPRET=1 pixi run --locked -e cpu-test \
  python -m pytest -q python/tests/qwen3_0_6b.py
# Alternatively use -e cuda-test or -e rocm-test with a matching GPU, and set
# KASHI_IREE_EXECUTION=direct or iree before running the same pytest command.
```

To test individual GPU operations, stream behavior, and mutation semantics:

```sh
pixi run --locked test-python -v python/tests/test_iree_execution.py
```

## Tests

The Pixi environment supplies PyTorch, IREE, vLLM, pytest, Catch2 v3, CMake, Ninja, nanobind, Python, and a C++ compiler. A GPU is not required on CPU platforms.

Run the basic suites from the repository root:

```sh
pixi install --locked
pixi run test-python
pixi run test-cpp
```

Python tests import the editable installation. Select a single test with
`pixi run test-python python/tests/test_import.py::test_native_extension_imports`.
Reinstall the editable package after C++ changes as shown above; the native
test command configures and builds its own tree on each run. Until genuine
C++ tests are added, it reports `no native tests yet` rather than claiming
that an empty suite passed.

Linux CPU CI sets `TRITON_INTERPRET=1` for the Python coverage run because
vLLM's Triton CPU sampler can fail with an illegal instruction on some CI
processors. The vLLM Qwen3 parity test still runs, but this job does not test
Triton's native CPU JIT. In a Linux CPU Pixi environment, reproduce that test:

```sh
TRITON_INTERPRET=1 pixi run --locked test-python python/tests/test_qwen3_integration.py
```

For native test selection, first run `pixi run test-cpp` to configure and
build, then use `pixi run ctest --test-dir build/test-cpp -C Debug -R '<test-name>'
--output-on-failure --no-tests=error`.

Put native units in `cpp/lib/`, public headers in `cpp/include/`, and Catch2
test sources in `cpp/tests/` (`.cpp`, `.cc`, or `.cxx`). CMake links the units
into `kashi_core`, links tests against that library and Catch2, and
registers each Catch2 case with CTest. If test sources exist but no cases are
discovered, the command fails.

## Coverage

The Linux GitHub Actions job collects coverage during its normal test run.
Run the same commands locally:

```sh
pixi run --locked coverage-python
pixi run --locked -e cpp-coverage coverage-cpp
```

Python coverage measures line and branch coverage of `kashi`. The C++ task
uses the locked Linux x86-64 Clang/LLVM environment and an instrumented
`build/coverage-cpp/` tree. It measures `cpp/lib/` and `cpp/include/` through
Catch2/CTest, not the native extension loaded by Python or code in dependencies.
It reports `no native tests yet` until there are native tests to run.

Reports are written to `build/coverage/`: `python.xml` and `python.md` for
Python, `cpp.txt` and (when coverage is available) `cpp.json` and `cpp.lcov` for C++. GitHub
Actions posts a summary and uploads the machine-readable reports and native
summary as `coverage-reports`. On pull requests, more than 90% of added or
modified executable lines must be covered across `python/src/kashi/`,
`cpp/lib/`, and `cpp/include/`. Linux CPU-only CI explicitly excludes
`python/src/kashi/iree_cuda.py` and `python/src/kashi/iree_runtime.py` from this
gate; both files remain visible in the full coverage report and require CUDA
tests for execution coverage. The summary names any excluded changed files.
All other changed source files must have coverage data. Lines are compared
against the PR base; test-only changes do not affect the percentage. Push runs
report coverage without enforcing it.

For a local GPU-inclusive changed-line check, run the coverage tasks above,
then run `python scripts/check_diff_coverage.py <base-commit>` without
`--exclude-gpu`.


## Native mutation testing

Every pull request runs Mull against added or modified C++ files under
`cpp/lib/` and `cpp/include/`, using the full contents of each selected file.
The PR check fails if any mutant survives across all tests. Manually dispatched
runs cover all project C++ code and report survivors without failing on them.
Baseline and infrastructure errors fail both modes.

Reports are in `build/mutation/reports/`.

Mutation testing requires Clang 19 and Mull 0.36.0 with its LLVM 19 frontend
plugin; see `.github/workflows/mutation.yml` for the pinned Linux installation.
Use the manually dispatched Linux workflow when the toolchain is unavailable locally.

## Runpod MI300X benchmark

`scripts/runpod_benchmark.py` runs the workload described in
[`specs/runpod-mi300x-vllm-benchmark.md`](specs/runpod-mi300x-vllm-benchmark.md).
It uses Python 3.10 or newer and does not require a local GPU or Docker.
It creates billable Pods; neither the deadline nor the cost estimate is a
hard spending cap. Consider a dedicated prepaid account with auto-pay disabled
and limited credits. The optional `HF_TOKEN` is sent in the Pod environment,
which is visible to users with account or API access.

```sh
export RUNPOD_API_KEY=...
python scripts/runpod_benchmark.py run --output-dir ./results
python scripts/runpod_benchmark.py run --profile --output-dir ./results
```


To wait for availability locally, add `--wait-for-capacity MINUTES` to
`run`. The wait is bounded and occurs before Pod creation;
the benchmark's own deadline starts afterward. The catalog does not reserve
capacity, so placement can still fail. No Runpod queued deployment is created.

The command defaults to MI300X, then MI350X when the first
is unavailable. Override the order with, for example:

```sh
python scripts/runpod_benchmark.py run --gpu 'AMD Instinct MI350 OAM' 'AMD Instinct MI300X OAM'
```

Selection uses a catalog snapshot and does not reserve a GPU. A rejected or
ambiguous create is not retried on another GPU. Prices may differ by GPU.

On interruption or host loss, run `python scripts/runpod_benchmark.py recover`
before another paid run. The controller also reconciles its durable local
journal automatically at startup. If cleanup cannot be confirmed, inspect
and terminate the exact Pod in the [Runpod console](https://www.runpod.io/console/pods).
After independently verifying termination, clear an unresolved entry with
`python scripts/runpod_benchmark.py attest RUN_ID --name EXACT_POD_NAME`.
The watchdog cannot stop a Pod whose container never starts or whose API is
unreachable; such a Pod may continue billing until manually terminated.

The output directory contains a `.tar.gz` artifact and a JSON cleanup sidecar.
Inspect the manifest and benchmark JSON with `tar -xzf FILE.tar.gz` and a JSON
viewer. Open compressed profiling traces directly in [Perfetto](https://ui.perfetto.dev/).
Files marked `partial` are diagnostics, not successful benchmark results.

## Layout

- `python/src/kashi/`: Python package and future FX integration.
- `python/tests/`: Python API and import tests.
- `cpp/bindings/`: Python native module.
- `cpp/include/`, `cpp/lib/`, `cpp/tests/`: C++ headers, native units and tests.
- `specs/`: project specifications.
