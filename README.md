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

### Reuse a local IREE SDK (macOS arm64)

The default build compiles IREE from source. To reuse its static libraries across
build trees, create an SDK from a completed, uninstrumented Release Ninja build
with native tests enabled. For example, from a Pixi shell on Apple silicon:

```sh
cmake -S . -B build/sdk-producer -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_BUILD_TYPE=Release -DKASHI_BUILD_TESTS=ON -DKASHI_IREE_SDK_DIR= \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -Dnanobind_DIR="$(python -m nanobind --cmake_dir)"
cmake --build build/sdk-producer --target _native kashi_native_tests
python scripts/package_iree_sdk.py build/sdk-producer build/kashi-iree-sdk.tar.zst
shasum -a 256 build/kashi-iree-sdk.tar.zst
mkdir -p build/sdk-extract
tar --zstd -xf build/kashi-iree-sdk.tar.zst -C build/sdk-extract
```

In a fresh CMake build directory, pass
`-DKASHI_IREE_SDK_DIR="$PWD/build/sdk-extract/iree-sdk"` with the same toolchain
and deployment target. For an editable reinstall, pass that option through
`CMAKE_ARGS` so it overrides any cached CMake value:

```sh
CMAKE_ARGS="-DKASHI_IREE_SDK_DIR=$PWD/build/sdk-extract/iree-sdk" \
  pixi run python -m pip install --no-deps -e .
```

The SDK includes upstream libraries, not Kashi's plugin object. The consumer
compiles that object and resolves the builtin registration symbol expected by
IREE's plugin manager. Rebuild the SDK when the IREE pin, toolchain, build flags,
or builtin registration contract changes, not for ordinary Kashi pass edits.
Linux and CI publishing still use the source-build path.

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
Clang-tidy reports unused direct C++ includes, checks project headers included
by those targets, and treats reported warnings as errors.
Run `pixi install --locked` before using the hooks; they use the installed tools.
The `clang-tidy` pre-commit hook runs the same CMake lint task on C++ changes.
The first run builds the pinned IREE dependencies; later runs reuse that build.
Set `KASHI_LINT_BUILD_DIR` to reuse an existing compatible Ninja build.

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

## Experimental IREE/PTX graph

`kashi.iree_backend.compile_iree_fx` explicitly opts an FX graph into the IREE
pipeline. It partitions around custom operators, compiles ordinary segments to
CUDA PTX, and returns a single FX graph containing opaque kernel calls. The
default `compile_fx` backend remains executable and does not invoke IREE.

For example, inspect the IREE graph during a normal `torch.compile` run:

```python
import torch
from kashi.backend import compile_fx
from kashi.iree_backend import compile_iree_fx

def inspect_iree(gm, example_inputs):
    result = compile_iree_fx(gm, output_dir="/tmp/kashi-iree", cuda_target="sm_80")
    print(result.graph.graph)
    print(result.kernels_ptx)
    return compile_fx(gm, example_inputs)

torch.compile(lambda x: torch.ops.aten.relu.default(x), backend=inspect_iree)(torch.ones(8))
```

Each call creates a unique artifact directory containing the partitioned and
driver MLIR, IREE phase dumps, compiler log, raw PTX files and `kernels.ptx`.
If `cuda_target` is omitted, `IREE_CUDA_TARGET` or the current CUDA device is
used. Compiler and unsupported-operation errors propagate; there is no silent
fallback. The returned graph is for inspection and further transformations:
its opaque kernel calls cannot execute until kernel launching is implemented.

The existing round-trip and vLLM tests continue to validate executable PyTorch
semantics. The IREE tests separately compile real kernels and check the returned
graph's custom-op boundaries, kernel operands and output wiring.
The CUDA/PTX compilation test is skipped on macOS; platform-independent IREE
partitioning and reconstruction tests still run there.

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
The native coverage build uses Release to avoid generating debug symbols for
embedded LLVM. CI removes the editable build tree after Python coverage to
leave room for the separate native build.

Reports are written to `build/coverage/`: `python.xml` and `python.md` for
Python, `cpp.txt` and (when coverage is available) `cpp.json` and `cpp.lcov` for C++. GitHub
Actions posts a summary and uploads the machine-readable reports and native
summary as `coverage-reports`. On pull requests, more than 90% of added or
modified executable lines must be covered across `python/src/kashi/`,
`cpp/lib/`, and `cpp/include/`. Lines are compared against the PR base; a
changed source file with no coverage data fails the check. Test-only changes
do not affect the percentage. Push runs report coverage without enforcing it.


## Native mutation testing

Every pull request runs Mull against added or modified C++ files under
`cpp/lib/` and `cpp/include/`, using the full contents of each selected file.
The PR check fails if any mutant survives across all tests. Manually dispatched
runs cover all project C++ code and report survivors without failing on them.
Baseline and infrastructure errors fail both modes.
CI installs only the native build dependencies and runs Mull without
installing the editable Python package.

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
