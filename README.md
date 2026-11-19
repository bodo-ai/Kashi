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

Reports are written to `build/coverage/`: `python.xml` and `python.md` for
Python, `cpp.txt` and (when coverage is available) `cpp.json` for C++. GitHub
Actions posts a summary and uploads the machine-readable reports and native
summary as `coverage-reports`. Test and tool failures fail the job; coverage
percentages are advisory.


## Native mutation testing

Every pull request runs Mull against added or modified C++ files under
`cpp/lib/` and `cpp/include/`, using the full contents of each selected file.
Manually dispatched runs cover all project C++ code. Survivors
are advisory; infrastructure and baseline failures fail the job.

Reports are in `build/mutation/reports/`.

Mutation testing requires Clang 19 and Mull 0.36.0 with its LLVM 19 frontend
plugin; see `.github/workflows/mutation.yml` for the pinned Linux installation.
Use the scheduled or manually dispatched Linux workflow when the toolchain is
unavailable locally.

## Layout

- `python/src/kashi/`: Python package and future FX integration.
- `python/tests/`: Python API and import tests.
- `cpp/bindings/`: Python native module.
- `cpp/include/`, `cpp/lib/`, `cpp/tests/`: C++ headers, native units and tests.
- `specs/`: project specifications.
