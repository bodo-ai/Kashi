# Kashi

Repository skeleton for a future compiler that captures PyTorch FX graphs in Python and optimizes MLIR in C++. The compiler pipeline is not implemented.

## Development setup

Install [Pixi](https://pixi.sh/latest/installation/). The supported development platforms are Linux x86-64 and macOS on Apple silicon and Intel. On macOS, install Xcode Command Line Tools (`xcode-select --install`), including the Apple SDK.

```sh
pixi install --locked
pixi run python -c 'import kashi._native'
```

Pixi installs the project in editable mode. Python source changes are picked up automatically. After changing C++ or CMake files, rebuild and reinstall with:

```sh
pixi run python -m pip install --no-deps -e .
```

The rebuild uses an isolated Python build environment.

Build a wheel with `pixi run python -m build --wheel`. Standard Python build tools also work without Pixi when the required compiler and build tools are available. The default environment does not require PyTorch or LLVM/MLIR; the compiler pipeline is not implemented yet.

## vLLM and MLIR development

The optional `vllm-mlir` environment adds the CPU dependencies for the FX–MLIR work without changing the default environment. It uses Python 3.14, vLLM 0.33.0, PyTorch 2.14.0, and IREE compiler/runtime 3.14.0. vLLM 0.33.0 requires PyTorch 2.14.0; the newer PyTorch 2.15.0 does not satisfy that requirement.

This environment targets Apple silicon with macOS 14 or newer. Upstream macOS CPU support is experimental.

```sh
pixi install --locked -e vllm-mlir
pixi run --locked -e vllm-mlir python -c 'import iree.compiler, iree.runtime, torch, vllm; from iree.compiler.extras.fx_importer import FxImporter'
```

## Tests

The locked Pixi environment supplies pytest, Catch2 v3, CMake, Ninja,
nanobind, Python and a C++ compiler. No GPU, PyTorch or LLVM/MLIR is needed.
Run the suites from the repository root:

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
