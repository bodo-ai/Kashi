# ML Compiler

Repository skeleton for a future compiler that captures PyTorch FX graphs in Python and optimizes MLIR in C++. The compiler pipeline is not implemented.

## Development setup

Install [Pixi](https://pixi.sh/latest/installation/). The supported development platforms are Linux x86-64 and macOS on Apple silicon and Intel. On macOS, install Xcode Command Line Tools (`xcode-select --install`), including the Apple SDK.

```sh
pixi install --locked
pixi run python -c 'import ml_compiler._native'
```

Pixi installs the project in editable mode. Python source changes are picked up automatically. After changing C++ or CMake files, rebuild and reinstall with:

```sh
pixi run python -m pip install --no-deps -e .
```

The rebuild uses an isolated Python build environment.

Build a wheel with `pixi run python -m build --wheel`. Standard Python build tools also work without Pixi when the required compiler and build tools are available. PyTorch and LLVM/MLIR are not needed for this skeleton; their integration is future work.

## Layout

- `python/src/ml_compiler/`: Python package and future FX integration.
- `python/tests/`: future Python tests.
- `cpp/bindings/`: Python native module.
- `cpp/include/`, `cpp/lib/`, `cpp/tests/`: future C++ headers, MLIR passes, and tests.
- `specs/`: project specifications.
