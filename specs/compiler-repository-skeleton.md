# Compiler Repository Skeleton

**Status:** Proposed

## Intent

Establish a Python and C++ repository layout and a working Python packaging path for a future LLM compiler. The intended compiler captures PyTorch FX graphs in Python, converts them to MLIR, runs C++ MLIR optimizations, and returns a result to Python. This specification does not decide the result representation or implement that pipeline.

## Approaches

- **Selected: scikit-build-core, CMake, and nanobind.** Standard Python packaging drives a CMake native build; C++ binding code exposes native functionality to Python. Nanobind matches the binding library used by MLIR's Python bindings.
- **Cython with CMake.** A viable alternative, particularly for compiling Python-like processing code, but it introduces a separate binding language for a primarily C++ compiler.
- **pybind11 with CMake.** A mature alternative; nanobind is the preferred fit for this new MLIR-oriented project.

## Layout

```text
pyproject.toml                  # Python package metadata, build backend, and Pixi workspace
pixi.lock                       # Locked development environment
CMakeLists.txt                  # Native build and installation
README.md                       # Scope, prerequisites, and build instructions
python/
  src/ml_compiler/              # Python package; future FX capture and API
  tests/                        # Future Python tests
cpp/
  bindings/                     # Python-to-C++ native extension
  include/                      # Future C++ headers
  lib/                          # Future MLIR transformations
  tests/                        # Future C++ and MLIR tests
specs/                          # Versioned specifications
```

Empty directories may use tracking files. Existing unrelated repository content remains unchanged.

## Packaging contract

- `pyproject.toml` uses `scikit-build-core` as its build backend and declares nanobind as a build dependency. CMake builds a private extension importable as `ml_compiler._native` and installs it into the Python package.
- Both a regular installation and an editable installation are supported. Documentation identifies a supported Python version, a C++ compiler, CMake, and a build tool, and gives commands to build and install locally.
- The extension may contain only the minimal module initialization needed to build and import it. No compiler behavior or placeholder compiler API is provided.
- The skeleton does not require PyTorch or an LLVM/MLIR installation to build: no compiler code uses them yet. Future work must specify how a compatible LLVM/MLIR installation is discovered and linked, and how its native libraries are distributed. Python object and MLIR ownership and interchange are also future decisions.
- A local wheel build is in scope. Publishing wheels, a cross-platform wheel matrix, and bundled LLVM/MLIR are not.

## Development dependencies

- Use Pixi as the development environment manager, with its workspace configuration in `pyproject.toml` and `conda-forge` as the channel. The supported development platforms are `linux-64`, `osx-arm64` (Apple silicon), and `osx-64` (Intel).
- Pixi provides Python, a C++ toolchain, CMake, Ninja, and the tools needed to build a wheel. The project is installed into the Pixi environment as an editable PyPI path dependency.
- Commit the generated `pixi.lock` and ignore `.pixi/`. Document `pixi install`, running Python through `pixi run`, building a wheel, and rebuilding after C++ changes. Document any required macOS system prerequisites, including the Apple SDK and Xcode Command Line Tools.
- Keep standard Python build requirements in `[build-system]` so the package can still be built without Pixi. Pixi locks the development environment; isolated Python builds can resolve their own build requirements and are not guaranteed to use versions from `pixi.lock`.
- Defer PyTorch and LLVM/MLIR dependencies until compiler functionality requires them; a conda package's availability alone does not establish that it provides the required headers, libraries, or ABI compatibility.

## Non-goals

- FX capture, graph conversion, MLIR passes, execution, or a compiler return type.
- A public compiler API, fake implementations, or placeholder optimization results.
- Choosing MLIR dialects, PyTorch integration strategy, LLVM version, or release distribution policy.

## Acceptance criteria

1. A fresh checkout contains the documented layout and explains the role of each major directory.
2. In a clean environment with documented prerequisites, a regular installation builds and `import ml_compiler._native` succeeds.
3. An editable installation also builds and imports the extension. The documentation explains how to rebuild after native source changes.
4. A wheel can be built locally and installed into a clean environment, where the extension imports successfully.
5. The package does not claim to implement the compiler pipeline or expose placeholder compiler behavior.
6. On each supported platform, `pixi install --locked` creates the documented environment from the checked-in lock file, and `pixi run python -c "import ml_compiler._native"` succeeds. Verify the macOS paths on both Apple silicon and Intel, or explicitly report any platform that could not be exercised.

## References

- [scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/)
- [nanobind packaging](https://nanobind.readthedocs.io/en/latest/packaging.html)
- [MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/)
- [Pixi Python project configuration](https://pixi.sh/latest/python/pyproject_toml/)
- [Pixi lock files](https://pixi.sh/latest/workspace/lock_file/)
