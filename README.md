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

Build a wheel with `pixi run python -m build --wheel`. Standard Python build tools also work without Pixi when the required compiler and build tools are available. PyTorch and LLVM/MLIR are not needed for this skeleton; their integration is future work.

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
