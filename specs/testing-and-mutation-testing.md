# Python and C++ Testing

**Status:** Proposed

## Intent

Give contributors a reproducible way to test Python and C++ changes and evaluate the effectiveness of native tests with Mull. The repository currently contains an importable, empty nanobind extension, no compiler implementation, and no tests. The initial testing infrastructure must be useful now without inventing compiler behavior solely to exercise it.

## Goals and non-goals

- Run fast, deterministic Python and native tests locally and on pull requests, using the existing Pixi, CMake, and scikit-build-core setup.
- Provide a separate, reproducible Clang/Mull build and a readable mutation report for project-owned C++ code exercised by native test executables.
- Keep development and testing dependencies out of the installed package's runtime requirements and ordinary wheel builds.
- Do not implement FX capture, MLIR, compiler APIs, or fake production functionality. Do not require PyTorch or LLVM/MLIR merely to run the initial tests.
- Python mutation testing, numeric coverage or mutation-score gates, benchmarks, releases, and testing future MLIR passes are outside this specification. A later specification can add them when the relevant functionality exists.

## Approaches

- **Proposed: pytest, Catch2/CTest, and a separate Mull build.** Python tests exercise the installed package; native tests run as CTest-registered executables; an opt-in Clang build instruments native production code for Mull. This uses the existing package and build systems without imposing Mull on regular development or distribution builds.
- **Run only Python tests through the extension.** This tests the public boundary but leaves C++ units difficult to test directly, and Mull's executable-based workflow becomes harder to apply. Reject as the sole native testing strategy.
- **Instrument every build and run mutation testing on every PR.** This provides continuous mutation feedback but couples normal builds to a specific Clang/plugin combination and adds cost and noise, especially while there is no mutable native logic. Reject for the initial workflow.

These are proposed engineering choices, not an assertion that the user has approved them.

## Test contract

### Python

- Use pytest, with tests under `python/tests/` and discovery configured in `pyproject.toml`. Execute against the installed package, not an ad hoc source-path override. Pixi supplies pytest as a development dependency, with `pixi.lock` updated accordingly.
- The initial suite checks that the installed `ml_compiler` package and its `_native` extension can be imported. When public behavior exists, test results, boundaries, and failure cases through the Python API, including the Python/native boundary where relevant.
- Tests must be isolated and deterministic, and must not require network access, GPUs, PyTorch, or LLVM/MLIR unless a future test explicitly declares and provides those prerequisites.
- A documented `pixi run test-python` runs the suite and returns nonzero for failures or no collected tests.

### C++

- Use Catch2 v3 and CTest for native unit tests under `cpp/tests/`. Add CMake support for registering native test executables and exposing project headers and libraries to them as real native units are introduced.
- The explicit native-test build enables testing and locates Catch2 through the development environment. The default package/wheel build does not require Catch2, build tests, or include test executables in the wheel. The Pixi environment provides Catch2 and everything needed for a direct CMake configure, including Python and nanobind; the lock file records the resolved dependencies.
- A documented `pixi run test-cpp` configures and builds a separate native test tree, runs its registered CTest tests, prints failure output, and returns nonzero on configuration, build, discovery, or test failures.
- There is currently no C++ compiler logic to assert against. Until a genuine native test is added, the command and CI must explicitly report `no native tests yet`, rather than treating an empty CTest run as a passing native suite. Once native test sources are present, zero discovered tests is an error. Add native tests alongside the first real C++ behavior; do not introduce dummy production APIs or permanent tautological tests to populate the suite.
- Native tests should target behavior and boundaries rather than implementation details. Keep test names and failures actionable and fixtures independent of execution order.

### Packaging boundary

The existing editable and regular build paths continue to work. Python tests use the editable installation for development; CI also builds a wheel, installs it into a clean environment outside the source tree, and checks that both the package and `_native` import. The wheel check must not accidentally import the checkout or the editable installation.

## Mull mutation testing

- Provide an opt-in `pixi run test-mutation` entry point using a dedicated build directory, separate from ordinary builds and the editable extension. Its documented prerequisites include Mull, Clang, and the matching Mull LLVM frontend plugin. Pin a mutually compatible Mull/Clang/plugin toolchain for the Linux CI job and document its installation, version checks, and any local platform limitations. Ordinary Python and native test commands must not require Mull or Clang.
- Compile native production translation units that are linked into native test executables with Clang, debug information, and the matching Mull compiler plugin; preserve the compile information required by the chosen Mull version. Apply instrumentation only to this build, not to third-party sources, test code, or normal distribution artifacts. Select project-owned `cpp/lib/` and `cpp/include/` for mutation; exclude `cpp/tests/`, dependencies, generated files, and binding glue. Python tests and native-extension binding glue are not part of this initial mutation score.
- Configure a checked-in Mull policy with explicit mutation operators and path filters. The initial operators must cover at least arithmetic and comparison/boundary mutations supported by the pinned Mull version. Document the selected operators and exclusions; do not suppress survivors merely to inflate the score.
- Run a passing baseline and invoke `mull-runner` on each relevant instrumented native test executable, not on `ctest` itself. Preserve each executable's needed arguments, environment, and working directory. When there are multiple executables, combine results using Mull's SQLite reporter and `mull-reporter`, so a mutant killed by any relevant executable is not incorrectly reported as a survivor. Start each invocation from a clean report database, retain a human-readable report and the machine-readable results, and surface source locations, killed/survived counts, and score when available.
- Survivors are advisory: use Mull's survivor-allowing mode rather than a score threshold. A failed baseline, tool or plugin mismatch, instrumentation/configuration failure, failed runner, or missing expected report is an error, not an advisory success. Mutation results must distinguish survivors from tool errors and timeouts; do not silently count infrastructure failures as killed mutants.
- With no native tests or no mutable project code, report `no mutation results yet`, do not claim a score or 100% success, and explain why. Once mutable, covered project code and native tests exist, an unexpected zero-mutant run is an error requiring investigation, not a passing mutation result.
- Run Mull manually and in a Linux CI job on a schedule and by explicit dispatch. Publish reports as CI artifacts even when survivors exist; do not make a mutation score a PR merge gate. Infrastructure failures still fail the mutation job. Normal PR checks remain independent of mutation tooling.

## CI and developer workflow

- On pull requests and pushes, use the locked Pixi environment to run the Python and native workflows and the clean-wheel import check on Linux x86-64, macOS Apple silicon, and macOS Intel. The native job reports the explicit empty-suite state until genuine tests exist. Each platform's failures are visible; do not silently skip an unavailable runner and claim platform coverage.
- Document prerequisites, exact setup and commands for both test suites, how to select a single test, how to rebuild after C++ changes, and how to run and interpret Mull locally. Explain that Mull may have narrower platform support than the ordinary test suites and how to use the Linux job when it is unavailable locally.
- Keep test tools and mutation configuration confined to development and CI. A clean `pixi install --locked` and the existing documented build and import flows remain supported.

## Acceptance criteria

1. A clean locked Pixi environment runs `pixi run test-python`, discovers the import test, and fails when that test is made to fail. The installed native extension is actually imported.
2. The documented native command configures successfully without requiring a wheel build, reports the initial empty-suite state honestly, and runs and propagates failures from real CTest-registered Catch2 tests when present. A temporary native test exercising a temporary C++ unit demonstrates discovery, execution, and failure propagation without leaving test-only production code in the repository.
3. Normal editable and wheel builds work without Catch2 or Mull in the isolated package build environment. The wheel installs in a clean environment and imports outside the checkout.
4. A temporary C++ unit and native test demonstrate the Mull path: a passing baseline produces at least one reported killed mutant and one reported survivor, with locations in the temporary production unit rather than its test or a dependency. A failing baseline and a missing or incompatible plugin produce nonzero results. Remove the temporary probe afterward.
5. The initial real-repository mutation invocation reports no results rather than a fabricated score while native tests or mutable native code are absent. When real applicable native tests and mutable code exist, reports include their results and unexpected zero-mutant runs fail.
6. CI implements the stated triggers, platform coverage, separation of normal and mutation jobs, result reporting, and artifact publication. Verify all available platforms; explicitly report any platform or CI scenario that cannot be exercised rather than claiming it passed.
7. README or development documentation explains all commands and prerequisites, and the lock file is updated for development dependencies.

## References

- [pytest integration practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Catch2](https://github.com/catchorg/Catch2)
- [CTest manual](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
- [Mull installation](https://mull.readthedocs.io/en/latest/Installation.html)
- [Mull CMake integration](https://mull.readthedocs.io/en/latest/tutorials/CMakeIntegration.html)
- [Mull CTest integration](https://mull.readthedocs.io/en/latest/tutorials/CTestIntegration.html)
- [Mull multiple test targets](https://mull.readthedocs.io/en/latest/tutorials/MultipleTestTargets.html)
- [Mull configuration](https://mull.readthedocs.io/en/latest/MullConfig.html)
- [Mull runner options](https://mull.readthedocs.io/en/latest/command-line/mull-runner.html)
