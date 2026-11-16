# Rename to Kashi

**Status:** Proposed

## Intent

Rename the project and its developer-facing interfaces to Kashi. Preserve the existing repository skeleton, build capabilities, and test behavior; this change does not implement a compiler.

## Approaches

- **Selected: complete cutover.** Rename the distribution, Python package, native build identifiers, and their callers together. This gives the project one consistent name without compatibility code.
- **Display-name-only rename.** Retaining the old technical identifiers leaves the former name in everyday use and does not meet the rename goal.
- **Compatibility transition.** Keeping aliases adds unnecessary complexity for a new project and does not meet the clean-cutover goal.

## Rename contract

- Use **Kashi** as the project display name and `kashi` as the Python distribution and import package name. The native extension remains `_native`, importable as `kashi._native`.
- Move the Python package to `python/src/kashi/`; configure wheel packaging and native installation to use that path and package. Update the Pixi editable dependency and regenerate the lock file using Pixi.
- Rename the CMake project, project-owned targets, options, and variables consistently using `kashi` and `KASHI`. Update scripts, tests, and CI invocations that consume them. Preserve the behavior of native tests and mutation testing.
- Update the README, examples, tests, workflows, and authoritative specifications to use the new names.
- Remove the old project name and identifiers throughout the tracked repository, including filenames and specifications. Do not retain compatibility aliases or migration references using the old name.
- Preserve the supported platforms, build backend, dependency policy, and existing functionality. Do not rename unrelated external dependencies or generic compiler terminology.

## Non-goals

- Implementing compiler functionality or changing the Python/native API beyond the rename.
- Publishing a package, transferring a repository, changing remote URLs, or claiming ownership of the name on external services.
- Renaming local checkout directories or deleting existing build artifacts and installed packages from developer machines.

## Acceptance criteria

1. A clean editable installation using `pixi install --locked` imports both `kashi` and `kashi._native`; the checked-in lock file matches the renamed project configuration.
2. A wheel built from the project identifies the distribution as `kashi` and, when installed in a clean environment, imports `kashi._native`.
3. The Python and C++ test commands and the mutation-testing configuration use the renamed identifiers and retain their existing behavior. CI exercises the renamed installed package.
4. The tracked repository contains no instances of the former project name or identifiers, including spelling and capitalization variants, in file contents or paths. Generated local build outputs and Git history are outside the tracked repository scope.
5. The documented supported platform set remains unchanged; report any platform or toolchain that could not be verified.
