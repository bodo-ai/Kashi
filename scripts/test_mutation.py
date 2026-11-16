#!/usr/bin/env python3
"""Build native tests with Mull and combine their mutation results."""

import json
import os
import re
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "mutation"
REPORTS = BUILD / "reports"
DATABASE = REPORTS / "mutations.sqlite"
REPORT = REPORTS / "report.txt"
CONFIG = ROOT / "mull.yml"


class MutationError(Exception):
    pass


def run(command, *, cwd=ROOT, env=None, capture=False):
    print("+", shlex.join(str(part) for part in command), flush=True)
    result = subprocess.run(
        [str(part) for part in command], cwd=cwd, env=env,
        text=True, errors="replace", stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if result.returncode:
        if capture and result.stdout:
            print(result.stdout, end="", file=sys.stderr)
        raise MutationError(f"command failed with exit code {result.returncode}: {command[0]}")
    return result.stdout if capture else ""


def executable(variable, default):
    name = os.environ.get(variable, default)
    found = shutil.which(name)
    if not found:
        raise MutationError(f"{variable}: executable not found: {name}")
    # Keep the executable name: clang++ may be a symlink to clang, and its
    # invocation name controls C++ linking defaults.
    return str(Path(found).absolute())


def test_environment(properties, base):
    env = base.copy()
    for assignment in properties.get("ENVIRONMENT", []):
        name, sep, value = assignment.partition("=")
        if not sep or not name:
            raise MutationError(f"invalid CTest ENVIRONMENT entry: {assignment!r}")
        env[name] = value
    original = env.copy()
    for modification in properties.get("ENVIRONMENT_MODIFICATION", []):
        name, sep, operation_value = modification.partition("=")
        operation, colon, value = operation_value.partition(":")
        if not name or not sep or not colon:
            raise MutationError(f"invalid CTest ENVIRONMENT_MODIFICATION: {modification!r}")
        if operation == "reset":
            if name in original:
                env[name] = original[name]
            else:
                env.pop(name, None)
        elif operation == "unset":
            env.pop(name, None)
        elif operation == "set":
            env[name] = value
        elif operation in ("string_append", "path_list_append", "cmake_list_append"):
            separator = {"string_append": "", "path_list_append": os.pathsep,
                         "cmake_list_append": ";"}[operation]
            env[name] = env[name] + separator + value if name in env else value
        elif operation in ("string_prepend", "path_list_prepend", "cmake_list_prepend"):
            separator = {"string_prepend": "", "path_list_prepend": os.pathsep,
                         "cmake_list_prepend": ";"}[operation]
            env[name] = value + separator + env[name] if name in env else value
        else:
            raise MutationError(f"unsupported CTest environment operation: {operation}")
    # The compiler plugin, runner and reporter must use the same policy.
    env["MULL_CONFIG"] = base["MULL_CONFIG"]
    return env


def instrumented_executable(command, cwd):
    # CTest can launch the test through a wrapper (e.g. cmake -E env). Mull
    # must inspect the native executable, while executing the complete command.
    candidates = []
    for arg in command:
        path = Path(arg)
        if not path.is_absolute():
            path = cwd / path
        try:
            path = path.resolve()
            if not path.is_relative_to(BUILD) or not path.is_file() or not os.access(path, os.X_OK):
                continue
            with path.open("rb") as file:
                magic = file.read(4)
            if magic == b"\x7fELF" or magic in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"):
                candidates.append(path)
        except OSError:
            continue
    if len(set(candidates)) != 1:
        raise MutationError(f"cannot identify one native build executable in CTest command: {command!r}")
    return candidates[0]


def library_search_options(binary, cwd, env):
    # Mull does not consult the dynamic loader's RPATH or default directories.
    # Resolve the actual dependencies using the same environment as the test.
    directories = {str(Path(sys.prefix) / "lib")}
    if sys.platform.startswith("linux"):
        output = run(["ldd", binary], cwd=cwd, env=env, capture=True)
        for line in output.splitlines():
            if "=> not found" in line:
                raise MutationError(f"missing test executable dependency: {line.strip()}")
            match = re.search(r"(?:=>\s+|^\s*)(/\S+)", line)
            if match:
                directories.add(str(Path(match.group(1)).parent))
    else:
        directories.add("/usr/lib")
    return [arg for directory in sorted(directories) for arg in ("--ld-search-path", directory)]


def checked_tests(tests):
    enabled = []
    for test in tests:
        properties = {item["name"]: item["value"] for item in test.get("properties", [])}
        if properties.get("DISABLED", False):
            continue
        # These CTest properties change how success is determined, which Mull
        # cannot reproduce just by executing the test program.
        for name in ("WILL_FAIL", "PASS_REGULAR_EXPRESSION", "FAIL_REGULAR_EXPRESSION",
                     "SKIP_REGULAR_EXPRESSION", "FIXTURES_REQUIRED",
                     "FIXTURES_SETUP", "FIXTURES_CLEANUP"):
            if name in properties and properties[name] not in (False, None, "", []):
                raise MutationError(f"CTest test {test['name']!r} uses unsupported {name}")
        command = test.get("command", [])
        if not command:
            raise MutationError(f"CTest test {test['name']!r} has no executable command")
        cwd = Path(properties.get("WORKING_DIRECTORY", BUILD))
        if not cwd.is_absolute():
            cwd = BUILD / cwd
        enabled.append((test["name"], command, cwd, properties))
    return enabled


def check_results():
    if not DATABASE.is_file():
        raise MutationError("Mull did not produce the expected SQLite report; no mutants were found")
    try:
        with sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT mutant_id, execution_status FROM mutant"
            ).fetchall()
            versions = {row[0] for row in db.execute(
                "SELECT value FROM information WHERE key = 'Mull Version'"
            )}
    except sqlite3.Error as error:
        raise MutationError(f"invalid Mull SQLite report: {error}") from error
    for version in versions:
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) and version != "0.36.0":
            raise MutationError(f"SQLite results came from incompatible Mull {version}")
    if not rows:
        raise MutationError("unexpected zero-mutant run; inspect instrumentation and path filters")
    bad = [(name, status) for name, status in rows if status not in (1, 2, 4, 5, 8)]
    if bad:
        details = "\n".join(f"  {name}: status {status}" for name, status in bad)
        raise MutationError(
            "Mull produced timeout or invalid/infrastructure results; no score is reported:\n" + details
        )
    # Match Mull's cross-executable aggregation: a failure or crash in any
    # invocation kills a mutant; a timeout is never silently treated as killed.
    grouped = {}
    for name, status in rows:
        grouped.setdefault(name, set()).add(status)
    killed = sum(bool(statuses & {1, 4, 5}) for statuses in grouped.values())
    survived = sum(not statuses & {1, 4, 5} and 2 in statuses for statuses in grouped.values())
    not_covered = len(grouped) - killed - survived
    if not killed and not survived:
        raise MutationError("zero covered mutants; inspect instrumentation and test coverage")
    return killed, survived, not_covered


def changed_files(base):
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if resolved.returncode:
        raise MutationError(f"cannot resolve mutation base {base!r}: {resolved.stderr.strip()}")
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AMT", "--no-renames", "-z",
         resolved.stdout.strip(), "--", "cpp/lib", "cpp/include"],
        cwd=ROOT, capture_output=True, check=True,
    )
    suffixes = {".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
    return [path for raw in diff.stdout.split(b"\0") if raw
            if (path := Path(os.fsdecode(raw))).suffix in suffixes and (ROOT / path).is_file()]


def changed_policy(paths):
    def escape(path):
        return re.sub(r"([\\.^$|?*+(){}\[\]])", r"\\\1", str(path))

    patterns = [f"^(?:{escape(ROOT / path)}|{escape(path)})$" for path in paths]
    include = "includePaths:\n" + "".join(f"  - {json.dumps(pattern)}\n" for pattern in patterns)
    policy, count = re.subn(r"(?m)^includePaths:\s*\n(?:[ \t]+[^\n]*\n)*", lambda _: include,
                            CONFIG.read_text())
    if count != 1:
        raise MutationError("Mull policy must contain exactly one includePaths section")
    destination = BUILD / "mull-changed.yml"
    destination.write_text(policy)
    return destination


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    for path in (DATABASE, REPORT, REPORTS / "mutations.txt"):
        path.unlink(missing_ok=True)

    base = os.environ.get("MULL_DIFF_BASE")
    selected = changed_files(base) if base else None
    if selected == []:
        message = "no mutation results yet: no changed project native files"
        REPORT.write_text(message + "\n")
        print(message)
        return

    source_suffixes = {".cpp", ".cc", ".cxx"}
    sources = [path for path in (ROOT / "cpp" / "tests").rglob("*")
               if path.is_file() and path.suffix in source_suffixes]
    if not sources:
        message = "no mutation results yet: no native tests yet"
        REPORT.write_text(message + "\n")
        print(message)
        return

    production = [path for path in (ROOT / "cpp" / "lib").rglob("*")
                  if path.is_file() and path.suffix in source_suffixes]
    headers = [path for path in (ROOT / "cpp" / "include").rglob("*")
               if path.is_file() and path.suffix in (".h", ".hh", ".hpp", ".hxx")]
    if not production and not headers:
        message = "no mutation results yet: no project native code to mutate"
        REPORT.write_text(message + "\n")
        print(message)
        return

    runner = executable("MULL_RUNNER", "mull-runner-19")
    reporter = executable("MULL_REPORTER", "mull-reporter-19")
    compiler = executable("MULL_CXX", "clang++-19")
    plugin = Path(os.environ.get("MULL_PLUGIN", "/usr/lib/mull-ir-frontend-19")).resolve()
    if not plugin.is_file():
        raise MutationError(f"Mull LLVM frontend plugin not found: {plugin}")
    plugin_data = plugin.read_bytes()
    plugin_version = re.search(rb"mull-plugin-version:([0-9]+\.[0-9]+\.[0-9]+)\x00", plugin_data)
    plugin_llvm = re.search(rb"mull-plugin-llvm:([0-9]+\.[0-9]+\.[0-9]+)\x00", plugin_data)
    if not plugin_version or plugin_version.group(1) != b"0.36.0" or not plugin_llvm or not plugin_llvm.group(1).startswith(b"19."):
        raise MutationError(f"expected Mull 0.36.0 LLVM 19 frontend plugin: {plugin}")
    print(f"Mull plugin 0.36.0, LLVM {plugin_llvm.group(1).decode()}: {plugin}", flush=True)
    env = os.environ.copy()
    env["MULL_CONFIG"] = str(changed_policy(selected) if selected is not None else CONFIG)
    compiler_version = run([compiler, "--version"], env=env, capture=True)
    if "clang version 19." not in compiler_version and "clang version 19\n" not in compiler_version:
        raise MutationError(f"expected Clang 19 compatible with the Mull plugin; got: {compiler_version.strip()}")
    print(compiler_version.strip(), flush=True)
    for tool in (runner, reporter):
        version_output = run([tool, "--version"], env=env, capture=True)
        print(version_output.strip(), flush=True)
        first_line = version_output.splitlines()[0] if version_output else ""
        version = re.search(r"\b([0-9]+\.[0-9]+\.[0-9]+)\b", first_line)
        if version and version.group(1) != "0.36.0":
            raise MutationError(f"expected Mull 0.36.0, got {first_line}")
    nanobind = run([sys.executable, "-m", "nanobind", "--cmake_dir"], env=env, capture=True).strip()
    prefix = ";".join([sys.prefix, *filter(None, env.get("CMAKE_PREFIX_PATH", "").split(os.pathsep))])
    run(["cmake", "-S", ROOT, "-B", BUILD, "-G", "Ninja",
         "-DKASHI_BUILD_TESTS=ON", "-DKASHI_MUTATION=ON",
         f"-DKASHI_MULL_PLUGIN={plugin}", f"-DCMAKE_CXX_COMPILER={compiler}",
         "-DCMAKE_BUILD_TYPE=Debug", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
         f"-DPython_EXECUTABLE={sys.executable}", f"-Dnanobind_DIR={nanobind}",
         f"-DCMAKE_PREFIX_PATH={prefix}"], env=env)
    # Recompile when the policy changes as it is consumed by the plugin.
    build_output = run(["cmake", "--build", BUILD, "--clean-first"], env=env, capture=True)
    print(build_output, end="", flush=True)
    if "Using some defaults" in build_output or "Failed to load passes" in build_output:
        raise MutationError("Mull plugin or configuration failed during compilation")
    output = run(["ctest", "--test-dir", BUILD, "--show-only=json-v1"], env=env, capture=True)
    try:
        tests = checked_tests(json.loads(output)["tests"])
    except (ValueError, KeyError) as error:
        raise MutationError(f"invalid CTest discovery output: {error}") from error
    if not tests:
        raise MutationError("native test sources exist but CTest discovered no enabled tests")
    run(["ctest", "--test-dir", BUILD, "--output-on-failure", "--no-tests=error"], env=env)

    all_empty = True
    for index, (name, command, cwd, properties) in enumerate(tests):
        test_env = test_environment(properties, env)
        binary = instrumented_executable(command, cwd)
        launch = Path(command[0])
        if not launch.is_absolute():
            launch = cwd / launch
        runner_options = library_search_options(binary, cwd, test_env)
        args = command[1:]
        skip_code = properties.get("SKIP_RETURN_CODE")
        skip_marker = BUILD / f"mull-test-command-{index}.skipped"
        skip_marker.unlink(missing_ok=True)
        if launch.resolve() != binary or skip_code is not None:
            wrapper = BUILD / f"mull-test-command-{index}.py"
            # Preserve the complete CTest command and Mull's environment. A
            # skipped mutant is not evidence that the mutant was killed.
            wrapper.write_text(
                f"#!{sys.executable}\nimport os, pathlib, subprocess, sys\n"
                + f"command = {command!r}\n"
                + "result = subprocess.call(command, env=os.environ)\n"
                + f"if result == {skip_code!r}: pathlib.Path({str(skip_marker)!r}).touch()\n"
                + "if result < 0: os.kill(os.getpid(), -result)\n"
                + "sys.exit(result)\n"
            )
            wrapper.chmod(0o755)
            runner_options.extend(["--test-program", wrapper])
            args = []
        print(f"Mutation test: {name}", flush=True)
        output = run([runner, "--allow-surviving", "--reporters", "SQLite",
                      "--report-dir", REPORTS, "--report-name", "mutations",
                      *runner_options, "--", binary, *args], cwd=cwd, env=test_env, capture=True)
        if skip_marker.exists():
            raise MutationError(f"CTest test {name!r} skipped during mutation testing")
        # Mull prints a fictitious infinite score for an empty invocation.
        print("\n".join(line for line in output.splitlines()
                        if "No mutants found. Mutation score" not in line), flush=True)
        all_empty &= "No mutants found. Mutation score" in output

    if selected is not None and all_empty and not DATABASE.exists():
        message = "no mutation results yet: no mutants found in changed files; check coverage and instrumentation"
        REPORT.write_text(message + "\n")
        print(message)
        return

    killed, survived, not_covered = check_results()
    run([reporter, "--allow-surviving", "--reporters", "IDE", "--ide-reporter-show-killed",
         "--report-dir", REPORTS, "--report-name", "mutations", DATABASE], env=env)
    generated = REPORTS / "mutations.txt"
    if not generated.is_file() or not generated.stat().st_size:
        raise MutationError("Mull did not produce the expected human-readable report")
    summary = (f"Killed: {killed}; survived: {survived}; not covered: {not_covered}\n"
               f"Mutation score: {100 * killed / (killed + survived + not_covered):.1f}%\n")
    REPORT.write_text(summary + generated.read_text())
    print(summary + f"Human-readable report: {REPORT}\nSQLite results: {DATABASE}")


if __name__ == "__main__":
    try:
        main()
    except (MutationError, OSError, subprocess.CalledProcessError) as error:
        message = f"mutation testing error: {error}"
        print(message, file=sys.stderr)
        REPORTS.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(message + "\n")
        sys.exit(1)
