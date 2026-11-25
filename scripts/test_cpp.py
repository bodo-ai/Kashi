#!/usr/bin/env python3
"""Configure, build, and run the project's native Catch2 tests."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build" / "test-cpp"
COVERAGE_BUILD_DIR = ROOT / "build" / "coverage-cpp"
REPORT_DIR = ROOT / "build" / "coverage"
TEST_DIR = ROOT / "cpp" / "tests"
SOURCE_EXTENSIONS = {".cpp", ".cc", ".cxx"}


def report_coverage(profile_dir: Path, build_dir: Path) -> None:
    profiles = sorted(profile_dir.glob("*.profraw"))
    if not profiles:
        raise ValueError("CTest produced no LLVM coverage profiles")
    merged = profile_dir / "merged.profdata"
    subprocess.run(
        ["llvm-profdata", "merge", "-sparse", "-o", merged, *profiles], check=True
    )

    sources = sorted(
        path
        for directory in (ROOT / "cpp" / "lib", ROOT / "cpp" / "include")
        for path in directory.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    if not sources:
        message = "no project-owned native sources to report"
        print(message)
        (REPORT_DIR / "cpp.txt").write_text(message + "\n")
        return

    args = [
        build_dir / "kashi_native_tests",
        f"-instr-profile={merged}",
        "--sources",
        *sources,
    ]
    summary = subprocess.check_output(
        ["llvm-cov", "report", *args], cwd=ROOT, text=True
    )
    exported = subprocess.check_output(
        ["llvm-cov", "export", "--summary-only", *args], cwd=ROOT, text=True
    )
    if not any(record["files"] for record in json.loads(exported)["data"]):
        message = "no project-owned native sources mapped in the test executable"
        print(message)
        (REPORT_DIR / "cpp.txt").write_text(message + "\n")
        return
    print(summary, end="")
    (REPORT_DIR / "cpp.txt").write_text(summary)
    (REPORT_DIR / "cpp.json").write_text(exported)
    (REPORT_DIR / "cpp.lcov").write_bytes(
        subprocess.check_output(
            ["llvm-cov", "export", "--format=lcov", *args], cwd=ROOT
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage", action="store_true", help="generate LLVM source coverage"
    )
    coverage = parser.parse_args().coverage
    build_dir = COVERAGE_BUILD_DIR if coverage else BUILD_DIR
    # Avoid building debug symbols for the entire embedded LLVM dependency.
    config = "Release" if coverage else "Debug"
    try:
        cmake_args = []
        build_env = None
        if coverage:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            for name in ("cpp.txt", "cpp.json", "cpp.lcov"):
                (REPORT_DIR / name).unlink(missing_ok=True)
            cmake_args = [
                "-G",
                "Ninja",
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DKASHI_COVERAGE=ON",
            ]
            # Catch2 may invoke the binary during discovery; do not count that run.
            build_env = dict(
                os.environ, LLVM_PROFILE_FILE=str(build_dir / "discovery.profraw")
            )
        nanobind_dir = subprocess.run(
            [sys.executable, "-m", "nanobind", "--cmake_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not nanobind_dir:
            raise ValueError("nanobind did not report its CMake package directory")

        subprocess.run(
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(build_dir),
                *cmake_args,
                "-DKASHI_BUILD_TESTS=ON",
                "-DKASHI_MUTATION=OFF",
                f"-DCMAKE_BUILD_TYPE={config}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-Dnanobind_DIR={nanobind_dir}",
                f"-DCMAKE_PREFIX_PATH={sys.prefix}",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", config],
            check=True,
            env=build_env,
        )

        ctest = ["ctest", "--test-dir", str(build_dir), "-C", config]
        discovery = subprocess.run(
            [*ctest, "--show-only=json-v1"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        tests = json.loads(discovery.stdout).get("tests", [])
        if not tests:
            has_sources = any(
                path.is_file() and path.suffix in SOURCE_EXTENSIONS
                for path in TEST_DIR.rglob("*")
            )
            if has_sources:
                print(
                    "error: native test sources exist but CTest discovered no tests",
                    file=sys.stderr,
                )
                return 1
            print("no native tests yet")
            if coverage:
                (REPORT_DIR / "cpp.txt").write_text("no native tests yet\n")
            return 0

        command = [*ctest, "--output-on-failure", "--no-tests=error"]
        if coverage:
            with TemporaryDirectory(dir=build_dir) as directory:
                profile_dir = Path(directory)
                env = dict(
                    os.environ, LLVM_PROFILE_FILE=str(profile_dir / "%m-%p.profraw")
                )
                subprocess.run(command, check=True, env=env)
                report_coverage(profile_dir, build_dir)
        else:
            subprocess.run(command, check=True)
        return 0
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        print(f"error: command failed: {error.cmd}", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
