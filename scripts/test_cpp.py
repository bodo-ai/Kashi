#!/usr/bin/env python3
"""Configure, build, and run the project's native Catch2 tests."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build" / "test-cpp"
TEST_DIR = ROOT / "cpp" / "tests"
SOURCE_EXTENSIONS = {".cpp", ".cc", ".cxx"}


def main() -> int:
    try:
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
                str(BUILD_DIR),
                "-DKASHI_BUILD_TESTS=ON",
                "-DKASHI_MUTATION=OFF",
                "-DCMAKE_BUILD_TYPE=Debug",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-Dnanobind_DIR={nanobind_dir}",
                f"-DCMAKE_PREFIX_PATH={sys.prefix}",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--config", "Debug"], check=True
        )

        ctest = ["ctest", "--test-dir", str(BUILD_DIR), "-C", "Debug"]
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
            return 0

        subprocess.run([*ctest, "--output-on-failure", "--no-tests=error"], check=True)
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
