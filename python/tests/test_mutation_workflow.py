"""Regression tests for the mutation command's empty-project detection."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("test_suffix", "unit_suffix"), [(".cc", ".cpp"), (".cpp", ".cxx")]
)
def test_native_sources_require_mutation_toolchain(tmp_path, test_suffix, unit_suffix):
    root = Path(__file__).resolve().parents[2]
    script_dir = tmp_path / "scripts"
    test_dir = tmp_path / "cpp" / "tests"
    unit_dir = tmp_path / "cpp" / "lib"
    for directory in (script_dir, test_dir, unit_dir):
        directory.mkdir(parents=True)
    shutil.copy2(root / "scripts" / "test_mutation.py", script_dir)
    (test_dir / f"example{test_suffix}").write_text("// native test\n")
    (unit_dir / f"example{unit_suffix}").write_text("// native unit\n")

    env = os.environ.copy()
    env["MULL_RUNNER"] = str(tmp_path / "missing-mull-runner")
    result = subprocess.run(
        [sys.executable, str(script_dir / "test_mutation.py")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "MULL_RUNNER" in result.stderr
    assert "no mutation results yet" not in result.stdout


@pytest.fixture
def mutation_git_checkout(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for directory in ("scripts", "cpp/lib", "cpp/include", "cpp/tests"):
        (tmp_path / directory).mkdir(parents=True)
    shutil.copy2(root / "scripts" / "test_mutation.py", tmp_path / "scripts")
    (tmp_path / "cpp/lib/existing.cpp").write_text("int existing() { return 1; }\n")
    (tmp_path / "cpp/tests/example.cpp").write_text("// test source\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "base",
        ],
        cwd=tmp_path,
        check=True,
    )
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    env = os.environ.copy()
    env.update(MULL_DIFF_BASE=base, MULL_RUNNER=str(tmp_path / "missing-mull-runner"))
    return tmp_path, env


def run_mutation(checkout, env):
    return subprocess.run(
        [sys.executable, str(checkout / "scripts/test_mutation.py")],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_changed_mode_ignores_nonproduction_files_and_deletions(mutation_git_checkout):
    checkout, env = mutation_git_checkout
    (checkout / "cpp/lib/existing.cpp").unlink()
    (checkout / "cpp/tests/example.cpp").write_text("// changed test\n")
    result = run_mutation(checkout, env)
    assert result.returncode == 0
    assert "no mutation results yet" in result.stdout
    assert "changed" in result.stdout


@pytest.mark.parametrize("path", ["cpp/lib/existing.cpp", "cpp/include/new header.hpp"])
def test_changed_production_files_require_mutation_toolchain(
    mutation_git_checkout, path
):
    checkout, env = mutation_git_checkout
    (checkout / path).write_text("int changed() { return 2; }\n")
    subprocess.run(["git", "add", "--", path], cwd=checkout, check=True)
    result = run_mutation(checkout, env)
    assert result.returncode != 0
    assert "MULL_RUNNER" in result.stderr


def test_changed_mode_rejects_unknown_base(mutation_git_checkout):
    checkout, env = mutation_git_checkout
    env["MULL_DIFF_BASE"] = "not-a-real-commit"
    result = run_mutation(checkout, env)
    assert result.returncode != 0
    assert "not-a-real-commit" in result.stderr
