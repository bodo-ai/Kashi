#!/usr/bin/env python3
"""Require more than 90% line coverage on changed production code."""

import argparse
import os
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree

REPORTS = Path("build/coverage")
PYTHON = Path("python/src/kashi")
NATIVE = (Path("cpp/lib"), Path("cpp/include"))
# These modules implement GPU-only execution; CPU CI cannot exercise their
# device paths. Keep the default gate comprehensive for runs with a GPU.
GPU_ONLY = frozenset(
    {
        PYTHON / "iree_cuda.py",
        PYTHON / "iree_hip.py",
        PYTHON / "iree_runtime.py",
    }
)
SUFFIXES = {".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
HUNK = re.compile(rb"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def git(*args):
    return subprocess.check_output(["git", "--literal-pathspecs", *args])


def python_lines(root):
    document = ElementTree.parse(REPORTS / "python.xml")
    sources = [Path(source.text) for source in document.findall("./sources/source")]
    result = {}
    for entry in document.findall("./packages/package/classes/class"):
        name = Path(entry.attrib["filename"])
        candidates = [root / name, *(root / source / name for source in sources)]
        path = next((p for p in candidates if p.is_file()), root / name)
        if path.is_relative_to(root):
            lines = result.setdefault(path.relative_to(root), {})
            for line in entry.findall("./lines/line"):
                lines[int(line.attrib["number"])] = int(line.attrib["hits"])
    return result


def native_lines(root):
    result = {}
    with (REPORTS / "cpp.lcov").open() as report:
        for record in report:
            if record.startswith("SF:"):
                path = root / record[3:].rstrip("\n")
                lines = result.setdefault(path, {})
            elif record.startswith("DA:"):
                number, count, *_ = record[3:].split(",")
                number, count = int(number), int(count)
                lines[number] = max(lines.get(number, 0), count)
    return {
        path.relative_to(root): lines
        for path, lines in result.items()
        if path.is_relative_to(root)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="PR base commit; HEAD is the tested merge commit")
    parser.add_argument(
        "--exclude-gpu",
        action="store_true",
        help="exclude GPU executor modules when coverage was collected on CPU-only CI",
    )
    args = parser.parse_args()
    base = args.base
    root = Path.cwd()
    base = (
        git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}")
        .strip()
        .decode()
    )
    names = git(
        "diff",
        "--name-status",
        "--find-renames",
        "--diff-filter=AMRT",
        "-z",
        base,
        "HEAD",
        "--",
        PYTHON,
        *NATIVE,
    )
    fields = iter(names.split(b"\0")[:-1])
    reports = {}
    covered = total = 0
    missing = False
    print("## Changed-line coverage (>90% required)\n")
    for status in fields:
        old_path = Path(os.fsdecode(next(fields)))
        renamed = status.startswith(b"R")
        path = Path(os.fsdecode(next(fields))) if renamed else old_path
        if path.is_relative_to(PYTHON) and path.suffix == ".py":
            reader = python_lines
        elif path.suffix in SUFFIXES and any(path.is_relative_to(p) for p in NATIVE):
            reader = native_lines
        else:
            continue
        revisions = (
            [f"{base}:{old_path}", f"HEAD:{path}"]
            if renamed
            else [base, "HEAD", "--", path]
        )
        diff = git("diff", "--no-ext-diff", "--no-textconv", "--unified=0", *revisions)
        hunks = [(int(start), int(count or 1)) for start, count in HUNK.findall(diff)]
        if not any(count for _, count in hunks):
            continue
        if args.exclude_gpu and path in GPU_ONLY:
            print(f"- `{path}`: excluded from CPU-only coverage gate (GPU execution)")
            continue
        if reader not in reports:
            try:
                reports[reader] = reader(root)
            except FileNotFoundError:
                reports[reader] = {}
        lines = reports[reader].get(path)
        if lines is None:
            print(f"- `{path}`: no line coverage data")
            missing = True
            continue
        uncovered = []
        for start, count in hunks:
            for number in range(start, start + count):
                if number in lines:
                    total += 1
                    if lines[number] > 0:
                        covered += 1
                    else:
                        uncovered.append(str(number))
        if uncovered:
            print(f"- `{path}`: uncovered lines {', '.join(uncovered)}")
    if total:
        print(
            f"\n{covered}/{total} changed executable lines covered ({100 * covered / total:.2f}%)."
        )
    elif not missing:
        print("No changed executable lines to measure.")
    return int(missing or (total > 0 and 10 * covered <= 9 * total))


if __name__ == "__main__":
    raise SystemExit(main())
