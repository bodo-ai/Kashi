"""Package a Release Ninja build of Kashi's pinned IREE dependency for macOS arm64."""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".inc", ".def", ".inl", ".cuh"}
PLUGIN = "iree/lib/libkashi_iree_plugin.a"


def rule(graph, pattern):
    match = re.search(
        r"^build " + pattern + r"[^\n]*\n((?:  [^\n]*\n)+)", graph, re.MULTILINE
    )
    if match is None:
        raise ValueError(f"Missing Ninja rule: {pattern}")
    return dict(re.findall(r"^  (\w+) = (.*)$", match[1], re.MULTILINE))


def copy_headers(source, destination):
    for folder, dirs, files in os.walk(source):
        dirs[:] = [name for name in dirs if name not in {".git", "CMakeFiles"}]
        for name in files:
            path = Path(folder) / name
            if path.suffix in HEADER_SUFFIXES or name.startswith(("LICENSE", "NOTICE")):
                output = destination / path.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)


def package(build, output):
    cache = dict(
        re.findall(
            r"^([^/#\n][^:\n]*):\w+=(.*)$",
            (build / "CMakeCache.txt").read_text(),
            re.MULTILINE,
        )
    )
    if (
        cache.get("CMAKE_BUILD_TYPE") != "Release"
        or cache.get("CMAKE_GENERATOR") != "Ninja"
    ):
        raise ValueError("Expected a Release Ninja source build")
    if (
        cache.get("KashiIree_DIR")
        or cache.get("KASHI_MUTATION") == "ON"
        or cache.get("KASHI_COVERAGE") == "ON"
    ):
        raise ValueError("Build the SDK from an uninstrumented source build")
    graph = (build / "build.ninja").read_text()
    native = rule(graph, r"_native\S*\.so: CXX_MODULE_LIBRARY_LINKER_")
    if shlex.split(native.get("ARCH_FLAGS", ""))[:2] != [
        "-arch",
        "arm64",
    ] or "-bundle" not in native.get("LINK_FLAGS", ""):
        raise ValueError("SDK packaging currently supports macOS arm64 only")
    core = rule(graph, r"CMakeFiles/kashi_core.dir/.*iree_partition\.cpp\.o: ")
    compiler_plugin = rule(
        graph, r"iree/.*kashi_iree_plugin.dir/.*registration\.cpp\.o: "
    )
    tests = rule(graph, r"kashi_native_tests .*: CXX_EXECUTABLE_LINKER_")
    test_compile = rule(graph, r"CMakeFiles/kashi_native_tests.dir/.*\.o: ")
    generated = build / "iree"
    source = next(
        path
        for path in (
            Path(t[2:])
            for t in shlex.split(compiler_plugin["INCLUDES"])
            if t.startswith("-I")
        )
        if (path / "third_party/llvm-project/llvm/include").is_dir()
    )
    revision = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()
    module = next(build.glob("_native*.so"))
    metadata = subprocess.check_output(
        ["xcrun", "vtool", "-show-build", module], text=True
    )
    deployment = re.search(r"^\s*minos\s+(\S+)", metadata, re.MULTILINE)[1]

    with tempfile.TemporaryDirectory(prefix="kashi-sdk-") as temp:
        stage = Path(temp) / "iree-sdk"
        roots = {source: "source", generated: "generated"}
        for root, name in roots.items():
            copy_headers(root, stage / name)
        config = [
            "include_guard(GLOBAL)",
            f'set(KASHI_SDK_IREE_REVISION "{revision}")',
            f'set(KASHI_SDK_MACOS_DEPLOYMENT_TARGET "{deployment}")',
            'set(KASHI_IREE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/source")',
            "find_library(KASHI_SDK_LZMA NAMES lzma REQUIRED)",
        ]
        for name, compile_rule in (
            ("core", core),
            ("plugin", compiler_plugin),
            ("tests", test_compile),
        ):
            config.append(f"add_library(KashiIree::{name} INTERFACE IMPORTED)")
            includes = []
            args = iter(
                shlex.split(compile_rule["INCLUDES"] + " " + compile_rule["FLAGS"])
            )
            for arg in args:
                if arg != "-isystem" and not arg.startswith("-I"):
                    continue
                path = Path(next(args) if arg in {"-I", "-isystem"} else arg[2:])
                for root, label in roots.items():
                    if path.is_relative_to(root):
                        relative = Path(label) / path.relative_to(root)
                        (stage / relative).mkdir(parents=True, exist_ok=True)
                        includes.append(f'"${{CMAKE_CURRENT_LIST_DIR}}/{relative}"')
                        break
            config.append(
                f"target_include_directories(KashiIree::{name} INTERFACE {' '.join(dict.fromkeys(includes))})"
            )
            definitions = " ".join(
                t[2:]
                for t in shlex.split(compile_rule.get("DEFINES", ""))
                if t.startswith("-DIREE_")
            )
            if definitions:
                config.append(
                    f"target_compile_definitions(KashiIree::{name} INTERFACE {definitions})"
                )

        for name, link_rule in (("native_link", native), ("test_link", tests)):
            config += [
                f"add_library(KashiIree::{name} INTERFACE IMPORTED)",
                f"target_link_libraries(KashiIree::{name} INTERFACE",
            ]
            for item in shlex.split(link_rule["LINK_LIBRARIES"]):
                if item in {PLUGIN, "libkashi_core.a", "libnanobind-static.a"} or Path(
                    item
                ).name.startswith("libCatch2"):
                    continue
                if item.startswith("iree/") and item.endswith(".a"):
                    destination = stage / item
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(build / item, destination)
                    config.append(f'  "${{CMAKE_CURRENT_LIST_DIR}}/{item}"')
                elif item.endswith("/liblzma.tbd"):
                    config.append('  "${KASHI_SDK_LZMA}"')
                elif item == "-lm":
                    config.append("  -lm")
                else:
                    raise ValueError(f"Unrecognized link dependency: {item}")
            config.append(")")
        (stage / "KashiIreeConfig.cmake").write_text("\n".join(config) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "--zstd", "-cf", output, "-C", temp, stage.name], check=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "build",
        type=Path,
        help="Release Ninja build directory with native tests enabled",
    )
    parser.add_argument("archive", type=Path, help="Output .tar.zst archive")
    args = parser.parse_args()
    package(args.build.resolve(), args.archive.resolve())


if __name__ == "__main__":
    main()
