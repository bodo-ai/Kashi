"""Package a Release Ninja build of Kashi's pinned IREE dependency."""

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


def copy_headers(source, destination, excluded_dir):
    for folder, dirs, files in os.walk(source):
        dirs[:] = [
            name
            for name in dirs
            if name not in {".git", "CMakeFiles"}
            and Path(folder) / name != excluded_dir
        ]
        for name in files:
            path = Path(folder) / name
            if path.suffix in HEADER_SUFFIXES or name.startswith(("LICENSE", "NOTICE")):
                output = destination / path.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)


def platform_metadata(build, native):
    module = next(build.glob("_native*.so"))
    with module.open("rb") as stream:
        header = stream.read(20)
    if header.startswith(b"\x7fELF"):
        # ELF64 little-endian, EM_X86_64.
        if header[4:6] != b"\x02\x01" or header[18:20] != b"\x3e\x00":
            raise ValueError("Linux SDK packaging requires an x86-64 build")
        return "Linux", "x86_64", None
    flags = shlex.split(native.get("ARCH_FLAGS", ""))
    architectures = [flags[i + 1] for i, flag in enumerate(flags) if flag == "-arch"]
    if architectures != ["arm64"] or "-bundle" not in native.get("LINK_FLAGS", ""):
        raise ValueError("Expected macOS arm64 or Linux x86-64")
    metadata = subprocess.check_output(
        ["xcrun", "vtool", "-show-build", module], text=True
    )
    deployment = re.search(r"^\s*minos\s+(\S+)", metadata, re.MULTILINE)[1]
    return "Darwin", "arm64", deployment


def system_library(item, dependencies):
    name = Path(item).name
    if name == "liblzma.tbd":
        dependencies[name] = "find_library(KASHI_SDK_LZMA NAMES lzma REQUIRED)"
        return "${KASHI_SDK_LZMA}"
    if Path(item).is_absolute() and re.fullmatch(
        r"lib[\w+.-]+\.(?:a|so(?:\.[\d.]+)?)", name
    ):
        variable = "KASHI_SDK_" + re.sub(r"\W", "_", name)
        # Look up the same library filename (including any SONAME), never the
        # producer's sysroot path or a different shared/static variant.
        dependencies[name] = (
            f'find_file({variable} NAMES "{name}" '
            "HINTS ${CMAKE_LIBRARY_PATH} ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES} "
            "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} "
            "PATHS ENV CONDA_PREFIX PATH_SUFFIXES lib lib64 REQUIRED)"
        )
        return "${" + variable + "}"
    if re.fullmatch(r"-l[\w+.-]+|-pthread", item):
        return item
    raise ValueError(f"Unrecognized link dependency: {item}")


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
        cache.get("KASHI_IREE_SDK_DIR")
        or cache.get("KASHI_MUTATION") == "ON"
        or cache.get("KASHI_COVERAGE") == "ON"
    ):
        raise ValueError("Build the SDK from an uninstrumented source build")
    graph = (build / "build.ninja").read_text()
    native = rule(graph, r"_native\S*\.so: CXX_MODULE_LIBRARY_LINKER_")
    system, arch, deployment = platform_metadata(build, native)
    cuda = cache["IREE_TARGET_BACKEND_CUDA"]
    if cuda not in {"ON", "OFF"} or (cuda == "ON" and system != "Linux"):
        raise ValueError(f"Unexpected CUDA setting for {system}: {cuda}")
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

    cuda_notice = Path(cache["KASHI_CUDA_NOTICE"]) if cuda == "ON" else None
    cuda_toolkit = cuda_notice.parent if cuda_notice else None
    with tempfile.TemporaryDirectory(prefix="kashi-sdk-") as temp:
        stage = Path(temp) / "iree-sdk"
        roots = {source: "source", generated: "generated"}
        for root, name in roots.items():
            copy_headers(root, stage / name, cuda_toolkit)
        if cuda_notice:
            notice = stage / "licenses/CUDA.txt"
            notice.parent.mkdir()
            shutil.copy2(cuda_notice, notice)
        config = [
            "include_guard(GLOBAL)",
            f'set(KASHI_SDK_IREE_REVISION "{revision}")',
            f'set(KASHI_SDK_SYSTEM "{system}")',
            f'set(KASHI_SDK_ARCH "{arch}")',
            f'set(KASHI_SDK_CUDA "{cuda}")',
            'set(KASHI_IREE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/source")',
        ]
        if deployment:
            config.append(f'set(KASHI_SDK_MACOS_DEPLOYMENT_TARGET "{deployment}")')
        dependencies = {}
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
                # CMake derives a new runtime search path from the resolved
                # shared libraries; never preserve the producer's RPATH.
                if item.startswith("-Wl,-rpath,"):
                    continue
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
                else:
                    dependency = system_library(item, dependencies)
                    config.append(f'  "{dependency}"')
            config.append(")")
        config[1:1] = dependencies.values()
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
