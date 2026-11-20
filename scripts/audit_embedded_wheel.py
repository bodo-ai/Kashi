"""Audit the repaired wheel before it is considered publishable."""

import argparse
import subprocess
import tempfile
import zipfile
from pathlib import Path


def _output(*args):
    """Run a platform binary inspection tool and return its text output."""
    return subprocess.check_output(args, text=True)


def audit(wheel: Path, limit_mib: int):
    """Reject repaired wheels that violate the packaging or upload contract."""
    linux = "manylinux" in wheel.name and "x86_64" in wheel.name
    mac = "macosx" in wheel.name and "arm64" in wheel.name
    assert linux or mac, f"unexpected repaired platform tag: {wheel.name}"
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as directory:
        names = archive.namelist()
        # Installers relocate these two wheel data schemes into site-packages.
        for name in names:
            root, _, remainder = name.partition("/")
            scheme, _, relative = remainder.partition("/")
            installed = (
                relative
                if root.endswith(".data") and scheme in {"platlib", "purelib"}
                else name
            )
            assert not installed.startswith("iree/"), f"{name} overlaps stock IREE"
        forbidden = ("libireecompiler", "iree-compile", ".tar.gz", ".tar.zst")
        assert not any(item in name.lower() for item in forbidden for name in names)
        notices = {"IREE", "LLVM", "MLIR", "Torch-MLIR", "FlatCC"}
        if linux:
            notices.add("CUDA")
        assert {f"kashi/licenses/{name}.txt" for name in notices} <= set(names)
        (extension,) = [
            name
            for name in names
            if name.startswith("kashi/_native") and name.endswith(".so")
        ]
        (metadata,) = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert (
            "Requires-Dist: iree-base-compiler==3.14.0"
            in archive.read(metadata).decode()
        )
        path = str(archive.extract(extension, directory))
        if linux:
            dependencies = _output("readelf", "-d", path)
            versions = _output("readelf", "--version-info", path)
            assert "GLIBC_ABI_GNU2_TLS" not in versions
            symbols = _output("nm", "-D", "-C", path)
            exports = _output("nm", "-D", "--defined-only", path)
        else:
            dependencies = _output("otool", "-L", path)
            symbols = _output("nm", "-gu", "-C", path)
            exports = _output("nm", "-gU", path)
        print(dependencies)
        assert "libireecompiler" not in dependencies.lower()
        assert not any(
            name in symbols for name in ("ireeCompiler", "llvm::", "mlir::")
        ), symbols
        expected = "PyInit__native" if linux else "_PyInit__native"
        assert [line.split()[-1] for line in exports.splitlines() if line.strip()] == [
            expected
        ], exports
    size = wheel.stat().st_size
    print(f"{wheel.name}: {size / 2**20:.2f} MiB ({size} bytes)")
    assert size <= limit_mib * 2**20, (
        "wheel exceeds the configured PyPI project upload limit"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--limit-mib", type=int, default=100)
    args = parser.parse_args()
    audit(args.wheel, args.limit_mib)
