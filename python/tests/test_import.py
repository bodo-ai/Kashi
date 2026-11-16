"""Smoke tests for the installed Python package and native extension."""


def test_package_imports():
    import kashi


def test_native_extension_imports():
    import kashi._native
