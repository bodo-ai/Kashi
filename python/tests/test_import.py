"""Smoke tests for the installed Python package and native extension."""


def test_package_imports():
    import ml_compiler


def test_native_extension_imports():
    import ml_compiler._native
