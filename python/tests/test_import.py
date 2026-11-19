"""Smoke tests for the installed Python package and native extension."""

from importlib import import_module


def test_package_imports():
    import_module("kashi")


def test_native_extension_imports():
    import_module("kashi._native")
