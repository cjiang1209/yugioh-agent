"""Test that the OCG core library loads correctly."""

import ctypes


def test_library_loads(lib):
    """Library should load without errors."""
    assert lib is not None


def test_get_version(lib):
    """OCG_GetVersion should return valid version numbers."""
    major = ctypes.c_int()
    minor = ctypes.c_int()
    lib.OCG_GetVersion(ctypes.byref(major), ctypes.byref(minor))
    # edo9300 fork uses version 11.x
    assert major.value >= 1
    assert minor.value >= 0
