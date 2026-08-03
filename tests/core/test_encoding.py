"""Tests for yugioh_core.encoding's multi-byte decoders.

The observation's encoding fields are numpy uint8 arrays, so every decoder
here is fed numpy scalars in practice. `hi << 8` on a numpy uint8 wraps mod
256 instead of widening, which would silently truncate the high byte to 0 --
the decoders must int()-cast before shifting.
"""

from __future__ import annotations

import numpy as np

from yugioh_core.encoding import decode_u16, decode_u32


def test_decode_u16_widens_numpy_uint8_instead_of_wrapping():
    """A bare `arr[1] << 8` on np.uint8 yields 0; 8000 proves the widening."""
    arr = np.array([0x40, 0x1F], dtype=np.uint8)
    assert decode_u16(arr, 0) == 8000


def test_decode_u16_plain_python_ints():
    assert decode_u16([0x40, 0x1F], 0) == 8000
    assert decode_u16([0, 0], 0) == 0
    assert decode_u16([0xFF, 0xFF], 0) == 0xFFFF


def test_decode_u16_reads_from_the_given_offset():
    assert decode_u16([0, 0, 0x40, 0x1F], 2) == 8000


def test_decode_u32_widens_numpy_uint8_instead_of_wrapping():
    arr = np.array([0xA3, 0xA9, 0x57, 0x05], dtype=np.uint8)  # 89631139 LE
    assert decode_u32(arr, 0) == 89631139
