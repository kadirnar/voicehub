"""Fixed-width arithmetic used by the native WebRTC VAD port.

The reference implementation is written in C and deliberately relies on
two's-complement casts at a few DSP boundaries.  Keeping those
operations explicit makes the Python implementation both readable and
bit-exact.
"""

from __future__ import annotations


def int16(value: int) -> int:
    """Return ``value`` with C ``int16_t`` wrapping semantics."""
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def int32(value: int) -> int:
    """Return ``value`` with C ``int32_t`` wrapping semantics."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def uint32(value: int) -> int:
    """Return ``value`` with C ``uint32_t`` wrapping semantics."""
    return value & 0xFFFFFFFF


def saturate_int16(value: int) -> int:
    """Clamp an integer to the signed 16-bit range."""
    return max(-32768, min(32767, value))


def divide_int32_by_int16(numerator: int, denominator: int) -> int:
    """Match C integer division, which truncates toward zero."""
    if denominator == 0:
        return 0x7FFFFFFF
    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        quotient = -quotient
    return int32(quotient)


def norm_int32(value: int) -> int:
    """Count safe left shifts using ``WebRtcSpl_NormW32`` semantics."""
    value = int32(value)
    if value == 0:
        return 0
    normalized = (~value) & 0xFFFFFFFF if value < 0 else value
    return 31 - normalized.bit_length()


def norm_uint32(value: int) -> int:
    """Count leading zeroes using ``WebRtcSpl_NormU32`` semantics."""
    value = uint32(value)
    return 0 if value == 0 else 32 - value.bit_length()


__all__ = [
    "divide_int32_by_int16",
    "int16",
    "int32",
    "norm_int32",
    "norm_uint32",
    "saturate_int16",
    "uint32",
]
