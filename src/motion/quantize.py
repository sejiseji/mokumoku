from __future__ import annotations


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def quantize_signed(value: float, scale: float, limit: int) -> int:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return clamp_int(int(round(value * scale)), -limit, limit)


def quantize_unit(value: float, levels: int) -> int:
    if levels <= 1:
        raise ValueError("levels must be greater than one")
    return clamp_int(int(round(value * (levels - 1))), 0, levels - 1)


def pack_signed(value: int) -> int:
    return value & 0xFF


def unpack_signed(value: int) -> int:
    return value - 256 if value >= 128 else value

