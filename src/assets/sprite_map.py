from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CloudSpriteFamily(Enum):
    INTERNAL = "internal"
    EDGE = "edge"
    BOTTOM = "bottom"
    UPDRAFT = "updraft"
    STRETCH = "stretch"
    FRAGMENT = "fragment"
    FADE = "fade"
    SERENDIPITY = "serendipity"
    CHARGE = "charge"


@dataclass(frozen=True)
class SpriteRect:
    image: int
    u: int
    v: int
    width: int
    height: int
    colkey: int = 0


CLOUD_SIZE_ORDER = ("s", "m", "l", "xl")
CLOUD_SIZE_PIXELS = {
    "s": 16,
    "m": 24,
    "l": 32,
    "xl": 40,
}


def cloud_sprite_rect(family: CloudSpriteFamily, size_class: str) -> SpriteRect:
    if size_class not in CLOUD_SIZE_PIXELS:
        raise ValueError(f"unknown cloud size class: {size_class}")
    size_index = CLOUD_SIZE_ORDER.index(size_class)
    family_index = list(CloudSpriteFamily).index(family)
    size = CLOUD_SIZE_PIXELS[size_class]
    return SpriteRect(
        image=0,
        u=size_index * 48,
        v=family_index * 48,
        width=size,
        height=size,
    )


def size_class_for_screen_radius(radius: float) -> str:
    if radius < 8.0:
        return "s"
    if radius < 13.0:
        return "m"
    if radius < 18.0:
        return "l"
    return "xl"
