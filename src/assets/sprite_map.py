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
CLOUD_SPRITE_VARIANT_COUNT = 3


def cloud_sprite_rect(
    family: CloudSpriteFamily,
    size_class: str,
    variant: int = 0,
) -> SpriteRect:
    if size_class not in CLOUD_SIZE_PIXELS:
        raise ValueError(f"unknown cloud size class: {size_class}")
    if not 0 <= variant < CLOUD_SPRITE_VARIANT_COUNT:
        raise ValueError(f"unknown cloud sprite variant: {variant}")
    size_index = CLOUD_SIZE_ORDER.index(size_class)
    family_index = list(CloudSpriteFamily).index(family)
    size = CLOUD_SIZE_PIXELS[size_class]
    return SpriteRect(
        image=variant,
        u=size_index * 48,
        v=family_index * 48,
        width=size,
        height=size,
    )


def size_class_for_screen_radius(radius: float) -> str:
    if radius < 10.0:
        return "s"
    if radius < 15.0:
        return "m"
    if radius < 20.0:
        return "l"
    return "xl"
