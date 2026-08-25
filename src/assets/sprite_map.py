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


@dataclass(frozen=True, slots=True)
class SpriteRect:
    image: int
    u: int
    v: int
    width: int
    height: int
    colkey: int = 0


CLOUD_FAMILY_ORDER = (
    CloudSpriteFamily.INTERNAL,
    CloudSpriteFamily.EDGE,
    CloudSpriteFamily.BOTTOM,
    CloudSpriteFamily.UPDRAFT,
    CloudSpriteFamily.STRETCH,
    CloudSpriteFamily.FRAGMENT,
    CloudSpriteFamily.FADE,
    CloudSpriteFamily.SERENDIPITY,
    CloudSpriteFamily.CHARGE,
)

CLOUD_SIZE_ORDER = ("s", "m", "l", "xl")
CLOUD_SIZE_PIXELS = {
    "s": 16,
    "m": 24,
    "l": 32,
    "xl": 40,
}
CLOUD_SPRITE_VARIANT_COUNT = 3

CLOUD_ATLAS_CELL_SIZE = 40
CLOUD_ATLAS_COLUMNS = 6
CLOUD_ATLAS_ROWS = 6
CLOUD_ATLAS_SLOT_COUNT = CLOUD_ATLAS_COLUMNS * CLOUD_ATLAS_ROWS

_REQUIRED_SLOT_COUNT = len(CLOUD_FAMILY_ORDER) * len(CLOUD_SIZE_ORDER)
if _REQUIRED_SLOT_COUNT > CLOUD_ATLAS_SLOT_COUNT:
    raise RuntimeError(
        "cloud atlas overflow: "
        f"required={_REQUIRED_SLOT_COUNT}, available={CLOUD_ATLAS_SLOT_COUNT}"
    )


def cloud_sprite_slot(
    family: CloudSpriteFamily,
    size_class: str,
) -> int:
    if family not in CLOUD_FAMILY_ORDER:
        raise ValueError(f"unknown cloud sprite family: {family}")
    if size_class not in CLOUD_SIZE_PIXELS:
        raise ValueError(f"unknown cloud size class: {size_class}")

    family_index = CLOUD_FAMILY_ORDER.index(family)
    size_index = CLOUD_SIZE_ORDER.index(size_class)
    return family_index * len(CLOUD_SIZE_ORDER) + size_index


def cloud_sprite_rect(
    family: CloudSpriteFamily,
    size_class: str,
    variant: int = 0,
) -> SpriteRect:
    if not 0 <= variant < CLOUD_SPRITE_VARIANT_COUNT:
        raise ValueError(f"unknown cloud sprite variant: {variant}")

    slot_index = cloud_sprite_slot(family, size_class)
    cell_column = slot_index % CLOUD_ATLAS_COLUMNS
    cell_row = slot_index // CLOUD_ATLAS_COLUMNS

    cell_u = cell_column * CLOUD_ATLAS_CELL_SIZE
    cell_v = cell_row * CLOUD_ATLAS_CELL_SIZE

    sprite_size = CLOUD_SIZE_PIXELS[size_class]
    center_padding = (CLOUD_ATLAS_CELL_SIZE - sprite_size) // 2

    return SpriteRect(
        image=variant,
        u=cell_u + center_padding,
        v=cell_v + center_padding,
        width=sprite_size,
        height=sprite_size,
    )


def size_class_for_screen_radius(radius: float) -> str:
    if radius < 10.0:
        return "s"
    if radius < 15.0:
        return "m"
    if radius < 20.0:
        return "l"
    return "xl"
