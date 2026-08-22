from __future__ import annotations

import sys
from pathlib import Path

import pyxel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESOURCE_PATH = PROJECT_ROOT / "assets" / "mokumoku.pyxres"


def draw_blob(image, x: int, y: int, size: int, family, variant: int = 0) -> None:
    from src.assets.sprite_map import CloudSpriteFamily

    cx = x + size // 2
    cy = y + size // 2
    scale = size / 16.0

    def unit(value: float) -> int:
        return int(round(value * scale))

    def puff(px: int, py: int, radius: int, color: int) -> None:
        image.circ(cx + unit(px), cy + unit(py), max(1, unit(radius)), color)

    def cut(px: int, py: int, radius: int) -> None:
        puff(px, py, radius, 0)

    def mark(px: int, py: int, color: int) -> None:
        image.pset(cx + unit(px), cy + unit(py), color)

    def lumpy(
        base: int,
        light: int = 7,
        accent: int = 8,
        puffs: tuple[tuple[int, int, int], ...] = (),
        cuts: tuple[tuple[int, int, int], ...] = (),
    ) -> None:
        puffs = morph_puffs(puffs, variant)
        cuts = morph_cuts(cuts, variant)
        for px, py, radius in puffs:
            puff(px, py, radius + 1, base)
        for px, py, radius in puffs:
            puff(px - 1, py - 1, radius, light)
        for px, py, radius in cuts:
            cut(px, py, radius)
        mark(-3, -3, accent)

    def morph_puffs(
        puffs: tuple[tuple[int, int, int], ...],
        variant: int,
    ) -> tuple[tuple[int, int, int], ...]:
        if variant == 1:
            return tuple(
                (px + (1 if index % 2 == 0 else 0), py - (1 if index % 3 == 0 else 0), radius)
                for index, (px, py, radius) in enumerate(puffs)
            )
        if variant == 2:
            return tuple(
                (px - (1 if index % 2 == 1 else 0), py + (1 if index % 3 == 1 else 0), radius)
                for index, (px, py, radius) in enumerate(puffs)
            )
        return puffs

    def morph_cuts(
        cuts: tuple[tuple[int, int, int], ...],
        variant: int,
    ) -> tuple[tuple[int, int, int], ...]:
        if variant == 1:
            return tuple((px + 1, py, radius) for px, py, radius in cuts)
        if variant == 2:
            return tuple((px - 1, py + 1, radius) for px, py, radius in cuts)
        return cuts

    if family is CloudSpriteFamily.INTERNAL:
        lumpy(
            6,
            7,
            8,
            puffs=((-5, 1, 4), (-1, -3, 5), (5, -1, 3), (2, 4, 4), (-4, 5, 3)),
            cuts=((7, 4, 2), (-7, -4, 2)),
        )
    elif family is CloudSpriteFamily.EDGE:
        lumpy(
            5,
            7,
            6,
            puffs=((-6, 0, 4), (-1, -4, 4), (5, -2, 3), (3, 4, 4), (-5, 5, 3)),
            cuts=((7, 2, 3), (4, -6, 2)),
        )
    elif family is CloudSpriteFamily.BOTTOM:
        lumpy(
            5,
            7,
            4,
            puffs=((-6, 1, 4), (-1, -3, 4), (5, 0, 3), (-3, 4, 3), (4, 4, 3)),
            cuts=((-7, -5, 2), (7, -4, 2)),
        )
        image.line(cx + unit(-7), cy + unit(6), cx + unit(7), cy + unit(6), 4)
    elif family is CloudSpriteFamily.UPDRAFT:
        lumpy(
            6,
            7,
            8,
            puffs=((-5, 2, 4), (-1, -4, 4), (2, -7, 3), (5, 1, 3), (0, 4, 4)),
            cuts=((-8, -3, 2), (7, 5, 2)),
        )
        mark(1, -7, 15)
    elif family is CloudSpriteFamily.STRETCH:
        lumpy(
            6,
            7,
            8,
            puffs=((-7, 1, 3), (-3, -2, 4), (2, -1, 4), (7, 1, 3), (1, 4, 3)),
            cuts=((-8, -4, 2), (8, 4, 2)),
        )
    elif family is CloudSpriteFamily.FRAGMENT:
        puff(-3, 0, 3, 5)
        puff(2, 1, 3, 6)
        puff(-1, -4, 2, 7)
        cut(5, -3, 2)
    elif family is CloudSpriteFamily.FADE:
        puff(-3, 0, 3, 6)
        puff(2, 1, 2, 5)
        mark(-2, -1, 7)
        mark(4, 3, 5)
    elif family is CloudSpriteFamily.SERENDIPITY:
        lumpy(
            7,
            8,
            15,
            puffs=((-5, 1, 4), (-1, -4, 4), (5, -1, 3), (1, 4, 4)),
            cuts=((7, 4, 2),),
        )
    elif family is CloudSpriteFamily.CHARGE:
        lumpy(
            6,
            7,
            10,
            puffs=((-5, 1, 4), (-1, -3, 4), (5, 0, 3), (1, 4, 4)),
            cuts=((-7, -4, 2),),
        )
        image.line(cx + unit(-2), cy + unit(-6), cx + unit(2), cy + unit(-1), 10)
        image.line(cx + unit(2), cy + unit(-1), cx + unit(-1), cy + unit(0), 10)
        image.line(cx + unit(-1), cy + unit(0), cx + unit(3), cy + unit(5), 10)


def generate_resource() -> Path:
    from src.assets.sprite_map import (
        CLOUD_SIZE_ORDER,
        CLOUD_SPRITE_VARIANT_COUNT,
        CloudSpriteFamily,
        cloud_sprite_rect,
    )

    RESOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pyxel.init(256, 256, title="mokumoku resource generator", headless=True)
    for variant in range(CLOUD_SPRITE_VARIANT_COUNT):
        pyxel.images[variant].cls(0)

    for variant in range(CLOUD_SPRITE_VARIANT_COUNT):
        image = pyxel.images[variant]
        for family in CloudSpriteFamily:
            for size_class in CLOUD_SIZE_ORDER:
                rect = cloud_sprite_rect(family, size_class, variant)
                draw_blob(image, rect.u, rect.v, rect.width, family, variant)

    pyxel.save(
        str(RESOURCE_PATH),
        exclude_tilemaps=True,
        exclude_sounds=True,
        exclude_musics=True,
    )
    return RESOURCE_PATH


def main() -> int:
    output = generate_resource()
    print(f"wrote {output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
