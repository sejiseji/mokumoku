from __future__ import annotations

import sys
from pathlib import Path

import pyxel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESOURCE_PATH = PROJECT_ROOT / "assets" / "mokumoku.pyxres"


def stamp_pixel_rows(
    image,
    *,
    target_u: int,
    target_v: int,
    rows: tuple[str, ...],
) -> None:
    for local_y, row in enumerate(rows):
        for local_x, char in enumerate(row):
            image.pset(
                target_u + local_x,
                target_v + local_y,
                int(char, 16),
            )


def generate_resource() -> Path:
    from src.assets.cloud_sprite_text import (
        CLOUD_SPRITE_TEXT,
        validate_sprite_catalog,
    )
    from src.assets.sprite_map import (
        CLOUD_FAMILY_ORDER,
        CLOUD_SIZE_ORDER,
        CLOUD_SPRITE_VARIANT_COUNT,
        cloud_sprite_rect,
    )

    validate_sprite_catalog()
    RESOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    pyxel.init(
        256,
        256,
        title="mokumoku resource generator",
        headless=True,
    )

    for variant in range(CLOUD_SPRITE_VARIANT_COUNT):
        pyxel.images[variant].cls(0)

    for family in CLOUD_FAMILY_ORDER:
        for size_class in CLOUD_SIZE_ORDER:
            variants = CLOUD_SPRITE_TEXT[(family, size_class)]

            for variant in range(CLOUD_SPRITE_VARIANT_COUNT):
                rect = cloud_sprite_rect(
                    family,
                    size_class,
                    variant,
                )
                stamp_pixel_rows(
                    pyxel.images[rect.image],
                    target_u=rect.u,
                    target_v=rect.v,
                    rows=variants.at(variant),
                )

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
