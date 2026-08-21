from __future__ import annotations

import sys
from pathlib import Path

import pyxel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESOURCE_PATH = PROJECT_ROOT / "assets" / "mokumoku.pyxres"


def draw_blob(image, x: int, y: int, size: int, family) -> None:
    from src.assets.sprite_map import CloudSpriteFamily

    cx = x + size // 2
    cy = y + size // 2
    rx = max(3, size // 2 - 2)
    ry = max(3, int(rx * 0.68))

    if family is CloudSpriteFamily.INTERNAL:
        image.elli(cx - rx + 1, cy - ry, rx * 2 - 2, ry * 2, 6)
        image.elli(cx - rx // 2, cy - ry - 2, rx, ry + 4, 7)
        image.pset(cx - rx // 2, cy - ry // 2, 8)
    elif family is CloudSpriteFamily.EDGE:
        image.elli(cx - rx + 1, cy - ry + 1, rx * 2 - 2, ry * 2, 5)
        image.elli(cx - rx + 2, cy - ry, rx * 2 - 5, ry * 2 - 2, 7)
        image.ellib(cx - rx + 1, cy - ry + 1, rx * 2 - 2, ry * 2, 6)
    elif family is CloudSpriteFamily.BOTTOM:
        image.elli(cx - rx, cy - ry + 2, rx * 2, ry * 2, 5)
        image.elli(cx - rx + 2, cy - ry, rx * 2 - 4, ry + 4, 7)
        image.line(cx - rx + 3, cy + ry - 1, cx + rx - 3, cy + ry - 1, 4)
    elif family is CloudSpriteFamily.UPDRAFT:
        image.elli(cx - rx + 2, cy - ry + 3, rx * 2 - 4, ry * 2 - 2, 6)
        image.elli(cx - rx // 2, cy - ry - 3, rx, ry + 5, 8)
        image.line(cx, cy + ry - 1, cx, cy - ry + 1, 8)
    elif family is CloudSpriteFamily.STRETCH:
        image.elli(cx - rx, cy - ry // 2, rx * 2, ry, 6)
        image.elli(cx - rx + 2, cy - ry // 2 - 1, rx * 2 - 4, ry - 1, 7)
        image.pset(cx + rx - 4, cy, 8)
    elif family is CloudSpriteFamily.FRAGMENT:
        image.elli(cx - rx + 4, cy - ry + 3, rx * 2 - 8, ry * 2 - 5, 5)
        image.pset(cx - 2, cy - 2, 7)
        image.pset(cx + 2, cy, 6)
        image.pset(cx - 4, cy + 2, 6)
    elif family is CloudSpriteFamily.FADE:
        image.elli(cx - rx + 4, cy - ry + 4, rx * 2 - 8, ry * 2 - 8, 6)
        image.pset(cx - 3, cy - 1, 7)
        image.pset(cx + 4, cy + 2, 5)
    elif family is CloudSpriteFamily.SERENDIPITY:
        image.elli(cx - rx + 1, cy - ry, rx * 2 - 2, ry * 2, 7)
        image.ellib(cx - rx + 2, cy - ry + 1, rx * 2 - 4, ry * 2 - 2, 8)
        image.pset(cx, cy - ry + 2, 15)
    elif family is CloudSpriteFamily.CHARGE:
        image.line(cx - 2, cy - ry + 2, cx + 2, cy - 1, 10)
        image.line(cx + 2, cy - 1, cx - 1, cy, 10)
        image.line(cx - 1, cy, cx + 3, cy + ry - 3, 10)


def generate_resource() -> Path:
    from src.assets.sprite_map import CLOUD_SIZE_ORDER, CloudSpriteFamily, cloud_sprite_rect

    RESOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pyxel.init(256, 256, title="mokumoku resource generator", headless=True)
    image = pyxel.images[0]
    image.cls(0)

    for family in CloudSpriteFamily:
        for size_class in CLOUD_SIZE_ORDER:
            rect = cloud_sprite_rect(family, size_class)
            draw_blob(image, rect.u, rect.v, rect.width, family)

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
