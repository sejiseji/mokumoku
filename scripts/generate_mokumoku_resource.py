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
    r = max(3, size // 4)

    def puff(px: int, py: int, radius: int, color: int) -> None:
        image.circ(cx + px, cy + py, max(1, radius), color)

    def lumpy(base: int, light: int = 7, accent: int = 8, stretch: int = 0) -> None:
        puff(-r - 1 - stretch, 1, r + 1, base)
        puff(0, -r // 2, r + 2, base)
        puff(r + stretch, 1, r + 1, base)
        puff(-r // 2, r // 2 + 1, r + 1, base)
        puff(r // 2 + stretch, r // 2 + 1, r, base)
        puff(-r - 1 - stretch, 0, r, light)
        puff(0, -r // 2 - 1, r + 1, light)
        puff(r - 1 + stretch, 0, r, light)
        puff(-r // 2, r // 2, r, light)
        image.pset(cx - r // 2, cy - r // 2, accent)

    if family is CloudSpriteFamily.INTERNAL:
        lumpy(6, 7, 8)
    elif family is CloudSpriteFamily.EDGE:
        lumpy(5, 7, 6, stretch=1)
    elif family is CloudSpriteFamily.BOTTOM:
        lumpy(5, 7, 4)
        image.line(cx - r - 3, cy + r + 1, cx + r + 4, cy + r + 1, 4)
    elif family is CloudSpriteFamily.UPDRAFT:
        lumpy(6, 7, 8)
        puff(0, -r - 3, max(2, r - 1), 8)
    elif family is CloudSpriteFamily.STRETCH:
        lumpy(6, 7, 8, stretch=2)
    elif family is CloudSpriteFamily.FRAGMENT:
        puff(-r // 2, -1, max(2, r - 1), 5)
        puff(r // 2, 1, max(2, r - 2), 6)
        puff(-1, -r // 2, max(2, r - 2), 7)
    elif family is CloudSpriteFamily.FADE:
        puff(-r // 2, 0, max(2, r - 1), 6)
        puff(r // 2, 1, max(2, r - 2), 5)
        image.pset(cx - 2, cy - 1, 7)
    elif family is CloudSpriteFamily.SERENDIPITY:
        lumpy(7, 8, 15)
    elif family is CloudSpriteFamily.CHARGE:
        lumpy(6, 7, 10)
        image.line(cx - 2, cy - r - 1, cx + 2, cy - 1, 10)
        image.line(cx + 2, cy - 1, cx - 1, cy, 10)
        image.line(cx - 1, cy, cx + 3, cy + r, 10)


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
