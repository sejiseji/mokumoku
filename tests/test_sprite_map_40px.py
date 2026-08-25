from __future__ import annotations

import unittest

from src.assets.sprite_map import (
    CLOUD_ATLAS_CELL_SIZE,
    CLOUD_ATLAS_COLUMNS,
    CLOUD_ATLAS_ROWS,
    CLOUD_FAMILY_ORDER,
    CLOUD_SIZE_ORDER,
    CLOUD_SIZE_PIXELS,
    CLOUD_SPRITE_VARIANT_COUNT,
    cloud_sprite_rect,
    cloud_sprite_slot,
)


class SpriteMap40pxTests(unittest.TestCase):
    def test_cloud_atlas_uses_exactly_36_unique_slots(self) -> None:
        slots = {
            cloud_sprite_slot(family, size_class)
            for family in CLOUD_FAMILY_ORDER
            for size_class in CLOUD_SIZE_ORDER
        }
        self.assertEqual(slots, set(range(36)))

    def test_cloud_sprite_rects_fit_inside_256_image_bank(self) -> None:
        for variant in range(CLOUD_SPRITE_VARIANT_COUNT):
            for family in CLOUD_FAMILY_ORDER:
                for size_class in CLOUD_SIZE_ORDER:
                    rect = cloud_sprite_rect(family, size_class, variant)
                    self.assertEqual(rect.image, variant)
                    self.assertGreaterEqual(rect.u, 0)
                    self.assertGreaterEqual(rect.v, 0)
                    self.assertLess(rect.u, 256)
                    self.assertLess(rect.v, 256)
                    self.assertLessEqual(rect.u + rect.width, 256)
                    self.assertLessEqual(rect.v + rect.height, 256)

    def test_each_sprite_is_centered_inside_40px_cell(self) -> None:
        for family in CLOUD_FAMILY_ORDER:
            for size_class in CLOUD_SIZE_ORDER:
                slot = cloud_sprite_slot(family, size_class)
                cell_column = slot % CLOUD_ATLAS_COLUMNS
                cell_row = slot // CLOUD_ATLAS_COLUMNS
                rect = cloud_sprite_rect(family, size_class)

                expected_size = CLOUD_SIZE_PIXELS[size_class]
                expected_padding = (CLOUD_ATLAS_CELL_SIZE - expected_size) // 2

                self.assertEqual(
                    rect.u,
                    cell_column * CLOUD_ATLAS_CELL_SIZE + expected_padding,
                )
                self.assertEqual(
                    rect.v,
                    cell_row * CLOUD_ATLAS_CELL_SIZE + expected_padding,
                )
                self.assertEqual(rect.width, expected_size)
                self.assertEqual(rect.height, expected_size)

    def test_atlas_grid_fits_in_pyxel_image_bank(self) -> None:
        self.assertLessEqual(CLOUD_ATLAS_COLUMNS * CLOUD_ATLAS_CELL_SIZE, 256)
        self.assertLessEqual(CLOUD_ATLAS_ROWS * CLOUD_ATLAS_CELL_SIZE, 256)


if __name__ == "__main__":
    unittest.main()
