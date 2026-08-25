from __future__ import annotations

import unittest

from src.assets.cloud_sprite_text import (
    CLOUD_SPRITE_TEXT,
    connected_component_count,
    opaque_bbox,
    opaque_centroid,
    opaque_positions,
    validate_sprite_catalog,
)
from src.assets.sprite_map import (
    CLOUD_FAMILY_ORDER,
    CLOUD_SIZE_ORDER,
    CLOUD_SIZE_PIXELS,
    CloudSpriteFamily,
)

PRIMARY_CLOUD_FAMILIES = (
    CloudSpriteFamily.INTERNAL,
    CloudSpriteFamily.EDGE,
    CloudSpriteFamily.BOTTOM,
    CloudSpriteFamily.UPDRAFT,
    CloudSpriteFamily.STRETCH,
)


class CloudSpriteTextTests(unittest.TestCase):
    def test_catalog_validation_passes(self) -> None:
        validate_sprite_catalog()

    def test_catalog_is_complete(self) -> None:
        self.assertEqual(
            set(CLOUD_SPRITE_TEXT),
            {
                (family, size_class)
                for family in CLOUD_FAMILY_ORDER
                for size_class in CLOUD_SIZE_ORDER
            },
        )

    def test_all_variants_keep_the_same_canvas_size(self) -> None:
        for family in CLOUD_FAMILY_ORDER:
            for size_class in CLOUD_SIZE_ORDER:
                expected_size = CLOUD_SIZE_PIXELS[size_class]
                variants = CLOUD_SPRITE_TEXT[(family, size_class)]

                for variant in range(3):
                    rows = variants.at(variant)
                    self.assertEqual(len(rows), expected_size)
                    self.assertTrue(all(len(row) == expected_size for row in rows))

    def test_variants_keep_visual_center_and_bbox_stable(self) -> None:
        for family in CLOUD_FAMILY_ORDER:
            for size_class in CLOUD_SIZE_ORDER:
                variants = CLOUD_SPRITE_TEXT[(family, size_class)]
                base_center = opaque_centroid(variants.variant_0)
                base_bbox = opaque_bbox(variants.variant_0)

                for variant in (1, 2):
                    center = opaque_centroid(variants.at(variant))
                    current_bbox = opaque_bbox(variants.at(variant))

                    self.assertLessEqual(abs(center[0] - base_center[0]), 0.35)
                    self.assertLessEqual(abs(center[1] - base_center[1]), 0.35)
                    self.assertTrue(
                        all(
                            abs(current - base) <= 1
                            for current, base in zip(
                                current_bbox,
                                base_bbox,
                                strict=True,
                            )
                        )
                    )

    def test_primary_cloud_variants_are_subtle_but_not_identical(self) -> None:
        for family in PRIMARY_CLOUD_FAMILIES:
            for size_class in CLOUD_SIZE_ORDER:
                variants = CLOUD_SPRITE_TEXT[(family, size_class)]
                base = variants.variant_0

                for variant in (1, 2):
                    rows = variants.at(variant)
                    changed_pixels = sum(
                        base_char != variant_char
                        for base_row, variant_row in zip(base, rows, strict=True)
                        for base_char, variant_char in zip(
                            base_row,
                            variant_row,
                            strict=True,
                        )
                    )
                    self.assertGreaterEqual(changed_pixels, 1)
                    self.assertLessEqual(changed_pixels, 6)

    def test_primary_cloud_shapes_are_single_connected_masses(self) -> None:
        for family in PRIMARY_CLOUD_FAMILIES:
            for size_class in CLOUD_SIZE_ORDER:
                variants = CLOUD_SPRITE_TEXT[(family, size_class)]

                for variant in range(3):
                    self.assertEqual(connected_component_count(variants.at(variant)), 1)

    def test_horizontal_cloud_families_are_wider_than_tall(self) -> None:
        families = (
            CloudSpriteFamily.INTERNAL,
            CloudSpriteFamily.EDGE,
            CloudSpriteFamily.BOTTOM,
            CloudSpriteFamily.STRETCH,
        )

        for family in families:
            for size_class in CLOUD_SIZE_ORDER:
                left, top, right, bottom = opaque_bbox(
                    CLOUD_SPRITE_TEXT[(family, size_class)].variant_0
                )
                width = right - left + 1
                height = bottom - top + 1
                self.assertGreaterEqual(width, height * 1.35)

    def test_updraft_keeps_width_while_growing_upward(self) -> None:
        for size_class in CLOUD_SIZE_ORDER:
            left, top, right, bottom = opaque_bbox(
                CLOUD_SPRITE_TEXT[(CloudSpriteFamily.UPDRAFT, size_class)].variant_0
            )
            width = right - left + 1
            height = bottom - top + 1

            self.assertGreaterEqual(width, height)
            self.assertLessEqual(width, height * 1.25)

    def test_internal_uses_only_transparent_body_and_highlight(self) -> None:
        for size_class in CLOUD_SIZE_ORDER:
            variants = CLOUD_SPRITE_TEXT[(CloudSpriteFamily.INTERNAL, size_class)]
            for variant in range(3):
                self.assertLessEqual(set("".join(variants.at(variant))), set("067"))

    def test_bottom_uses_deep_shadow_without_charge_colors(self) -> None:
        for size_class in CLOUD_SIZE_ORDER:
            variants = CLOUD_SPRITE_TEXT[(CloudSpriteFamily.BOTTOM, size_class)]
            for variant in range(3):
                used = set("".join(variants.at(variant)))
                self.assertIn("4", used)
                self.assertNotIn("A", used)
                self.assertNotIn("F", used)

    def test_charge_contains_only_a_small_charge_accent(self) -> None:
        for size_class in CLOUD_SIZE_ORDER:
            variants = CLOUD_SPRITE_TEXT[(CloudSpriteFamily.CHARGE, size_class)]
            for variant in range(3):
                rows = variants.at(variant)
                charge_pixels = sum(row.count("A") for row in rows)
                self.assertGreaterEqual(charge_pixels, 1)
                self.assertLessEqual(charge_pixels, 4)

    def test_fragment_is_visibly_smaller_than_internal(self) -> None:
        for size_class in CLOUD_SIZE_ORDER:
            internal = len(
                opaque_positions(
                    CLOUD_SPRITE_TEXT[(CloudSpriteFamily.INTERNAL, size_class)].variant_0
                )
            )
            fragment = len(
                opaque_positions(
                    CLOUD_SPRITE_TEXT[(CloudSpriteFamily.FRAGMENT, size_class)].variant_0
                )
            )
            self.assertLess(fragment, internal * 0.55)

    def test_fade_contains_transparent_holes_inside_opaque_bbox(self) -> None:
        for size_class in CLOUD_SIZE_ORDER:
            rows = CLOUD_SPRITE_TEXT[(CloudSpriteFamily.FADE, size_class)].variant_0
            left, top, right, bottom = opaque_bbox(rows)
            inner_zero_count = sum(
                rows[y][x] == "0"
                for y in range(top + 1, bottom)
                for x in range(left + 1, right)
            )
            self.assertGreaterEqual(inner_zero_count, 1)


if __name__ == "__main__":
    unittest.main()
