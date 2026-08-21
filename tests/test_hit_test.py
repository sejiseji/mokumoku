from __future__ import annotations

import unittest

from src import config
from src.camera.camera import build_camera_basis
from src.camera.projection import project_point
from src.input.hit_test import HitTarget, collect_hit_candidates, hit_test
from src.math3d import Vec3


class HitTestTests(unittest.TestCase):
    def test_hit_test_chooses_frontmost_overlapping_target(self) -> None:
        camera = build_camera_basis(config.CAMERA_FRONT_YAW)
        far = HitTarget(stable_id=1, position=Vec3(0.0, 120.0, 0.0), radius=10.0)
        near = HitTarget(
            stable_id=2,
            position=far.position - camera.forward * 20.0,
            radius=10.0,
        )
        projection = project_point(far.position, camera)

        hit = hit_test(projection.screen_x, projection.screen_y, [far, near], camera)

        self.assertIsNotNone(hit)
        self.assertEqual(hit.target.stable_id, 2)

    def test_previous_selection_wins_when_depths_are_close(self) -> None:
        camera = build_camera_basis(config.CAMERA_FRONT_YAW)
        base_depth = 430.0
        previous = HitTarget(
            stable_id=1,
            position=camera.position + camera.forward * base_depth,
            radius=10.0,
        )
        closer = HitTarget(
            stable_id=2,
            position=camera.position + camera.forward * (base_depth - 1.0),
            radius=10.0,
        )
        projection = project_point(previous.position, camera)

        hit = hit_test(
            projection.screen_x,
            projection.screen_y,
            [previous, closer],
            camera,
            previous_selected_id=1,
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit.target.stable_id, 1)

    def test_candidates_exclude_points_outside_touch_radius(self) -> None:
        camera = build_camera_basis(config.CAMERA_FRONT_YAW)
        target = HitTarget(stable_id=1, position=Vec3(0.0, 100.0, 0.0), radius=6.0)
        projection = project_point(target.position, camera)

        candidates = collect_hit_candidates(
            projection.screen_x + 200.0,
            projection.screen_y,
            [target],
            camera,
        )

        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
