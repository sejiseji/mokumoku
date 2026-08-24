from __future__ import annotations

import unittest

from src import config
from src.camera.camera import (
    CameraController,
    CameraControlMode,
    CameraPreset,
    build_camera_basis,
)
from src.camera.interaction_plane import (
    depth_locked_drag_target,
    screen_to_world_at_depth,
    screen_to_world_in_cloud_bounds,
)
from src.camera.projection import camera_depth, project_point
from src.math3d import Vec3


class CameraProjectionTests(unittest.TestCase):
    def test_front_projection_maps_x_to_screen_right(self) -> None:
        camera = build_camera_basis(config.CAMERA_FRONT_YAW)
        center = project_point(Vec3(0.0, 0.0, 0.0), camera)
        right = project_point(Vec3(20.0, 0.0, 0.0), camera)

        self.assertTrue(center.visible)
        self.assertTrue(right.visible)
        self.assertGreater(right.screen_x, center.screen_x)

    def test_front_projection_maps_y_to_screen_up(self) -> None:
        camera = build_camera_basis(config.CAMERA_FRONT_YAW)
        center = project_point(Vec3(0.0, 0.0, 0.0), camera)
        higher = project_point(Vec3(0.0, 20.0, 0.0), camera)

        self.assertTrue(center.visible)
        self.assertTrue(higher.visible)
        self.assertLess(higher.screen_y, center.screen_y)

    def test_left_and_right_camera_swap_side_depth_order(self) -> None:
        left_camera = build_camera_basis(config.CAMERA_LEFT_YAW)
        right_camera = build_camera_basis(config.CAMERA_RIGHT_YAW)
        west = Vec3(-30.0, 0.0, 0.0)
        east = Vec3(30.0, 0.0, 0.0)

        self.assertGreater(camera_depth(west, left_camera), camera_depth(east, left_camera))
        self.assertLess(camera_depth(west, right_camera), camera_depth(east, right_camera))

    def test_point_behind_camera_is_invisible(self) -> None:
        camera = build_camera_basis(config.CAMERA_FRONT_YAW)
        behind = camera.position - camera.forward * 10.0

        projection = project_point(behind, camera)

        self.assertFalse(projection.visible)
        self.assertLessEqual(projection.depth, config.NEAR_CLIP)

    def test_screen_to_world_round_trip_at_depth(self) -> None:
        camera = build_camera_basis(config.CAMERA_RIGHT_YAW)
        point = Vec3(24.0, 120.0, -9.0)
        projection = project_point(point, camera)

        restored = screen_to_world_at_depth(
            projection.screen_x,
            projection.screen_y,
            projection.depth,
            camera,
        )

        self.assertLess(restored.distance_to(point), 1e-9)

    def test_screen_to_world_in_cloud_bounds_preserves_screen_position(self) -> None:
        camera = build_camera_basis(config.CAMERA_FRONT_YAW)

        point = screen_to_world_in_cloud_bounds(300.0, 80.0, config.CAMERA_DISTANCE, camera)
        projection = project_point(point, camera)

        self.assertGreaterEqual(point.y, config.MIN_CLOUD_Y)
        self.assertLessEqual(point.y, config.MAX_CLOUD_Y)
        self.assertGreaterEqual(point.z, config.CLOUD_DEPTH_MIN)
        self.assertLessEqual(point.z, config.CLOUD_DEPTH_MAX)
        self.assertAlmostEqual(projection.screen_x, 300.0, delta=1.0)
        self.assertAlmostEqual(projection.screen_y, 80.0, delta=1.0)

    def test_depth_locked_drag_does_not_move_along_camera_forward(self) -> None:
        for yaw in [config.CAMERA_LEFT_YAW, config.CAMERA_FRONT_YAW, config.CAMERA_RIGHT_YAW]:
            with self.subTest(yaw=yaw):
                camera = build_camera_basis(yaw)
                start = Vec3(0.0, 150.0, 0.0)
                projection = project_point(start, camera)

                target = depth_locked_drag_target(
                    start,
                    projection.screen_x + 32.0,
                    projection.screen_y - 18.0,
                    camera,
                )

                self.assertAlmostEqual(camera_depth(start, camera), camera_depth(target, camera))
                self.assertGreater((target - start).dot(camera.right), 0.0)

    def test_camera_transition_locks_cloud_input_until_complete(self) -> None:
        controller = CameraController()

        changed = controller.request_preset(CameraPreset.RIGHT)

        self.assertTrue(changed)
        self.assertFalse(controller.can_accept_cloud_input())

        controller.update(config.CAMERA_TRANSITION_SECONDS)

        self.assertTrue(controller.can_accept_cloud_input())
        self.assertEqual(controller.current_yaw, config.CAMERA_RIGHT_YAW)

    def test_camera_request_yaw_clamps_to_continuous_limits(self) -> None:
        controller = CameraController()

        controller.request_yaw(config.CAMERA_MAX_YAW + 50.0)
        controller.update(config.CAMERA_TRANSITION_SECONDS)

        self.assertEqual(controller.current_yaw, config.CAMERA_MAX_YAW)
        self.assertEqual(controller.mode, CameraControlMode.IDLE)

    def test_camera_dial_drag_locks_cloud_input_and_settles(self) -> None:
        controller = CameraController()

        controller.begin_dial_drag(18.0)
        self.assertEqual(controller.mode, CameraControlMode.DIAL_DRAG)
        self.assertFalse(controller.can_accept_cloud_input())

        controller.update(1.0 / config.FPS)
        self.assertGreater(controller.current_yaw, 0.0)
        controller.update_dial_drag(24.0)
        controller.end_dial_drag()

        for _ in range(30):
            controller.update(1.0 / config.FPS)

        self.assertTrue(controller.can_accept_cloud_input())
        self.assertAlmostEqual(controller.current_yaw, 24.0, delta=0.05)

    def test_camera_relative_request_uses_current_continuous_yaw(self) -> None:
        controller = CameraController(current_yaw=18.0, target_yaw=18.0)

        changed = controller.request_relative(-1)
        controller.update(config.CAMERA_TRANSITION_SECONDS)

        self.assertTrue(changed)
        self.assertEqual(controller.current_yaw, config.CAMERA_FRONT_YAW)


if __name__ == "__main__":
    unittest.main()
