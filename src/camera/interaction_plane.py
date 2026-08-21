from __future__ import annotations

from src import config
from src.camera.camera import CameraBasis
from src.camera.projection import camera_depth
from src.math3d import Vec3


def screen_to_world_at_depth(
    screen_x: float,
    screen_y: float,
    depth: float,
    camera: CameraBasis,
    viewport_center_x: float = config.VIEWPORT_CENTER_X,
    horizon_y: float = config.HORIZON_Y,
    focal_length: float = config.CAMERA_FOCAL_LENGTH,
) -> Vec3:
    if depth <= config.NEAR_CLIP:
        raise ValueError("depth must be in front of the near clip plane")
    scale = focal_length / depth
    return (
        camera.position
        + camera.forward * depth
        + camera.right * ((screen_x - viewport_center_x) / scale)
        - camera.up * ((screen_y - horizon_y) / scale)
    )


def depth_locked_drag_target(
    start_point: Vec3,
    screen_x: float,
    screen_y: float,
    camera: CameraBasis,
) -> Vec3:
    depth = camera_depth(start_point, camera)
    return screen_to_world_at_depth(screen_x, screen_y, depth, camera)
