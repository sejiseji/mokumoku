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


def screen_to_world_in_cloud_bounds(
    screen_x: float,
    screen_y: float,
    desired_depth: float,
    camera: CameraBasis,
) -> Vec3:
    ray = (
        camera.forward
        + camera.right * ((screen_x - config.VIEWPORT_CENTER_X) / config.CAMERA_FOCAL_LENGTH)
        - camera.up * ((screen_y - config.HORIZON_Y) / config.CAMERA_FOCAL_LENGTH)
    )
    lower = config.NEAR_CLIP + 0.01
    upper = config.CAMERA_DISTANCE * 2.0

    for origin, direction, minimum, maximum in (
        (camera.position.y, ray.y, config.MIN_CLOUD_Y, config.MAX_CLOUD_Y),
        (camera.position.z, ray.z, config.CLOUD_DEPTH_MIN, config.CLOUD_DEPTH_MAX),
    ):
        if abs(direction) < 0.0001:
            if origin < minimum or origin > maximum:
                return clamp_cloud_position(
                    screen_to_world_at_depth(screen_x, screen_y, desired_depth, camera)
                )
            continue
        first = (minimum - origin) / direction
        second = (maximum - origin) / direction
        axis_lower = min(first, second)
        axis_upper = max(first, second)
        lower = max(lower, axis_lower)
        upper = min(upper, axis_upper)

    if lower > upper:
        return clamp_cloud_position(
            screen_to_world_at_depth(screen_x, screen_y, desired_depth, camera)
        )

    depth = min(max(desired_depth, lower), upper)
    return clamp_cloud_position(screen_to_world_at_depth(screen_x, screen_y, depth, camera))


def clamp_cloud_position(position: Vec3) -> Vec3:
    return Vec3(
        position.x,
        min(max(position.y, config.MIN_CLOUD_Y), config.MAX_CLOUD_Y),
        min(max(position.z, config.CLOUD_DEPTH_MIN), config.CLOUD_DEPTH_MAX),
    )


def depth_locked_drag_target(
    start_point: Vec3,
    screen_x: float,
    screen_y: float,
    camera: CameraBasis,
) -> Vec3:
    depth = camera_depth(start_point, camera)
    return screen_to_world_at_depth(screen_x, screen_y, depth, camera)
