from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.camera.camera import CameraBasis
from src.math3d import Vec3


@dataclass(frozen=True)
class ProjectedPoint:
    screen_x: float
    screen_y: float
    depth: float
    scale: float
    visible: bool


def camera_depth(point: Vec3, camera: CameraBasis) -> float:
    return (point - camera.position).dot(camera.forward)


def project_point(
    point: Vec3,
    camera: CameraBasis,
    viewport_center_x: float = config.VIEWPORT_CENTER_X,
    horizon_y: float = config.HORIZON_Y,
    focal_length: float = config.CAMERA_FOCAL_LENGTH,
    near_clip: float = config.NEAR_CLIP,
) -> ProjectedPoint:
    q = point - camera.position
    depth = q.dot(camera.forward)
    if depth <= near_clip:
        return ProjectedPoint(0.0, 0.0, depth, 0.0, False)

    camera_x = q.dot(camera.right)
    camera_y = q.dot(camera.up)
    scale = focal_length / max(near_clip, depth)
    return ProjectedPoint(
        screen_x=viewport_center_x + camera_x * scale,
        screen_y=horizon_y - camera_y * scale,
        depth=depth,
        scale=scale,
        visible=True,
    )
