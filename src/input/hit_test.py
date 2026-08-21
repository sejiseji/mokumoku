from __future__ import annotations

import math
from dataclasses import dataclass

from src import config
from src.camera.camera import CameraBasis
from src.camera.projection import ProjectedPoint, project_point
from src.math3d import Vec3


@dataclass(frozen=True)
class HitTarget:
    stable_id: int
    position: Vec3
    radius: float


@dataclass(frozen=True)
class HitCandidate:
    target: HitTarget
    projection: ProjectedPoint
    screen_distance: float


def collect_hit_candidates(
    screen_x: float,
    screen_y: float,
    targets: list[HitTarget],
    camera: CameraBasis,
    padding: float = config.TOUCH_HIT_PADDING,
) -> list[HitCandidate]:
    candidates: list[HitCandidate] = []
    for target in targets:
        projection = project_point(target.position, camera)
        if not projection.visible:
            continue
        screen_radius = max(1.0, target.radius * projection.scale)
        distance = math.hypot(projection.screen_x - screen_x, projection.screen_y - screen_y)
        if distance <= screen_radius + padding:
            candidates.append(HitCandidate(target, projection, distance))
    return candidates


def choose_frontmost_candidate(
    candidates: list[HitCandidate],
    previous_selected_id: int | None = None,
    depth_epsilon: float = config.HIT_DEPTH_EPSILON,
) -> HitCandidate | None:
    if not candidates:
        return None

    nearest_depth = min(candidate.projection.depth for candidate in candidates)
    if previous_selected_id is not None:
        for candidate in candidates:
            if (
                candidate.target.stable_id == previous_selected_id
                and abs(candidate.projection.depth - nearest_depth) <= depth_epsilon
            ):
                return candidate

    return min(
        candidates,
        key=lambda candidate: (
            candidate.projection.depth,
            candidate.screen_distance,
            candidate.target.stable_id,
        ),
    )


def hit_test(
    screen_x: float,
    screen_y: float,
    targets: list[HitTarget],
    camera: CameraBasis,
    previous_selected_id: int | None = None,
) -> HitCandidate | None:
    return choose_frontmost_candidate(
        collect_hit_candidates(screen_x, screen_y, targets, camera),
        previous_selected_id=previous_selected_id,
    )
