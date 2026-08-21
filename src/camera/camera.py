from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

from src import config
from src.math3d import ZERO, Vec3


class CameraPreset(Enum):
    LEFT = auto()
    FRONT = auto()
    RIGHT = auto()


PRESET_YAWS = {
    CameraPreset.LEFT: config.CAMERA_LEFT_YAW,
    CameraPreset.FRONT: config.CAMERA_FRONT_YAW,
    CameraPreset.RIGHT: config.CAMERA_RIGHT_YAW,
}


@dataclass(frozen=True)
class CameraBasis:
    right: Vec3
    up: Vec3
    forward: Vec3
    position: Vec3
    yaw_degrees: float
    pitch_degrees: float


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def build_camera_basis(
    yaw_degrees: float,
    pitch_degrees: float = config.CAMERA_PITCH,
    distance: float = config.CAMERA_DISTANCE,
    target: Vec3 = ZERO,
) -> CameraBasis:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    cos_pitch = math.cos(pitch)

    forward = Vec3(
        math.sin(yaw) * cos_pitch,
        -math.sin(pitch),
        math.cos(yaw) * cos_pitch,
    ).normalized()
    right = Vec3(math.cos(yaw), 0.0, -math.sin(yaw)).normalized()
    up = forward.cross(right).normalized()
    position = target - forward * distance

    return CameraBasis(
        right=right,
        up=up,
        forward=forward,
        position=position,
        yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees,
    )


@dataclass
class CameraController:
    preset: CameraPreset = CameraPreset.FRONT
    current_yaw: float = config.CAMERA_FRONT_YAW
    target_yaw: float = config.CAMERA_FRONT_YAW
    start_yaw: float = config.CAMERA_FRONT_YAW
    transition_elapsed: float = 0.0
    transition_seconds: float = config.CAMERA_TRANSITION_SECONDS

    def basis(self) -> CameraBasis:
        return build_camera_basis(self.current_yaw)

    def is_transitioning(self) -> bool:
        return (
            self.transition_elapsed < self.transition_seconds
            and self.current_yaw != self.target_yaw
        )

    def can_accept_cloud_input(self) -> bool:
        return not self.is_transitioning()

    def request_preset(self, preset: CameraPreset) -> bool:
        if preset == self.preset and not self.is_transitioning():
            return False
        self.preset = preset
        self.start_yaw = self.current_yaw
        self.target_yaw = PRESET_YAWS[preset]
        self.transition_elapsed = 0.0
        return True

    def request_relative(self, direction: int) -> bool:
        presets = [CameraPreset.LEFT, CameraPreset.FRONT, CameraPreset.RIGHT]
        index = presets.index(self.preset)
        next_index = max(0, min(len(presets) - 1, index + direction))
        return self.request_preset(presets[next_index])

    def cycle(self) -> bool:
        presets = [CameraPreset.LEFT, CameraPreset.FRONT, CameraPreset.RIGHT]
        return self.request_preset(presets[(presets.index(self.preset) + 1) % len(presets)])

    def update(self, dt: float) -> None:
        if self.current_yaw == self.target_yaw:
            self.transition_elapsed = self.transition_seconds
            return
        self.transition_elapsed += dt
        t = smoothstep(self.transition_elapsed / self.transition_seconds)
        self.current_yaw = self.start_yaw + (self.target_yaw - self.start_yaw) * t
        if self.transition_elapsed >= self.transition_seconds:
            self.current_yaw = self.target_yaw
            self.transition_elapsed = self.transition_seconds
