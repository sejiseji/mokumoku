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


class CameraControlMode(Enum):
    IDLE = auto()
    PRESET_TRANSITION = auto()
    DIAL_DRAG = auto()
    DIAL_SETTLE = auto()


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
    mode: CameraControlMode = CameraControlMode.IDLE

    def basis(self) -> CameraBasis:
        return build_camera_basis(self.current_yaw)

    def is_transitioning(self) -> bool:
        return self.mode is not CameraControlMode.IDLE

    def is_dial_active(self) -> bool:
        return self.mode is CameraControlMode.DIAL_DRAG

    def can_accept_cloud_input(self) -> bool:
        return self.mode is CameraControlMode.IDLE

    def begin_dial_drag(self, yaw: float) -> None:
        self.mode = CameraControlMode.DIAL_DRAG
        self.target_yaw = self._clamp_yaw(yaw)
        self.start_yaw = self.current_yaw
        self.transition_elapsed = 0.0

    def update_dial_drag(self, yaw: float) -> None:
        if self.mode is CameraControlMode.DIAL_DRAG:
            self.target_yaw = self._clamp_yaw(yaw)

    def end_dial_drag(self) -> None:
        if self.mode is not CameraControlMode.DIAL_DRAG:
            return
        if abs(self.current_yaw - self.target_yaw) <= config.CAMERA_YAW_SETTLE_EPSILON:
            self.current_yaw = self.target_yaw
            self._sync_preset_to_nearest_yaw()
            self.mode = CameraControlMode.IDLE
            return
        self.mode = CameraControlMode.DIAL_SETTLE

    def request_yaw(
        self,
        yaw: float,
        *,
        transition_seconds: float | None = None,
    ) -> bool:
        target_yaw = self._clamp_yaw(yaw)
        if (
            self.mode is CameraControlMode.IDLE
            and abs(self.current_yaw - target_yaw) <= config.CAMERA_YAW_SETTLE_EPSILON
        ):
            self.current_yaw = target_yaw
            self.target_yaw = target_yaw
            self._sync_preset_to_nearest_yaw()
            return False
        self.start_yaw = self.current_yaw
        self.target_yaw = target_yaw
        self.transition_elapsed = 0.0
        self.transition_seconds = (
            config.CAMERA_TRANSITION_SECONDS
            if transition_seconds is None
            else max(1.0 / config.FPS, transition_seconds)
        )
        self.mode = CameraControlMode.PRESET_TRANSITION
        return True

    def request_preset(self, preset: CameraPreset) -> bool:
        self.preset = preset
        return self.request_yaw(PRESET_YAWS[preset])

    def request_left(self) -> bool:
        return self.request_preset(CameraPreset.LEFT)

    def request_front(self) -> bool:
        return self.request_preset(CameraPreset.FRONT)

    def request_right(self) -> bool:
        return self.request_preset(CameraPreset.RIGHT)

    def request_relative(self, direction: int) -> bool:
        presets = [CameraPreset.LEFT, CameraPreset.FRONT, CameraPreset.RIGHT]
        yaw = self.target_yaw if self.mode is not CameraControlMode.IDLE else self.current_yaw
        epsilon = 0.5
        if direction < 0:
            candidates = [
                preset for preset in presets if PRESET_YAWS[preset] < yaw - epsilon
            ]
            return self.request_preset(candidates[-1] if candidates else presets[0])
        if direction > 0:
            candidates = [
                preset for preset in presets if PRESET_YAWS[preset] > yaw + epsilon
            ]
            return self.request_preset(candidates[0] if candidates else presets[-1])
        return False

    def cycle(self) -> bool:
        presets = [CameraPreset.LEFT, CameraPreset.FRONT, CameraPreset.RIGHT]
        current = self._nearest_preset()
        return self.request_preset(presets[(presets.index(current) + 1) % len(presets)])

    def update(self, dt: float) -> None:
        if self.mode is CameraControlMode.IDLE:
            return

        if self.mode is CameraControlMode.DIAL_DRAG:
            follow = 1.0 - math.exp(-config.CAMERA_DIAL_FOLLOW_RATE * dt)
            self.current_yaw += (self.target_yaw - self.current_yaw) * follow
            self.current_yaw = self._clamp_yaw(self.current_yaw)
            return

        if self.mode is CameraControlMode.DIAL_SETTLE:
            follow = 1.0 - math.exp(-config.CAMERA_DIAL_SETTLE_RATE * dt)
            self.current_yaw += (self.target_yaw - self.current_yaw) * follow
            self.current_yaw = self._clamp_yaw(self.current_yaw)
            if abs(self.current_yaw - self.target_yaw) <= config.CAMERA_YAW_SETTLE_EPSILON:
                self.current_yaw = self.target_yaw
                self._sync_preset_to_nearest_yaw()
                self.mode = CameraControlMode.IDLE
            return

        if self.mode is CameraControlMode.PRESET_TRANSITION:
            self.transition_elapsed += dt
            t = smoothstep(self.transition_elapsed / self.transition_seconds)
            self.current_yaw = self.start_yaw + (self.target_yaw - self.start_yaw) * t
            self.current_yaw = self._clamp_yaw(self.current_yaw)
            if self.transition_elapsed >= self.transition_seconds:
                self.current_yaw = self.target_yaw
                self.transition_elapsed = self.transition_seconds
                self._sync_preset_to_nearest_yaw()
                self.mode = CameraControlMode.IDLE

    @staticmethod
    def _clamp_yaw(yaw: float) -> float:
        return max(config.CAMERA_MIN_YAW, min(config.CAMERA_MAX_YAW, yaw))

    def _nearest_preset(self) -> CameraPreset:
        yaw = self.target_yaw if self.mode is not CameraControlMode.IDLE else self.current_yaw
        return min(CameraPreset, key=lambda preset: abs(PRESET_YAWS[preset] - yaw))

    def _sync_preset_to_nearest_yaw(self) -> None:
        self.preset = self._nearest_preset()
