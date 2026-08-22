from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random

from src import config
from src.motion.quantize import (
    clamp_int,
    pack_signed,
    quantize_signed,
    quantize_unit,
    unpack_signed,
)
from src.motion.weather_formula import sample_weather_motion


@dataclass(frozen=True, slots=True)
class RainTrajectory:
    kind: int
    dx: bytes
    dy: bytes

    def point(self, step: int) -> tuple[int, int]:
        index = max(0, min(len(self.dx) - 1, step))
        return unpack_signed(self.dx[index]), self.dy[index]


@dataclass(frozen=True, slots=True)
class LightningBranch:
    start_index: int
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class LightningTemplate:
    main_points: tuple[tuple[int, int], ...]
    branches: tuple[LightningBranch, ...]


@dataclass(slots=True)
class WeatherMotionAtlas:
    phase_count: int
    group_count: int
    seed: int
    cloud_active_dx: bytearray
    cloud_active_dy: bytearray
    cloud_active_pulse: bytearray
    cloud_settling_dx: bytearray
    cloud_settling_dy: bytearray
    cloud_settling_pulse: bytearray
    cloud_mature_dx: bytearray
    cloud_mature_dy: bytearray
    cloud_mature_pulse: bytearray
    cloud_shape_level: bytearray
    cloud_growth_ease: bytes
    rain_sway: bytearray
    rain_length: bytearray
    rain_density: bytearray
    amplitude_table: tuple[bytes, ...]
    rain_trajectories: tuple[RainTrajectory, ...]
    lightning_templates: tuple[LightningTemplate, ...]

    @classmethod
    def build(
        cls,
        *,
        phase_count: int = config.MOTION_PHASE_COUNT,
        group_count: int = config.MOTION_GROUP_COUNT,
        seed: int = 0,
    ) -> WeatherMotionAtlas:
        if phase_count <= 0 or group_count <= 0:
            raise ValueError("phase_count and group_count must be positive")

        (
            cloud_active_dx,
            cloud_active_dy,
            cloud_active_pulse,
            cloud_settling_dx,
            cloud_settling_dy,
            cloud_settling_pulse,
            cloud_mature_dx,
            cloud_mature_dy,
            cloud_mature_pulse,
        ) = build_cloud_banks(phase_count, group_count, seed)
        cloud_shape_level = build_cloud_shape_bank(phase_count, group_count, seed)
        cloud_growth_ease = build_cloud_growth_ease()

        rain_sway, rain_length, rain_density = build_rain_band_banks(seed)
        return cls(
            phase_count=phase_count,
            group_count=group_count,
            seed=seed,
            cloud_active_dx=cloud_active_dx,
            cloud_active_dy=cloud_active_dy,
            cloud_active_pulse=cloud_active_pulse,
            cloud_settling_dx=cloud_settling_dx,
            cloud_settling_dy=cloud_settling_dy,
            cloud_settling_pulse=cloud_settling_pulse,
            cloud_mature_dx=cloud_mature_dx,
            cloud_mature_dy=cloud_mature_dy,
            cloud_mature_pulse=cloud_mature_pulse,
            cloud_shape_level=cloud_shape_level,
            cloud_growth_ease=cloud_growth_ease,
            rain_sway=rain_sway,
            rain_length=rain_length,
            rain_density=rain_density,
            amplitude_table=build_amplitude_table(),
            rain_trajectories=build_rain_trajectories(seed),
            lightning_templates=build_lightning_templates(seed),
        )

    def cloud_active_offset(self, group: int, phase: int) -> tuple[int, int, int]:
        index = self.index(group, phase)
        return (
            unpack_signed(self.cloud_active_dx[index]),
            unpack_signed(self.cloud_active_dy[index]),
            self.cloud_active_pulse[index],
        )

    def cloud_settling_offset(self, group: int, phase: int) -> tuple[int, int, int]:
        index = self.index(group, phase)
        return (
            unpack_signed(self.cloud_settling_dx[index]),
            unpack_signed(self.cloud_settling_dy[index]),
            self.cloud_settling_pulse[index],
        )

    def cloud_mature_offset(self, group: int, phase: int) -> tuple[int, int, int]:
        index = self.index(group, phase)
        return (
            unpack_signed(self.cloud_mature_dx[index]),
            unpack_signed(self.cloud_mature_dy[index]),
            self.cloud_mature_pulse[index],
        )

    def cloud_shape(self, group: int, phase: int) -> int:
        return self.cloud_shape_level[self.index(group, phase)]

    def index(self, group: int, phase: int) -> int:
        return (group % self.group_count) * self.phase_count + (phase % self.phase_count)


def build_cloud_banks(
    phase_count: int,
    group_count: int,
    seed: int,
) -> tuple[bytearray, ...]:
    active_dx = bytearray()
    active_dy = bytearray()
    active_pulse = bytearray()
    settling_dx = bytearray()
    settling_dy = bytearray()
    settling_pulse = bytearray()
    mature_dx = bytearray()
    mature_dy = bytearray()
    mature_pulse = bytearray()

    for group in range(group_count):
        sample_index = seed * 131 + group * 17 + 23
        raw_x_values: list[float] = []
        raw_y_values: list[float] = []
        pulse_values: list[float] = []
        for phase in range(phase_count):
            phase01 = phase / phase_count
            x, y, pulse = sample_weather_motion(sample_index, group, phase01)
            raw_x_values.append(0.75 * x)
            raw_y_values.append(0.62 * y)
            pulse_values.append(pulse)

        smooth_x = smooth_periodic(raw_x_values, config.CLOUD_MOTION_SMOOTH_RADIUS)
        smooth_y = smooth_periodic(raw_y_values, config.CLOUD_MOTION_SMOOTH_RADIUS)
        for phase in range(phase_count):
            x = smooth_x[phase]
            y = smooth_y[phase]
            pulse = pulse_values[phase]

            active_dx.append(pack_signed(quantize_signed(x, 112.0, 127)))
            active_dy.append(pack_signed(quantize_signed(y, 112.0, 127)))
            active_pulse.append(quantize_pulse(pulse))

            settling_dx.append(pack_signed(quantize_signed(x, 84.0, 127)))
            settling_dy.append(pack_signed(quantize_signed(y, 84.0, 127)))
            settling_pulse.append(quantize_pulse(0.84 + (pulse - 0.84) * 0.55))

            mature_dx.append(pack_signed(quantize_signed(x, 70.0, 127)))
            mature_dy.append(pack_signed(quantize_signed(y, 70.0, 127)))
            mature_pulse.append(quantize_pulse(0.94 + (pulse - 0.94) * 0.28))

    return (
        active_dx,
        active_dy,
        active_pulse,
        settling_dx,
        settling_dy,
        settling_pulse,
        mature_dx,
        mature_dy,
        mature_pulse,
    )


def build_cloud_shape_bank(phase_count: int, group_count: int, seed: int) -> bytearray:
    shape = bytearray()
    for group in range(group_count):
        sample_index = seed * 211 + group * 31 + 11
        raw_values: list[float] = []
        phase_shift = (sample_index % 997) / 997.0 * math.tau
        for phase in range(phase_count):
            phase01 = phase / phase_count
            _x, _y, pulse = sample_weather_motion(sample_index, group + 9, phase01)
            slow_breath = 0.5 + 0.5 * math.sin(math.tau * phase01 + phase_shift)
            pulse01 = max(0.0, min(1.0, (pulse - 0.68) / (1.50 - 0.68)))
            organic = slow_breath * 0.72 + pulse01 * 0.28
            raw_values.append(max(0.0, min(1.0, organic)))

        smooth_values = smooth_periodic(raw_values, config.CLOUD_SHAPE_SMOOTH_RADIUS)
        for value in smooth_values:
            if value >= 0.66:
                shape.append(2)
            elif value >= 0.34:
                shape.append(1)
            else:
                shape.append(0)
    return shape


def build_cloud_growth_ease() -> bytes:
    levels = bytearray()
    attack_frame = max(1, config.CLOUD_TAP_PULSE_ATTACK_FRAMES)
    final_frame = max(attack_frame + 1, config.CLOUD_TAP_PULSE_DURATION_FRAMES)
    for frame in range(config.CLOUD_GROWTH_EASE_FRAMES):
        if frame <= attack_frame:
            t = frame / attack_frame
            value = 10.0 * ease_out_sine(t)
        else:
            t = (frame - attack_frame) / (final_frame - attack_frame)
            value = 10.0 * (1.0 - smoothstep(t))
        levels.append(clamp_int(int(round(value)), 0, 10))
    levels[0] = 0
    levels[-1] = 0
    return bytes(levels)


def ease_out_sine(value: float) -> float:
    return math.sin(max(0.0, min(1.0, value)) * math.pi * 0.5)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def smooth_periodic(values: list[float], radius: int) -> list[float]:
    if radius <= 0:
        return list(values)
    count = len(values)
    result: list[float] = []
    for index in range(count):
        total = 0.0
        weight_sum = 0.0
        for offset in range(-radius, radius + 1):
            weight = radius + 1 - abs(offset)
            total += values[(index + offset) % count] * weight
            weight_sum += weight
        result.append(total / weight_sum)
    return result


def quantize_pulse(pulse: float) -> int:
    normalized = (pulse - 0.68) / (1.50 - 0.68)
    return quantize_unit(max(0.0, min(1.0, normalized)), 16)


def build_amplitude_table() -> tuple[bytes, ...]:
    rows: list[bytes] = []
    for activation in range(config.MOTION_AMPLITUDE_LEVELS):
        activation01 = activation / (config.MOTION_AMPLITUDE_LEVELS - 1)
        row = bytearray()
        for incubation in range(config.MOTION_AMPLITUDE_LEVELS):
            incubation01 = incubation / (config.MOTION_AMPLITUDE_LEVELS - 1)
            amplitude = 16.0 * (0.25 + 0.75 * activation01) * (1.0 - 0.70 * incubation01)
            row.append(clamp_int(int(round(amplitude)), 0, 16))
        rows.append(bytes(row))
    return tuple(rows)


def build_rain_band_banks(seed: int) -> tuple[bytearray, bytearray, bytearray]:
    sway = bytearray()
    length = bytearray()
    density = bytearray()
    for group in range(config.RAIN_BAND_GROUPS):
        sample_index = seed * 197 + group * 29 + 7
        for phase in range(config.RAIN_PHASE_COUNT):
            phase01 = phase / config.RAIN_PHASE_COUNT
            x, _y, pulse = sample_weather_motion(sample_index, group + 3, phase01)
            sway.append(pack_signed(quantize_signed(x, 2.2, 3)))
            length.append(clamp_int(2 + int(round(pulse * 3.8)), 2, 8))
            density.append(clamp_int(int(round((pulse - 0.68) * 8.0)), 0, 8))
    return sway, length, density


def build_rain_trajectories(seed: int) -> tuple[RainTrajectory, ...]:
    rng = Random(seed ^ 0x51A7)
    trajectories: list[RainTrajectory] = []
    for trajectory_id in range(config.RAIN_TRAJECTORY_COUNT):
        kind = trajectory_id % 4
        if kind == 0:
            wind = 0.0
            wobble = 1.2
        elif kind == 1:
            wind = -5.5
            wobble = 1.6
        elif kind == 2:
            wind = 5.5
            wobble = 1.6
        else:
            wind = rng.choice((-8.0, 8.0))
            wobble = 3.0
        phase_bias = rng.uniform(-1.0, 1.0)
        dx_values = bytearray()
        dy_values = bytearray()
        previous_y = 0
        for step in range(config.RAIN_TRAJECTORY_STEPS):
            s = step / (config.RAIN_TRAJECTORY_STEPS - 1)
            motion, _y, _pulse = sample_weather_motion(
                seed * 31 + trajectory_id * 11,
                trajectory_id + kind,
                (s + phase_bias) % 1.0,
            )
            dx = int(round(wind * s * s + motion * wobble * (1.0 - s)))
            y = int(round(58.0 * s * s))
            previous_y = max(previous_y, y)
            dx_values.append(pack_signed(clamp_int(dx, -16, 16)))
            dy_values.append(clamp_int(previous_y, 0, 96))
        trajectories.append(RainTrajectory(kind, bytes(dx_values), bytes(dy_values)))
    return tuple(trajectories)


def build_lightning_templates(seed: int) -> tuple[LightningTemplate, ...]:
    rng = Random(seed ^ 0xA113)
    templates: list[LightningTemplate] = []
    for template_id in range(config.LIGHTNING_TEMPLATE_COUNT):
        main_points: list[tuple[int, int]] = []
        lateral_scale = rng.randint(14, 42)
        for point_index in range(config.LIGHTNING_TEMPLATE_POINTS):
            s = point_index / (config.LIGHTNING_TEMPLATE_POINTS - 1)
            u = int(round(255 * s))
            if point_index in (0, config.LIGHTNING_TEMPLATE_POINTS - 1):
                v = 0
            else:
                motion, _y, _pulse = sample_weather_motion(
                    seed * 43 + template_id * 5,
                    template_id,
                    s,
                )
                envelope = max(0.0, 1.0 - abs(s * 2.0 - 1.0))
                v = int(round(motion * lateral_scale * envelope))
                v = clamp_int(v, -127, 127)
            main_points.append((u, v))

        branch_count = rng.randint(0, config.LIGHTNING_TEMPLATE_MAX_BRANCHES)
        branches: list[LightningBranch] = []
        for branch_index in range(branch_count):
            start_index = rng.randint(5, config.LIGHTNING_TEMPLATE_POINTS - 7)
            start_u, start_v = main_points[start_index]
            direction = -1 if rng.random() < 0.5 else 1
            branch_length = rng.randint(4, 8)
            branch_points: list[tuple[int, int]] = []
            for step in range(branch_length):
                s = step / max(1, branch_length - 1)
                u = clamp_int(start_u + int(round(35 * s)), 0, 255)
                v = clamp_int(
                    start_v + direction * int(round((12 + branch_index * 5) * s)),
                    -127,
                    127,
                )
                branch_points.append((u, v))
            branches.append(LightningBranch(start_index, tuple(branch_points)))
        templates.append(LightningTemplate(tuple(main_points), tuple(branches)))
    return tuple(templates)
