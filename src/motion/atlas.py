from __future__ import annotations

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
        for phase in range(phase_count):
            phase01 = phase / phase_count
            x, y, pulse = sample_weather_motion(sample_index, group, phase01)
            active_dx.append(pack_signed(quantize_signed(x, 1.55, 2)))
            active_dy.append(pack_signed(quantize_signed(y, 1.85, 2)))
            active_pulse.append(quantize_pulse(pulse))

            settling_dx.append(pack_signed(quantize_signed(x, 0.92, 1)))
            settling_dy.append(pack_signed(quantize_signed(y, 1.08, 1)))
            settling_pulse.append(quantize_pulse(0.84 + (pulse - 0.84) * 0.55))

            mature_dx.append(pack_signed(quantize_signed(x, 0.55, 1)))
            mature_dy.append(pack_signed(quantize_signed(y, 0.64, 1)))
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

