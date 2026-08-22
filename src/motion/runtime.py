from __future__ import annotations

from dataclasses import dataclass, field

from src import config


@dataclass
class WeatherMotionRuntime:
    cluster_x: dict[int, int] = field(default_factory=dict)
    cluster_y: dict[int, int] = field(default_factory=dict)
    local_x: dict[int, int] = field(default_factory=dict)
    local_y: dict[int, int] = field(default_factory=dict)
    cluster_x_changed_at: dict[int, int] = field(default_factory=dict)
    cluster_y_changed_at: dict[int, int] = field(default_factory=dict)
    local_x_changed_at: dict[int, int] = field(default_factory=dict)
    local_y_changed_at: dict[int, int] = field(default_factory=dict)
    growth_started_at: dict[int, int] = field(default_factory=dict)

    def cluster_offset(
        self,
        cluster_key: int,
        raw_x: int,
        raw_y: int,
        frame: int,
        motion_state: int,
        size_class: str,
    ) -> tuple[int, int]:
        candidate_x = hysteresis_step(
            self.cluster_x.get(cluster_key, 0),
            raw_x,
            config.CLOUD_CLUSTER_ENTER_THRESHOLD,
            config.CLOUD_CLUSTER_EXIT_THRESHOLD,
        )
        candidate_y = hysteresis_step(
            self.cluster_y.get(cluster_key, 0),
            raw_y,
            config.CLOUD_CLUSTER_ENTER_THRESHOLD,
            config.CLOUD_CLUSTER_EXIT_THRESHOLD,
        )
        dx = gated_motion_step(
            cluster_key,
            candidate_x,
            frame,
            self.cluster_x,
            self.cluster_x_changed_at,
            motion_interval(config.CLOUD_CLUSTER_X_INTERVAL_FRAMES, motion_state, size_class),
        )
        dy = gated_motion_step(
            cluster_key,
            candidate_y,
            frame,
            self.cluster_y,
            self.cluster_y_changed_at,
            motion_interval(config.CLOUD_CLUSTER_Y_INTERVAL_FRAMES, motion_state, size_class),
        )
        return dx, dy

    def local_offset(
        self,
        node_id: int,
        raw_x: int,
        raw_y: int,
        frame: int,
        motion_state: int,
        size_class: str,
    ) -> tuple[int, int]:
        candidate_x = hysteresis_step(
            self.local_x.get(node_id, 0),
            raw_x,
            config.CLOUD_LOCAL_ENTER_THRESHOLD,
            config.CLOUD_LOCAL_EXIT_THRESHOLD,
        )
        candidate_y = hysteresis_step(
            self.local_y.get(node_id, 0),
            raw_y,
            config.CLOUD_LOCAL_ENTER_THRESHOLD,
            config.CLOUD_LOCAL_EXIT_THRESHOLD,
        )
        dx = gated_motion_step(
            node_id,
            candidate_x,
            frame,
            self.local_x,
            self.local_x_changed_at,
            motion_interval(config.CLOUD_LOCAL_X_INTERVAL_FRAMES, motion_state, size_class),
        )
        dy = gated_motion_step(
            node_id,
            candidate_y,
            frame,
            self.local_y,
            self.local_y_changed_at,
            motion_interval(config.CLOUD_LOCAL_Y_INTERVAL_FRAMES, motion_state, size_class),
        )
        return dx, dy

    def trigger_growth(self, node_id: int | None, frame: int) -> None:
        if node_id is None:
            return
        self.growth_started_at[node_id] = frame

    def growth_level(self, node_id: int, frame: int, growth_ease: bytes) -> int:
        started_at = self.growth_started_at.get(node_id)
        if started_at is None:
            return 0
        age = frame - started_at
        if age < 0 or age >= len(growth_ease):
            return 0
        return growth_ease[age]


def hysteresis_step(current: int, raw: int, enter: int, exit: int) -> int:
    if current == 0:
        if raw > enter:
            return 1
        if raw < -enter:
            return -1
        return 0
    if current == 1 and raw < exit:
        return 0
    if current == -1 and raw > -exit:
        return 0
    return current


def gated_motion_step(
    key: int,
    candidate: int,
    frame: int,
    values: dict[int, int],
    changed_at: dict[int, int],
    min_interval_frames: int,
) -> int:
    current = values.get(key)
    if current is None:
        values[key] = candidate
        changed_at[key] = frame
        return candidate
    if candidate == current:
        return current

    last_changed = changed_at.get(key, frame)
    if frame < last_changed or frame - last_changed >= min_interval_frames:
        values[key] = candidate
        changed_at[key] = frame
        return candidate
    return current


def motion_interval(table: tuple[tuple[int, ...], ...], motion_state: int, size_class: str) -> int:
    state_index = max(0, min(len(table) - 1, int(motion_state)))
    size_index = motion_size_index(size_class)
    row = table[state_index]
    return row[max(0, min(len(row) - 1, size_index))]


def motion_size_index(size_class: str) -> int:
    try:
        return config.CLOUD_MOTION_SIZE_CLASSES.index(size_class)
    except ValueError:
        return config.CLOUD_MOTION_SIZE_CLASSES.index("m")
