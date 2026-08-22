from __future__ import annotations

import math
from dataclasses import dataclass, field

from src import config


@dataclass(slots=True)
class AmbientMorphEvent:
    start_frame: int
    duration_frames: int
    node_ids: tuple[int, ...]
    sequence_id: int


@dataclass(slots=True)
class ClusterMorphScheduler:
    next_event_frame: int
    event_index: int = 0
    active_event: AmbientMorphEvent | None = None


@dataclass
class WeatherMotionRuntime:
    growth_started_at: dict[int, int] = field(default_factory=dict)
    morph_schedulers: dict[int, ClusterMorphScheduler] = field(default_factory=dict)

    def ambient_morph_variants(
        self,
        cluster_key: int,
        candidate_node_ids: tuple[int, ...],
        frame: int,
        motion_state: int,
        node_count: int,
    ) -> dict[int, int]:
        if not candidate_node_ids:
            return {}

        scheduler = self.morph_schedulers.get(cluster_key)
        if scheduler is None:
            scheduler = ClusterMorphScheduler(
                next_event_frame=initial_morph_frame(cluster_key, motion_state)
            )
            self.morph_schedulers[cluster_key] = scheduler

        event = scheduler.active_event
        if event is not None and frame >= event.start_frame + event.duration_frames:
            scheduler.active_event = None
            scheduler.next_event_frame = frame + morph_interval(
                cluster_key,
                scheduler.event_index,
                motion_state,
            )

        if scheduler.active_event is None and frame >= scheduler.next_event_frame:
            node_ids = choose_morph_node_ids(
                cluster_key,
                scheduler.event_index,
                candidate_node_ids,
                motion_state,
                node_count,
            )
            scheduler.active_event = AmbientMorphEvent(
                start_frame=frame,
                duration_frames=config.CLOUD_AMBIENT_MORPH_DURATION_FRAMES,
                node_ids=node_ids,
                sequence_id=scheduler.event_index,
            )
            scheduler.event_index += 1

        event = scheduler.active_event
        if event is None:
            return {}

        age = frame - event.start_frame
        if age < 0 or age >= event.duration_frames:
            return {}
        variant = ambient_morph_variant(age)
        if variant == 0:
            return {}
        return {node_id: variant for node_id in event.node_ids}

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

    def active_morph_node_ids(self) -> tuple[int, ...]:
        node_ids: list[int] = []
        for scheduler in self.morph_schedulers.values():
            if scheduler.active_event is not None:
                node_ids.extend(scheduler.active_event.node_ids)
        return tuple(sorted(node_ids))

    def next_morph_event_frame(self) -> int | None:
        frames = [scheduler.next_event_frame for scheduler in self.morph_schedulers.values()]
        if not frames:
            return None
        return min(frames)


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


def initial_morph_frame(cluster_key: int, motion_state: int) -> int:
    return stable_range(cluster_key, 0, morph_interval_bounds(motion_state))


def morph_interval(cluster_key: int, event_index: int, motion_state: int) -> int:
    return stable_range(cluster_key, event_index + 1, morph_interval_bounds(motion_state))


def morph_interval_bounds(motion_state: int) -> tuple[int, int]:
    if motion_state <= 0:
        return config.CLOUD_ACTIVE_MORPH_INTERVAL_FRAMES
    if motion_state == 1:
        return config.CLOUD_SETTLING_MORPH_INTERVAL_FRAMES
    return config.CLOUD_MATURE_MORPH_INTERVAL_FRAMES


def morph_ratio(motion_state: int) -> float:
    if motion_state <= 0:
        return config.CLOUD_ACTIVE_MORPH_MAX_RATIO
    if motion_state == 1:
        return config.CLOUD_SETTLING_MORPH_MAX_RATIO
    return config.CLOUD_MATURE_MORPH_MAX_RATIO


def choose_morph_node_ids(
    cluster_key: int,
    event_index: int,
    candidate_node_ids: tuple[int, ...],
    motion_state: int,
    node_count: int,
) -> tuple[int, ...]:
    max_nodes = min(
        config.CLOUD_AMBIENT_MORPH_MAX_NODES,
        max(1, math.ceil(node_count * morph_ratio(motion_state))),
    )
    priority_pool = candidate_node_ids[: max(max_nodes * 3, max_nodes)]
    ranked = sorted(
        priority_pool,
        key=lambda node_id: stable_hash(cluster_key, event_index, node_id),
    )
    return tuple(sorted(ranked[:max_nodes]))


def ambient_morph_variant(age: int) -> int:
    hold = max(1, config.CLOUD_AMBIENT_MORPH_FRAME_HOLD)
    sequence = (0, 1, 2, 1, 0)
    index = max(0, min(len(sequence) - 1, age // hold))
    return sequence[index]


def stable_range(cluster_key: int, event_index: int, bounds: tuple[int, int]) -> int:
    lower, upper = bounds
    if upper <= lower:
        return lower
    span = upper - lower + 1
    return lower + stable_hash(cluster_key, event_index, 0xA47) % span


def stable_hash(cluster_key: int, event_index: int, salt: int) -> int:
    value = (cluster_key * 0x45D9F3B) ^ (event_index * 0x119DE1F3) ^ (salt * 0x1B873593)
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF
