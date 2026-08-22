from __future__ import annotations

import math
from dataclasses import dataclass, field

from src import config
from src.cloud.model import CloudState


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


@dataclass(slots=True)
class GrowthPulse:
    node_id: int
    start_frame: int
    duration_frames: int
    strength_level: int
    graph_distance: int


@dataclass
class WeatherMotionRuntime:
    growth_pulses: dict[int, GrowthPulse] = field(default_factory=dict)
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
        return {
            node_id: variant
            for node_id in event.node_ids
            if not self.response_blocks_ambient(node_id, frame)
        }

    def trigger_growth(self, node_id: int | None, frame: int) -> None:
        if node_id is None:
            return
        self.growth_pulses[node_id] = GrowthPulse(
            node_id=node_id,
            start_frame=frame,
            duration_frames=config.CLOUD_TAP_PULSE_DURATION_FRAMES,
            strength_level=config.CLOUD_PULSE_STRENGTH_BY_DISTANCE[0],
            graph_distance=0,
        )

    def trigger_growth_wave(
        self,
        state: CloudState,
        node_id: int | None,
        frame: int,
    ) -> None:
        if node_id is None or node_id not in state.nodes:
            return
        for target_id, distance in graph_distances(
            state,
            node_id,
            config.CLOUD_PULSE_PROPAGATION_MAX_DISTANCE,
        ).items():
            node = state.nodes.get(target_id)
            if node is None or node.is_pruning or node.fade <= 0.0:
                continue
            strength = config.CLOUD_PULSE_STRENGTH_BY_DISTANCE[
                min(distance, len(config.CLOUD_PULSE_STRENGTH_BY_DISTANCE) - 1)
            ]
            self.growth_pulses[target_id] = GrowthPulse(
                node_id=target_id,
                start_frame=frame + distance * config.CLOUD_PULSE_PROPAGATION_DELAY_FRAMES,
                duration_frames=config.CLOUD_TAP_PULSE_DURATION_FRAMES,
                strength_level=strength,
                graph_distance=distance,
            )

    def growth_level(self, node_id: int, frame: int, growth_ease: bytes) -> int:
        pulse = self.growth_pulses.get(node_id)
        if pulse is None:
            return 0
        age = frame - pulse.start_frame
        if age < 0 or age >= len(growth_ease):
            return 0
        return int(round(growth_ease[age] * pulse.strength_level / 10.0))

    def response_blocks_ambient(self, node_id: int, frame: int) -> bool:
        pulse = self.growth_pulses.get(node_id)
        if pulse is None:
            return False
        block_start = pulse.start_frame - config.CLOUD_PULSE_PROPAGATION_DELAY_FRAMES
        block_end = (
            pulse.start_frame
            + pulse.duration_frames
            + config.POST_RESPONSE_AMBIENT_COOLDOWN_FRAMES
        )
        return block_start <= frame < block_end

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

    def growth_pulse_count(self, frame: int) -> int:
        return sum(
            1
            for pulse in self.growth_pulses.values()
            if pulse.start_frame <= frame < pulse.start_frame + pulse.duration_frames
        )


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


def graph_distances(state: CloudState, origin_node_id: int, max_distance: int) -> dict[int, int]:
    if origin_node_id not in state.nodes:
        return {}
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in state.nodes}
    for edge in state.live_edges():
        adjacency.setdefault(edge.node_a, set()).add(edge.node_b)
        adjacency.setdefault(edge.node_b, set()).add(edge.node_a)

    distances = {origin_node_id: 0}
    frontier = [origin_node_id]
    while frontier:
        current = frontier.pop(0)
        next_distance = distances[current] + 1
        if next_distance > max_distance:
            continue
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            frontier.append(neighbor)
    return distances


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
