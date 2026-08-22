from __future__ import annotations

from enum import IntEnum

from src import config
from src.cloud.model import CloudNode, CloudState
from src.motion.atlas import WeatherMotionAtlas
from src.motion.runtime import WeatherMotionRuntime, hysteresis_step


class CloudMotionState(IntEnum):
    ACTIVE = 0
    SETTLING = 1
    MATURE = 2


def cloud_motion_state_for_node(node: CloudNode) -> CloudMotionState:
    if node.incubation >= 0.68 or (node.noise <= 0.08 and node.activation <= 0.18):
        return CloudMotionState.MATURE
    if node.incubation > 0.0 or node.activation < 0.35 or node.noise < 0.18:
        return CloudMotionState.SETTLING
    return CloudMotionState.ACTIVE


def cloud_render_offset(
    node: CloudNode,
    state: CloudState,
    atlas: WeatherMotionAtlas,
    frame: int,
    runtime: WeatherMotionRuntime | None = None,
) -> tuple[float, float, float]:
    if node.id not in state.nodes:
        return 0.0, 0.0, 1.0

    motion_state = cloud_motion_state_for_node(node)
    held_frame = frame - (frame % config.CLOUD_LOCAL_MOTION_HOLD_FRAMES)
    node_phase = phase_index(atlas, held_frame, (node.sprite_seed * 73) & 0x7FFFFFFF)
    cluster_phase = phase_index(atlas, frame, cluster_seed(node))
    sync_level = sync_level_for_node(node, motion_state)
    effective_node_phase = lerp_phase(node_phase, cluster_phase, sync_level, atlas.phase_count)

    node_group = node.sprite_seed % atlas.group_count
    cluster_group = cluster_seed(node) % atlas.group_count
    local_raw_x, local_raw_y, local_pulse = cloud_bank_offset(
        atlas,
        motion_state,
        node_group,
        effective_node_phase,
    )
    cluster_raw_x, cluster_raw_y, cluster_pulse = cloud_bank_offset(
        atlas,
        motion_state,
        cluster_group,
        cluster_phase,
    )

    _cluster_weight, local_weight = motion_weights(motion_state)
    cluster_key = cluster_seed(node)
    if runtime is None:
        cluster_dx, cluster_dy = stateless_cluster_offset(cluster_raw_x, cluster_raw_y)
    else:
        cluster_dx, cluster_dy = runtime.cluster_offset(cluster_key, cluster_raw_x, cluster_raw_y)

    local_dx = 0
    local_dy = 0
    if motion_state is CloudMotionState.ACTIVE:
        if runtime is None:
            local_dx, local_dy = stateless_local_offset(local_raw_x, local_raw_y)
        else:
            local_dx, local_dy = runtime.local_offset(node.id, local_raw_x, local_raw_y)

    dx = clamp_motion_offset(cluster_dx + local_dx * local_weight)
    dy = clamp_motion_offset(cluster_dy + local_dy * local_weight)
    pulse = (cluster_pulse * (1.0 - local_weight) + local_pulse * local_weight) / 15.0
    radius_ratio = 1.0 + (pulse - 0.5) * 0.02

    return dx, dy, radius_ratio


def cloud_shape_level(node: CloudNode, atlas: WeatherMotionAtlas, frame: int) -> int:
    motion_state = cloud_motion_state_for_node(node)
    node_phase = shape_phase_index(atlas, frame, (node.sprite_seed * 97) & 0x7FFFFFFF)
    cluster_phase = shape_phase_index(atlas, frame, cluster_seed(node) * 3)
    sync_level = sync_level_for_node(node, motion_state)
    effective_phase = lerp_phase(node_phase, cluster_phase, sync_level, atlas.phase_count)
    if motion_state is CloudMotionState.ACTIVE:
        group = node.sprite_seed % atlas.group_count
    else:
        group = cluster_seed(node) % atlas.group_count
    return atlas.cloud_shape(group, effective_phase)


def cloud_bank_offset(
    atlas: WeatherMotionAtlas,
    motion_state: CloudMotionState,
    group: int,
    phase: int,
) -> tuple[int, int, int]:
    if motion_state is CloudMotionState.ACTIVE:
        return atlas.cloud_active_offset(group, phase)
    if motion_state is CloudMotionState.SETTLING:
        return atlas.cloud_settling_offset(group, phase)
    return atlas.cloud_mature_offset(group, phase)


def phase_index(atlas: WeatherMotionAtlas, frame: int, phase_offset: int) -> int:
    return period_phase_index(
        atlas,
        frame,
        phase_offset,
        config.CLOUD_MOTION_PERIOD_SECONDS,
    )


def shape_phase_index(atlas: WeatherMotionAtlas, frame: int, phase_offset: int) -> int:
    return period_phase_index(
        atlas,
        frame,
        phase_offset,
        config.CLOUD_SHAPE_PERIOD_SECONDS,
    )


def period_phase_index(
    atlas: WeatherMotionAtlas,
    frame: int,
    phase_offset: int,
    period_seconds: float,
) -> int:
    period_frames = max(1, int(period_seconds * config.FPS))
    motion_phase = (frame * atlas.phase_count) // period_frames
    if atlas.phase_count & (atlas.phase_count - 1) == 0:
        return (motion_phase + phase_offset) & (atlas.phase_count - 1)
    return (motion_phase + phase_offset) % atlas.phase_count


def cluster_seed(node: CloudNode) -> int:
    return node.lineage_id * 4099 + node.cluster_id * 131


def sync_level_for_node(node: CloudNode, motion_state: CloudMotionState) -> int:
    if motion_state is CloudMotionState.ACTIVE:
        return 0
    incubation_level = clamp_level(node.incubation)
    if motion_state is CloudMotionState.MATURE:
        return max(12, incubation_level)
    return max(4, incubation_level)


def lerp_phase(node_phase: int, cluster_phase: int, amount: int, phase_count: int) -> int:
    amount = max(0, min(15, amount))
    half = phase_count // 2
    diff = ((cluster_phase - node_phase + half) % phase_count) - half
    return (node_phase + (diff * amount) // 15) % phase_count


def motion_weights(motion_state: CloudMotionState) -> tuple[float, float]:
    if motion_state is CloudMotionState.ACTIVE:
        cluster = config.CLOUD_ACTIVE_CLUSTER_WEIGHT
    elif motion_state is CloudMotionState.SETTLING:
        cluster = config.CLOUD_SETTLING_CLUSTER_WEIGHT
    else:
        cluster = config.CLOUD_MATURE_CLUSTER_WEIGHT
    return cluster, 1.0 - cluster


def amplitude_for_node(atlas: WeatherMotionAtlas, node: CloudNode) -> int:
    activation_level = clamp_level(node.activation)
    incubation_level = clamp_level(node.incubation)
    return atlas.amplitude_table[activation_level][incubation_level]


def clamp_level(value: float) -> int:
    return max(0, min(config.MOTION_AMPLITUDE_LEVELS - 1, int(value * 15.999)))


def stateless_cluster_offset(raw_x: int, raw_y: int) -> tuple[int, int]:
    return (
        hysteresis_step(
            0,
            raw_x,
            config.CLOUD_CLUSTER_ENTER_THRESHOLD,
            config.CLOUD_CLUSTER_EXIT_THRESHOLD,
        ),
        hysteresis_step(
            0,
            raw_y,
            config.CLOUD_CLUSTER_ENTER_THRESHOLD,
            config.CLOUD_CLUSTER_EXIT_THRESHOLD,
        ),
    )


def stateless_local_offset(raw_x: int, raw_y: int) -> tuple[int, int]:
    return (
        hysteresis_step(
            0,
            raw_x,
            config.CLOUD_LOCAL_ENTER_THRESHOLD,
            config.CLOUD_LOCAL_EXIT_THRESHOLD,
        ),
        hysteresis_step(
            0,
            raw_y,
            config.CLOUD_LOCAL_ENTER_THRESHOLD,
            config.CLOUD_LOCAL_EXIT_THRESHOLD,
        ),
    )


def clamp_motion_offset(value: float) -> float:
    return max(-1.0, min(1.0, value))
