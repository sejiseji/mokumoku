from __future__ import annotations

from enum import IntEnum

from src import config
from src.cloud.model import CloudNode, CloudState
from src.motion.atlas import WeatherMotionAtlas


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
) -> tuple[float, float, float]:
    if node.id not in state.nodes:
        return 0.0, 0.0, 1.0

    motion_state = cloud_motion_state_for_node(node)
    node_phase = phase_index(atlas, frame, (node.sprite_seed * 73) & 0x7FFFFFFF)
    cluster_phase = phase_index(atlas, frame, cluster_seed(node))
    sync_level = sync_level_for_node(node, motion_state)
    effective_node_phase = lerp_phase(node_phase, cluster_phase, sync_level, atlas.phase_count)

    node_group = node.sprite_seed % atlas.group_count
    cluster_group = cluster_seed(node) % atlas.group_count
    local_dx, local_dy, local_pulse = cloud_bank_offset(
        atlas,
        motion_state,
        node_group,
        effective_node_phase,
    )
    cluster_dx, cluster_dy, cluster_pulse = cloud_bank_offset(
        atlas,
        motion_state,
        cluster_group,
        cluster_phase,
    )

    cluster_weight, local_weight = motion_weights(motion_state)
    amplitude = amplitude_for_node(atlas, node) / 16.0
    dx = ((cluster_dx * cluster_weight + local_dx * local_weight) / 16.0) * amplitude
    dy = ((cluster_dy * cluster_weight + local_dy * local_weight) / 16.0) * amplitude
    dx = clamp_offset(dx * config.CLOUD_MOTION_RENDER_SCALE)
    dy = clamp_offset(dy * config.CLOUD_MOTION_RENDER_SCALE)
    pulse = ((cluster_pulse * cluster_weight + local_pulse * local_weight) / 16.0) / 15.0
    radius_ratio = 1.0 + (pulse - 0.5) * 0.04 * amplitude

    return dx, dy, radius_ratio


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
    if atlas.phase_count & (atlas.phase_count - 1) == 0:
        return (frame + phase_offset) & (atlas.phase_count - 1)
    return (frame + phase_offset) % atlas.phase_count


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


def motion_weights(motion_state: CloudMotionState) -> tuple[int, int]:
    if motion_state is CloudMotionState.ACTIVE:
        return 8, 8
    if motion_state is CloudMotionState.SETTLING:
        return 11, 5
    return 14, 2


def amplitude_for_node(atlas: WeatherMotionAtlas, node: CloudNode) -> int:
    activation_level = clamp_level(node.activation)
    incubation_level = clamp_level(node.incubation)
    return atlas.amplitude_table[activation_level][incubation_level]


def clamp_level(value: float) -> int:
    return max(0, min(config.MOTION_AMPLITUDE_LEVELS - 1, int(value * 15.999)))


def clamp_offset(value: float) -> float:
    return max(-config.CLOUD_MOTION_MAX_OFFSET_PX, min(config.CLOUD_MOTION_MAX_OFFSET_PX, value))
