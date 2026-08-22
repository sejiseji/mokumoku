from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from src import config
from src.math3d import Vec3


class StimulusKind(Enum):
    TAP = auto()


class ReactionEventKind(Enum):
    NODE_PULSE = auto()
    SECONDARY_SPROUT = auto()
    SEED_IGNITION = auto()


class ReactionGrade(Enum):
    SINGLE = auto()
    SOFT_CHAIN = auto()
    BLOOM = auto()
    CASCADE = auto()


@dataclass(slots=True, frozen=True)
class ReactionViewSnapshot:
    camera_right: Vec3
    camera_up: Vec3
    camera_forward: Vec3
    camera_position: Vec3
    pointer_screen_x: float
    pointer_screen_y: float


@dataclass(slots=True, frozen=True)
class CloudStimulus:
    reaction_id: int
    source_kind: StimulusKind
    primary_node_id: int | None
    lineage_id: int | None
    screen_x: float
    screen_y: float
    world_position: Vec3
    strength: float
    radius_px: int
    created_frame: int
    seed: int
    view_snapshot: ReactionViewSnapshot


@dataclass(order=True, slots=True, frozen=True)
class ReactionEvent:
    execute_frame: int
    stable_order: int
    reaction_id: int = field(compare=False)
    kind: ReactionEventKind = field(compare=False)
    target_node_id: int | None = field(compare=False)
    source_node_id: int | None = field(compare=False)
    generation: int = field(compare=False)
    energy: float = field(compare=False)
    direction_index: int = field(compare=False, default=0)


@dataclass(slots=True, frozen=True)
class ReactionSummary:
    reaction_id: int
    lineage_id: int | None
    primary_node_id: int | None
    reacted_nodes: int
    sprouts: int
    ignited_seeds: int
    highest_generation: int
    duration_frames: int
    reaction_grade: ReactionGrade


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp01(t)


def stable_hash(*parts: int) -> int:
    value = 0xA511E9B3
    for index, part in enumerate(parts):
        value ^= (part + index * 0x9E3779B9) & 0xFFFFFFFF
        value = (value * 0x85EBCA6B) & 0xFFFFFFFF
        value ^= value >> 13
        value = (value * 0xC2B2AE35) & 0xFFFFFFFF
        value ^= value >> 16
    return value & 0xFFFFFFFF


def stable_hash01(*parts: int) -> float:
    return stable_hash(*parts) / 0xFFFFFFFF


def stable_range(lower: int, upper: int, *parts: int) -> int:
    if upper <= lower:
        return lower
    return lower + stable_hash(*parts) % (upper - lower + 1)


def response_coefficient(
    moisture: float,
    activation: float,
    updraft: float,
    noise: float,
    coherence: float,
) -> float:
    return clamp01(
        0.30
        + 0.22 * moisture
        + 0.18 * activation
        + 0.14 * updraft
        + 0.08 * noise
        + 0.08 * coherence
    )


def propagation_selectivity(
    edge_strength: float,
    target_stability: float,
    attribute_similarity: float,
    target_incubation: float,
) -> float:
    edge_quality = clamp01(
        0.50 * edge_strength
        + 0.30 * target_stability
        + 0.20 * attribute_similarity
    )
    return lerp(1.0, edge_quality, target_incubation)


def child_reaction_limit(activation: float, noise: float, incubation: float) -> int:
    if incubation >= 0.65:
        return config.MATURE_MAX_CHILD_REACTIONS
    if activation >= 0.50 or noise >= 0.45:
        return config.ACTIVE_MAX_CHILD_REACTIONS
    return config.SETTLING_MAX_CHILD_REACTIONS


def chain_delay(reaction_id: int, source_id: int, target_id: int, generation: int) -> int:
    jitter = stable_range(
        0,
        config.CHAIN_DELAY_JITTER_FRAMES,
        reaction_id,
        source_id,
        target_id,
        generation,
    )
    return config.CHAIN_DELAY_FRAMES + jitter


def seed_ignition_delay(reaction_id: int, seed_id: int, distance_ratio: float) -> int:
    distance_delay = round(
        clamp01(distance_ratio) * config.SEED_IGNITION_DISTANCE_DELAY_MAX
    )
    jitter = stable_range(0, 2, reaction_id, seed_id, 0x51ED)
    return config.SEED_IGNITION_BASE_DELAY_FRAMES + distance_delay + jitter


def reaction_grade(
    reacted_nodes: int,
    sprouts: int,
    ignited_seeds: int,
) -> ReactionGrade:
    secondary = sprouts + ignited_seeds
    if sprouts == 0 and ignited_seeds == 0 and reacted_nodes <= 1:
        return ReactionGrade.SINGLE
    if reacted_nodes <= 4 and secondary <= 2:
        return ReactionGrade.SOFT_CHAIN
    if reacted_nodes <= 9 and secondary <= 5:
        return ReactionGrade.BLOOM
    return ReactionGrade.CASCADE
