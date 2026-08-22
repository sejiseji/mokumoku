from __future__ import annotations

import math
from dataclasses import dataclass

from src import config
from src.assets.sprite_map import (
    CLOUD_SPRITE_VARIANT_COUNT,
    CloudSpriteFamily,
    SpriteRect,
    cloud_sprite_rect,
    size_class_for_screen_radius,
)
from src.camera.camera import CameraBasis
from src.camera.projection import ProjectedPoint, project_point
from src.cloud.model import CloudEdge, CloudNode, CloudState
from src.enums import EdgeKind
from src.motion.atlas import WeatherMotionAtlas
from src.motion.cloud_motion import cloud_motion_state_for_node, cloud_render_offset, cluster_seed
from src.motion.runtime import WeatherMotionRuntime


@dataclass(frozen=True)
class EdgePayload:
    edge: CloudEdge
    point_a: ProjectedPoint
    point_b: ProjectedPoint


@dataclass(frozen=True)
class BridgePayload:
    edge: CloudEdge
    point: ProjectedPoint
    sprite: SpriteRect
    family: CloudSpriteFamily
    strain_t: float
    distance_ratio: float
    visual_radius: float


@dataclass(frozen=True)
class NodePayload:
    node: CloudNode
    projection: ProjectedPoint
    sprite: SpriteRect
    offset_x: float = 0.0
    offset_y: float = 0.0
    mesh_intensity: float = 0.0
    mesh_phase: float = 0.0
    shape_level: int = 0
    growth_level: int = 0


@dataclass(frozen=True)
class RenderItem:
    depth: float
    layer_bias: int
    stable_id: int
    payload: EdgePayload | BridgePayload | NodePayload


def collect_cloud_render_items(
    state: CloudState,
    camera: CameraBasis,
    frame: int = 0,
    motion_atlas: WeatherMotionAtlas | None = None,
    motion_runtime: WeatherMotionRuntime | None = None,
) -> list[RenderItem]:
    items: list[RenderItem] = []
    morph_variants: dict[int, int] = {}
    if motion_atlas is not None and motion_runtime is not None:
        morph_variants = collect_ambient_morph_variants(
            state,
            camera,
            frame,
            motion_runtime,
        )

    for edge in state.live_edges():
        node_a = state.nodes[edge.node_a]
        node_b = state.nodes[edge.node_b]
        projection_a = project_point(node_a.position, camera)
        projection_b = project_point(node_b.position, camera)
        if projection_a.visible and projection_b.visible:
            bridges = cloud_bridge_payloads(
                edge,
                node_a,
                node_b,
                projection_a,
                projection_b,
                camera,
            )
            items.append(
                RenderItem(
                    depth=max(projection_a.depth, projection_b.depth),
                    layer_bias=0,
                    stable_id=edge.id,
                    payload=EdgePayload(edge, projection_a, projection_b),
                )
            )
            for bridge_index, bridge in enumerate(bridges):
                items.append(
                    RenderItem(
                        depth=bridge.point.depth,
                        layer_bias=1,
                        stable_id=edge.id * 10 + bridge_index,
                        payload=bridge,
                    )
                )

    for node in state.live_nodes():
        projection = project_point(node.position, camera)
        if not projection.visible:
            continue
        screen_radius = node.radius * projection.scale
        size_class = size_class_for_screen_radius(screen_radius)
        if motion_atlas is None:
            offset_x, offset_y, _radius_ratio = cloud_node_wobble(node, state, frame)
            growth_level = 0
        else:
            offset_x, offset_y, _radius_ratio = cloud_render_offset(
                node,
                state,
                motion_atlas,
                frame,
                motion_runtime,
                size_class,
            )
            if motion_runtime is None:
                growth_level = 0
            else:
                growth_level = motion_runtime.growth_level(
                    node.id,
                    frame,
                    motion_atlas.cloud_growth_ease,
                )
        mesh_intensity = single_node_mesh_intensity(node, state, screen_radius)
        mesh_phase = single_node_mesh_phase(node, frame)
        family = choose_cloud_sprite_family(node, state, camera, projection)
        morph_variant = morph_variants.get(node.id, 0)
        sprite = cloud_sprite_rect(
            family,
            size_class_for_screen_radius(screen_radius * max(0.45, node.fade)),
            morph_variant,
        )
        items.append(
            RenderItem(
                depth=projection.depth,
                layer_bias=2,
                stable_id=node.id,
                payload=NodePayload(
                    node,
                    projection,
                    sprite,
                    offset_x,
                    offset_y,
                    mesh_intensity,
                    mesh_phase,
                    morph_variant,
                    growth_level,
                ),
            )
        )

    items.sort(key=lambda item: (-item.depth, item.layer_bias, item.stable_id))
    return items


def collect_ambient_morph_variants(
    state: CloudState,
    camera: CameraBasis,
    frame: int,
    motion_runtime: WeatherMotionRuntime,
) -> dict[int, int]:
    cluster_candidates: dict[int, list[tuple[int, int]]] = {}
    cluster_states: dict[int, int] = {}
    cluster_counts: dict[int, int] = {}
    for node in state.live_nodes():
        projection = project_point(node.position, camera)
        if not projection.visible:
            continue
        family = choose_cloud_sprite_family(node, state, camera, projection)
        priority = ambient_morph_priority(family)
        cluster_counts[node.cluster_id] = cluster_counts.get(node.cluster_id, 0) + 1
        motion_state = int(cloud_motion_state_for_node(node))
        cluster_states[node.cluster_id] = min(
            motion_state,
            cluster_states.get(node.cluster_id, motion_state),
        )
        if priority is None:
            continue
        cluster_candidates.setdefault(node.cluster_id, []).append((priority, node.id))

    variants: dict[int, int] = {}
    for cluster_id, candidates in cluster_candidates.items():
        sorted_candidates = tuple(
            node_id for _priority, node_id in sorted(candidates, key=lambda item: item)
        )
        representative = state.nodes[sorted_candidates[0]]
        variants.update(
            motion_runtime.ambient_morph_variants(
                cluster_seed(representative),
                sorted_candidates,
                frame,
                cluster_states.get(cluster_id, int(cloud_motion_state_for_node(representative))),
                cluster_counts.get(cluster_id, len(sorted_candidates)),
            )
        )
    return variants


def ambient_morph_priority(family: CloudSpriteFamily) -> int | None:
    if family is CloudSpriteFamily.UPDRAFT:
        return 0
    if family in (CloudSpriteFamily.EDGE, CloudSpriteFamily.STRETCH, CloudSpriteFamily.FRAGMENT):
        return 1
    if family is CloudSpriteFamily.BOTTOM:
        return 2
    return None


def cloud_bridge_payloads(
    edge: CloudEdge,
    node_a: CloudNode,
    node_b: CloudNode,
    projection_a: ProjectedPoint,
    projection_b: ProjectedPoint,
    camera: CameraBasis,
) -> list[BridgePayload]:
    if edge.strain >= config.CLOUD_BRIDGE_MAX_STRAIN:
        return []

    distance = (
        (projection_b.screen_x - projection_a.screen_x) ** 2
        + (projection_b.screen_y - projection_a.screen_y) ** 2
    ) ** 0.5
    radius_a = node_a.radius * projection_a.scale
    radius_b = node_b.radius * projection_b.scale
    radius_sum = radius_a + radius_b
    if distance <= radius_sum * config.CLOUD_BRIDGE_OVERLAP_HIDE_RATIO:
        return []

    strain_span = max(0.001, config.CLOUD_BRIDGE_MAX_STRAIN - config.CLOUD_BRIDGE_MIN_STRAIN)
    strain_t = max(0.0, min(1.0, (edge.strain - config.CLOUD_BRIDGE_MIN_STRAIN) / strain_span))
    distance_ratio = distance / max(1.0, radius_sum)
    bridge_count = cloud_bridge_count(distance_ratio, edge.strain, strain_t)
    offsets = cloud_bridge_offsets(bridge_count)
    family = cloud_bridge_family(edge.strain, strain_t)
    radius_scale = (
        config.CLOUD_BRIDGE_NECK_RADIUS_SCALE
        if family is CloudSpriteFamily.STRETCH
        else config.CLOUD_BRIDGE_FULL_RADIUS_SCALE
    )
    base_radius = min(radius_a, radius_b) * (
        radius_scale - strain_t * (radius_scale - config.CLOUD_BRIDGE_NECK_RADIUS_SCALE)
    )
    payloads: list[BridgePayload] = []
    for offset in offsets:
        point = node_a.position.lerp(node_b.position, offset)
        projection = project_point(point, camera)
        if not projection.visible:
            continue
        edge_taper = 0.26 if family is CloudSpriteFamily.INTERNAL else 0.12
        center_factor = 1.0 - abs(offset - 0.5) * edge_taper
        visual_radius = max(1.0, base_radius * center_factor)
        variant = (edge.id + len(payloads)) % CLOUD_SPRITE_VARIANT_COUNT
        sprite = cloud_sprite_rect(
            family,
            size_class_for_screen_radius(visual_radius),
            variant,
        )
        payloads.append(
            BridgePayload(edge, projection, sprite, family, strain_t, distance_ratio, visual_radius)
        )
    return payloads


def cloud_bridge_count(distance_ratio: float, strain: float, strain_t: float) -> int:
    if strain >= config.CLOUD_BRIDGE_NECK_STRAIN or strain_t >= 0.72:
        return 1
    if strain_t >= 0.45:
        return 2 if distance_ratio >= config.CLOUD_BRIDGE_TRIPLE_RATIO else 1
    if distance_ratio >= config.CLOUD_BRIDGE_QUAD_RATIO:
        return 4
    if distance_ratio >= config.CLOUD_BRIDGE_TRIPLE_RATIO:
        return 3
    if distance_ratio >= config.CLOUD_BRIDGE_DOUBLE_RATIO:
        return 2
    return 1


def cloud_bridge_offsets(bridge_count: int) -> tuple[float, ...]:
    if bridge_count <= 1:
        return (0.5,)
    if bridge_count == 2:
        return (0.40, 0.60)
    if bridge_count == 3:
        return (0.32, 0.50, 0.68)
    return (0.26, 0.42, 0.58, 0.74)


def cloud_bridge_family(strain: float, strain_t: float) -> CloudSpriteFamily:
    if strain >= config.CLOUD_BRIDGE_NECK_STRAIN or strain_t >= 0.45:
        return CloudSpriteFamily.STRETCH
    return CloudSpriteFamily.INTERNAL


def cloud_node_wobble(
    node: CloudNode,
    state: CloudState | int | None = None,
    frame: int = 0,
) -> tuple[float, float, float]:
    if isinstance(state, int):
        state = None
    return 0.0, 0.0, 1.0


def node_edge_count(node_id: int, state: CloudState) -> int:
    return sum(
        1
        for edge in state.edges.values()
        if edge.node_a == node_id or edge.node_b == node_id
    )


def single_node_mesh_intensity(
    node: CloudNode,
    state: CloudState,
    screen_radius: float,
) -> float:
    if node.is_pruning or node.fade <= 0.0:
        return 0.0
    if node_edge_count(node.id, state) > 0:
        return 0.0

    size_t = screen_radius / max(1.0, config.CLOUD_SINGLE_MESH_RADIUS_MAX)
    size_factor = max(0.0, min(1.0, 1.0 - size_t))
    seed_mass_range = max(1.0, config.RETENTION_GROWN_MASS)
    growth_t = max(0.0, min(1.0, (node.mass - config.SEED_MASS) / seed_mass_range))
    growth_factor = 1.0 - growth_t * 0.65
    return max(0.0, min(1.0, size_factor * growth_factor * node.fade))


def single_node_mesh_phase(node: CloudNode, frame: int) -> float:
    seconds = frame / config.FPS
    phase = (node.sprite_seed % 6283) / 1000.0
    return seconds * math.tau / config.CLOUD_SINGLE_MESH_PERIOD + phase


def projected_cloud_sprite_family(
    node: CloudNode,
    edge_count: int,
    projection: ProjectedPoint,
    neighbor_projections: list[ProjectedPoint],
) -> CloudSpriteFamily:
    gap = config.CLOUD_ROLE_EXPOSURE_GAP_PX
    has_above = any(other.screen_y < projection.screen_y - gap for other in neighbor_projections)
    has_below = any(other.screen_y > projection.screen_y + gap for other in neighbor_projections)
    has_left = any(other.screen_x < projection.screen_x - gap for other in neighbor_projections)
    has_right = any(other.screen_x > projection.screen_x + gap for other in neighbor_projections)

    if edge_count >= 4 and has_above and has_below and has_left and has_right:
        return CloudSpriteFamily.INTERNAL
    if has_above and not has_below:
        return CloudSpriteFamily.BOTTOM
    if has_below and not has_above:
        return CloudSpriteFamily.UPDRAFT
    if edge_count >= 3 and has_left and has_right and (has_above or has_below):
        return CloudSpriteFamily.INTERNAL
    if node.noise < 0.14 and edge_count >= 3:
        return CloudSpriteFamily.INTERNAL
    return CloudSpriteFamily.EDGE


def choose_cloud_sprite_family(
    node: CloudNode,
    state: CloudState,
    camera: CameraBasis | None = None,
    projection: ProjectedPoint | None = None,
) -> CloudSpriteFamily:
    if node.is_pruning or node.fade < 0.7:
        return CloudSpriteFamily.FADE
    edge_count = 0
    neighbor_projections: list[ProjectedPoint] = []
    for edge in state.edges.values():
        if edge.node_a == node.id:
            edge_count += 1
            neighbor = state.nodes[edge.node_b]
        elif edge.node_b == node.id:
            edge_count += 1
            neighbor = state.nodes[edge.node_a]
        else:
            continue
        if camera is not None:
            neighbor_projection = project_point(neighbor.position, camera)
            if neighbor_projection.visible:
                neighbor_projections.append(neighbor_projection)

    if projection is not None and neighbor_projections:
        return projected_cloud_sprite_family(node, edge_count, projection, neighbor_projections)

    if node.updraft > 0.35:
        return CloudSpriteFamily.UPDRAFT
    if node.density > 1.35:
        return CloudSpriteFamily.BOTTOM
    if edge_count == 0:
        return CloudSpriteFamily.FRAGMENT
    if node.noise < 0.14:
        return CloudSpriteFamily.INTERNAL
    if any(
        edge.kind is EdgeKind.PRIMARY and (edge.node_a == node.id or edge.node_b == node.id)
        for edge in state.edges.values()
    ):
        return CloudSpriteFamily.EDGE
    return CloudSpriteFamily.STRETCH
