from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

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
from src.motion.runtime import TouchResponseKind, WeatherMotionRuntime


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
class BodyPayload:
    source_id: int
    cluster_id: int
    point: ProjectedPoint
    sprite: SpriteRect
    visual_radius: float


class CloudDepthLayer(IntEnum):
    FRONT = 0
    MIDDLE = 1
    BACK = 2


@dataclass(frozen=True)
class SurfaceMetrics:
    exposure: float
    exposure_mask: int
    neighbor_count: int


@dataclass(frozen=True)
class NodePayload:
    node: CloudNode
    projection: ProjectedPoint
    sprite: SpriteRect
    family: CloudSpriteFamily
    depth_layer: CloudDepthLayer
    offset_x: float = 0.0
    offset_y: float = 0.0
    mesh_intensity: float = 0.0
    mesh_phase: float = 0.0
    shape_level: int = 0
    growth_level: int = 0
    response_kind: TouchResponseKind | None = None


@dataclass(frozen=True)
class RenderItem:
    depth: float
    layer_bias: int
    stable_id: int
    payload: EdgePayload | BridgePayload | BodyPayload | NodePayload


def depth_sort_bucket(depth: float) -> int:
    bucket_size = max(0.001, config.DEPTH_SORT_BUCKET_SIZE)
    return int(round(depth / bucket_size))


def render_item_sort_key(item: RenderItem) -> tuple[int, int, int]:
    return -depth_sort_bucket(item.depth), item.layer_bias, item.stable_id


def cloud_depth_layer(
    node: CloudNode,
    projection: ProjectedPoint,
    cluster_depth_ranges: dict[int, tuple[float, float]],
) -> CloudDepthLayer:
    depth_range = cluster_depth_ranges.get(node.cluster_id)
    if depth_range is None:
        return CloudDepthLayer.MIDDLE
    front_depth, back_depth = depth_range
    depth_span = back_depth - front_depth
    if depth_span < config.CLOUD_DEPTH_LAYER_MIN_SPAN:
        return CloudDepthLayer.MIDDLE

    depth_t = (projection.depth - front_depth) / depth_span
    if depth_t <= config.CLOUD_DEPTH_FRONT_THRESHOLD:
        return CloudDepthLayer.FRONT
    if depth_t >= config.CLOUD_DEPTH_BACK_THRESHOLD:
        return CloudDepthLayer.BACK
    return CloudDepthLayer.MIDDLE


def visible_lineage_node_counts(
    visible_nodes: list[tuple[CloudNode, ProjectedPoint]],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node, _projection in visible_nodes:
        counts[node.lineage_id] = counts.get(node.lineage_id, 0) + 1
    return counts


def cloud_surface_metrics(
    visible_nodes: list[tuple[CloudNode, ProjectedPoint]],
) -> dict[int, SurfaceMetrics]:
    metrics: dict[int, SurfaceMetrics] = {}
    direction_count = max(4, config.CLOUD_SURFACE_DIRECTION_COUNT)
    directions = tuple(
        (
            math.cos(math.tau * index / direction_count),
            math.sin(math.tau * index / direction_count),
        )
        for index in range(direction_count)
    )
    projected_radii = {
        node.id: max(1.0, node.radius * projection.scale)
        for node, projection in visible_nodes
    }

    for node, projection in visible_nodes:
        radius = projected_radii[node.id]
        neighbor_count = 0
        exposed_mask = 0
        for other, other_projection in visible_nodes:
            if other.id == node.id or other.lineage_id != node.lineage_id:
                continue
            other_radius = projected_radii[other.id]
            distance = math.hypot(
                other_projection.screen_x - projection.screen_x,
                other_projection.screen_y - projection.screen_y,
            )
            if (
                distance
                <= (radius + other_radius) * config.CLOUD_SURFACE_NEIGHBOR_RADIUS_SCALE
            ):
                neighbor_count += 1

        for index, (dx, dy) in enumerate(directions):
            sample_x = projection.screen_x + dx * radius
            sample_y = projection.screen_y + dy * radius
            covered = False
            for other, other_projection in visible_nodes:
                if other.id == node.id or other.lineage_id != node.lineage_id:
                    continue
                other_radius = projected_radii[other.id]
                center_distance = math.hypot(
                    other_projection.screen_x - projection.screen_x,
                    other_projection.screen_y - projection.screen_y,
                )
                if (
                    center_distance
                    > (radius + other_radius)
                    * config.CLOUD_SURFACE_NEIGHBOR_RADIUS_SCALE
                ):
                    continue
                sample_distance = math.hypot(
                    other_projection.screen_x - sample_x,
                    other_projection.screen_y - sample_y,
                )
                if sample_distance <= other_radius * config.CLOUD_SURFACE_OCCLUSION_RADIUS_SCALE:
                    covered = True
                    break
            if not covered:
                exposed_mask |= 1 << index
        exposure = exposed_mask.bit_count() / direction_count
        metrics[node.id] = SurfaceMetrics(exposure, exposed_mask, neighbor_count)
    return metrics


def protected_surface_lobe_ids(
    visible_nodes: list[tuple[CloudNode, ProjectedPoint]],
    frame: int,
    motion_runtime: WeatherMotionRuntime | None,
) -> set[int]:
    if motion_runtime is None:
        return set()
    protected: set[int] = set()
    for node, _projection in visible_nodes:
        if motion_runtime.response_kind(node.id, frame) is not None:
            protected.add(node.id)
    return protected


def select_surface_lobe_ids(
    visible_nodes: list[tuple[CloudNode, ProjectedPoint]],
    surface_metrics: dict[int, SurfaceMetrics],
    visible_group_counts: dict[int, int],
    protected_node_ids: set[int],
) -> set[int]:
    selected: set[int] = set()
    candidates_by_group: dict[int, list[tuple[float, CloudNode, ProjectedPoint]]] = {}
    for node, projection in visible_nodes:
        metric = surface_metrics.get(node.id, SurfaceMetrics(1.0, 0, 0))
        group_count = visible_group_counts.get(node.lineage_id, 1)
        if (
            group_count < config.CLOUD_SURFACE_MIN_CLUSTER_NODES
            or metric.neighbor_count < 3
            or node.id in protected_node_ids
            or node.is_pruning
            or node.fade < 0.7
            or is_visible_dormant_seed(node, metric)
        ):
            selected.add(node.id)
            continue
        if metric.exposure < config.CLOUD_SURFACE_INTERNAL_EXPOSURE:
            continue
        score = surface_lobe_score(node, projection, metric)
        candidates_by_group.setdefault(node.lineage_id, []).append((score, node, projection))

    for candidates in candidates_by_group.values():
        candidates.sort(key=lambda item: (-item[0], item[1].id))
        max_lobes = max(
            config.CLOUD_SURFACE_MIN_LOBES,
            int(math.ceil(len(candidates) * config.CLOUD_SURFACE_MAX_LOBE_RATIO)),
        )
        accepted: list[tuple[CloudNode, ProjectedPoint]] = []
        for _score, node, projection in candidates:
            if len(accepted) >= max_lobes:
                break
            radius = max(1.0, node.radius * projection.scale)
            suppressed = False
            for accepted_node, accepted_projection in accepted:
                accepted_radius = max(1.0, accepted_node.radius * accepted_projection.scale)
                distance = math.hypot(
                    accepted_projection.screen_x - projection.screen_x,
                    accepted_projection.screen_y - projection.screen_y,
                )
                if (
                    distance
                    < max(radius, accepted_radius)
                    * config.CLOUD_SURFACE_SUPPRESSION_RADIUS_RATIO
                ):
                    suppressed = True
                    break
            if suppressed:
                continue
            selected.add(node.id)
            accepted.append((node, projection))
    return selected


def surface_lobe_score(
    node: CloudNode,
    projection: ProjectedPoint,
    metric: SurfaceMetrics,
) -> float:
    screen_radius = node.radius * projection.scale
    radius_score = min(1.0, screen_radius / 20.0)
    upper_score = min(1.0, max(0.0, node.updraft))
    stability_score = min(1.0, node.incubation)
    seed_score = ((node.sprite_seed % 997) / 997.0) * 0.08
    return (
        0.42 * metric.exposure
        + 0.24 * radius_score
        + 0.14 * upper_score
        + 0.12 * stability_score
        + seed_score
    )


def is_visible_dormant_seed(node: CloudNode, metric: SurfaceMetrics) -> bool:
    if node.mass > config.SEED_MASS + 2.0:
        return False
    if node.excitation >= config.CLOUD_BODY_SEED_ABSORB_EXCITATION:
        return False
    return metric.neighbor_count <= 1


def cloud_body_payloads(
    state: CloudState,
    visible_nodes: list[tuple[CloudNode, ProjectedPoint]],
    surface_metrics: dict[int, SurfaceMetrics],
    visible_group_counts: dict[int, int],
    camera: CameraBasis,
) -> list[BodyPayload]:
    payloads: list[BodyPayload] = []
    for node, projection in visible_nodes:
        metric = surface_metrics.get(node.id, SurfaceMetrics(1.0, 0, 0))
        if not should_draw_body_stamp(node, state, metric, visible_group_counts):
            continue
        visual_radius = node.radius * projection.scale * config.CLOUD_BODY_RADIUS_SCALE
        payloads.append(
            BodyPayload(
                source_id=node.id,
                cluster_id=node.cluster_id,
                point=projection,
                sprite=cloud_sprite_rect(
                    CloudSpriteFamily.INTERNAL,
                    size_class_for_screen_radius(visual_radius * max(0.45, node.fade)),
                    0,
                ),
                visual_radius=visual_radius,
            )
        )
    payloads.extend(cloud_body_gap_payloads(visible_nodes, camera))
    return payloads


def should_draw_body_stamp(
    node: CloudNode,
    state: CloudState,
    metric: SurfaceMetrics,
    visible_group_counts: dict[int, int],
) -> bool:
    if node.is_pruning or node.fade <= 0.0:
        return False
    if node_edge_count(node.id, state) > 0:
        return True
    if metric.neighbor_count >= 2:
        return True
    if visible_group_counts.get(node.lineage_id, 1) >= config.CLOUD_SURFACE_MIN_CLUSTER_NODES:
        return node.excitation >= config.CLOUD_BODY_SEED_ABSORB_EXCITATION
    return node.mass > config.SEED_MASS + config.TAP_MASS_GAIN * 0.45


def cloud_body_gap_payloads(
    visible_nodes: list[tuple[CloudNode, ProjectedPoint]],
    camera: CameraBasis,
) -> list[BodyPayload]:
    pair_candidates: dict[int, list[tuple[float, BodyPayload]]] = {}
    for index, (node_a, projection_a) in enumerate(visible_nodes):
        radius_a = max(1.0, node_a.radius * projection_a.scale)
        for node_b, projection_b in visible_nodes[index + 1 :]:
            if node_a.lineage_id != node_b.lineage_id:
                continue
            radius_b = max(1.0, node_b.radius * projection_b.scale)
            distance = math.hypot(
                projection_b.screen_x - projection_a.screen_x,
                projection_b.screen_y - projection_a.screen_y,
            )
            radius_sum = radius_a + radius_b
            distance_ratio = distance / max(1.0, radius_sum)
            if distance_ratio < config.CLOUD_BODY_GAP_MIN_DISTANCE_RATIO:
                continue
            if distance_ratio > config.CLOUD_BODY_GAP_MAX_DISTANCE_RATIO:
                continue
            midpoint = node_a.position.lerp(node_b.position, 0.5)
            midpoint_projection = project_point(midpoint, camera)
            if not midpoint_projection.visible:
                continue
            visual_radius = min(radius_a, radius_b) * config.CLOUD_BODY_GAP_RADIUS_SCALE
            source_id = 1_000_000 + min(node_a.id, node_b.id) * 4099 + max(node_a.id, node_b.id)
            payload = BodyPayload(
                source_id=source_id,
                cluster_id=node_a.cluster_id,
                point=midpoint_projection,
                sprite=cloud_sprite_rect(
                    CloudSpriteFamily.INTERNAL,
                    size_class_for_screen_radius(visual_radius),
                    0,
                ),
                visual_radius=visual_radius,
            )
            pair_candidates.setdefault(node_a.lineage_id, []).append(
                (abs(distance_ratio - 0.72), payload)
            )
    payloads: list[BodyPayload] = []
    for candidates in pair_candidates.values():
        candidates.sort(key=lambda item: (item[0], item[1].source_id))
        payloads.extend(
            payload
            for _score, payload in candidates[: config.CLOUD_BODY_GAP_MAX_PER_CLUSTER]
        )
    return payloads


def collect_cloud_render_items(
    state: CloudState,
    camera: CameraBasis,
    frame: int = 0,
    motion_atlas: WeatherMotionAtlas | None = None,
    motion_runtime: WeatherMotionRuntime | None = None,
) -> list[RenderItem]:
    items: list[RenderItem] = []
    morph_variants: dict[int, int] = {}
    visible_nodes: list[tuple[CloudNode, ProjectedPoint]] = []
    cluster_depth_ranges: dict[int, tuple[float, float]] = {}
    for node in state.live_nodes():
        projection = project_point(node.position, camera)
        if not projection.visible:
            continue
        visible_nodes.append((node, projection))
        current_range = cluster_depth_ranges.get(node.cluster_id)
        if current_range is None:
            cluster_depth_ranges[node.cluster_id] = (projection.depth, projection.depth)
        else:
            cluster_depth_ranges[node.cluster_id] = (
                min(current_range[0], projection.depth),
                max(current_range[1], projection.depth),
            )
    if motion_runtime is not None:
        motion_runtime.prune_sprite_role_states({node.id for node, _projection in visible_nodes})
    if motion_atlas is not None and motion_runtime is not None:
        morph_variants = collect_ambient_morph_variants(
            state,
            camera,
            frame,
            motion_runtime,
        )
    surface_metrics = cloud_surface_metrics(visible_nodes)
    visible_lineage_counts = visible_lineage_node_counts(visible_nodes)
    protected_lobe_ids = protected_surface_lobe_ids(
        visible_nodes,
        frame,
        motion_runtime,
    )
    surface_lobe_ids = select_surface_lobe_ids(
        visible_nodes,
        surface_metrics,
        visible_lineage_counts,
        protected_lobe_ids,
    )

    for body_index, body in enumerate(
        cloud_body_payloads(
            state,
            visible_nodes,
            surface_metrics,
            visible_lineage_counts,
            camera,
        )
    ):
        items.append(
            RenderItem(
                depth=body.point.depth,
                layer_bias=1,
                stable_id=body.source_id * 10_000 + body_index,
                payload=body,
            )
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
                        layer_bias=2,
                        stable_id=edge.id * 10 + bridge_index,
                        payload=bridge,
                    )
                )

    for node, projection in visible_nodes:
        screen_radius = node.radius * projection.scale
        size_class = size_class_for_screen_radius(screen_radius)
        if motion_atlas is None:
            offset_x, offset_y, _radius_ratio = cloud_node_wobble(node, state, frame)
            growth_level = 0
            response_kind = None
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
                response_kind = None
            else:
                growth_level = motion_runtime.growth_level(
                    node.id,
                    frame,
                    motion_atlas.cloud_growth_ease,
                )
                response_kind = motion_runtime.response_kind(node.id, frame)
        if node.id not in surface_lobe_ids and response_kind is None and growth_level <= 0:
            continue
        mesh_intensity = single_node_mesh_intensity(node, state, screen_radius)
        mesh_phase = single_node_mesh_phase(node, frame)
        family_scores = cloud_sprite_family_scores(
            node,
            state,
            camera,
            projection,
            surface_metrics,
        )
        family = select_cloud_sprite_family(family_scores)
        if motion_runtime is not None:
            family = motion_runtime.stabilize_sprite_family(
                node.id,
                family,
                family_scores,
                frame,
            )
        depth_layer = cloud_depth_layer(node, projection, cluster_depth_ranges)
        morph_variant = morph_variants.get(node.id, 0)
        sprite = cloud_sprite_rect(
            family,
            size_class_for_screen_radius(screen_radius * max(0.45, node.fade)),
            morph_variant,
        )
        items.append(
            RenderItem(
                depth=projection.depth,
                layer_bias=3,
                stable_id=node.id,
                payload=NodePayload(
                    node,
                    projection,
                    sprite,
                    family,
                    depth_layer,
                    offset_x,
                    offset_y,
                    mesh_intensity,
                    mesh_phase,
                    morph_variant,
                    growth_level,
                    response_kind,
                ),
            )
        )

    items.sort(key=render_item_sort_key)
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


def projected_cloud_sprite_family_scores(
    node: CloudNode,
    edge_count: int,
    projection: ProjectedPoint,
    neighbor_projections: list[ProjectedPoint],
) -> dict[CloudSpriteFamily, float]:
    gap = config.CLOUD_ROLE_EXPOSURE_GAP_PX
    has_above = any(other.screen_y < projection.screen_y - gap for other in neighbor_projections)
    has_below = any(other.screen_y > projection.screen_y + gap for other in neighbor_projections)
    has_left = any(other.screen_x < projection.screen_x - gap for other in neighbor_projections)
    has_right = any(other.screen_x > projection.screen_x + gap for other in neighbor_projections)

    scores: dict[CloudSpriteFamily, float] = {CloudSpriteFamily.EDGE: 0.52}
    if edge_count >= 4 and has_above and has_below and has_left and has_right:
        scores[CloudSpriteFamily.INTERNAL] = 1.0
    if has_above and not has_below:
        scores[CloudSpriteFamily.BOTTOM] = 1.0
    if has_below and not has_above:
        scores[CloudSpriteFamily.UPDRAFT] = 1.0
    if edge_count >= 3 and has_left and has_right and (has_above or has_below):
        scores[CloudSpriteFamily.INTERNAL] = max(
            scores.get(CloudSpriteFamily.INTERNAL, 0.0),
            0.86,
        )
    if node.noise < 0.14 and edge_count >= 3:
        scores[CloudSpriteFamily.INTERNAL] = max(
            scores.get(CloudSpriteFamily.INTERNAL, 0.0),
            0.78,
        )
    return scores


def projected_cloud_sprite_family(
    node: CloudNode,
    edge_count: int,
    projection: ProjectedPoint,
    neighbor_projections: list[ProjectedPoint],
) -> CloudSpriteFamily:
    return select_cloud_sprite_family(
        projected_cloud_sprite_family_scores(
            node,
            edge_count,
            projection,
            neighbor_projections,
        )
    )


def select_cloud_sprite_family(
    scores: dict[CloudSpriteFamily, float],
) -> CloudSpriteFamily:
    return max(scores, key=lambda family: (scores[family], -list(CloudSpriteFamily).index(family)))


def cloud_sprite_family_scores(
    node: CloudNode,
    state: CloudState,
    camera: CameraBasis | None = None,
    projection: ProjectedPoint | None = None,
    surface_metrics: dict[int, SurfaceMetrics] | None = None,
) -> dict[CloudSpriteFamily, float]:
    if node.is_pruning or node.fade < 0.7:
        return {CloudSpriteFamily.FADE: 1.0}
    edge_count = 0
    has_primary_edge = False
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
        if edge.kind is EdgeKind.PRIMARY:
            has_primary_edge = True
        if camera is not None:
            neighbor_projection = project_point(neighbor.position, camera)
            if neighbor_projection.visible:
                neighbor_projections.append(neighbor_projection)

    if projection is not None and neighbor_projections:
        scores = projected_cloud_sprite_family_scores(
            node,
            edge_count,
            projection,
            neighbor_projections,
        )
        apply_surface_metric_family_bias(node, scores, surface_metrics)
        return scores

    scores: dict[CloudSpriteFamily, float] = {CloudSpriteFamily.STRETCH: 0.42}
    if node.updraft > 0.35:
        scores[CloudSpriteFamily.UPDRAFT] = min(1.0, 0.76 + (node.updraft - 0.35) * 0.48)
    if node.density > 1.35:
        scores[CloudSpriteFamily.BOTTOM] = min(1.0, 0.78 + (node.density - 1.35) * 0.42)
    if edge_count == 0:
        scores[CloudSpriteFamily.FRAGMENT] = 0.72
    if node.noise < 0.14 and edge_count > 0:
        scores[CloudSpriteFamily.INTERNAL] = 0.74
    if has_primary_edge:
        scores[CloudSpriteFamily.EDGE] = 0.68
    apply_surface_metric_family_bias(node, scores, surface_metrics)
    return scores


def apply_surface_metric_family_bias(
    node: CloudNode,
    scores: dict[CloudSpriteFamily, float],
    surface_metrics: dict[int, SurfaceMetrics] | None,
) -> None:
    if surface_metrics is None:
        return
    metric = surface_metrics.get(node.id)
    if metric is None:
        return
    if metric.exposure < config.CLOUD_SURFACE_INTERNAL_EXPOSURE:
        scores[CloudSpriteFamily.INTERNAL] = max(
            scores.get(CloudSpriteFamily.INTERNAL, 0.0),
            1.05,
        )
        scores[CloudSpriteFamily.EDGE] = min(scores.get(CloudSpriteFamily.EDGE, 0.0), 0.34)
        scores.pop(CloudSpriteFamily.FRAGMENT, None)
        return
    if metric.exposure < config.CLOUD_SURFACE_WEAK_EXPOSURE:
        scores[CloudSpriteFamily.INTERNAL] = max(
            scores.get(CloudSpriteFamily.INTERNAL, 0.0),
            0.82,
        )
        scores[CloudSpriteFamily.EDGE] = max(scores.get(CloudSpriteFamily.EDGE, 0.0), 0.48)
        scores.pop(CloudSpriteFamily.FRAGMENT, None)
        return
    if metric.exposure >= config.CLOUD_SURFACE_STRONG_EXPOSURE:
        scores[CloudSpriteFamily.EDGE] = max(scores.get(CloudSpriteFamily.EDGE, 0.0), 0.78)
        if node.updraft > 0.35:
            scores[CloudSpriteFamily.UPDRAFT] = max(
                scores.get(CloudSpriteFamily.UPDRAFT, 0.0),
                0.86,
            )


def choose_cloud_sprite_family(
    node: CloudNode,
    state: CloudState,
    camera: CameraBasis | None = None,
    projection: ProjectedPoint | None = None,
) -> CloudSpriteFamily:
    return select_cloud_sprite_family(
        cloud_sprite_family_scores(node, state, camera, projection)
    )
