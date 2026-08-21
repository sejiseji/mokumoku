from __future__ import annotations

import math
from dataclasses import dataclass

from src import config
from src.assets.sprite_map import (
    CloudSpriteFamily,
    SpriteRect,
    cloud_sprite_rect,
    size_class_for_screen_radius,
)
from src.camera.camera import CameraBasis
from src.camera.projection import ProjectedPoint, project_point
from src.cloud.model import CloudEdge, CloudNode, CloudState
from src.enums import EdgeKind


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


@dataclass(frozen=True)
class NodePayload:
    node: CloudNode
    projection: ProjectedPoint
    sprite: SpriteRect
    offset_x: float = 0.0
    offset_y: float = 0.0
    mesh_intensity: float = 0.0
    mesh_phase: float = 0.0


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
) -> list[RenderItem]:
    items: list[RenderItem] = []
    for edge in state.live_edges():
        node_a = state.nodes[edge.node_a]
        node_b = state.nodes[edge.node_b]
        projection_a = project_point(node_a.position, camera)
        projection_b = project_point(node_b.position, camera)
        if projection_a.visible and projection_b.visible:
            bridge = cloud_bridge_payload(
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
            if bridge is not None:
                items.append(
                    RenderItem(
                        depth=bridge.point.depth,
                        layer_bias=1,
                        stable_id=edge.id,
                        payload=bridge,
                    )
                )

    for node in state.live_nodes():
        projection = project_point(node.position, camera)
        if not projection.visible:
            continue
        offset_x, offset_y, _radius_ratio = cloud_node_wobble(node, state, frame)
        screen_radius = node.radius * projection.scale
        mesh_intensity = single_node_mesh_intensity(node, state, screen_radius)
        mesh_phase = single_node_mesh_phase(node, frame)
        sprite = cloud_sprite_rect(
            choose_cloud_sprite_family(node, state, camera, projection),
            size_class_for_screen_radius(screen_radius * max(0.45, node.fade)),
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
                ),
            )
        )

    items.sort(key=lambda item: (-item.depth, item.layer_bias, item.stable_id))
    return items


def cloud_bridge_payload(
    edge: CloudEdge,
    node_a: CloudNode,
    node_b: CloudNode,
    projection_a: ProjectedPoint,
    projection_b: ProjectedPoint,
    camera: CameraBasis,
) -> BridgePayload | None:
    if edge.strain >= config.CLOUD_BRIDGE_MAX_STRAIN:
        return None

    distance = (
        (projection_b.screen_x - projection_a.screen_x) ** 2
        + (projection_b.screen_y - projection_a.screen_y) ** 2
    ) ** 0.5
    radius_a = node_a.radius * projection_a.scale
    radius_b = node_b.radius * projection_b.scale
    if distance <= (radius_a + radius_b) * 0.55:
        return None

    midpoint = node_a.position.lerp(node_b.position, 0.5)
    projection = project_point(midpoint, camera)
    if not projection.visible:
        return None

    strain_span = max(0.001, config.CLOUD_BRIDGE_MAX_STRAIN - config.CLOUD_BRIDGE_MIN_STRAIN)
    strain_t = max(0.0, min(1.0, (edge.strain - config.CLOUD_BRIDGE_MIN_STRAIN) / strain_span))
    bridge_radius = min(radius_a, radius_b) * (0.72 - strain_t * 0.34)
    sprite = cloud_sprite_rect(
        CloudSpriteFamily.INTERNAL,
        size_class_for_screen_radius(bridge_radius),
    )
    return BridgePayload(edge, projection, sprite)


def cloud_node_wobble(
    node: CloudNode,
    state: CloudState | int | None = None,
    frame: int = 0,
) -> tuple[float, float, float]:
    if isinstance(state, int):
        frame = state
        state = None
    cluster_seed = node.cluster_id * 977
    if state is not None:
        cluster_seed += len(state.nodes)
    cluster_phase = (cluster_seed % 6283) / 1000.0
    local_phase = (node.sprite_seed % 6283) / 1000.0
    seconds = frame / config.FPS
    fade = max(0.0, min(1.0, node.fade))
    activation = 0.35 + node.activation * 0.65
    incubation_factor = max(0.0, min(1.0, node.incubation))
    calm_factor = 1.0 - incubation_factor * 0.8
    cluster_amplitude = config.CLOUD_CLUSTER_WOBBLE_OFFSET_PX * fade * calm_factor
    local_amplitude = config.CLOUD_LOCAL_WOBBLE_OFFSET_PX * fade * activation
    local_amplitude *= calm_factor

    cluster_x = (
        math.sin(seconds * math.tau / config.CLOUD_CLUSTER_WOBBLE_PERIOD_X + cluster_phase)
        * cluster_amplitude
    )
    cluster_y = (
        math.cos(
            seconds * math.tau / config.CLOUD_CLUSTER_WOBBLE_PERIOD_Y
            + cluster_phase * 1.21
        )
        * cluster_amplitude
        * 0.7
    )
    local_x = (
        math.sin(seconds * math.tau / config.CLOUD_LOCAL_WOBBLE_PERIOD_X + local_phase)
        * local_amplitude
    )
    local_y = (
        math.cos(
            seconds * math.tau / config.CLOUD_LOCAL_WOBBLE_PERIOD_Y + local_phase * 1.31
        )
        * local_amplitude
        * 0.75
    )
    return cluster_x + local_x, cluster_y + local_y, 1.0


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
        is_top = all(other.screen_y > projection.screen_y + 2.0 for other in neighbor_projections)
        is_bottom = all(
            other.screen_y < projection.screen_y - 2.0 for other in neighbor_projections
        )
        if is_bottom or node.density > 1.35:
            return CloudSpriteFamily.BOTTOM
        if is_top or node.updraft > 0.35:
            return CloudSpriteFamily.UPDRAFT
        if edge_count >= 3:
            return CloudSpriteFamily.INTERNAL

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
