from __future__ import annotations

from dataclasses import dataclass

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
class NodePayload:
    node: CloudNode
    projection: ProjectedPoint
    sprite: SpriteRect


@dataclass(frozen=True)
class RenderItem:
    depth: float
    layer_bias: int
    stable_id: int
    payload: EdgePayload | NodePayload


def collect_cloud_render_items(state: CloudState, camera: CameraBasis) -> list[RenderItem]:
    items: list[RenderItem] = []
    for edge in state.live_edges():
        node_a = state.nodes[edge.node_a]
        node_b = state.nodes[edge.node_b]
        projection_a = project_point(node_a.position, camera)
        projection_b = project_point(node_b.position, camera)
        if projection_a.visible and projection_b.visible:
            items.append(
                RenderItem(
                    depth=max(projection_a.depth, projection_b.depth),
                    layer_bias=0,
                    stable_id=edge.id,
                    payload=EdgePayload(edge, projection_a, projection_b),
                )
            )

    for node in state.live_nodes():
        projection = project_point(node.position, camera)
        if not projection.visible:
            continue
        screen_radius = node.radius * projection.scale
        sprite = cloud_sprite_rect(
            choose_cloud_sprite_family(node, state),
            size_class_for_screen_radius(screen_radius * max(0.45, node.fade)),
        )
        items.append(
            RenderItem(
                depth=projection.depth,
                layer_bias=1,
                stable_id=node.id,
                payload=NodePayload(node, projection, sprite),
            )
        )

    items.sort(key=lambda item: (-item.depth, item.layer_bias, item.stable_id))
    return items


def choose_cloud_sprite_family(node: CloudNode, state: CloudState) -> CloudSpriteFamily:
    if node.is_pruning or node.fade < 0.7:
        return CloudSpriteFamily.FADE
    if node.parent_node_id is None:
        return CloudSpriteFamily.INTERNAL
    edge_count = sum(
        1
        for edge in state.edges.values()
        if edge.node_a == node.id or edge.node_b == node.id
    )
    if edge_count == 0:
        return CloudSpriteFamily.FRAGMENT
    if node.updraft > 0.35:
        return CloudSpriteFamily.UPDRAFT
    if node.density > 1.35:
        return CloudSpriteFamily.BOTTOM
    if node.noise < 0.14:
        return CloudSpriteFamily.INTERNAL
    if any(
        edge.kind is EdgeKind.PRIMARY and (edge.node_a == node.id or edge.node_b == node.id)
        for edge in state.edges.values()
    ):
        return CloudSpriteFamily.EDGE
    return CloudSpriteFamily.STRETCH
