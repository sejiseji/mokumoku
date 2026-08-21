from __future__ import annotations

from src import config
from src.cloud.graph import recompute_clusters
from src.cloud.model import CloudNode, CloudState
from src.math3d import Vec3


def build_adjacency(state: CloudState) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in state.nodes}
    for edge in state.edges.values():
        if edge.node_a in state.nodes and edge.node_b in state.nodes:
            adjacency[edge.node_a].add(edge.node_b)
            adjacency[edge.node_b].add(edge.node_a)
    return adjacency


def update_incubation(state: CloudState, dt: float) -> None:
    adjacency = build_adjacency(state)
    mature_positions: dict[int, Vec3] = {}

    for node in state.nodes.values():
        if node.is_pruning:
            node.fade = max(0.0, node.fade - dt / config.PRUNE_FADE_SECONDS)
            continue

        if node.untouched_time >= config.INCUBATION_START_SECONDS:
            node.incubation += dt * config.INCUBATION_GAIN_RATE
            node.noise = max(0.0, node.noise - dt * config.NOISE_DECAY_RATE)
            if node.untouched_time >= config.NATURAL_MASS_DECAY_START_SECONDS:
                node.mass = max(0.0, node.mass - dt * config.NATURAL_MASS_DECAY_RATE)

        if node.incubation > 0.0 and adjacency.get(node.id):
            mature_positions[node.id] = smoothed_position(node, adjacency[node.id], state, dt)

        if node.mass <= config.PRUNE_MASS_THRESHOLD:
            node.is_pruning = True

    for node_id, position in mature_positions.items():
        if node_id in state.nodes and not state.nodes[node_id].is_pruning:
            node = state.nodes[node_id]
            node.previous_position = node.position
            node.position = position
            node.velocity = (node.position - node.previous_position) * config.FPS

    merge_redundant_nodes(state)
    remove_faded_nodes(state)
    refresh_all_clusters(state)


def smoothed_position(
    node: CloudNode,
    neighbor_ids: set[int],
    state: CloudState,
    dt: float,
) -> Vec3:
    total = Vec3(0.0, 0.0, 0.0)
    count = 0
    for neighbor_id in neighbor_ids:
        neighbor = state.nodes[neighbor_id]
        if neighbor.fade > 0.0:
            total += neighbor.position
            count += 1
    if count == 0:
        return node.position
    average = total / count
    smoothing = min(1.0, config.SMOOTHING_RATE * dt * max(1.0, node.incubation))
    return node.position.lerp(average, smoothing)


def merge_redundant_nodes(state: CloudState) -> list[int]:
    removed: list[int] = []
    node_ids = sorted(state.nodes)
    for index, node_a_id in enumerate(node_ids):
        if node_a_id not in state.nodes:
            continue
        node_a = state.nodes[node_a_id]
        if node_a.is_pruning or node_a.untouched_time < config.REDUNDANT_MERGE_MIN_UNTOUCHED:
            continue
        for node_b_id in node_ids[index + 1 :]:
            if node_b_id not in state.nodes:
                continue
            node_b = state.nodes[node_b_id]
            if node_b.lineage_id != node_a.lineage_id or node_b.is_pruning:
                continue
            if node_b.untouched_time < config.REDUNDANT_MERGE_MIN_UNTOUCHED:
                continue
            if node_a.position.distance_to(node_b.position) > config.REDUNDANT_MERGE_DISTANCE:
                continue
            absorb_node(state, keep_id=node_a_id, remove_id=node_b_id)
            removed.append(node_b_id)
            break
    return removed


def absorb_node(state: CloudState, keep_id: int, remove_id: int) -> None:
    keep = state.nodes[keep_id]
    remove = state.nodes[remove_id]
    total_mass = max(0.001, keep.mass + remove.mass)
    keep.position = (keep.position * keep.mass + remove.position * remove.mass) / total_mass
    keep.mass = total_mass
    keep.moisture = max(keep.moisture, remove.moisture)
    keep.density = (keep.density + remove.density) * 0.5
    keep.noise = min(keep.noise, remove.noise)
    keep.incubation = max(keep.incubation, remove.incubation)

    for edge in list(state.edges.values()):
        if edge.node_a == remove_id:
            edge.node_a = keep_id
        if edge.node_b == remove_id:
            edge.node_b = keep_id
        if edge.node_a == edge.node_b:
            del state.edges[edge.id]

    del state.nodes[remove_id]


def remove_faded_nodes(state: CloudState) -> list[int]:
    removed: list[int] = []
    for node_id, node in list(state.nodes.items()):
        if node.is_pruning and node.fade <= 0.0:
            removed.append(node_id)
            del state.nodes[node_id]
    if removed:
        for edge_id, edge in list(state.edges.items()):
            if edge.node_a in removed or edge.node_b in removed:
                del state.edges[edge_id]
    return removed


def refresh_all_clusters(state: CloudState) -> None:
    for lineage_id in list(state.lineages):
        recompute_clusters(state, lineage_id)
