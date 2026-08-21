from __future__ import annotations

from collections import deque

from src import config
from src.cloud.model import CloudCluster, CloudEdge, CloudNode, CloudState
from src.enums import EdgeKind
from src.math3d import Vec3
from src.rng import RandomSource


def node_degree(state: CloudState, node_id: int) -> int:
    return sum(
        1
        for edge in state.edges.values()
        if edge.node_a == node_id or edge.node_b == node_id
    )


def edge_length(state: CloudState, edge: CloudEdge) -> float:
    return state.nodes[edge.node_a].position.distance_to(state.nodes[edge.node_b].position)


def add_edge(
    state: CloudState,
    lineage_id: int,
    cluster_id: int,
    node_a: int,
    node_b: int,
    kind: EdgeKind,
) -> CloudEdge | None:
    if node_a == node_b or not state.can_add_edge():
        return None
    if node_degree(state, node_a) >= config.MAX_NODE_DEGREE:
        return None
    if node_degree(state, node_b) >= config.MAX_NODE_DEGREE:
        return None
    for edge in state.edges.values():
        if {edge.node_a, edge.node_b} == {node_a, node_b}:
            return edge

    rest_length = state.nodes[node_a].position.distance_to(state.nodes[node_b].position)
    edge = CloudEdge(
        id=state.next_edge_id,
        lineage_id=lineage_id,
        cluster_id=cluster_id,
        node_a=node_a,
        node_b=node_b,
        kind=kind,
        rest_length=max(1.0, rest_length),
        stiffness=0.35 if kind is EdgeKind.PRIMARY else 0.18,
        strength=1.0,
        moisture_conductivity=0.5,
        temperature_conductivity=0.25,
        charge_conductivity=0.25,
        age=0.0,
        strain=0.0,
    )
    state.edges[edge.id] = edge
    state.next_edge_id += 1
    return edge


def create_node(
    state: CloudState,
    lineage_id: int,
    cluster_id: int,
    position: Vec3,
    rng: RandomSource,
    parent_node_id: int | None = None,
    generation: int = 0,
) -> CloudNode:
    if not state.can_add_node():
        raise RuntimeError("node limit reached")

    node = CloudNode(
        id=state.next_node_id,
        lineage_id=lineage_id,
        cluster_id=cluster_id,
        position=position,
        previous_position=position,
        velocity=Vec3(0.0, 0.0, 0.0),
        mass=config.SEED_MASS,
        moisture=0.45,
        temperature=0.5,
        density=config.SEED_DENSITY,
        charge=0.0,
        updraft=0.0,
        activation=0.65,
        noise=0.35,
        incubation=0.0,
        untouched_time=0.0,
        age=0.0,
        fade=1.0,
        is_pruning=False,
        parent_node_id=parent_node_id,
        generation=generation,
        origin_evidence=state.lineages[lineage_id].total_origin_evidence,
        sprite_seed=rng.randint(0, 2**31 - 1),
        sprite_family="placeholder",
        size_class="m",
    )
    state.nodes[node.id] = node
    state.next_node_id += 1
    return node


def recompute_clusters(state: CloudState, lineage_id: int) -> None:
    lineage = state.lineages[lineage_id]
    old_cluster_age = {cluster.id: cluster.age for cluster in state.clusters.values()}
    state.clusters = {
        cluster_id: cluster
        for cluster_id, cluster in state.clusters.items()
        if cluster.lineage_id != lineage_id
    }

    lineage_node_ids = [
        node.id
        for node in state.nodes.values()
        if node.lineage_id == lineage_id and node.fade > 0.0
    ]
    if not lineage_node_ids:
        lineage.active_cluster_ids.clear()
        lineage.extinct = True
        return

    adjacency: dict[int, set[int]] = {node_id: set() for node_id in lineage_node_ids}
    edge_ids_by_pair: dict[frozenset[int], list[int]] = {}
    for edge in state.edges.values():
        if edge.lineage_id != lineage_id:
            continue
        if edge.node_a not in adjacency or edge.node_b not in adjacency:
            continue
        adjacency[edge.node_a].add(edge.node_b)
        adjacency[edge.node_b].add(edge.node_a)
        edge_ids_by_pair.setdefault(frozenset((edge.node_a, edge.node_b)), []).append(edge.id)

    visited: set[int] = set()
    lineage.active_cluster_ids.clear()
    for start_id in lineage_node_ids:
        if start_id in visited:
            continue
        queue = deque([start_id])
        visited.add(start_id)
        component: list[int] = []
        component_edge_ids: set[int] = set()
        while queue:
            node_id = queue.popleft()
            component.append(node_id)
            for other_id in adjacency[node_id]:
                component_edge_ids.update(edge_ids_by_pair[frozenset((node_id, other_id))])
                if other_id not in visited:
                    visited.add(other_id)
                    queue.append(other_id)

        cluster_id = state.next_cluster_id
        state.next_cluster_id += 1
        total_mass = sum(state.nodes[node_id].mass for node_id in component)
        centroid = sum_positions(state, component) / max(1.0, float(len(component)))
        cluster = CloudCluster(
            id=cluster_id,
            lineage_id=lineage_id,
            node_ids=sorted(component),
            edge_ids=sorted(component_edge_ids),
            centroid=centroid,
            total_mass=total_mass,
            age=old_cluster_age.get(cluster_id, 0.0),
        )
        state.clusters[cluster_id] = cluster
        lineage.active_cluster_ids.add(cluster_id)
        for node_id in component:
            state.nodes[node_id].cluster_id = cluster_id
        for edge_id in component_edge_ids:
            state.edges[edge_id].cluster_id = cluster_id


def sum_positions(state: CloudState, node_ids: list[int]) -> Vec3:
    total = Vec3(0.0, 0.0, 0.0)
    for node_id in node_ids:
        total += state.nodes[node_id].position
    return total


def break_overstretched_edges(state: CloudState) -> list[int]:
    broken: list[int] = []
    for edge in list(state.edges.values()):
        current_length = edge_length(state, edge)
        edge.strain = current_length / max(1.0, edge.rest_length)
        if edge.strain >= config.EDGE_BREAK_RATIO:
            broken.append(edge.id)
            del state.edges[edge.id]
    for lineage_id in list(state.lineages):
        recompute_clusters(state, lineage_id)
    return broken


def split_node_from_cluster(state: CloudState, node_id: int) -> list[int]:
    removed: list[int] = []
    for edge in list(state.edges.values()):
        if edge.node_a == node_id or edge.node_b == node_id:
            removed.append(edge.id)
            del state.edges[edge.id]
    if node_id in state.nodes:
        recompute_clusters(state, state.nodes[node_id].lineage_id)
    return removed


def try_merge_clusters(state: CloudState) -> list[int]:
    added_edges: list[int] = []
    for lineage in state.lineages.values():
        cluster_ids = sorted(lineage.active_cluster_ids)
        for index, cluster_a_id in enumerate(cluster_ids):
            cluster_a = state.clusters[cluster_a_id]
            for cluster_b_id in cluster_ids[index + 1 :]:
                cluster_b = state.clusters[cluster_b_id]
                best_pair: tuple[float, int, int] | None = None
                for node_a_id in cluster_a.node_ids:
                    node_a = state.nodes[node_a_id]
                    for node_b_id in cluster_b.node_ids:
                        node_b = state.nodes[node_b_id]
                        relative_speed = node_a.velocity.distance_to(node_b.velocity)
                        if relative_speed > config.MERGE_MAX_RELATIVE_SPEED:
                            continue
                        distance = node_a.position.distance_to(node_b.position)
                        if distance <= config.MERGE_DISTANCE and (
                            best_pair is None or distance < best_pair[0]
                        ):
                            best_pair = (distance, node_a_id, node_b_id)
                if best_pair is not None:
                    _, node_a_id, node_b_id = best_pair
                    edge = add_edge(
                        state,
                        lineage.id,
                        cluster_a_id,
                        node_a_id,
                        node_b_id,
                        EdgeKind.CROSSLINK,
                    )
                    if edge is not None:
                        added_edges.append(edge.id)
                        recompute_clusters(state, lineage.id)
                        return added_edges
    return added_edges
