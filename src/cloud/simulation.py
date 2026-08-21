from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.camera.camera import CameraBasis
from src.camera.interaction_plane import (
    clamp_cloud_position,
    depth_locked_drag_target,
    screen_to_world_in_cloud_bounds,
)
from src.camera.projection import project_point
from src.cloud.graph import (
    add_edge,
    break_overstretched_edges,
    create_node,
    recompute_clusters,
    split_node_from_cluster,
    try_merge_clusters,
)
from src.cloud.incubation import update_incubation
from src.cloud.model import CloudLineage, CloudNode, CloudState, OriginEvidence
from src.enums import EdgeKind
from src.input.hit_test import HitTarget, hit_test
from src.math3d import Vec3
from src.rng import RandomSource


@dataclass(frozen=True)
class CloudOperationResult:
    kind: str
    node_id: int | None = None
    edge_ids: tuple[int, ...] = ()


class CloudSimulation:
    def __init__(self, rng: RandomSource) -> None:
        self.rng = rng
        self.state = CloudState()
        self.elapsed_time = 0.0

    def has_active_cloud(self) -> bool:
        return self.state.active_lineage() is not None

    def make_hit_targets(self) -> list[HitTarget]:
        return [
            HitTarget(node.id, node.position, node.radius)
            for node in self.state.nodes.values()
            if node.fade > 0.0
        ]

    def hit_node(
        self,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
        previous_selected_id: int | None = None,
    ) -> CloudNode | None:
        candidate = hit_test(
            screen_x,
            screen_y,
            self.make_hit_targets(),
            camera,
            previous_selected_id=previous_selected_id,
        )
        if candidate is None:
            return None
        return self.state.nodes[candidate.target.stable_id]

    def create_seed_at_screen(
        self,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
    ) -> CloudOperationResult:
        if self.state.active_lineage() is not None:
            return CloudOperationResult("ignored_existing_lineage")
        lineage_id = self.state.next_lineage_id
        self.state.next_lineage_id += 1
        cluster_id = self.state.next_cluster_id
        self.state.next_cluster_id += 1
        self.state.lineages[lineage_id] = CloudLineage(
            id=lineage_id,
            active_cluster_ids={cluster_id},
            started_at=self.elapsed_time,
            ended_at=None,
            extinct=False,
            total_origin_evidence=OriginEvidence(),
        )
        position = screen_to_world_in_cloud_bounds(
            screen_x,
            screen_y,
            config.CAMERA_DISTANCE,
            camera,
        )
        node = create_node(self.state, lineage_id, cluster_id, position, self.rng)
        recompute_clusters(self.state, lineage_id)
        return CloudOperationResult("seed", node_id=node.id)

    def tap_screen(
        self, screen_x: float, screen_y: float, camera: CameraBasis
    ) -> CloudOperationResult:
        node = self.hit_node(screen_x, screen_y, camera)
        if node is None:
            if self.state.active_lineage() is None:
                return self.create_seed_at_screen(screen_x, screen_y, camera)
            nearest = self.nearest_projected_node(screen_x, screen_y, camera)
            if nearest is None:
                return CloudOperationResult("ripple")
            nearest_projection = project_point(nearest.position, camera)
            screen_distance = (
                (nearest_projection.screen_x - screen_x) ** 2
                + (nearest_projection.screen_y - screen_y) ** 2
            ) ** 0.5
            child = self.add_child_near_screen(
                nearest,
                screen_x,
                screen_y,
                camera,
                connect=screen_distance <= config.TAP_ATTACH_DISTANCE,
            )
            if child is None:
                return CloudOperationResult("node_limit")
            self.touch_neighborhood(child.id)
            return CloudOperationResult("child", node_id=child.id)

        self.grow_node(node.id)
        return CloudOperationResult("tap", node_id=node.id)

    def long_press_node(self, node_id: int) -> CloudOperationResult:
        node = self.state.nodes[node_id]
        node.moisture = min(1.0, node.moisture + config.BASE_CONDENSATION_GAIN * 0.05)
        node.density = min(3.0, node.density + config.BASE_DENSITY_GAIN)
        node.mass += config.BASE_CONDENSATION_GAIN
        node.activation = min(1.0, node.activation + 0.2)
        self.touch_neighborhood(node_id)
        return CloudOperationResult("long_press", node_id=node_id)

    def drag_node_to_screen(
        self,
        node_id: int,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
    ) -> CloudOperationResult:
        node = self.state.nodes[node_id]
        target = depth_locked_drag_target(node.position, screen_x, screen_y, camera)
        self.move_node(node_id, target)
        self.touch_neighborhood(node_id)
        broken = break_overstretched_edges(self.state)
        merged = try_merge_clusters(self.state)
        return CloudOperationResult("drag", node_id=node_id, edge_ids=tuple(broken + merged))

    def flick_node(self, node_id: int, velocity: Vec3) -> CloudOperationResult:
        node = self.state.nodes[node_id]
        node.velocity = velocity
        removed = split_node_from_cluster(self.state, node_id)
        self.touch_neighborhood(node_id)
        return CloudOperationResult("flick", node_id=node_id, edge_ids=tuple(removed))

    def grow_node(self, node_id: int) -> None:
        node = self.state.nodes[node_id]
        node.mass += config.TAP_MASS_GAIN
        node.activation = min(1.0, node.activation + config.TAP_ACTIVATION_GAIN)
        node.noise = min(1.0, node.noise + 0.08)
        self.touch_neighborhood(node_id)

    def add_child_near_screen(
        self,
        parent: CloudNode,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
        connect: bool = True,
    ) -> CloudNode | None:
        if not self.state.can_add_node():
            return None
        projection = project_point(parent.position, camera)
        if not projection.visible:
            return None
        target_x = screen_x
        target_y = screen_y
        if connect:
            target_x = projection.screen_x + (
                screen_x - projection.screen_x
            ) * config.CONNECTED_TAP_SCREEN_BLEND
            target_y = projection.screen_y + (
                screen_y - projection.screen_y
            ) * config.CONNECTED_TAP_SCREEN_BLEND
        position = screen_to_world_in_cloud_bounds(target_x, target_y, projection.depth, camera)
        depth_offset = self.rng.uniform(
            -config.CHILD_DEPTH_OFFSET_MAX,
            config.CHILD_DEPTH_OFFSET_MAX,
        )
        if connect:
            position = position + camera.right * self.rng.uniform(-2.0, 2.0)
            position = position + Vec3(0.0, 0.0, depth_offset)
            position = clamp_cloud_position(position)
        child = create_node(
            self.state,
            parent.lineage_id,
            parent.cluster_id,
            position,
            self.rng,
            parent_node_id=parent.id if connect else None,
            generation=parent.generation + 1,
        )
        if connect:
            add_edge(
                self.state,
                parent.lineage_id,
                parent.cluster_id,
                parent.id,
                child.id,
                EdgeKind.PRIMARY,
            )
        recompute_clusters(self.state, parent.lineage_id)
        return child

    def move_node(self, node_id: int, target: Vec3) -> None:
        node = self.state.nodes[node_id]
        delta = target - node.position
        node.previous_position = node.position
        node.position = target
        node.velocity = delta * config.FPS
        for edge in self.state.edges.values():
            if edge.node_a == node_id:
                follower = self.state.nodes[edge.node_b]
            elif edge.node_b == node_id:
                follower = self.state.nodes[edge.node_a]
            else:
                continue
            follower.previous_position = follower.position
            follow_delta = delta * config.SPRING_FOLLOW_RATIO * edge.stiffness
            follower.position = follower.position + follow_delta
            follower.velocity = (follower.position - follower.previous_position) * config.FPS

    def nearest_projected_node(
        self,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
        max_distance: float | None = None,
    ) -> CloudNode | None:
        best: tuple[float, int] | None = None
        for node in self.state.nodes.values():
            projection = project_point(node.position, camera)
            if not projection.visible:
                continue
            dx = projection.screen_x - screen_x
            dy = projection.screen_y - screen_y
            distance = (dx * dx + dy * dy) ** 0.5
            if max_distance is not None and distance > max_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, node.id)
        return None if best is None else self.state.nodes[best[1]]

    def touch_neighborhood(self, node_id: int, graph_distance: int = 2) -> None:
        adjacency: dict[int, set[int]] = {node_id: set() for node_id in self.state.nodes}
        for edge in self.state.edges.values():
            adjacency.setdefault(edge.node_a, set()).add(edge.node_b)
            adjacency.setdefault(edge.node_b, set()).add(edge.node_a)

        frontier = {node_id}
        visited = {node_id}
        for _ in range(graph_distance + 1):
            next_frontier: set[int] = set()
            for current_id in frontier:
                node = self.state.nodes[current_id]
                node.untouched_time = 0.0
                node.incubation = 0.0
                if node.fade > 0.0 and not node.is_pruning:
                    node.is_pruning = False
                for neighbor_id in adjacency.get(current_id, set()):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
            frontier = next_frontier

    def update(self, dt: float) -> None:
        self.elapsed_time += dt
        for node in self.state.nodes.values():
            node.age += dt
            node.untouched_time += dt
            node.activation = max(0.0, node.activation - dt * 0.12)
        for edge in self.state.edges.values():
            edge.age += dt
        try_merge_clusters(self.state)
        update_incubation(self.state, dt)
        self.finalize_extinct_lineages()

    def advance_time(self, seconds: float, step: float = 1.0 / 60.0) -> None:
        remaining = seconds
        while remaining > 0.0:
            dt = min(step, remaining)
            self.update(dt)
            remaining -= dt

    def finalize_extinct_lineages(self) -> None:
        for lineage in self.state.lineages.values():
            has_live_node = any(
                node.lineage_id == lineage.id and node.fade > 0.0
                for node in self.state.nodes.values()
            )
            if not has_live_node:
                lineage.extinct = True
                if lineage.ended_at is None:
                    lineage.ended_at = self.elapsed_time
