from __future__ import annotations

import math
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
    desired_edge_rest_length,
    node_degree,
    recompute_clusters,
    split_node_from_cluster,
    try_merge_clusters,
)
from src.cloud.incubation import update_incubation
from src.cloud.model import CloudLineage, CloudNode, CloudState, OriginEvidence
from src.cloud.reaction import (
    CloudStimulus,
    ReactionEvent,
    ReactionEventKind,
    ReactionSummary,
    ReactionViewSnapshot,
    StimulusKind,
    chain_delay,
    child_reaction_limit,
    clamp01,
    propagation_selectivity,
    reaction_grade,
    response_coefficient,
    seed_ignition_delay,
    stable_hash,
    stable_hash01,
)
from src.enums import EdgeKind
from src.input.hit_test import HitTarget, hit_test
from src.math3d import Vec3
from src.rng import RandomSource


@dataclass(frozen=True)
class CloudOperationResult:
    kind: str
    node_id: int | None = None
    edge_ids: tuple[int, ...] = ()
    reaction_id: int | None = None
    created_frame: int = 0
    reaction_events: tuple[ReactionEvent, ...] = ()
    reaction_summary: ReactionSummary | None = None
    reacted_node_ids: tuple[int, ...] = ()
    spawned_node_ids: tuple[int, ...] = ()
    resonant_node_ids: tuple[int, ...] = ()


class CloudSimulation:
    def __init__(self, rng: RandomSource) -> None:
        self.rng = rng
        self.state = CloudState()
        self.elapsed_time = 0.0
        self.next_reaction_id = 1
        self.last_reaction_summaries: list[ReactionSummary] = []

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
        return self.create_dormant_seed_at_screen(screen_x, screen_y, camera)

    def create_dormant_seed_at_screen(
        self,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
    ) -> CloudOperationResult:
        if not self.state.can_add_node():
            return CloudOperationResult("node_limit")
        lineage = self.state.active_lineage()
        if lineage is None:
            lineage_id = self.state.next_lineage_id
            self.state.next_lineage_id += 1
            self.state.lineages[lineage_id] = CloudLineage(
                id=lineage_id,
                active_cluster_ids=set(),
                started_at=self.elapsed_time,
                ended_at=None,
                extinct=False,
                total_origin_evidence=OriginEvidence(),
            )
        else:
            lineage_id = lineage.id
        cluster_id = self.state.next_cluster_id
        self.state.next_cluster_id += 1
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
            if not self.state.live_nodes():
                return self.create_dormant_seed_at_screen(screen_x, screen_y, camera)
            soft_node = self.nearest_projected_node(
                screen_x,
                screen_y,
                camera,
                max_distance=config.SOFT_HIT_RADIUS_PX,
            )
            if soft_node is not None:
                return self.stimulate_node(soft_node, screen_x, screen_y, camera)
            seed = self.nearest_dormant_seed(
                screen_x,
                screen_y,
                camera,
                max_distance=config.DORMANT_SEED_TAP_RADIUS_PX,
            )
            if seed is not None:
                return self.stimulate_node(seed, screen_x, screen_y, camera)
            return self.create_dormant_seed_at_screen(screen_x, screen_y, camera)

        return self.stimulate_node(node, screen_x, screen_y, camera)

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

    def stimulate_node(
        self,
        node: CloudNode,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
    ) -> CloudOperationResult:
        reaction_id = self.next_reaction_id
        self.next_reaction_id += 1
        created_frame = int(round(self.elapsed_time * config.FPS))
        projection = project_point(node.position, camera)
        depth = projection.depth if projection.visible else config.CAMERA_DISTANCE
        stimulus = CloudStimulus(
            reaction_id=reaction_id,
            source_kind=StimulusKind.TAP,
            primary_node_id=node.id,
            lineage_id=node.lineage_id,
            screen_x=screen_x,
            screen_y=screen_y,
            world_position=screen_to_world_in_cloud_bounds(screen_x, screen_y, depth, camera),
            strength=1.0,
            radius_px=config.STIMULUS_RADIUS_PX,
            created_frame=created_frame,
            seed=stable_hash(self.rng.seed, reaction_id, node.id),
            view_snapshot=ReactionViewSnapshot(
                camera_right=camera.right,
                camera_up=camera.up,
                camera_forward=camera.forward,
                camera_position=camera.position,
                pointer_screen_x=screen_x,
                pointer_screen_y=screen_y,
            ),
        )
        events, summary, reacted_ids, spawned_ids, resonant_ids, edge_ids = (
            self.resolve_stimulus(stimulus)
        )
        self.last_reaction_summaries.append(summary)
        self.last_reaction_summaries = self.last_reaction_summaries[-16:]
        return CloudOperationResult(
            "stimulus",
            node_id=node.id,
            edge_ids=edge_ids,
            reaction_id=reaction_id,
            created_frame=created_frame,
            reaction_events=events,
            reaction_summary=summary,
            reacted_node_ids=reacted_ids,
            spawned_node_ids=spawned_ids,
            resonant_node_ids=resonant_ids,
        )

    def resolve_stimulus(
        self,
        stimulus: CloudStimulus,
    ) -> tuple[
        tuple[ReactionEvent, ...],
        ReactionSummary,
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        if stimulus.primary_node_id is None or stimulus.primary_node_id not in self.state.nodes:
            summary = ReactionSummary(
                reaction_id=stimulus.reaction_id,
                lineage_id=stimulus.lineage_id,
                primary_node_id=stimulus.primary_node_id,
                reacted_nodes=0,
                sprouts=0,
                ignited_seeds=0,
                highest_generation=0,
                duration_frames=0,
                reaction_grade=reaction_grade(0, 0, 0),
            )
            return (), summary, (), (), (), ()

        events: list[ReactionEvent] = []
        reacted_ids: list[int] = []
        spawned_ids: list[int] = []
        resonant_ids: list[int] = []
        edge_ids: list[int] = []
        visited: set[int] = set()
        scheduled: set[int] = {stimulus.primary_node_id}
        frontier: list[tuple[int, int, float, int, int | None]] = [
            (stimulus.primary_node_id, 0, 1.0, stimulus.created_frame, None)
        ]
        energies: dict[int, float] = {stimulus.primary_node_id: 1.0}
        generations: dict[int, int] = {stimulus.primary_node_id: 0}
        highest_generation = 0

        while frontier and len(reacted_ids) < config.MAX_REACTED_NODES_PER_REACTION:
            frontier.sort(key=lambda item: (item[3], item[1], item[0]))
            node_id, generation, energy, execute_frame, source_id = frontier.pop(0)
            if node_id in visited or node_id not in self.state.nodes:
                continue
            node = self.state.nodes[node_id]
            if node.fade <= 0.0 or node.is_pruning:
                continue
            visited.add(node_id)
            reacted_ids.append(node_id)
            energies[node_id] = energy
            generations[node_id] = generation
            highest_generation = max(highest_generation, generation)
            self.apply_node_reaction(node_id, energy, generation)
            events.append(
                ReactionEvent(
                    execute_frame=execute_frame,
                    stable_order=len(events),
                    reaction_id=stimulus.reaction_id,
                    kind=ReactionEventKind.NODE_PULSE,
                    target_node_id=node_id,
                    source_node_id=source_id,
                    generation=generation,
                    energy=energy,
                )
            )
            if len(events) >= config.MAX_PENDING_REACTION_EVENTS:
                break
            if generation >= config.MAX_CHAIN_GENERATION:
                continue

            for next_id, next_energy in self.propagation_targets(
                stimulus.reaction_id,
                node_id,
                energy,
                generation,
            ):
                if next_id in visited or next_id in scheduled:
                    continue
                if len(scheduled) >= config.MAX_REACTED_NODES_PER_REACTION:
                    break
                delay = chain_delay(
                    stimulus.reaction_id,
                    node_id,
                    next_id,
                    generation + 1,
                )
                next_frame = execute_frame + delay
                if next_frame - stimulus.created_frame > config.MAX_REACTION_DURATION_FRAMES:
                    continue
                scheduled.add(next_id)
                frontier.append((next_id, generation + 1, next_energy, next_frame, node_id))

        seed_candidates = self.seed_resonance_candidates(stimulus)
        for seed_id, distance_ratio, seed_energy in seed_candidates[
            : config.MAX_SEED_IGNITIONS_PER_REACTION
        ]:
            if seed_id in visited or seed_id not in self.state.nodes:
                continue
            delay = seed_ignition_delay(stimulus.reaction_id, seed_id, distance_ratio)
            execute_frame = stimulus.created_frame + delay
            if execute_frame - stimulus.created_frame > config.MAX_REACTION_DURATION_FRAMES:
                continue
            self.apply_seed_ignition(seed_id, seed_energy)
            resonant_ids.append(seed_id)
            energies[seed_id] = seed_energy
            generations[seed_id] = 1
            events.append(
                ReactionEvent(
                    execute_frame=execute_frame,
                    stable_order=len(events),
                    reaction_id=stimulus.reaction_id,
                    kind=ReactionEventKind.SEED_IGNITION,
                    target_node_id=seed_id,
                    source_node_id=stimulus.primary_node_id,
                    generation=1,
                    energy=seed_energy,
                )
            )
            if len(events) >= config.MAX_PENDING_REACTION_EVENTS:
                break

        guaranteed_primary_sprout = len(reacted_ids) <= 1 and not resonant_ids
        sprout_sources = reacted_ids + resonant_ids
        for source_id in sprout_sources:
            if len(spawned_ids) >= config.MAX_SPROUTS_PER_REACTION:
                break
            if source_id not in self.state.nodes:
                continue
            source_generation = generations.get(source_id, 0)
            source_energy = energies.get(source_id, 0.5)
            guaranteed = guaranteed_primary_sprout and source_id == stimulus.primary_node_id
            planned_count = self.secondary_sprout_count(
                stimulus.reaction_id,
                source_id,
                source_energy,
                guaranteed,
            )
            for local_index in range(planned_count):
                if len(spawned_ids) >= config.MAX_SPROUTS_PER_REACTION:
                    break
                sprout = self.create_secondary_sprout(
                    stimulus,
                    source_id,
                    source_generation + 1,
                    local_index,
                )
                if sprout is None:
                    continue
                child_id, edge_id, direction_index = sprout
                spawned_ids.append(child_id)
                if edge_id is not None:
                    edge_ids.append(edge_id)
                energies[child_id] = max(0.30, source_energy * 0.75)
                highest_generation = max(highest_generation, source_generation + 1)
                events.append(
                    ReactionEvent(
                        execute_frame=stimulus.created_frame
                        + (source_generation + 1)
                        * config.SECONDARY_SPROUT_GENERATION_DELAY_FRAMES
                        + local_index * 3,
                        stable_order=len(events),
                        reaction_id=stimulus.reaction_id,
                        kind=ReactionEventKind.SECONDARY_SPROUT,
                        target_node_id=child_id,
                        source_node_id=source_id,
                        generation=source_generation + 1,
                        energy=energies[child_id],
                        direction_index=direction_index,
                    )
                )
                if len(events) >= config.MAX_PENDING_REACTION_EVENTS:
                    break

        events.sort()
        duration = 0
        if events:
            duration = max(event.execute_frame for event in events) - stimulus.created_frame
        summary = ReactionSummary(
            reaction_id=stimulus.reaction_id,
            lineage_id=stimulus.lineage_id,
            primary_node_id=stimulus.primary_node_id,
            reacted_nodes=len(reacted_ids),
            sprouts=len(spawned_ids),
            ignited_seeds=len(resonant_ids),
            highest_generation=highest_generation,
            duration_frames=duration,
            reaction_grade=reaction_grade(len(reacted_ids), len(spawned_ids), len(resonant_ids)),
        )
        return (
            tuple(events[: config.MAX_PENDING_REACTION_EVENTS]),
            summary,
            tuple(reacted_ids),
            tuple(spawned_ids),
            tuple(resonant_ids),
            tuple(edge_ids),
        )

    def apply_node_reaction(self, node_id: int, energy: float, generation: int) -> None:
        node = self.state.nodes[node_id]
        generation_falloff = max(0.38, 1.0 - generation * 0.18)
        node.mass += config.TAP_MASS_GAIN * (0.32 + 0.24 * energy) * generation_falloff
        node.activation = min(1.0, node.activation + 0.22 * energy * generation_falloff)
        node.moisture = min(1.0, node.moisture + 0.025 * energy)
        node.noise = min(1.0, node.noise + 0.035 * energy * (1.0 - node.incubation))
        self.touch_neighborhood(node_id, graph_distance=1)

    def apply_seed_ignition(self, node_id: int, energy: float) -> None:
        node = self.state.nodes[node_id]
        node.mass += config.TAP_MASS_GAIN * (0.24 + 0.24 * energy)
        node.activation = min(1.0, node.activation + 0.38 * energy)
        node.moisture = min(1.0, node.moisture + 0.05 * energy)
        node.noise = min(1.0, node.noise + 0.05 * energy)
        self.touch_neighborhood(node_id, graph_distance=0)

    def propagation_targets(
        self,
        reaction_id: int,
        source_id: int,
        source_energy: float,
        generation: int,
    ) -> list[tuple[int, float]]:
        source = self.state.nodes[source_id]
        candidates: list[tuple[float, int, float]] = []
        for edge in self.state.live_edges():
            if edge.node_a == source_id:
                target_id = edge.node_b
            elif edge.node_b == source_id:
                target_id = edge.node_a
            else:
                continue
            target = self.state.nodes[target_id]
            if target.fade <= 0.0 or target.is_pruning:
                continue
            coherence = clamp01(node_degree(self.state, target_id) / max(1, config.MAX_NODE_DEGREE))
            response = response_coefficient(
                target.moisture,
                target.activation,
                target.updraft,
                target.noise,
                coherence,
            )
            similarity = attribute_similarity(source, target)
            stability = clamp01(1.0 - target.noise * 0.45 + target.incubation * 0.35)
            selectivity = propagation_selectivity(
                edge.strength,
                stability,
                similarity,
                target.incubation,
            )
            effective_response = 0.55 + 0.45 * response
            next_energy = (
                source_energy
                * config.CHAIN_BASE_DECAY
                * edge.strength
                * effective_response
                * selectivity
            )
            if next_energy < config.CHAIN_ENERGY_THRESHOLD:
                continue
            tie = stable_hash(reaction_id, source_id, target_id, generation)
            candidates.append((next_energy, tie, target_id))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        limit = child_reaction_limit(source.activation, source.noise, source.incubation)
        return [(target_id, energy) for energy, _tie, target_id in candidates[:limit]]

    def secondary_sprout_count(
        self,
        reaction_id: int,
        source_id: int,
        energy: float,
        guaranteed: bool,
    ) -> int:
        if source_id not in self.state.nodes:
            return 0
        node = self.state.nodes[source_id]
        if node.fade <= 0.0 or node.is_pruning:
            return 0
        if not self.state.can_add_node() or not self.state.can_add_edge():
            return 0
        if node_degree(self.state, source_id) >= config.MAX_NODE_DEGREE:
            return 0
        density = self.local_density_factor(node.position, source_id)
        score = clamp01(
            0.16
            + 0.34 * energy
            + 0.24 * node.moisture
            + 0.18 * node.updraft
            + 0.12 * node.activation
            + 0.08 * node.noise
            - 0.22 * density
            - 0.12 * node.incubation
        )
        if guaranteed and score >= 0.18:
            return 1
        if stable_hash01(reaction_id, source_id, 0x5A07) > min(0.90, score):
            return 0
        count = 1
        if (
            score >= 0.74
            and node_degree(self.state, source_id) <= config.MAX_NODE_DEGREE - 2
            and stable_hash01(reaction_id, source_id, 0x5A08) < score - 0.44
        ):
            count += 1
        if (
            score >= 0.86
            and node.noise >= 0.52
            and stable_hash01(reaction_id, source_id, 0x5A09) < score - 0.68
        ):
            count += 1
        return min(config.SECONDARY_SPROUT_NODE_MAX, count)

    def create_secondary_sprout(
        self,
        stimulus: CloudStimulus,
        parent_id: int,
        generation: int,
        local_index: int,
    ) -> tuple[int, int | None, int] | None:
        if parent_id not in self.state.nodes:
            return None
        parent = self.state.nodes[parent_id]
        if not self.state.can_add_node() or not self.state.can_add_edge():
            return None
        if node_degree(self.state, parent_id) >= config.MAX_NODE_DEGREE:
            return None
        candidate = self.best_sprout_candidate(stimulus, parent, generation, local_index)
        if candidate is None:
            return None
        position, direction_index = candidate
        local_rng = RandomSource(stable_hash(stimulus.seed, parent_id, generation, local_index))
        child = create_node(
            self.state,
            parent.lineage_id,
            parent.cluster_id,
            position,
            local_rng,
            parent_node_id=parent.id,
            generation=parent.generation + 1,
        )
        child.mass = config.SEED_MASS * 0.62
        child.activation = min(1.0, parent.activation * 0.66 + 0.22)
        child.moisture = clamp01((parent.moisture + 0.50) * 0.5)
        child.updraft = clamp01(parent.updraft + 0.08 * upward_bias(direction_index))
        child.noise = clamp01(parent.noise * 0.72 + 0.10)
        edge = add_edge(
            self.state,
            parent.lineage_id,
            parent.cluster_id,
            parent.id,
            child.id,
            EdgeKind.PRIMARY,
        )
        recompute_clusters(self.state, parent.lineage_id)
        self.touch_neighborhood(child.id, graph_distance=1)
        return child.id, None if edge is None else edge.id, direction_index

    def best_sprout_candidate(
        self,
        stimulus: CloudStimulus,
        parent: CloudNode,
        generation: int,
        local_index: int,
    ) -> tuple[Vec3, int] | None:
        snapshot = stimulus.view_snapshot
        centroid = self.cluster_centroid(parent)
        outward = parent.position - centroid
        spacing = max(
            config.SECONDARY_SPROUT_BASE_OFFSET,
            parent.radius * 0.58 + config.MIN_NODE_RADIUS,
        )
        best: tuple[float, Vec3, int] | None = None
        for direction_index in range(config.SPROUT_DIRECTION_COUNT):
            angle = math.tau * direction_index / config.SPROUT_DIRECTION_COUNT
            right_weight = math.cos(angle)
            up_weight = math.sin(angle)
            depth_jitter = (
                stable_hash01(
                    stimulus.reaction_id,
                    parent.id,
                    generation,
                    local_index,
                    direction_index,
                )
                * 2.0
                - 1.0
            ) * config.SPROUT_DEPTH_JITTER_MAX
            raw_position = (
                parent.position
                + snapshot.camera_right * (right_weight * spacing)
                + snapshot.camera_up * (up_weight * spacing * 0.86)
                + snapshot.camera_forward * depth_jitter
            )
            position = clamp_cloud_position(raw_position)
            projection = project_point(
                position,
                CameraBasis(
                    right=snapshot.camera_right,
                    up=snapshot.camera_up,
                    forward=snapshot.camera_forward,
                    position=snapshot.camera_position,
                    yaw_degrees=0.0,
                    pitch_degrees=0.0,
                ),
            )
            score = self.sprout_candidate_score(
                parent,
                position,
                direction_index,
                right_weight,
                up_weight,
                outward,
            )
            if projection.visible:
                if projection.screen_y >= config.SKY_BOTTOM_Y - 6:
                    score -= 0.35
                if projection.screen_x < -16 or projection.screen_x > config.SCREEN_WIDTH + 16:
                    score -= 0.20
            else:
                score -= 0.45
            if best is None or score > best[0]:
                best = (score, position, direction_index)
        if best is None or best[0] < 0.18:
            return None
        return best[1], best[2]

    def sprout_candidate_score(
        self,
        parent: CloudNode,
        position: Vec3,
        direction_index: int,
        right_weight: float,
        up_weight: float,
        outward: Vec3,
    ) -> float:
        score = 0.50
        if parent.updraft >= 0.35:
            score += max(0.0, up_weight) * (0.35 + parent.updraft * 0.30)
        if parent.moisture >= 0.48 and parent.updraft < 0.45:
            score += abs(right_weight) * 0.28
        if parent.noise >= 0.42:
            score += stable_hash01(parent.id, direction_index, 0x1A2B) * 0.30
        if parent.incubation >= 0.55:
            score += max(0.0, -up_weight) * 0.10
        if outward.length() > 0.01:
            try:
                outward_direction = outward.normalized()
            except ValueError:
                outward_direction = Vec3(0.0, 0.0, 0.0)
            view_direction = (
                Vec3(right_weight, up_weight, 0.0).normalized()
                if abs(right_weight) + abs(up_weight) > 0.01
                else Vec3(0.0, 0.0, 0.0)
            )
            score += max(0.0, outward_direction.x * view_direction.x) * 0.12
        nearest = self.nearest_node_distance(position, exclude_node_id=parent.id)
        if nearest is not None:
            if nearest < config.MIN_NODE_RADIUS * 0.75:
                score -= 0.45
            elif nearest < config.MIN_NODE_RADIUS * 1.4:
                score -= 0.14
            elif nearest > 42.0:
                score -= 0.08
        return score

    def seed_resonance_candidates(
        self,
        stimulus: CloudStimulus,
    ) -> list[tuple[int, float, float]]:
        primary_id = stimulus.primary_node_id
        if primary_id is None or primary_id not in self.state.nodes:
            return []
        primary_projection = project_point(
            self.state.nodes[primary_id].position,
            CameraBasis(
                right=stimulus.view_snapshot.camera_right,
                up=stimulus.view_snapshot.camera_up,
                forward=stimulus.view_snapshot.camera_forward,
                position=stimulus.view_snapshot.camera_position,
                yaw_degrees=0.0,
                pitch_degrees=0.0,
            ),
        )
        candidates: list[tuple[float, int, float, float]] = []
        for node in self.state.live_nodes():
            if node.id == primary_id or not self.is_dormant_seed(node):
                continue
            projection = project_point(
                node.position,
                CameraBasis(
                    right=stimulus.view_snapshot.camera_right,
                    up=stimulus.view_snapshot.camera_up,
                    forward=stimulus.view_snapshot.camera_forward,
                    position=stimulus.view_snapshot.camera_position,
                    yaw_degrees=0.0,
                    pitch_degrees=0.0,
                ),
            )
            if not projection.visible:
                continue
            screen_distance = math.hypot(
                projection.screen_x - stimulus.screen_x,
                projection.screen_y - stimulus.screen_y,
            )
            if screen_distance > config.SEED_RESONANCE_RADIUS_PX:
                continue
            depth_delta = (
                abs(projection.depth - primary_projection.depth)
                if primary_projection.visible
                else 0.0
            )
            if depth_delta > config.SEED_RESONANCE_DEPTH_TOLERANCE:
                continue
            distance_ratio = screen_distance / max(1.0, config.SEED_RESONANCE_RADIUS_PX)
            energy = clamp01((1.0 - distance_ratio * 0.72) * (0.55 + 0.45 * node.moisture))
            roll = stable_hash01(stimulus.reaction_id, node.id, 0x51ED)
            if candidates and roll > energy:
                continue
            candidates.append((screen_distance, node.id, distance_ratio, energy))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [
            (node_id, distance_ratio, energy)
            for _distance, node_id, distance_ratio, energy in candidates
        ]

    def nearest_dormant_seed(
        self,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
        max_distance: float,
    ) -> CloudNode | None:
        best: tuple[float, int] | None = None
        for node in self.state.live_nodes():
            if not self.is_dormant_seed(node):
                continue
            projection = project_point(node.position, camera)
            if not projection.visible:
                continue
            distance = math.hypot(projection.screen_x - screen_x, projection.screen_y - screen_y)
            if distance > max_distance:
                continue
            if best is None or (distance, node.id) < best:
                best = (distance, node.id)
        return None if best is None else self.state.nodes[best[1]]

    def is_dormant_seed(self, node: CloudNode) -> bool:
        return (
            node.fade > 0.0
            and not node.is_pruning
            and node_degree(self.state, node.id) == 0
            and node.mass <= config.SEED_MASS + config.TAP_MASS_GAIN * 0.35
            and node.age >= 0.0
        )

    def local_density_factor(self, position: Vec3, exclude_node_id: int) -> float:
        count = 0
        for node in self.state.live_nodes():
            if node.id == exclude_node_id:
                continue
            if node.position.distance_to(position) <= 28.0:
                count += 1
        return clamp01(count / 5.0)

    def nearest_node_distance(
        self,
        position: Vec3,
        exclude_node_id: int,
    ) -> float | None:
        distances = [
            node.position.distance_to(position)
            for node in self.state.live_nodes()
            if node.id != exclude_node_id
        ]
        if not distances:
            return None
        return min(distances)

    def cluster_centroid(self, node: CloudNode) -> Vec3:
        cluster = self.state.clusters.get(node.cluster_id)
        if cluster is not None:
            return cluster.centroid
        return node.position

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
        self.relax_connected_edges(dt)
        try_merge_clusters(self.state)
        update_incubation(self.state, dt)
        self.finalize_extinct_lineages()

    def relax_connected_edges(self, dt: float) -> None:
        for edge in self.state.edges.values():
            if edge.node_a not in self.state.nodes or edge.node_b not in self.state.nodes:
                continue
            first = self.state.nodes[edge.node_a]
            second = self.state.nodes[edge.node_b]
            delta = second.position - first.position
            distance = delta.length()
            if distance <= 0.001:
                continue

            edge.rest_length = desired_edge_rest_length(self.state, edge.node_a, edge.node_b)
            edge.strain = distance / max(1.0, edge.rest_length)
            compression_limit = edge.rest_length * config.EDGE_COMPRESSION_RATIO
            if compression_limit <= distance <= edge.rest_length:
                continue

            target = edge.rest_length if distance > edge.rest_length else compression_limit
            correction = (distance - target) * min(1.0, config.EDGE_COHESION_RATE * dt)
            correction *= edge.stiffness
            direction = delta / distance
            movement = direction * (correction * 0.5)

            first.previous_position = first.position
            second.previous_position = second.position
            first.position = first.position + movement
            second.position = second.position - movement
            first.velocity = (first.position - first.previous_position) * config.FPS
            second.velocity = (second.position - second.previous_position) * config.FPS

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


def attribute_similarity(first: CloudNode, second: CloudNode) -> float:
    difference = (
        abs(first.moisture - second.moisture)
        + abs(first.updraft - second.updraft)
        + abs(first.noise - second.noise)
    ) / 3.0
    return clamp01(1.0 - difference)


def upward_bias(direction_index: int) -> float:
    if config.SPROUT_DIRECTION_COUNT <= 0:
        return 0.0
    angle = math.tau * direction_index / config.SPROUT_DIRECTION_COUNT
    return max(0.0, math.sin(angle))
