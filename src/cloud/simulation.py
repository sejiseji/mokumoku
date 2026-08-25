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
from src.camera.projection import camera_depth, project_point
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
    RadialActionKind,
    RadialActionPlan,
    RadialZone,
    ReactionEvent,
    ReactionEventKind,
    ReactionSummary,
    ReactionViewSnapshot,
    StimulusKind,
    chain_delay,
    charge_level,
    child_reaction_limit,
    clamp01,
    new_seed_budget,
    productive_budget,
    propagation_selectivity,
    radial_strength,
    radial_zone,
    reaction_grade,
    reaction_radius_px,
    response_coefficient,
    seed_ignition_delay,
    stable_hash,
    stable_hash01,
    wave_arrival_frame,
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


@dataclass(frozen=True)
class SeedEcologyResult:
    bloomed: bool
    visual_energy: float
    threshold: float


@dataclass(frozen=True)
class ClusterShapeSample:
    centroid: Vec3
    min_height: float
    max_height: float
    max_planar_radius: float
    sector_counts: tuple[int, int, int, int]
    horizontal_axis_count: int
    depth_axis_count: int
    node_count: int


class CloudSimulation:
    def __init__(self, rng: RandomSource) -> None:
        self.rng = rng
        self.state = CloudState()
        self.elapsed_time = 0.0
        self.next_reaction_id = 1
        self.pending_node_reveals: dict[int, int] = {}
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

    def radial_reaction_screen(
        self,
        screen_x: float,
        screen_y: float,
        hold_seconds: float,
        camera: CameraBasis,
        target_node_id: int | None = None,
    ) -> CloudOperationResult:
        reaction_id = self.next_reaction_id
        self.next_reaction_id += 1
        release_frame = int(round(self.elapsed_time * config.FPS))
        charge = charge_level(hold_seconds)
        radius_px = reaction_radius_px(charge)
        origin_world = screen_to_world_in_cloud_bounds(
            screen_x,
            screen_y,
            config.CAMERA_DISTANCE,
            camera,
        )
        plane_depth = camera_depth(origin_world, camera)
        local_density = self.local_reaction_density(
            screen_x,
            screen_y,
            radius_px,
            camera,
            plane_depth,
        )
        target_node = self.radial_primary_node(
            screen_x,
            screen_y,
            camera,
            target_node_id,
        )
        lineage_id = (
            target_node.lineage_id
            if target_node is not None
            else self.ensure_active_lineage().id
        )
        snapshot = ReactionViewSnapshot(
            camera_right=camera.right,
            camera_up=camera.up,
            camera_forward=camera.forward,
            camera_position=camera.position,
            pointer_screen_x=screen_x,
            pointer_screen_y=screen_y,
        )
        plans = self.plan_radial_actions(
            reaction_id,
            release_frame,
            screen_x,
            screen_y,
            origin_world,
            plane_depth,
            radius_px,
            charge,
            local_density,
            camera,
            lineage_id,
            target_node.id if target_node is not None else None,
        )
        return self.apply_radial_plans(
            plans,
            reaction_id,
            release_frame,
            charge,
            radius_px,
            local_density,
            lineage_id,
            snapshot,
        )

    def ensure_active_lineage(self) -> CloudLineage:
        lineage = self.state.active_lineage()
        if lineage is not None:
            return lineage
        lineage_id = self.state.next_lineage_id
        self.state.next_lineage_id += 1
        lineage = CloudLineage(
            id=lineage_id,
            active_cluster_ids=set(),
            started_at=self.elapsed_time,
            ended_at=None,
            extinct=False,
            total_origin_evidence=OriginEvidence(),
        )
        self.state.lineages[lineage_id] = lineage
        return lineage

    def radial_primary_node(
        self,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
        target_node_id: int | None,
    ) -> CloudNode | None:
        if target_node_id is not None and target_node_id in self.state.nodes:
            node = self.state.nodes[target_node_id]
            if node.fade > 0.0 and not node.is_pruning:
                return node
        node = self.hit_node(screen_x, screen_y, camera)
        if node is not None:
            return node
        return self.nearest_projected_node(
            screen_x,
            screen_y,
            camera,
            max_distance=config.SOFT_HIT_RADIUS_PX,
        )

    def local_reaction_density(
        self,
        screen_x: float,
        screen_y: float,
        radius_px: float,
        camera: CameraBasis,
        plane_depth: float,
    ) -> float:
        grid = max(3, config.DENSITY_SAMPLE_GRID_SIZE)
        covered = 0
        inside = 0
        step = (radius_px * 2.0) / max(1, grid - 1)
        for row in range(grid):
            sample_y = screen_y - radius_px + row * step
            for col in range(grid):
                sample_x = screen_x - radius_px + col * step
                sample_distance = math.hypot(sample_x - screen_x, sample_y - screen_y)
                if sample_distance > radius_px:
                    continue
                inside += 1
                if self.sample_covered_by_cloud(sample_x, sample_y, camera, plane_depth):
                    covered += 1
        if inside <= 0:
            return 0.0
        coverage = covered / inside
        coverage_density = coverage / max(0.01, config.DENSE_REFERENCE_COVERAGE)
        nearby_nodes = 0
        for node in self.state.live_nodes():
            projection = project_point(node.position, camera)
            if not projection.visible:
                continue
            if abs(projection.depth - plane_depth) > config.REACTION_DEPTH_TOLERANCE:
                continue
            if math.hypot(projection.screen_x - screen_x, projection.screen_y - screen_y) <= (
                radius_px
            ):
                nearby_nodes += 1
        node_reference = max(3.0, radius_px / 10.0)
        node_density = nearby_nodes / node_reference
        return clamp01(max(coverage_density, node_density * 0.85))

    def sample_covered_by_cloud(
        self,
        sample_x: float,
        sample_y: float,
        camera: CameraBasis,
        plane_depth: float,
    ) -> bool:
        for node in self.state.live_nodes():
            projection = project_point(node.position, camera)
            if not projection.visible:
                continue
            if abs(projection.depth - plane_depth) > config.REACTION_DEPTH_TOLERANCE:
                continue
            screen_radius = max(3.0, node.radius * projection.scale)
            if math.hypot(projection.screen_x - sample_x, projection.screen_y - sample_y) <= (
                screen_radius
            ):
                return True
        return False

    def plan_radial_actions(
        self,
        reaction_id: int,
        release_frame: int,
        screen_x: float,
        screen_y: float,
        origin_world: Vec3,
        plane_depth: float,
        radius_px: float,
        charge: float,
        local_density: float,
        camera: CameraBasis,
        lineage_id: int,
        primary_node_id: int | None,
    ) -> list[RadialActionPlan]:
        plans: list[RadialActionPlan] = []
        used_node_ids: set[int] = set()
        existing = self.radial_existing_candidates(
            reaction_id,
            screen_x,
            screen_y,
            radius_px,
            charge,
            camera,
            plane_depth,
        )
        if primary_node_id is not None:
            existing.sort(key=lambda item: (item[0].id != primary_node_id, item[2], item[0].id))
        productive_existing = [
            candidate for candidate in existing if not self.is_dormant_seed(candidate[0])
        ]
        dormant_candidates = [
            candidate for candidate in existing if self.is_dormant_seed(candidate[0])
        ]
        existing_budget = min(
            productive_budget(charge),
            max(1, round(productive_budget(charge) * (0.25 + 0.75 * local_density))),
        )
        for node, strength, distance_px, norm in productive_existing[:existing_budget]:
            used_node_ids.add(node.id)
            plans.append(
                self.make_radial_plan(
                    RadialActionKind.GROW_EXISTING,
                    reaction_id,
                    release_frame,
                    node.id,
                    None,
                    screen_x,
                    screen_y,
                    node.position,
                    distance_px,
                    norm,
                    strength,
                    charge,
                )
            )

        seed_ignition_budget = min(
            config.MAX_SEED_IGNITIONS_PER_REACTION,
            max(0, productive_budget(charge) - len(plans)),
        )
        for node, strength, distance_px, norm in dormant_candidates[:seed_ignition_budget]:
            used_node_ids.add(node.id)
            plans.append(
                self.make_radial_plan(
                    RadialActionKind.IGNITE_DORMANT,
                    reaction_id,
                    release_frame,
                    node.id,
                    None,
                    screen_x,
                    screen_y,
                    node.position,
                    distance_px,
                    norm,
                    max(0.16, strength),
                    charge,
                )
            )

        visual_candidates = [
            candidate for candidate in existing if candidate[0].id not in used_node_ids
        ]
        visual_budget = min(config.MAX_VISUAL_WAVE_HITS, len(visual_candidates))
        for node, strength, distance_px, norm in visual_candidates[:visual_budget]:
            used_node_ids.add(node.id)
            plans.append(
                self.make_radial_plan(
                    RadialActionKind.PULSE_EXISTING,
                    reaction_id,
                    release_frame,
                    node.id,
                    None,
                    screen_x,
                    screen_y,
                    node.position,
                    distance_px,
                    norm,
                    max(0.12, strength * 0.45),
                    charge,
                )
            )

        create_budget = new_seed_budget(charge, local_density)
        center_has_existing = any(
            math.hypot(
                project_point(node.position, camera).screen_x - screen_x,
                project_point(node.position, camera).screen_y - screen_y,
            )
            <= config.REACTION_REUSE_RADIUS_PX
            for node in self.state.live_nodes()
            if abs(camera_depth(node.position, camera) - plane_depth)
            <= config.REACTION_DEPTH_TOLERANCE
        )
        if not center_has_existing and local_density < 0.25:
            create_budget = max(config.BASE_NEW_SEEDS_PER_REACTION, create_budget)
        create_candidates = self.radial_create_candidates(
            reaction_id,
            screen_x,
            screen_y,
            radius_px,
            charge,
            camera,
            plane_depth,
            origin_world,
            lineage_id,
            primary_node_id,
        )
        selected_positions: list[tuple[float, float]] = []
        stack_source_id = (
            primary_node_id
            if self.radial_stack_factor(primary_node_id, camera) > 0.0
            else None
        )
        stack_attached_seeds = 0
        for screen_cx, screen_cy, position, norm, score in create_candidates:
            if len(selected_positions) >= create_budget:
                break
            if any(
                math.hypot(screen_cx - x, screen_cy - y)
                < config.MIN_GENERATED_SEED_DISTANCE_PX
                for x, y in selected_positions
            ):
                continue
            seed_order = len(selected_positions)
            strength = radial_strength(norm, charge) * (0.55 + 0.45 * score)
            selected_positions.append((screen_cx, screen_cy))
            source_node_id = None
            if (
                stack_source_id is not None
                and norm <= config.RADIAL_STACK_ATTACH_MAX_NORMALIZED_RADIUS
                and stack_attached_seeds < config.RADIAL_STACK_ATTACH_MAX_SEEDS
            ):
                source_node_id = stack_source_id
                stack_attached_seeds += 1
            plans.append(
                self.make_radial_plan(
                    RadialActionKind.CREATE_SEED,
                    reaction_id,
                    release_frame,
                    None,
                    source_node_id,
                    screen_cx,
                    screen_cy,
                    position,
                    math.hypot(screen_cx - screen_x, screen_cy - screen_y),
                    norm,
                    strength,
                    charge,
                    extra_delay_frames=seed_order
                    * config.RADIAL_SEED_REVEAL_STAGGER_FRAMES,
                )
            )
        plans.sort(
            key=lambda plan: (
                plan.execute_frame,
                plan.normalized_radius,
                plan.action_kind.name,
            )
        )
        return plans[: config.MAX_PENDING_REACTION_EVENTS]

    def make_radial_plan(
        self,
        action_kind: RadialActionKind,
        reaction_id: int,
        release_frame: int,
        target_node_id: int | None,
        source_node_id: int | None,
        screen_x: float,
        screen_y: float,
        world_position: Vec3,
        distance_px: float,
        normalized_radius: float,
        strength: float,
        charge: float,
        extra_delay_frames: int = 0,
    ) -> RadialActionPlan:
        execute_frame = wave_arrival_frame(
            release_frame,
            distance_px,
            reaction_id,
            target_node_id or int(screen_x * 17 + screen_y * 31),
        ) + max(0, extra_delay_frames)
        return RadialActionPlan(
            action_kind=action_kind,
            target_node_id=target_node_id,
            source_node_id=source_node_id,
            screen_x=screen_x,
            screen_y=screen_y,
            world_position=world_position,
            normalized_radius=clamp01(normalized_radius),
            radial_strength=radial_strength(normalized_radius, charge),
            effective_strength=clamp01(strength),
            execute_frame=execute_frame,
            zone=radial_zone(normalized_radius),
        )

    def radial_existing_candidates(
        self,
        reaction_id: int,
        screen_x: float,
        screen_y: float,
        radius_px: float,
        charge: float,
        camera: CameraBasis,
        plane_depth: float,
    ) -> list[tuple[CloudNode, float, float, float]]:
        candidates: list[tuple[float, int, CloudNode, float, float, float]] = []
        for node in self.state.live_nodes():
            if node.is_pruning:
                continue
            projection = project_point(node.position, camera)
            if not projection.visible:
                continue
            if abs(projection.depth - plane_depth) > config.REACTION_DEPTH_TOLERANCE:
                continue
            distance = math.hypot(projection.screen_x - screen_x, projection.screen_y - screen_y)
            if distance > radius_px:
                continue
            normalized = distance / max(1.0, radius_px)
            strength = radial_strength(normalized, charge)
            priority = (
                normalized,
                stable_hash(reaction_id, node.id, int(distance * 10.0)),
            )
            candidates.append((priority[0], priority[1], node, strength, distance, normalized))
        candidates.sort(key=lambda item: (item[0], item[1], item[2].id))
        return [
            (node, strength, distance, normalized)
            for _p, _t, node, strength, distance, normalized in candidates
        ]

    def radial_create_candidates(
        self,
        reaction_id: int,
        screen_x: float,
        screen_y: float,
        radius_px: float,
        charge: float,
        camera: CameraBasis,
        plane_depth: float,
        origin_world: Vec3,
        lineage_id: int,
        primary_node_id: int | None = None,
    ) -> list[tuple[float, float, Vec3, float, float]]:
        del origin_world
        candidates: list[tuple[float, float, float, float, Vec3, float]] = []
        birth_basis = radial_birth_basis(camera)
        shape_sample = self.cluster_shape_sample(lineage_id, birth_basis, primary_node_id)
        stack_t = self.radial_stack_factor(primary_node_id, camera)
        horizontal_radius = radial_birth_horizontal_radius(radius_px)
        if stack_t > 0.0:
            horizontal_radius *= lerp_config(
                1.0,
                config.RADIAL_STACK_HORIZONTAL_SCALE,
                stack_t,
            )
        vertical_ratio = lerp_config(
            config.RADIAL_BIRTH_VERTICAL_RATIO_MIN,
            config.RADIAL_BIRTH_VERTICAL_RATIO_MAX,
            charge,
        )
        depth_ratio = lerp_config(
            config.RADIAL_BIRTH_DEPTH_RATIO_MIN,
            config.RADIAL_BIRTH_DEPTH_RATIO_MAX,
            charge,
        )
        vertical_radius = horizontal_radius * vertical_ratio
        depth_radius = horizontal_radius * depth_ratio * lerp_config(
            1.0,
            config.RADIAL_STACK_DEPTH_SCALE,
            stack_t,
        )
        center_world = screen_to_world_in_cloud_bounds(
            screen_x,
            screen_y,
            plane_depth,
            camera,
        )
        center_world = clamp_radial_birth_center(
            center_world
            + birth_basis[1]
            * (
                vertical_radius
                * (
                    config.RADIAL_BIRTH_CENTER_UP_SHIFT_RATIO
                    + config.RADIAL_STACK_CENTER_UP_SHIFT_RATIO * stack_t
                )
            ),
            vertical_radius,
            depth_radius,
        )
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        raw_points: list[tuple[float, float, int]] = [(0.0, 0.0, 0)]
        for index in range(1, config.RADIAL_CANDIDATE_COUNT + 1):
            base_radius = math.sqrt(index / (config.RADIAL_CANDIDATE_COUNT + 1))
            angle = index * golden_angle
            jitter_r = (stable_hash01(reaction_id, index, 0xCA11) - 0.5) * 0.12
            jitter_a = (stable_hash01(reaction_id, index, 0xCA12) - 0.5) * 0.42
            radial = max(0.0, min(1.0, base_radius + jitter_r))
            raw_points.append((radial, angle + jitter_a, index))

        for normalized, angle, index in raw_points:
            distance = normalized * radius_px
            candidate_x = screen_x + math.cos(angle) * distance
            candidate_y = screen_y + math.sin(angle) * distance
            if stack_t > 0.0 and index > 0:
                candidate_y -= (
                    radius_px
                    * config.RADIAL_STACK_SCREEN_UP_SHIFT_RATIO
                    * stack_t
                    * (0.35 + normalized * 0.65)
                )
                distance = math.hypot(candidate_x - screen_x, candidate_y - screen_y)
            if candidate_y >= config.SKY_BOTTOM_Y - 6:
                continue
            if candidate_x < 8 or candidate_x > config.SCREEN_WIDTH - 8:
                continue
            if candidate_y < 16 or candidate_y > config.SKY_BOTTOM_Y - 10:
                continue
            if self.near_projected_node(
                candidate_x,
                candidate_y,
                camera,
                plane_depth,
                config.REACTION_REUSE_RADIUS_PX,
            ):
                continue
            if index == 0:
                position = screen_to_world_in_cloud_bounds(
                    candidate_x,
                    candidate_y,
                    plane_depth,
                    camera,
                )
            else:
                screen_u = (candidate_x - screen_x) / max(1.0, radius_px)
                screen_v = (screen_y - candidate_y) / max(1.0, radius_px)
                depth_factor = radial_birth_depth_factor(index)
                position = lift_radial_candidate_into_ellipsoid(
                    center_world,
                    birth_basis,
                    screen_u,
                    screen_v,
                    depth_factor,
                    horizontal_radius,
                    vertical_radius,
                    depth_radius,
                )
            score = self.radial_create_candidate_score(
                candidate_x,
                candidate_y,
                normalized,
                camera,
                plane_depth,
            )
            if index > 0:
                score += config.RADIAL_GLOBAL_SHAPE_SCORE_WEIGHT * (
                    self.cluster_shape_candidate_score(
                        shape_sample,
                        position,
                        birth_basis,
                        stack_t,
                    )
                )
            if stack_t > 0.0 and primary_node_id is not None:
                source = self.state.nodes.get(primary_node_id)
                if source is not None:
                    direction = safe_normalized(position - source.position, source.polarity)
                    score += (
                        config.RADIAL_STACK_POLARITY_SCORE_BONUS
                        * stack_t
                        * max(0.0, source.polarity.dot(direction))
                    )
                    outward = source.position - self.cluster_centroid(source)
                    if outward.length() > 0.01:
                        outward_direction = safe_normalized(outward, source.polarity)
                        score += (
                            config.RADIAL_STACK_OUTWARD_SCORE_BONUS
                            * stack_t
                            * max(0.0, outward_direction.dot(direction))
                        )
            if stack_t > 0.0:
                vertical_delta = (screen_y - candidate_y) / max(1.0, radius_px)
                if vertical_delta > 0.0:
                    score += config.RADIAL_STACK_UP_SCORE_BONUS * stack_t * clamp01(
                        vertical_delta
                    )
                else:
                    score += config.RADIAL_STACK_DOWN_SCORE_PENALTY * stack_t * vertical_delta
            if index == 0:
                score += 0.35
            score = clamp01(score)
            candidates.append((score, normalized, candidate_x, candidate_y, position, distance))
        candidates.sort(
            key=lambda item: (
                -item[0],
                abs(item[1] - 0.42),
                stable_hash(reaction_id, int(item[2] * 8.0), int(item[3] * 8.0)),
            )
        )
        return [
            (x, y, position, normalized, score)
            for score, normalized, x, y, position, _distance in candidates
        ]

    def radial_stack_factor(
        self,
        primary_node_id: int | None,
        camera: CameraBasis,
    ) -> float:
        if primary_node_id is None or primary_node_id not in self.state.nodes:
            return 0.0
        node = self.state.nodes[primary_node_id]
        if node.fade <= 0.0 or node.is_pruning:
            return 0.0
        projection = project_point(node.position, camera)
        if not projection.visible:
            return 0.0

        gap = max(6.0, node.radius * projection.scale * 0.45)
        has_above = False
        for other in self.state.live_nodes():
            if other.id == node.id or other.cluster_id != node.cluster_id:
                continue
            other_projection = project_point(other.position, camera)
            if not other_projection.visible:
                continue
            if other_projection.screen_y < projection.screen_y - gap:
                has_above = True
                break

        top_exposure = 0.0 if has_above else 1.0
        updraft_factor = clamp01((node.updraft - 0.25) / 0.50)
        activation_factor = clamp01((node.activation - 0.50) / 0.45) * 0.25
        return clamp01(max(top_exposure * 0.72, updraft_factor) + activation_factor)

    def radial_create_candidate_score(
        self,
        screen_x: float,
        screen_y: float,
        normalized_radius: float,
        camera: CameraBasis,
        plane_depth: float,
    ) -> float:
        nearest = self.nearest_projected_distance(screen_x, screen_y, camera, plane_depth)
        score = 0.48
        if nearest is None:
            score += 0.30
        elif nearest >= config.MIN_GENERATED_SEED_DISTANCE_PX * 1.8:
            score += 0.28
        elif nearest >= config.MIN_GENERATED_SEED_DISTANCE_PX:
            score += 0.10
        else:
            score -= 0.50
        if 0.22 <= normalized_radius <= 0.72:
            score += 0.14
        if normalized_radius <= 0.08:
            score += 0.18
        edge_margin = min(screen_x, config.SCREEN_WIDTH - screen_x, screen_y)
        if edge_margin < 16:
            score -= 0.22
        return clamp01(score)

    def cluster_shape_sample(
        self,
        lineage_id: int,
        basis: tuple[Vec3, Vec3, Vec3],
        primary_node_id: int | None = None,
    ) -> ClusterShapeSample | None:
        primary = self.state.nodes.get(primary_node_id) if primary_node_id is not None else None
        nodes = [
            node
            for node in self.state.live_nodes()
            if node.lineage_id == lineage_id
            and not node.is_pruning
            and node.fade > 0.0
            and (primary is None or node.cluster_id == primary.cluster_id)
        ]
        if len(nodes) < config.CLUSTER_SHAPE_MIN_NODES:
            return None

        centroid = Vec3(
            sum(node.position.x for node in nodes) / len(nodes),
            sum(node.position.y for node in nodes) / len(nodes),
            sum(node.position.z for node in nodes) / len(nodes),
        )
        horizontal, world_up, depth_axis = basis
        heights: list[float] = []
        sector_counts = [0, 0, 0, 0]
        horizontal_axis_count = 0
        depth_axis_count = 0
        max_planar_radius = 1.0
        for node in nodes:
            offset = node.position - centroid
            height = offset.dot(world_up)
            horizontal_offset = offset.dot(horizontal)
            depth_offset = offset.dot(depth_axis)
            planar_radius = math.hypot(horizontal_offset, depth_offset)
            heights.append(height)
            max_planar_radius = max(max_planar_radius, planar_radius)
            if planar_radius <= 0.01:
                continue
            sector_counts[cluster_shape_sector_index(horizontal_offset, depth_offset)] += 1
            if abs(horizontal_offset) >= abs(depth_offset):
                horizontal_axis_count += 1
            else:
                depth_axis_count += 1

        return ClusterShapeSample(
            centroid=centroid,
            min_height=min(heights),
            max_height=max(heights),
            max_planar_radius=max_planar_radius,
            sector_counts=tuple(sector_counts),
            horizontal_axis_count=horizontal_axis_count,
            depth_axis_count=depth_axis_count,
            node_count=len(nodes),
        )

    def cluster_shape_candidate_score(
        self,
        sample: ClusterShapeSample | None,
        position: Vec3,
        basis: tuple[Vec3, Vec3, Vec3],
        stack_t: float,
    ) -> float:
        if sample is None:
            return 0.0
        horizontal, world_up, depth_axis = basis
        offset = position - sample.centroid
        horizontal_offset = offset.dot(horizontal)
        depth_offset = offset.dot(depth_axis)
        planar_radius = math.hypot(horizontal_offset, depth_offset)
        height_span = max(1.0, sample.max_height - sample.min_height)
        height_t = clamp01((offset.dot(world_up) - sample.min_height) / height_span)
        radius_t = clamp01(
            planar_radius
            / max(
                config.SECONDARY_SPROUT_BASE_OFFSET * 2.0,
                sample.max_planar_radius + config.MIN_NODE_RADIUS,
            )
        )

        sector_index = cluster_shape_sector_index(horizontal_offset, depth_offset)
        sector_average = sum(sample.sector_counts) / 4.0
        sector_need = clamp01(
            (sector_average + 1.0 - sample.sector_counts[sector_index])
            / max(1.0, sector_average + 1.0)
        )

        if abs(horizontal_offset) >= abs(depth_offset):
            axis_count = sample.horizontal_axis_count
            opposite_axis_count = sample.depth_axis_count
        else:
            axis_count = sample.depth_axis_count
            opposite_axis_count = sample.horizontal_axis_count
        axis_need = clamp01(
            (opposite_axis_count + 1.0 - axis_count) / max(1.0, opposite_axis_count + 1.0)
        )

        target_width = cluster_shape_target_width(height_t, stack_t)
        profile_fit = clamp01(1.0 - abs(radius_t - target_width) / 0.62)
        upper_stack_fit = clamp01(height_t * (1.0 - max(0.0, radius_t - 0.70) * 1.8))
        return clamp01(
            0.40 * sector_need
            + 0.24 * axis_need
            + 0.24 * profile_fit
            + 0.12 * upper_stack_fit
        )

    def near_projected_node(
        self,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
        plane_depth: float,
        radius_px: float,
    ) -> bool:
        nearest = self.nearest_projected_distance(screen_x, screen_y, camera, plane_depth)
        return nearest is not None and nearest <= radius_px

    def nearest_projected_distance(
        self,
        screen_x: float,
        screen_y: float,
        camera: CameraBasis,
        plane_depth: float,
    ) -> float | None:
        distances: list[float] = []
        for node in self.state.live_nodes():
            projection = project_point(node.position, camera)
            if not projection.visible:
                continue
            if abs(projection.depth - plane_depth) > config.REACTION_DEPTH_TOLERANCE:
                continue
            distances.append(
                math.hypot(
                    projection.screen_x - screen_x,
                    projection.screen_y - screen_y,
                )
            )
        if not distances:
            return None
        return min(distances)

    def apply_radial_plans(
        self,
        plans: list[RadialActionPlan],
        reaction_id: int,
        release_frame: int,
        charge: float,
        radius_px: float,
        local_density: float,
        lineage_id: int,
        snapshot: ReactionViewSnapshot,
    ) -> CloudOperationResult:
        events: list[ReactionEvent] = []
        reacted_ids: list[int] = []
        spawned_ids: list[int] = []
        resonant_ids: list[int] = []
        visual_hits = 0
        productive_hits = 0
        created_seeds = 0
        secondary_sprouts = 0
        highest_generation = 0
        edge_ids: list[int] = []
        created_frame = release_frame
        for plan in plans:
            if len(events) >= config.MAX_PENDING_REACTION_EVENTS:
                break
            if plan.action_kind is RadialActionKind.CREATE_SEED:
                node = self.create_radial_seed(plan, lineage_id, reaction_id)
                if node is None:
                    continue
                spawned_ids.append(node.id)
                created_seeds += 1
                events.append(self.event_from_plan(plan, reaction_id, node.id))
                continue
            if plan.target_node_id is None or plan.target_node_id not in self.state.nodes:
                continue
            if plan.action_kind is RadialActionKind.PULSE_EXISTING:
                self.accumulate_visual_excitation(
                    plan.target_node_id,
                    plan.effective_strength * 0.45,
                )
                self.apply_seed_visual_stimulus(
                    plan.target_node_id,
                    plan.effective_strength,
                    release_frame,
                )
                visual_hits += 1
                events.append(self.event_from_plan(plan, reaction_id, plan.target_node_id))
                continue
            if plan.action_kind is RadialActionKind.IGNITE_DORMANT:
                ecology = self.apply_seed_ecology_energy(
                    plan.target_node_id,
                    plan.effective_strength,
                    plan.execute_frame,
                    0,
                    is_seed=True,
                )
                if ecology.bloomed:
                    self.apply_seed_ignition(plan.target_node_id, plan.effective_strength)
                    resonant_ids.append(plan.target_node_id)
                    events.append(self.event_from_plan(plan, reaction_id, plan.target_node_id))
                else:
                    visual_hits += 1
                    events.append(
                        self.event_from_plan(
                            plan,
                            reaction_id,
                            plan.target_node_id,
                            kind_override=ReactionEventKind.VISUAL_WAVE_HIT,
                            energy_override=ecology.visual_energy,
                        )
                    )
                continue
            if plan.action_kind is RadialActionKind.GROW_EXISTING:
                ecology = self.apply_seed_ecology_energy(
                    plan.target_node_id,
                    plan.effective_strength,
                    plan.execute_frame,
                    0,
                    is_seed=False,
                )
                if not ecology.bloomed:
                    visual_hits += 1
                    events.append(
                        self.event_from_plan(
                            plan,
                            reaction_id,
                            plan.target_node_id,
                            kind_override=ReactionEventKind.VISUAL_WAVE_HIT,
                            energy_override=ecology.visual_energy,
                        )
                    )
                    continue
                self.apply_node_reaction(plan.target_node_id, plan.effective_strength, 0)
                reacted_ids.append(plan.target_node_id)
                productive_hits += 1
                events.append(self.event_from_plan(plan, reaction_id, plan.target_node_id))
                graph_events = self.radial_graph_events(
                    reaction_id,
                    plan.target_node_id,
                    plan.execute_frame,
                    plan.effective_strength,
                )
                for graph_event in graph_events:
                    if len(events) >= config.MAX_PENDING_REACTION_EVENTS:
                        break
                    highest_generation = max(highest_generation, graph_event.generation)
                    if graph_event.target_node_id is not None:
                        if graph_event.kind is ReactionEventKind.GRAPH_NODE_PULSE:
                            reacted_ids.append(graph_event.target_node_id)
                        elif graph_event.kind is ReactionEventKind.VISUAL_WAVE_HIT:
                            visual_hits += 1
                    events.append(graph_event)
                sprout = self.create_radial_secondary_sprout(
                    reaction_id,
                    plan.target_node_id,
                    plan.effective_strength,
                    snapshot,
                    secondary_sprouts,
                )
                if sprout is not None:
                    child_id, edge_id, direction_index = sprout
                    spawned_ids.append(child_id)
                    secondary_sprouts += 1
                    if edge_id is not None:
                        edge_ids.append(edge_id)
                    events.append(
                        ReactionEvent(
                            execute_frame=plan.execute_frame
                            + config.SECONDARY_SPROUT_GENERATION_DELAY_FRAMES,
                            stable_order=len(events),
                            reaction_id=reaction_id,
                            kind=ReactionEventKind.SECONDARY_SPROUT,
                            target_node_id=child_id,
                            source_node_id=plan.target_node_id,
                            generation=1,
                            energy=max(0.20, plan.effective_strength * 0.72),
                            direction_index=direction_index,
                        )
                    )
        events.sort()
        duration = 0 if not events else max(event.execute_frame for event in events) - release_frame
        summary = ReactionSummary(
            reaction_id=reaction_id,
            lineage_id=lineage_id,
            primary_node_id=reacted_ids[0] if reacted_ids else None,
            reacted_nodes=len(set(reacted_ids)),
            sprouts=secondary_sprouts + created_seeds,
            ignited_seeds=len(set(resonant_ids)),
            highest_generation=highest_generation,
            duration_frames=duration,
            reaction_grade=reaction_grade(
                len(set(reacted_ids)),
                secondary_sprouts + created_seeds,
                len(set(resonant_ids)),
            ),
            charge_level=charge,
            max_radius_px=radius_px,
            local_density=local_density,
            visual_hits=visual_hits,
            productive_hits=productive_hits,
            created_seeds=created_seeds,
            secondary_sprouts=secondary_sprouts,
        )
        self.last_reaction_summaries.append(summary)
        self.last_reaction_summaries = self.last_reaction_summaries[-16:]
        return CloudOperationResult(
            "radial",
            node_id=summary.primary_node_id,
            edge_ids=tuple(edge_ids),
            reaction_id=reaction_id,
            created_frame=created_frame,
            reaction_events=tuple(events[: config.MAX_PENDING_REACTION_EVENTS]),
            reaction_summary=summary,
            reacted_node_ids=tuple(sorted(set(reacted_ids))),
            spawned_node_ids=tuple(spawned_ids),
            resonant_node_ids=tuple(sorted(set(resonant_ids))),
        )

    def event_from_plan(
        self,
        plan: RadialActionPlan,
        reaction_id: int,
        target_node_id: int | None,
        kind_override: ReactionEventKind | None = None,
        energy_override: float | None = None,
    ) -> ReactionEvent:
        if kind_override is not None:
            kind = kind_override
        elif plan.action_kind is RadialActionKind.PULSE_EXISTING:
            kind = ReactionEventKind.VISUAL_WAVE_HIT
        elif plan.action_kind is RadialActionKind.GROW_EXISTING:
            kind = ReactionEventKind.GROW_EXISTING
        elif plan.action_kind is RadialActionKind.IGNITE_DORMANT:
            kind = ReactionEventKind.IGNITE_DORMANT
        else:
            kind = ReactionEventKind.CREATE_SEED
        return ReactionEvent(
            execute_frame=plan.execute_frame,
            stable_order=stable_hash(reaction_id, target_node_id or int(plan.screen_x * 10.0)),
            reaction_id=reaction_id,
            kind=kind,
            target_node_id=target_node_id,
            source_node_id=plan.source_node_id,
            generation=plan.generation,
            energy=plan.effective_strength if energy_override is None else energy_override,
        )

    def create_radial_seed(
        self,
        plan: RadialActionPlan,
        lineage_id: int,
        reaction_id: int,
    ) -> CloudNode | None:
        if not self.state.can_add_node():
            return None
        source = (
            self.state.nodes.get(plan.source_node_id)
            if plan.source_node_id is not None
            else None
        )
        if source is None:
            cluster_id = self.state.next_cluster_id
            self.state.next_cluster_id += 1
            parent_node_id = None
            generation = 0
        else:
            lineage_id = source.lineage_id
            cluster_id = source.cluster_id
            parent_node_id = source.id
            generation = source.generation + 1
        local_rng = RandomSource(
            stable_hash(
                self.rng.seed,
                reaction_id,
                int(plan.screen_x * 10.0),
                int(plan.screen_y * 10.0),
            )
        )
        node = create_node(
            self.state,
            lineage_id,
            cluster_id,
            plan.world_position,
            local_rng,
            parent_node_id=parent_node_id,
            generation=generation,
        )
        strength = clamp01(plan.effective_strength)
        radial_t = clamp01(plan.normalized_radius)
        size_jitter = (stable_hash01(reaction_id, node.id, 0x512E) - 0.5) * 0.10
        radial_size = 1.38 - 0.74 * radial_t + size_jitter
        if plan.zone is RadialZone.CORE:
            node.mass = config.SEED_MASS * (radial_size + 0.30 * strength)
            node.activation = clamp01(0.44 + 0.50 * strength)
            node.moisture = clamp01(0.43 + 0.26 * strength)
            node.density = 0.90
        elif plan.zone is RadialZone.MIDDLE:
            node.mass = config.SEED_MASS * (radial_size + 0.22 * strength)
            node.activation = clamp01(0.26 + 0.46 * strength)
            node.moisture = clamp01(0.38 + 0.23 * strength)
            node.density = 1.00
        else:
            node.mass = config.SEED_MASS * (max(0.48, radial_size) + 0.12 * strength)
            node.activation = clamp01(0.10 + 0.30 * strength)
            node.moisture = clamp01(0.33 + 0.18 * strength)
            node.density = 1.12
        node.noise = clamp01(0.18 + 0.22 * strength)
        node.untouched_time = 0.0
        if source is not None:
            growth_direction = safe_normalized(node.position - source.position, source.polarity)
            node.polarity = inherited_child_polarity(source, growth_direction)
            node.updraft = clamp01(source.updraft + 0.07 * (1.0 - radial_t))
            node.moisture = clamp01((node.moisture + source.moisture * 0.55) / 1.55)
            add_edge(
                self.state,
                lineage_id,
                cluster_id,
                source.id,
                node.id,
                EdgeKind.PRIMARY,
            )
        if plan.normalized_radius > 0.08:
            node.fade = 0.0
            self.pending_node_reveals[node.id] = plan.execute_frame
        recompute_clusters(self.state, lineage_id)
        return node

    def radial_graph_events(
        self,
        reaction_id: int,
        source_id: int,
        source_frame: int,
        source_energy: float,
    ) -> list[ReactionEvent]:
        events: list[ReactionEvent] = []
        visited = {source_id}
        frontier: list[tuple[int, int, float, int]] = [(source_id, 0, source_energy, source_frame)]
        while frontier and len(visited) <= config.MAX_GRAPH_REACTED_NODES:
            node_id, generation, energy, execute_frame = frontier.pop(0)
            if generation >= config.MAX_GRAPH_CHAIN_GENERATION:
                continue
            for next_id, next_energy in self.propagation_targets(
                reaction_id,
                node_id,
                energy,
                generation,
            ):
                if next_id in visited:
                    continue
                visited.add(next_id)
                delay = chain_delay(reaction_id, node_id, next_id, generation + 1)
                next_frame = execute_frame + delay
                if next_frame - source_frame > config.MAX_REACTION_DURATION_FRAMES:
                    continue
                target = self.state.nodes[next_id]
                ecology = self.apply_seed_ecology_energy(
                    next_id,
                    next_energy,
                    next_frame,
                    generation + 1,
                    is_seed=self.is_dormant_seed(target),
                )
                kind = ReactionEventKind.VISUAL_WAVE_HIT
                event_energy = ecology.visual_energy
                if ecology.bloomed:
                    self.apply_node_reaction(next_id, next_energy, generation + 1)
                    kind = ReactionEventKind.GRAPH_NODE_PULSE
                    event_energy = next_energy
                events.append(
                    ReactionEvent(
                        execute_frame=next_frame,
                        stable_order=len(events),
                        reaction_id=reaction_id,
                        kind=kind,
                        target_node_id=next_id,
                        source_node_id=node_id,
                        generation=generation + 1,
                        energy=event_energy,
                    )
                )
                frontier.append((next_id, generation + 1, next_energy, next_frame))
                if len(events) >= config.MAX_GRAPH_REACTED_NODES:
                    return events
        return events

    def create_radial_secondary_sprout(
        self,
        reaction_id: int,
        source_id: int,
        source_energy: float,
        snapshot: ReactionViewSnapshot,
        sprout_index: int,
    ) -> tuple[int, int | None, int] | None:
        if sprout_index >= config.MAX_SECONDARY_SPROUTS_PER_REACTION:
            return None
        if source_id not in self.state.nodes:
            return None
        if source_energy < 0.54:
            return None
        stimulus = CloudStimulus(
            reaction_id=reaction_id,
            source_kind=StimulusKind.TAP,
            primary_node_id=source_id,
            lineage_id=self.state.nodes[source_id].lineage_id,
            screen_x=snapshot.pointer_screen_x,
            screen_y=snapshot.pointer_screen_y,
            world_position=self.state.nodes[source_id].position,
            strength=source_energy,
            radius_px=int(config.MAX_REACTION_RADIUS_PX),
            created_frame=int(round(self.elapsed_time * config.FPS)),
            seed=stable_hash(self.rng.seed, reaction_id, source_id, sprout_index),
            view_snapshot=snapshot,
        )
        if self.secondary_sprout_count(reaction_id, source_id, source_energy, False) <= 0:
            return None
        return self.create_secondary_sprout(
            stimulus,
            source_id,
            1,
            sprout_index,
        )

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
            energies[node_id] = energy
            generations[node_id] = generation
            ecology = self.apply_seed_ecology_energy(
                node_id,
                energy,
                execute_frame,
                generation,
                is_seed=self.is_dormant_seed(node),
            )
            if ecology.bloomed:
                reacted_ids.append(node_id)
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
            else:
                events.append(
                    ReactionEvent(
                        execute_frame=execute_frame,
                        stable_order=len(events),
                        reaction_id=stimulus.reaction_id,
                        kind=ReactionEventKind.VISUAL_WAVE_HIT,
                        target_node_id=node_id,
                        source_node_id=source_id,
                        generation=generation,
                        energy=ecology.visual_energy,
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
            energies[seed_id] = seed_energy
            generations[seed_id] = 1
            ecology = self.apply_seed_ecology_energy(
                seed_id,
                seed_energy,
                execute_frame,
                1,
                is_seed=True,
            )
            if not ecology.bloomed:
                events.append(
                    ReactionEvent(
                        execute_frame=execute_frame,
                        stable_order=len(events),
                        reaction_id=stimulus.reaction_id,
                        kind=ReactionEventKind.VISUAL_WAVE_HIT,
                        target_node_id=seed_id,
                        source_node_id=stimulus.primary_node_id,
                        generation=1,
                        energy=ecology.visual_energy,
                    )
                )
                if len(events) >= config.MAX_PENDING_REACTION_EVENTS:
                    break
                continue
            self.apply_seed_ignition(seed_id, seed_energy)
            resonant_ids.append(seed_id)
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

    def apply_seed_ecology_energy(
        self,
        node_id: int,
        energy: float,
        execute_frame: int,
        generation: int,
        is_seed: bool,
    ) -> SeedEcologyResult:
        if node_id not in self.state.nodes:
            return SeedEcologyResult(False, 0.0, config.SEED_BLOOM_BASE_THRESHOLD)
        node = self.state.nodes[node_id]
        if node.fade <= 0.0 or node.is_pruning:
            return SeedEcologyResult(False, 0.0, config.SEED_BLOOM_BASE_THRESHOLD)

        if execute_frame < node.refractory_until_frame:
            retained = energy * config.SEED_REFRACTORY_EXCITATION_RATIO
            node.excitation = min(config.SEED_EXCITATION_MAX, node.excitation + retained)
            self.apply_seed_visual_stimulus(node_id, retained, execute_frame)
            return SeedEcologyResult(
                False,
                min(config.SEED_REFRACTORY_VISUAL_ENERGY, max(0.08, retained)),
                self.seed_bloom_threshold(node, 0.0, 0.0),
            )

        activation_signal, inhibition_signal = self.local_seed_ecology_signals(
            node_id,
            execute_frame,
        )
        resonance = seed_resonance_sensitivity(node.trait_seed)
        incoming = energy * resonance + activation_signal
        if is_seed:
            incoming *= 0.92
        node.excitation = min(config.SEED_EXCITATION_MAX, node.excitation + incoming)
        threshold = self.seed_bloom_threshold(node, activation_signal, inhibition_signal)
        force_bloom = energy >= config.SEED_BLOOM_FORCE_ENERGY and generation == 0
        if not force_bloom and node.excitation < threshold:
            self.apply_seed_visual_stimulus(node_id, incoming, execute_frame)
            return SeedEcologyResult(
                False,
                max(0.08, min(0.44, node.excitation / max(0.01, threshold))),
                threshold,
            )

        node.excitation *= config.SEED_POST_BLOOM_EXCITATION_RATIO
        node.refractory_until_frame = execute_frame + config.SEED_BLOOM_REFRACTORY_FRAMES
        return SeedEcologyResult(True, min(1.0, energy), threshold)

    def apply_seed_visual_stimulus(
        self,
        node_id: int | None,
        energy: float,
        execute_frame: int,
    ) -> None:
        del execute_frame
        if node_id is None or node_id not in self.state.nodes:
            return
        node = self.state.nodes[node_id]
        if node.fade <= 0.0 or node.is_pruning:
            return
        amount = clamp01(energy)
        node.activation = min(1.0, node.activation + 0.035 * amount)
        primed_bonus = 0.03 if node.excitation >= config.SEED_PRIMED_THRESHOLD else 0.0
        node.charge = min(1.0, node.charge + 0.08 * amount + primed_bonus)
        node.untouched_time = 0.0

    def accumulate_visual_excitation(self, node_id: int, energy: float) -> None:
        if node_id not in self.state.nodes:
            return
        node = self.state.nodes[node_id]
        if node.fade <= 0.0 or node.is_pruning:
            return
        retained = energy * seed_resonance_sensitivity(node.trait_seed)
        node.excitation = min(config.SEED_EXCITATION_MAX, node.excitation + retained)

    def seed_bloom_threshold(
        self,
        node: CloudNode,
        activation_signal: float,
        inhibition_signal: float,
    ) -> float:
        crowding = self.local_density_factor(node.position, node.id)
        threshold = (
            config.SEED_BLOOM_BASE_THRESHOLD
            + 0.16 * crowding
            + 0.18 * inhibition_signal
            + 0.08 * node.incubation
            - 0.18 * node.moisture
            - 0.10 * node.activation
            - (0.10 + config.SEED_QUORUM_THRESHOLD_BONUS) * activation_signal
        )
        return max(0.34, min(1.08, threshold))

    def local_seed_ecology_signals(
        self,
        node_id: int,
        frame: int,
    ) -> tuple[float, float]:
        node = self.state.nodes[node_id]
        return self.seed_ecology_position_signals(node.position, node_id, frame)

    def seed_ecology_position_signals(
        self,
        position: Vec3,
        exclude_node_id: int,
        frame: int,
    ) -> tuple[float, float]:
        activation = 0.0
        inhibition = 0.0
        for other in self.state.live_nodes():
            if other.id == exclude_node_id or other.is_pruning:
                continue
            distance = position.distance_to(other.position)
            if distance <= config.SEED_NEAR_ACTIVATION_RADIUS:
                recent_bloom = 1.0 if other.refractory_until_frame > frame else 0.0
                activity = max(other.excitation, recent_bloom, other.activation * 0.35)
                activation += config.SEED_NEAR_ACTIVATION_GAIN * activity
            elif distance <= config.SEED_QUORUM_RADIUS and other.excitation >= (
                config.SEED_PRIMED_THRESHOLD
            ):
                primed = min(1.0, other.excitation / max(0.01, config.SEED_PRIMED_THRESHOLD))
                falloff = 1.0 - (
                    (distance - config.SEED_NEAR_ACTIVATION_RADIUS)
                    / max(
                        1.0,
                        config.SEED_QUORUM_RADIUS - config.SEED_NEAR_ACTIVATION_RADIUS,
                    )
                ) * 0.55
                activation += config.SEED_QUORUM_GAIN * primed * clamp01(falloff)
            elif (
                config.SEED_MID_INHIBITION_INNER_RADIUS
                < distance
                <= config.SEED_MID_INHIBITION_OUTER_RADIUS
            ):
                density = min(1.0, other.mass / max(1.0, config.RETENTION_GROWN_MASS))
                inhibition += config.SEED_MID_INHIBITION_GAIN * (0.50 + density * 0.50)
        return clamp01(activation), clamp01(inhibition)

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
        exposure = self.sprout_exposure_factor(node)
        if not guaranteed and exposure < config.SPROUT_MIN_EXPOSURE_FOR_UNGUARANTEED:
            return 0
        score = clamp01(
            0.16
            + 0.34 * energy
            + 0.24 * node.moisture
            + 0.18 * node.updraft
            + 0.12 * node.activation
            + 0.08 * node.noise
            + 0.14 * exposure
            - 0.22 * density
            - 0.18 * (1.0 - exposure) * density
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
        growth_direction = safe_normalized(child.position - parent.position, parent.polarity)
        child.polarity = inherited_child_polarity(parent, growth_direction)
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
        view_camera = CameraBasis(
            right=snapshot.camera_right,
            up=snapshot.camera_up,
            forward=snapshot.camera_forward,
            position=snapshot.camera_position,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
        )
        shape_basis = radial_birth_basis(view_camera)
        shape_sample = self.cluster_shape_sample(parent.lineage_id, shape_basis, parent.id)
        parent_projection = project_point(parent.position, view_camera)
        input_dx = stimulus.screen_x - parent_projection.screen_x
        input_dy = stimulus.screen_y - parent_projection.screen_y
        input_length = math.hypot(input_dx, input_dy)
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
            candidate_direction = safe_normalized(position - parent.position, parent.polarity)
            projection = project_point(position, view_camera)
            score = self.sprout_candidate_score(
                parent,
                position,
                direction_index,
                right_weight,
                up_weight,
                outward,
                candidate_direction,
                shape_sample,
                shape_basis,
            )
            if projection.visible:
                if input_length > 4.0 and parent_projection.visible:
                    candidate_dx = projection.screen_x - parent_projection.screen_x
                    candidate_dy = projection.screen_y - parent_projection.screen_y
                    candidate_length = math.hypot(candidate_dx, candidate_dy)
                    if candidate_length > 0.01:
                        input_alignment = (
                            (candidate_dx * input_dx + candidate_dy * input_dy)
                            / (candidate_length * input_length)
                        )
                        score += (
                            config.SPROUT_INPUT_DIRECTION_SCORE_WEIGHT
                            * max(0.0, input_alignment)
                        )
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
        candidate_direction: Vec3,
        shape_sample: ClusterShapeSample | None = None,
        shape_basis: tuple[Vec3, Vec3, Vec3] | None = None,
    ) -> float:
        score = 0.50
        polarity_match = parent.polarity.dot(candidate_direction)
        score += config.SPROUT_POLARITY_SCORE_WEIGHT * max(0.0, polarity_match)
        if polarity_match < -0.15:
            score += 0.12 * polarity_match
        if parent.updraft >= 0.35:
            score += max(0.0, up_weight) * (0.35 + parent.updraft * 0.30)
        if parent.moisture >= 0.48 and parent.updraft < 0.45:
            score += abs(right_weight) * 0.28
        if parent.noise >= 0.42:
            score += stable_hash01(parent.id, direction_index, 0x1A2B) * 0.30
        if parent.incubation >= 0.55:
            score += max(0.0, -up_weight) * 0.10
        exposure = self.sprout_exposure_factor(parent)
        score += config.SPROUT_EXPOSURE_SCORE_WEIGHT * exposure
        if outward.length() > 0.01:
            outward_direction = safe_normalized(outward, candidate_direction)
            outward_match = max(0.0, outward_direction.dot(candidate_direction))
            score += (
                outward_match * config.SPROUT_OUTWARD_SCORE_WEIGHT
            )
            score -= (
                config.SPROUT_INTERNAL_SCORE_PENALTY
                * (1.0 - exposure)
                * (1.0 - outward_match)
            )
        else:
            score -= config.SPROUT_INTERNAL_SCORE_PENALTY * (1.0 - exposure) * 0.50
        activation_signal, inhibition_signal = self.seed_ecology_position_signals(
            position,
            parent.id,
            int(round(self.elapsed_time * config.FPS)),
        )
        score += config.SPROUT_NEAR_ACTIVATION_SCORE_WEIGHT * activation_signal
        score -= config.SPROUT_MID_INHIBITION_SCORE_WEIGHT * inhibition_signal
        score += self.sprout_sector_score(parent, candidate_direction)
        if shape_basis is not None:
            score += config.SPROUT_GLOBAL_SHAPE_SCORE_WEIGHT * (
                self.cluster_shape_candidate_score(
                    shape_sample,
                    position,
                    shape_basis,
                    0.0,
                )
            )
        nearest = self.nearest_node_distance(position, exclude_node_id=parent.id)
        if nearest is not None:
            if nearest < config.MIN_NODE_RADIUS * 0.75:
                score -= 0.45
            elif nearest < config.MIN_NODE_RADIUS * 1.4:
                score -= 0.14
            elif nearest > 42.0:
                score -= 0.08
        return score

    def sprout_exposure_factor(self, parent: CloudNode) -> float:
        free_count = 0
        for direction_index in range(config.SPROUT_DIRECTION_COUNT):
            direction = canonical_sprout_direction(direction_index)
            if self.sprout_sector_occupancy(parent, direction) <= 0.05:
                free_count += 1
        sector_factor = free_count / max(1, config.SPROUT_DIRECTION_COUNT)
        degree_factor = 1.0 - clamp01(node_degree(self.state, parent.id) / config.MAX_NODE_DEGREE)
        return clamp01(sector_factor * 0.78 + degree_factor * 0.22)

    def sprout_sector_score(
        self,
        parent: CloudNode,
        candidate_direction: Vec3,
    ) -> float:
        occupied = self.sprout_sector_occupancy(parent, candidate_direction)
        return config.SPROUT_FREE_SECTOR_BONUS - (
            config.SPROUT_OCCUPIED_SECTOR_PENALTY * occupied
        )

    def sprout_sector_occupancy(
        self,
        parent: CloudNode,
        candidate_direction: Vec3,
    ) -> float:
        occupied = 0.0
        for other in self.state.live_nodes():
            if other.id == parent.id or other.is_pruning:
                continue
            offset = other.position - parent.position
            distance = offset.length()
            if distance <= 0.01 or distance > config.SPROUT_OCCUPIED_SECTOR_RADIUS:
                continue
            direction = safe_normalized(offset, candidate_direction)
            alignment = direction.dot(candidate_direction)
            if alignment < config.SPROUT_OCCUPIED_SECTOR_ALIGNMENT:
                continue
            closeness = 1.0 - distance / config.SPROUT_OCCUPIED_SECTOR_RADIUS
            occupied = max(occupied, alignment * closeness)
        return occupied

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
        current_frame = int(round(self.elapsed_time * config.FPS))
        self.activate_pending_nodes(current_frame)
        self.elapsed_time += dt
        for node in self.state.nodes.values():
            node.age += dt
            node.untouched_time += dt
            node.activation = max(0.0, node.activation - dt * 0.12)
            node.excitation = max(
                0.0,
                node.excitation - dt * config.SEED_EXCITATION_DECAY_RATE,
            )
            node.charge = max(0.0, node.charge - dt * 0.18)
        for edge in self.state.edges.values():
            edge.age += dt
        self.relax_connected_edges(dt)
        try_merge_clusters(self.state)
        update_incubation(self.state, dt)
        self.finalize_extinct_lineages()

    def activate_pending_nodes(self, current_frame: int) -> None:
        ready_ids = [
            node_id
            for node_id, reveal_frame in self.pending_node_reveals.items()
            if reveal_frame <= current_frame
        ]
        if not ready_ids:
            return
        touched_lineages: set[int] = set()
        for node_id in ready_ids:
            node = self.state.nodes.get(node_id)
            del self.pending_node_reveals[node_id]
            if node is None:
                continue
            node.fade = 1.0
            lineage = self.state.lineages.get(node.lineage_id)
            if lineage is not None:
                lineage.extinct = False
            touched_lineages.add(node.lineage_id)
        for lineage_id in touched_lineages:
            recompute_clusters(self.state, lineage_id)

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
                node.lineage_id == lineage.id
                and (node.fade > 0.0 or node.id in self.pending_node_reveals)
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


def seed_resonance_sensitivity(trait_seed: int) -> float:
    return 0.88 + stable_hash01(trait_seed, 0xEC01) * 0.24


def safe_normalized(vector: Vec3, fallback: Vec3) -> Vec3:
    try:
        return vector.normalized()
    except ValueError:
        try:
            return fallback.normalized()
        except ValueError:
            return Vec3(0.0, 1.0, 0.0)


def inherited_child_polarity(parent: CloudNode, growth_direction: Vec3) -> Vec3:
    return safe_normalized(
        parent.polarity * config.SPROUT_POLARITY_PARENT_WEIGHT
        + growth_direction * config.SPROUT_POLARITY_GROWTH_WEIGHT
        + Vec3(0.0, 1.0, 0.0) * config.SPROUT_POLARITY_UP_WEIGHT,
        parent.polarity,
    )


def canonical_sprout_direction(direction_index: int) -> Vec3:
    angle = math.tau * direction_index / max(1, config.SPROUT_DIRECTION_COUNT)
    return safe_normalized(
        Vec3(math.cos(angle), math.sin(angle) * 0.86, 0.0),
        Vec3(0.0, 1.0, 0.0),
    )


def radial_birth_basis(camera: CameraBasis) -> tuple[Vec3, Vec3, Vec3]:
    world_up = Vec3(0.0, 1.0, 0.0)
    horizontal = camera.right - world_up * camera.right.dot(world_up)
    try:
        horizontal = horizontal.normalized()
    except ValueError:
        horizontal = Vec3(1.0, 0.0, 0.0)
    depth_axis = horizontal.cross(world_up)
    if depth_axis.dot(camera.forward) < 0.0:
        depth_axis = depth_axis * -1.0
    try:
        depth_axis = depth_axis.normalized()
    except ValueError:
        depth_axis = Vec3(0.0, 0.0, 1.0)
    return horizontal, world_up, depth_axis


def radial_birth_horizontal_radius(radius_px: float) -> float:
    t = clamp01(
        (radius_px - config.MIN_REACTION_RADIUS_PX)
        / max(1.0, config.MAX_REACTION_RADIUS_PX - config.MIN_REACTION_RADIUS_PX)
    )
    return lerp_config(
        config.RADIAL_BIRTH_HORIZONTAL_RADIUS_MIN,
        config.RADIAL_BIRTH_HORIZONTAL_RADIUS_MAX,
        t,
    )


def radial_birth_depth_factor(index: int) -> float:
    sequence = (0.0, 0.82, -0.82, 0.45, -0.45, 0.68, -0.68, 0.24, -0.24)
    return sequence[index % len(sequence)]


def cluster_shape_sector_index(horizontal_offset: float, depth_offset: float) -> int:
    if abs(horizontal_offset) >= abs(depth_offset):
        return 0 if horizontal_offset >= 0.0 else 1
    return 2 if depth_offset >= 0.0 else 3


def cluster_shape_target_width(height_t: float, stack_t: float) -> float:
    base_target = 0.74 - 0.34 * clamp01(height_t)
    if 0.28 <= height_t <= 0.64:
        base_target += 0.07
    stacked_target = max(0.26, base_target * 0.74)
    return lerp_config(base_target, stacked_target, stack_t)


def lift_radial_candidate_into_ellipsoid(
    center: Vec3,
    basis: tuple[Vec3, Vec3, Vec3],
    screen_u: float,
    screen_v: float,
    depth_factor: float,
    horizontal_radius: float,
    vertical_radius: float,
    depth_radius: float,
) -> Vec3:
    disk_r2 = min(1.0, screen_u * screen_u + screen_v * screen_v)
    available_depth = depth_radius * math.sqrt(max(0.0, 1.0 - disk_r2))
    horizontal, world_up, depth_axis = basis
    return clamp_cloud_position(
        center
        + horizontal * (horizontal_radius * screen_u)
        + world_up * (vertical_radius * screen_v)
        + depth_axis * (available_depth * depth_factor)
    )


def clamp_radial_birth_center(
    center: Vec3,
    vertical_radius: float,
    depth_radius: float,
) -> Vec3:
    y_min = config.MIN_CLOUD_Y + vertical_radius
    y_max = config.MAX_CLOUD_Y - vertical_radius
    z_min = config.CLOUD_DEPTH_MIN + depth_radius
    z_max = config.CLOUD_DEPTH_MAX - depth_radius
    if y_min > y_max:
        y = (config.MIN_CLOUD_Y + config.MAX_CLOUD_Y) * 0.5
    else:
        y = min(max(center.y, y_min), y_max)
    if z_min > z_max:
        z = (config.CLOUD_DEPTH_MIN + config.CLOUD_DEPTH_MAX) * 0.5
    else:
        z = min(max(center.z, z_min), z_max)
    return Vec3(center.x, y, z)


def lerp_config(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp01(t)
