from __future__ import annotations

import unittest

from src import config
from src.camera.camera import build_camera_basis
from src.camera.projection import camera_depth, project_point
from src.cloud.graph import add_edge, create_node, recompute_clusters
from src.cloud.reaction import (
    ReactionEventKind,
    charge_level,
    new_seed_budget,
    radial_strength,
    reaction_radius_px,
)
from src.cloud.simulation import CloudSimulation
from src.enums import EdgeKind
from src.math3d import Vec3
from src.rng import RandomSource


class RadialReactionTests(unittest.TestCase):
    def test_charge_controls_reaction_radius(self) -> None:
        cold = charge_level(0.0)
        short = charge_level(config.REACTION_CHARGE_START_SECONDS + 0.12)
        full = charge_level(config.REACTION_CHARGE_FULL_SECONDS + 0.5)

        self.assertEqual(cold, 0.0)
        self.assertLess(short, full)
        self.assertEqual(full, 1.0)
        self.assertAlmostEqual(
            reaction_radius_px(cold),
            config.MIN_REACTION_RADIUS_PX,
        )
        self.assertAlmostEqual(
            reaction_radius_px(full),
            config.MAX_REACTION_RADIUS_PX,
        )

    def test_radial_strength_falls_from_center_to_outer_ring(self) -> None:
        center = radial_strength(0.0, 0.0)
        middle = radial_strength(0.5, 0.0)
        outer = radial_strength(1.0, 0.0)

        self.assertGreater(center, middle)
        self.assertGreater(middle, outer)
        self.assertGreaterEqual(outer, config.OUTER_STRENGTH_FLOOR)

    def test_seed_budget_scales_with_charge_to_nine(self) -> None:
        cold = new_seed_budget(0.0, 0.0)
        middle = new_seed_budget(0.5, 0.0)
        full = new_seed_budget(1.0, 0.0)
        dense_full = new_seed_budget(1.0, 1.0)

        self.assertEqual(cold, config.BASE_NEW_SEEDS_PER_REACTION)
        self.assertGreater(middle, cold)
        self.assertEqual(full, config.MAX_NEW_SEEDS_PER_REACTION)
        self.assertGreaterEqual(dense_full, config.BASE_NEW_SEEDS_PER_REACTION)
        self.assertLess(dense_full, full)

    def test_empty_short_reaction_creates_five_radial_seeds(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)

        result = simulation.radial_reaction_screen(160.0, 190.0, 0.05, camera)

        self.assertEqual(result.kind, "radial")
        self.assertIsNotNone(result.reaction_summary)
        self.assertEqual(
            result.reaction_summary.created_seeds,
            config.BASE_NEW_SEEDS_PER_REACTION,
        )
        self.assertLess(result.reaction_summary.local_density, 0.25)
        self.assertEqual(len(simulation.state.nodes), config.BASE_NEW_SEEDS_PER_REACTION)
        created = simulation.state.nodes[result.spawned_node_ids[0]]
        projection = project_point(created.position, camera)
        self.assertAlmostEqual(projection.screen_x, 160.0, delta=1.0)
        self.assertAlmostEqual(projection.screen_y, 190.0, delta=1.0)
        create_events = [
            event for event in result.reaction_events if event.kind is ReactionEventKind.CREATE_SEED
        ]
        self.assertEqual(len(create_events), config.BASE_NEW_SEEDS_PER_REACTION)
        create_frames = [event.execute_frame for event in create_events]
        self.assertGreaterEqual(
            max(create_frames) - min(create_frames),
            (config.BASE_NEW_SEEDS_PER_REACTION - 1)
            * config.RADIAL_SEED_REVEAL_STAGGER_FRAMES,
        )
        hidden_ids = [
            node_id
            for node_id in result.spawned_node_ids
            if simulation.state.nodes[node_id].fade <= 0.0
        ]
        self.assertGreaterEqual(len(hidden_ids), config.BASE_NEW_SEEDS_PER_REACTION - 1)

    def test_empty_long_reaction_creates_multiple_radial_seeds(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)

        short = simulation.radial_reaction_screen(60.0, 180.0, 0.05, camera)
        long = simulation.radial_reaction_screen(220.0, 180.0, 0.95, camera)

        self.assertIsNotNone(short.reaction_summary)
        self.assertIsNotNone(long.reaction_summary)
        self.assertGreater(
            long.reaction_summary.created_seeds,
            short.reaction_summary.created_seeds,
        )
        self.assertEqual(
            long.reaction_summary.created_seeds,
            config.MAX_NEW_SEEDS_PER_REACTION,
        )
        self.assertLessEqual(
            long.reaction_summary.created_seeds,
            config.MAX_NEW_SEEDS_PER_REACTION,
        )
        created_events = [
            event for event in long.reaction_events if event.kind is ReactionEventKind.CREATE_SEED
        ]
        self.assertEqual(len(created_events), long.reaction_summary.created_seeds)

    def test_radial_seed_birth_volume_has_front_and_back_depth(self) -> None:
        simulation = CloudSimulation(RandomSource(97531))
        camera = build_camera_basis(0.0)

        result = simulation.radial_reaction_screen(160.0, 190.0, 0.95, camera)

        self.assertGreaterEqual(len(result.spawned_node_ids), 5)
        center_id = result.spawned_node_ids[0]
        center_depth = camera_depth(simulation.state.nodes[center_id].position, camera)
        depth_offsets = [
            camera_depth(simulation.state.nodes[node_id].position, camera) - center_depth
            for node_id in result.spawned_node_ids[1:]
        ]
        self.assertGreater(max(depth_offsets), 8.0)
        self.assertLess(min(depth_offsets), -8.0)
        for node_id in result.spawned_node_ids:
            node = simulation.state.nodes[node_id]
            self.assertGreaterEqual(node.position.z, config.CLOUD_DEPTH_MIN)
            self.assertLessEqual(node.position.z, config.CLOUD_DEPTH_MAX)

    def test_center_radial_seed_remains_at_tap_position_after_depth_lift(self) -> None:
        simulation = CloudSimulation(RandomSource(86420))
        camera = build_camera_basis(0.0)

        result = simulation.radial_reaction_screen(174.0, 186.0, 0.95, camera)

        center = simulation.state.nodes[result.spawned_node_ids[0]]
        projection = project_point(center.position, camera)
        self.assertAlmostEqual(projection.screen_x, 174.0, delta=1.0)
        self.assertAlmostEqual(projection.screen_y, 186.0, delta=1.0)

    def test_dense_reaction_creates_new_seed_and_interferes_with_existing_nodes(self) -> None:
        dense = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        root = dense.radial_reaction_screen(160.0, 190.0, 0.05, camera)
        self.assertTrue(root.spawned_node_ids)
        current = dense.state.nodes[root.spawned_node_ids[0]]
        for index in range(10):
            projection = project_point(current.position, camera)
            child = dense.add_child_near_screen(
                current,
                projection.screen_x + 14.0 + (index % 3) * 4.0,
                projection.screen_y + (index % 2) * 7.0,
                camera,
            )
            self.assertIsNotNone(child)
            current = child

        dense_result = dense.radial_reaction_screen(160.0, 190.0, 0.95, camera)

        self.assertIsNotNone(dense_result.reaction_summary)
        self.assertGreater(dense_result.reaction_summary.local_density, 0.45)
        self.assertGreaterEqual(dense_result.reaction_summary.created_seeds, 1)
        self.assertGreaterEqual(dense_result.reaction_summary.productive_hits, 1)
        self.assertTrue(
            any(
                event.kind is ReactionEventKind.CREATE_SEED
                for event in dense_result.reaction_events
            )
        )
        self.assertTrue(
            any(
                event.kind is ReactionEventKind.GROW_EXISTING
                for event in dense_result.reaction_events
            )
        )

    def test_wave_events_arrive_from_center_to_outer_radius(self) -> None:
        simulation = CloudSimulation(RandomSource(2468))
        camera = build_camera_basis(0.0)

        result = simulation.radial_reaction_screen(160.0, 190.0, 0.95, camera)
        create_events = [
            event for event in result.reaction_events if event.kind is ReactionEventKind.CREATE_SEED
        ]

        self.assertGreaterEqual(len(create_events), 2)
        frames = [event.execute_frame for event in create_events]
        self.assertEqual(frames, sorted(frames))
        self.assertLess(frames[0], frames[-1])
        hidden_ids = [
            node_id
            for node_id in result.spawned_node_ids
            if simulation.state.nodes[node_id].fade <= 0.0
        ]
        self.assertGreaterEqual(len(hidden_ids), 1)

        first_hidden_id = hidden_ids[0]
        first_hidden_frame = min(
            event.execute_frame
            for event in create_events
            if event.target_node_id == first_hidden_id
        )
        self.assertGreater(first_hidden_frame, result.created_frame)
        self.assertEqual(simulation.state.nodes[first_hidden_id].fade, 0.0)
        for _ in range(first_hidden_frame - result.created_frame + 1):
            simulation.update(1.0 / config.FPS)
        self.assertEqual(simulation.state.nodes[first_hidden_id].fade, 1.0)

    def test_radial_seed_initial_size_varies_by_distance(self) -> None:
        simulation = CloudSimulation(RandomSource(13579))
        camera = build_camera_basis(0.0)

        result = simulation.radial_reaction_screen(160.0, 190.0, 0.95, camera)

        self.assertGreaterEqual(len(result.spawned_node_ids), 3)
        seeds_by_distance = sorted(
            (
                distance_from_origin(simulation, camera, node_id, 160.0, 190.0),
                simulation.state.nodes[node_id].radius,
            )
            for node_id in result.spawned_node_ids
        )
        inner_distance, inner_radius = seeds_by_distance[0]
        outer_distance, outer_radius = seeds_by_distance[-1]

        self.assertLess(inner_distance, outer_distance)
        self.assertGreater(inner_radius, outer_radius + 2.0)

    def test_reacting_to_top_node_stacks_new_seed_volume_upward(self) -> None:
        simulation = CloudSimulation(RandomSource(24680))
        camera = build_camera_basis(0.0)
        root_result = simulation.radial_reaction_screen(160.0, 210.0, 0.05, camera)
        root = simulation.state.nodes[root_result.spawned_node_ids[0]]
        upper = create_node(
            simulation.state,
            root.lineage_id,
            root.cluster_id,
            root.position + Vec3(0.0, 28.0, 0.0),
            simulation.rng,
            parent_node_id=root.id,
            generation=1,
        )
        upper.updraft = 0.62
        add_edge(
            simulation.state,
            root.lineage_id,
            root.cluster_id,
            root.id,
            upper.id,
            EdgeKind.PRIMARY,
        )
        recompute_clusters(simulation.state, root.lineage_id)
        upper_projection = project_point(upper.position, camera)

        result = simulation.radial_reaction_screen(
            upper_projection.screen_x,
            upper_projection.screen_y,
            0.95,
            camera,
            target_node_id=upper.id,
        )

        self.assertGreaterEqual(len(result.spawned_node_ids), 2)
        spawned_projections = [
            project_point(simulation.state.nodes[node_id].position, camera)
            for node_id in result.spawned_node_ids
        ]
        upper_seed_count = sum(
            1
            for projection in spawned_projections
            if projection.screen_y < upper_projection.screen_y - 4.0
        )
        lower_seed_count = sum(
            1
            for projection in spawned_projections
            if projection.screen_y > upper_projection.screen_y + 4.0
        )

        self.assertGreater(upper_seed_count, lower_seed_count)
        self.assertLess(
            sum(projection.screen_y for projection in spawned_projections)
            / len(spawned_projections),
            upper_projection.screen_y,
        )

    def test_radial_reaction_plan_is_deterministic(self) -> None:
        first = radial_signature()
        second = radial_signature()

        self.assertEqual(first, second)


def radial_signature() -> tuple:
    simulation = CloudSimulation(RandomSource(8642))
    camera = build_camera_basis(0.0)
    result = simulation.radial_reaction_screen(160.0, 190.0, 0.95, camera)
    assert result.reaction_summary is not None
    return (
        result.reaction_summary.created_seeds,
        result.reaction_summary.productive_hits,
        result.reaction_summary.local_density,
        tuple(
            (
                event.execute_frame,
                event.kind.name,
                event.target_node_id,
                event.source_node_id,
                event.generation,
                round(event.energy, 4),
            )
            for event in result.reaction_events
        ),
        tuple(
            (
                round(project_point(simulation.state.nodes[node_id].position, camera).screen_x, 3),
                round(project_point(simulation.state.nodes[node_id].position, camera).screen_y, 3),
            )
            for node_id in result.spawned_node_ids
        ),
    )


def distance_from_origin(
    simulation: CloudSimulation,
    camera,
    node_id: int,
    origin_x: float,
    origin_y: float,
) -> float:
    projection = project_point(simulation.state.nodes[node_id].position, camera)
    return ((projection.screen_x - origin_x) ** 2 + (projection.screen_y - origin_y) ** 2) ** 0.5


if __name__ == "__main__":
    unittest.main()
