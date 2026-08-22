from __future__ import annotations

import unittest

from src import config
from src.camera.camera import build_camera_basis
from src.camera.projection import project_point
from src.cloud.reaction import (
    ReactionEventKind,
    charge_level,
    radial_strength,
    reaction_radius_px,
)
from src.cloud.simulation import CloudSimulation
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

    def test_empty_short_reaction_creates_center_seed(self) -> None:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)

        result = simulation.radial_reaction_screen(160.0, 190.0, 0.05, camera)

        self.assertEqual(result.kind, "radial")
        self.assertIsNotNone(result.reaction_summary)
        self.assertEqual(result.reaction_summary.created_seeds, 1)
        self.assertLess(result.reaction_summary.local_density, 0.25)
        self.assertEqual(len(simulation.state.nodes), 1)
        created = simulation.state.nodes[result.spawned_node_ids[0]]
        projection = project_point(created.position, camera)
        self.assertAlmostEqual(projection.screen_x, 160.0, delta=1.0)
        self.assertAlmostEqual(projection.screen_y, 190.0, delta=1.0)

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
        self.assertLessEqual(
            long.reaction_summary.created_seeds,
            config.MAX_NEW_SEEDS_PER_REACTION,
        )
        created_events = [
            event for event in long.reaction_events if event.kind is ReactionEventKind.CREATE_SEED
        ]
        self.assertEqual(len(created_events), long.reaction_summary.created_seeds)

    def test_dense_reaction_prefers_existing_nodes_over_new_seeds(self) -> None:
        sparse = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        sparse_result = sparse.radial_reaction_screen(160.0, 190.0, 0.95, camera)
        self.assertIsNotNone(sparse_result.reaction_summary)

        dense = CloudSimulation(RandomSource(12345))
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
        self.assertGreaterEqual(dense_result.reaction_summary.productive_hits, 1)
        self.assertLessEqual(
            dense_result.reaction_summary.created_seeds,
            sparse_result.reaction_summary.created_seeds,
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


if __name__ == "__main__":
    unittest.main()
