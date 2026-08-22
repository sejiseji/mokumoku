from __future__ import annotations

import unittest

from src import config
from src.camera.camera import build_camera_basis
from src.camera.projection import project_point
from src.cloud.reaction import ReactionEventKind
from src.cloud.simulation import CloudSimulation
from src.rng import RandomSource


class CloudReactionTests(unittest.TestCase):
    def make_seed(self) -> tuple[CloudSimulation, object, int]:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertEqual(result.kind, "seed")
        self.assertIsNotNone(result.node_id)
        return simulation, camera, result.node_id

    def test_direct_tap_creates_stimulus_and_guaranteed_sprout_for_single_seed(self) -> None:
        simulation, camera, node_id = self.make_seed()
        node = simulation.state.nodes[node_id]
        projection = project_point(node.position, camera)

        result = simulation.tap_screen(projection.screen_x, projection.screen_y, camera)

        self.assertEqual(result.kind, "stimulus")
        self.assertIsNotNone(result.reaction_id)
        self.assertEqual(result.node_id, node_id)
        self.assertGreaterEqual(len(result.spawned_node_ids), 1)
        self.assertGreaterEqual(result.reaction_summary.reacted_nodes, 1)
        self.assertGreaterEqual(result.reaction_summary.sprouts, 1)
        self.assertEqual(result.reaction_summary.ignited_seeds, 0)
        self.assertTrue(
            any(event.kind is ReactionEventKind.NODE_PULSE for event in result.reaction_events)
        )
        self.assertTrue(
            any(
                event.kind is ReactionEventKind.SECONDARY_SPROUT
                for event in result.reaction_events
            )
        )

    def test_soft_hit_near_cloud_uses_stimulus_instead_of_blank_seed(self) -> None:
        simulation, camera, node_id = self.make_seed()
        node = simulation.state.nodes[node_id]
        projection = project_point(node.position, camera)

        result = simulation.tap_screen(
            projection.screen_x + config.SOFT_HIT_RADIUS_PX - 2,
            projection.screen_y,
            camera,
        )

        self.assertEqual(result.kind, "stimulus")
        self.assertEqual(result.node_id, node_id)
        self.assertEqual(len(simulation.state.lineages), 1)

    def test_neighbor_pulse_events_are_delayed_by_graph_distance(self) -> None:
        simulation, camera, root_id = self.make_seed()
        root = simulation.state.nodes[root_id]
        root_projection = project_point(root.position, camera)
        child = simulation.add_child_near_screen(
            root,
            root_projection.screen_x + 30.0,
            root_projection.screen_y,
            camera,
        )
        self.assertIsNotNone(child)
        child_projection = project_point(child.position, camera)
        grandchild = simulation.add_child_near_screen(
            child,
            child_projection.screen_x + 30.0,
            child_projection.screen_y,
            camera,
        )
        self.assertIsNotNone(grandchild)

        result = simulation.tap_screen(root_projection.screen_x, root_projection.screen_y, camera)
        pulse_events = [
            event
            for event in result.reaction_events
            if event.kind is ReactionEventKind.NODE_PULSE
        ]
        frame_by_node = {
            event.target_node_id: event.execute_frame for event in pulse_events
        }

        self.assertEqual(frame_by_node[root_id], result.created_frame)
        self.assertGreater(frame_by_node[child.id], frame_by_node[root_id])
        self.assertGreater(frame_by_node[grandchild.id], frame_by_node[child.id])
        self.assertEqual(len(frame_by_node), len(set(frame_by_node)))
        self.assertLessEqual(
            result.reaction_summary.highest_generation,
            config.MAX_CHAIN_GENERATION,
        )

    def test_seed_resonance_ignites_nearby_dormant_seed_with_delay(self) -> None:
        simulation, camera, root_id = self.make_seed()
        root = simulation.state.nodes[root_id]
        root_projection = project_point(root.position, camera)

        first = simulation.tap_screen(root_projection.screen_x, root_projection.screen_y, camera)
        self.assertGreaterEqual(len(first.spawned_node_ids), 1)
        seed = simulation.tap_screen(
            root_projection.screen_x + 68.0,
            root_projection.screen_y,
            camera,
        )
        self.assertEqual(seed.kind, "seed")
        self.assertIsNotNone(seed.node_id)

        result = simulation.tap_screen(root_projection.screen_x, root_projection.screen_y, camera)
        resonance_events = [
            event
            for event in result.reaction_events
            if event.kind is ReactionEventKind.SEED_IGNITION
        ]

        self.assertIn(seed.node_id, result.resonant_node_ids)
        self.assertEqual(resonance_events[0].target_node_id, seed.node_id)
        self.assertGreater(resonance_events[0].execute_frame, result.created_frame)
        self.assertGreaterEqual(result.reaction_summary.ignited_seeds, 1)

    def test_reaction_budgets_cap_chain_size(self) -> None:
        simulation, camera, root_id = self.make_seed()
        current = simulation.state.nodes[root_id]
        for _ in range(config.MAX_REACTED_NODES_PER_REACTION + 4):
            projection = project_point(current.position, camera)
            child = simulation.add_child_near_screen(
                current,
                projection.screen_x + 24.0,
                projection.screen_y,
                camera,
            )
            self.assertIsNotNone(child)
            current = child

        root = simulation.state.nodes[root_id]
        root_projection = project_point(root.position, camera)
        result = simulation.tap_screen(root_projection.screen_x, root_projection.screen_y, camera)

        self.assertLessEqual(
            result.reaction_summary.reacted_nodes,
            config.MAX_REACTED_NODES_PER_REACTION,
        )
        self.assertLessEqual(
            len(result.reaction_events),
            config.MAX_PENDING_REACTION_EVENTS,
        )

    def test_reaction_plan_is_deterministic_for_same_seed_and_input(self) -> None:
        first = reaction_signature()
        second = reaction_signature()

        self.assertEqual(first, second)


def reaction_signature() -> tuple:
    simulation = CloudSimulation(RandomSource(2468))
    camera = build_camera_basis(0.0)
    root_result = simulation.tap_screen(160.0, 190.0, camera)
    assert root_result.node_id is not None
    root = simulation.state.nodes[root_result.node_id]
    projection = project_point(root.position, camera)
    result = simulation.tap_screen(projection.screen_x, projection.screen_y, camera)
    assert result.reaction_summary is not None
    event_signature = tuple(
        (
            event.execute_frame,
            event.kind.name,
            event.target_node_id,
            event.source_node_id,
            event.generation,
            round(event.energy, 4),
            event.direction_index,
        )
        for event in result.reaction_events
    )
    return (
        result.reaction_summary.reacted_nodes,
        result.reaction_summary.sprouts,
        result.reaction_summary.ignited_seeds,
        result.reaction_summary.highest_generation,
        result.reaction_summary.reaction_grade.name,
        result.spawned_node_ids,
        result.resonant_node_ids,
        event_signature,
    )


if __name__ == "__main__":
    unittest.main()
