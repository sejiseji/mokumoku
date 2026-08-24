from __future__ import annotations

import unittest

from src import config
from src.camera.camera import build_camera_basis
from src.camera.projection import project_point
from src.cloud.graph import create_node, recompute_clusters
from src.cloud.reaction import ReactionEventKind
from src.cloud.simulation import CloudSimulation
from src.math3d import Vec3
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
            if event.kind
            in (ReactionEventKind.NODE_PULSE, ReactionEventKind.VISUAL_WAVE_HIT)
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

    def test_seed_resonance_primes_then_ignites_nearby_dormant_seed(self) -> None:
        simulation, camera, root_id = self.make_seed()
        root = simulation.state.nodes[root_id]
        root_projection = project_point(root.position, camera)

        first = simulation.tap_screen(root_projection.screen_x, root_projection.screen_y, camera)
        self.assertGreaterEqual(len(first.spawned_node_ids), 1)
        seed = simulation.tap_screen(
            root_projection.screen_x + config.DORMANT_SEED_TAP_RADIUS_PX + 14.0,
            root_projection.screen_y,
            camera,
        )
        self.assertEqual(seed.kind, "seed")
        self.assertIsNotNone(seed.node_id)
        seed_node = simulation.state.nodes[seed.node_id]
        old_mass = seed_node.mass

        primed = simulation.tap_screen(root_projection.screen_x, root_projection.screen_y, camera)
        primed_events = [
            event
            for event in primed.reaction_events
            if event.target_node_id == seed.node_id
        ]
        self.assertNotIn(seed.node_id, primed.resonant_node_ids)
        self.assertTrue(
            any(event.kind is ReactionEventKind.VISUAL_WAVE_HIT for event in primed_events)
        )
        self.assertGreater(seed_node.excitation, 0.0)
        self.assertAlmostEqual(seed_node.mass, old_mass)

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

    def test_refractory_blocks_immediate_rebloom(self) -> None:
        simulation, camera, node_id = self.make_seed()
        node = simulation.state.nodes[node_id]
        projection = project_point(node.position, camera)

        first = simulation.tap_screen(projection.screen_x, projection.screen_y, camera)
        self.assertGreaterEqual(first.reaction_summary.reacted_nodes, 1)
        mass_after_first = node.mass
        refractory_until = node.refractory_until_frame

        second = simulation.tap_screen(projection.screen_x, projection.screen_y, camera)

        self.assertGreater(refractory_until, second.created_frame)
        self.assertNotIn(node_id, second.reacted_node_ids)
        self.assertAlmostEqual(node.mass, mass_after_first)
        self.assertTrue(
            any(
                event.kind is ReactionEventKind.VISUAL_WAVE_HIT
                and event.target_node_id == node_id
                for event in second.reaction_events
            )
        )

    def test_seed_excitation_decays_over_time(self) -> None:
        simulation, camera, root_id = self.make_seed()
        root = simulation.state.nodes[root_id]
        root_projection = project_point(root.position, camera)

        seed = simulation.tap_screen(
            root_projection.screen_x + config.DORMANT_SEED_TAP_RADIUS_PX + 14.0,
            root_projection.screen_y,
            camera,
        )
        self.assertIsNotNone(seed.node_id)
        seed_node = simulation.state.nodes[seed.node_id]

        simulation.tap_screen(root_projection.screen_x, root_projection.screen_y, camera)
        primed_excitation = seed_node.excitation

        simulation.advance_time(1.0)

        self.assertGreater(primed_excitation, seed_node.excitation)

    def test_primed_seed_quorum_can_trigger_delayed_bloom(self) -> None:
        isolated = CloudSimulation(RandomSource(24680))
        isolated_target = create_ecology_seed(isolated, Vec3(0.0, 150.0, 0.0))
        isolated_target.excitation = config.SEED_PRIMED_THRESHOLD

        isolated_result = isolated.apply_seed_ecology_energy(
            isolated_target.id,
            0.16,
            24,
            1,
            is_seed=True,
        )

        simulation = CloudSimulation(RandomSource(24680))
        target = create_ecology_seed(simulation, Vec3(0.0, 150.0, 0.0))
        first_neighbor = create_ecology_seed(simulation, Vec3(22.0, 150.0, 0.0))
        second_neighbor = create_ecology_seed(simulation, Vec3(-24.0, 150.0, 0.0))
        target.excitation = config.SEED_PRIMED_THRESHOLD
        first_neighbor.excitation = config.SEED_PRIMED_THRESHOLD * 1.25
        second_neighbor.excitation = config.SEED_PRIMED_THRESHOLD * 1.25

        quorum_result = simulation.apply_seed_ecology_energy(
            target.id,
            0.16,
            24,
            1,
            is_seed=True,
        )

        self.assertFalse(isolated_result.bloomed)
        self.assertTrue(quorum_result.bloomed)
        self.assertGreater(target.refractory_until_frame, 24)

    def test_mid_range_seed_inhibition_raises_bloom_threshold(self) -> None:
        simulation = CloudSimulation(RandomSource(24681))
        target = create_ecology_seed(simulation, Vec3(0.0, 150.0, 0.0))
        baseline = simulation.seed_bloom_threshold(target, 0.0, 0.0)
        blocker = create_ecology_seed(simulation, Vec3(28.0, 150.0, 0.0))
        blocker.mass = config.RETENTION_GROWN_MASS * 1.4

        _activation, inhibition = simulation.local_seed_ecology_signals(target.id, 24)
        inhibited = simulation.seed_bloom_threshold(target, 0.0, inhibition)

        self.assertGreater(inhibition, 0.0)
        self.assertGreater(inhibited, baseline)

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


def create_ecology_seed(simulation: CloudSimulation, position: Vec3):
    lineage = simulation.ensure_active_lineage()
    cluster_id = simulation.state.next_cluster_id
    simulation.state.next_cluster_id += 1
    node = create_node(
        simulation.state,
        lineage.id,
        cluster_id,
        position,
        simulation.rng,
    )
    recompute_clusters(simulation.state, lineage.id)
    return node


if __name__ == "__main__":
    unittest.main()
