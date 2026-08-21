from __future__ import annotations

import unittest

from src import config
from src.camera.camera import build_camera_basis
from src.cloud.graph import add_edge, create_node, recompute_clusters
from src.cloud.incubation import build_adjacency, node_retention_score, retained_decay_ratio
from src.cloud.simulation import CloudSimulation
from src.enums import EdgeKind
from src.math3d import Vec3
from src.rng import RandomSource


class IncubationExtinctionTests(unittest.TestCase):
    def make_seeded_simulation(self) -> tuple[CloudSimulation, object, int]:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertIsNotNone(result.node_id)
        return simulation, camera, result.node_id

    def add_chain_node(self, simulation: CloudSimulation, parent_id: int, offset_x: float) -> int:
        parent = simulation.state.nodes[parent_id]
        node = create_node(
            simulation.state,
            parent.lineage_id,
            parent.cluster_id,
            parent.position + Vec3(offset_x, 0.0, 0.0),
            simulation.rng,
            parent_node_id=parent.id,
            generation=parent.generation + 1,
        )
        add_edge(
            simulation.state,
            parent.lineage_id,
            parent.cluster_id,
            parent.id,
            node.id,
            EdgeKind.PRIMARY,
        )
        recompute_clusters(simulation.state, parent.lineage_id)
        return node.id

    def test_touch_resets_graph_distance_two_only(self) -> None:
        simulation, _, first_id = self.make_seeded_simulation()
        second_id = self.add_chain_node(simulation, first_id, 18.0)
        third_id = self.add_chain_node(simulation, second_id, 18.0)
        fourth_id = self.add_chain_node(simulation, third_id, 18.0)
        for node in simulation.state.nodes.values():
            node.untouched_time = 9.0
            node.incubation = 1.0

        simulation.touch_neighborhood(first_id)

        self.assertEqual(simulation.state.nodes[first_id].untouched_time, 0.0)
        self.assertEqual(simulation.state.nodes[second_id].untouched_time, 0.0)
        self.assertEqual(simulation.state.nodes[third_id].untouched_time, 0.0)
        self.assertEqual(simulation.state.nodes[fourth_id].untouched_time, 9.0)

    def test_untouched_parts_incubate_and_noise_decays(self) -> None:
        simulation, _, node_id = self.make_seeded_simulation()
        node = simulation.state.nodes[node_id]
        node.noise = 0.8

        simulation.advance_time(config.INCUBATION_START_SECONDS + 1.0)

        self.assertGreater(node.incubation, 0.0)
        self.assertLess(node.noise, 0.8)

    def test_incubation_is_independent_of_camera_visibility(self) -> None:
        first, _, first_node_id = self.make_seeded_simulation()
        second, _, second_node_id = self.make_seeded_simulation()

        first.advance_time(6.0)
        second.advance_time(6.0)

        first_node = first.state.nodes[first_node_id]
        second_node = second.state.nodes[second_node_id]
        self.assertEqual(first_node.incubation, second_node.incubation)
        self.assertEqual(first_node.noise, second_node.noise)

    def test_smoothing_moves_mature_node_toward_neighbor(self) -> None:
        simulation, _, first_id = self.make_seeded_simulation()
        second_id = self.add_chain_node(simulation, first_id, 30.0)
        first = simulation.state.nodes[first_id]
        old_distance = first.position.distance_to(simulation.state.nodes[second_id].position)
        first.untouched_time = config.INCUBATION_START_SECONDS + 1.0
        first.incubation = 1.0

        simulation.update(1.0)

        new_distance = first.position.distance_to(simulation.state.nodes[second_id].position)
        self.assertLess(new_distance, old_distance)

    def test_retention_score_rewards_connected_grown_mature_nodes(self) -> None:
        simulation, _, first_id = self.make_seeded_simulation()
        self.add_chain_node(simulation, first_id, 18.0)
        first = simulation.state.nodes[first_id]
        first.mass = config.SEED_MASS + config.RETENTION_GROWN_MASS
        first.incubation = 1.0
        first.noise = 0.0
        adjacency = build_adjacency(simulation.state)

        retention = node_retention_score(first, adjacency[first_id], simulation.state)

        self.assertGreater(retention, 0.45)
        self.assertLess(retained_decay_ratio(retention), 1.0)

    def test_retained_connected_node_decays_slower_than_isolated_node(self) -> None:
        isolated, _, isolated_id = self.make_seeded_simulation()
        connected, _, connected_id = self.make_seeded_simulation()
        self.add_chain_node(connected, connected_id, 18.0)

        for simulation, node_id in [(isolated, isolated_id), (connected, connected_id)]:
            node = simulation.state.nodes[node_id]
            node.mass = config.SEED_MASS + config.RETENTION_GROWN_MASS
            node.untouched_time = config.NATURAL_MASS_DECAY_START_SECONDS + 1.0
            node.incubation = 1.0
            node.noise = 0.0

        isolated.update(1.0)
        connected.update(1.0)

        self.assertGreater(
            connected.state.nodes[connected_id].mass,
            isolated.state.nodes[isolated_id].mass,
        )

    def test_redundant_mature_nodes_merge_without_losing_structure(self) -> None:
        simulation, _, first_id = self.make_seeded_simulation()
        second_id = self.add_chain_node(simulation, first_id, 3.0)
        for node_id in [first_id, second_id]:
            node = simulation.state.nodes[node_id]
            node.untouched_time = config.REDUNDANT_MERGE_MIN_UNTOUCHED + 0.5
            node.incubation = 1.0

        simulation.update(1.0)

        self.assertEqual(len(simulation.state.nodes), 1)
        self.assertEqual(len(simulation.state.edges), 0)
        self.assertFalse(simulation.state.active_lineage().extinct)

    def test_pruning_uses_fade_before_node_removal(self) -> None:
        simulation, _, node_id = self.make_seeded_simulation()
        node = simulation.state.nodes[node_id]
        node.mass = config.PRUNE_MASS_THRESHOLD - 0.1

        simulation.update(1.0 / 60.0)

        self.assertIn(node_id, simulation.state.nodes)
        self.assertTrue(node.is_pruning)
        self.assertEqual(node.fade, 1.0)

        simulation.update(config.PRUNE_FADE_SECONDS * 0.5)

        self.assertIn(node_id, simulation.state.nodes)
        self.assertGreater(node.fade, 0.0)
        self.assertLess(node.fade, 1.0)

    def test_complete_extinction_ends_lineage_and_allows_new_seed(self) -> None:
        simulation, camera, first_node_id = self.make_seeded_simulation()
        lineage = simulation.state.active_lineage()
        self.assertIsNotNone(lineage)
        simulation.state.nodes[first_node_id].mass = config.PRUNE_MASS_THRESHOLD - 0.1

        simulation.advance_time(config.PRUNE_FADE_SECONDS + 0.2)

        self.assertEqual(simulation.state.nodes, {})
        self.assertTrue(lineage.extinct)
        self.assertIsNotNone(lineage.ended_at)

        result = simulation.tap_screen(160.0, 190.0, camera)

        self.assertEqual(result.kind, "seed")
        self.assertEqual(len(simulation.state.lineages), 2)
        self.assertFalse(simulation.state.active_lineage().extinct)

    def test_single_cloud_can_disappear_naturally_when_left_alone(self) -> None:
        simulation, _, _ = self.make_seeded_simulation()

        simulation.advance_time(30.0)

        self.assertEqual(simulation.state.nodes, {})
        self.assertIsNone(simulation.state.active_lineage())


if __name__ == "__main__":
    unittest.main()
