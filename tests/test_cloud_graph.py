from __future__ import annotations

import unittest

from src import config
from src.camera.camera import build_camera_basis
from src.camera.projection import camera_depth, project_point
from src.cloud.graph import desired_edge_rest_length, node_degree
from src.cloud.simulation import CloudSimulation
from src.math3d import Vec3
from src.rng import RandomSource


def first_spawned_node_id(result) -> int:
    assert result.spawned_node_ids
    return result.spawned_node_ids[0]


def seed_stimulus_x(projection) -> float:
    return projection.screen_x + config.DORMANT_SEED_TAP_RADIUS_PX - 4.0


class CloudGraphTests(unittest.TestCase):
    def make_simulation(self) -> tuple[CloudSimulation, object]:
        simulation = CloudSimulation(RandomSource(12345))
        camera = build_camera_basis(0.0)
        return simulation, camera

    def seed_cloud(self) -> tuple[CloudSimulation, object, int]:
        simulation, camera = self.make_simulation()
        result = simulation.tap_screen(160.0, 190.0, camera)
        self.assertEqual(result.kind, "seed")
        self.assertIsNotNone(result.node_id)
        return simulation, camera, result.node_id

    def test_seed_creates_one_active_lineage_and_cluster(self) -> None:
        simulation, _, node_id = self.seed_cloud()
        lineage = simulation.state.active_lineage()

        self.assertIsNotNone(lineage)
        self.assertEqual(len(simulation.state.nodes), 1)
        self.assertEqual(simulation.state.nodes[node_id].lineage_id, lineage.id)
        self.assertEqual(len(lineage.active_cluster_ids), 1)

    def test_blank_tap_adds_detached_seed_without_second_lineage(self) -> None:
        simulation, camera, _ = self.seed_cloud()

        result = simulation.tap_screen(300.0, 80.0, camera)

        self.assertEqual(result.kind, "seed")
        self.assertIsNotNone(result.node_id)
        self.assertEqual(len(simulation.state.lineages), 1)
        self.assertEqual(len(simulation.state.nodes), 2)
        self.assertEqual(len(simulation.state.edges), 0)
        projection = project_point(simulation.state.nodes[result.node_id].position, camera)
        self.assertAlmostEqual(projection.screen_x, 300.0, delta=1.0)
        self.assertAlmostEqual(projection.screen_y, 80.0, delta=1.0)

    def test_second_blank_tap_inside_resonance_radius_stays_at_tap_position(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        first_projection = project_point(simulation.state.nodes[node_id].position, camera)
        tap_x = first_projection.screen_x + config.DORMANT_SEED_TAP_RADIUS_PX + 18.0
        tap_y = first_projection.screen_y
        self.assertLess(
            tap_x - first_projection.screen_x,
            config.SEED_RESONANCE_RADIUS_PX,
        )

        result = simulation.tap_screen(tap_x, tap_y, camera)

        self.assertEqual(result.kind, "seed")
        self.assertIsNotNone(result.node_id)
        self.assertEqual(len(result.spawned_node_ids), 0)
        projection = project_point(simulation.state.nodes[result.node_id].position, camera)
        self.assertAlmostEqual(projection.screen_x, tap_x, delta=1.0)
        self.assertAlmostEqual(projection.screen_y, tap_y, delta=1.0)

    def test_tap_on_node_grows_local_node(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        node = simulation.state.nodes[node_id]
        projection = project_point(node.position, camera)
        old_mass = node.mass

        result = simulation.tap_screen(projection.screen_x, projection.screen_y, camera)

        self.assertEqual(result.kind, "stimulus")
        self.assertIsNotNone(result.reaction_id)
        self.assertGreaterEqual(result.reaction_summary.reacted_nodes, 1)
        self.assertGreater(node.mass, old_mass)
        self.assertEqual(node.untouched_time, 0.0)

    def test_near_cloud_tap_adds_child_with_primary_edge(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        projection = project_point(simulation.state.nodes[node_id].position, camera)

        result = simulation.tap_screen(seed_stimulus_x(projection), projection.screen_y, camera)

        self.assertEqual(result.kind, "stimulus")
        self.assertGreaterEqual(len(result.spawned_node_ids), 1)
        self.assertEqual(len(simulation.state.nodes), 2)
        self.assertEqual(len(simulation.state.edges), 1)
        self.assertEqual(node_degree(simulation.state, node_id), 1)

    def test_primary_edge_rest_length_uses_node_radii_for_overlap(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        projection = project_point(simulation.state.nodes[node_id].position, camera)

        result = simulation.tap_screen(seed_stimulus_x(projection), projection.screen_y, camera)
        self.assertGreaterEqual(len(result.spawned_node_ids), 1)
        edge = next(iter(simulation.state.edges.values()))

        self.assertAlmostEqual(
            edge.rest_length,
            desired_edge_rest_length(simulation.state, edge.node_a, edge.node_b),
        )
        expected = (
            simulation.state.nodes[edge.node_a].radius
            + simulation.state.nodes[edge.node_b].radius
        ) * config.EDGE_REST_RADIUS_RATIO
        self.assertAlmostEqual(edge.rest_length, expected)

    def test_edge_cohesion_pulls_connected_nodes_toward_overlap(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        projection = project_point(simulation.state.nodes[node_id].position, camera)
        child = simulation.tap_screen(seed_stimulus_x(projection), projection.screen_y, camera)
        child_id = first_spawned_node_id(child)
        parent = simulation.state.nodes[node_id]
        child_node = simulation.state.nodes[child_id]
        old_distance = parent.position.distance_to(child_node.position)

        simulation.update(1.0 / 6.0)

        new_distance = parent.position.distance_to(child_node.position)
        self.assertLess(new_distance, old_distance)

    def test_long_press_condenses_node(self) -> None:
        simulation, _, node_id = self.seed_cloud()
        node = simulation.state.nodes[node_id]
        old_density = node.density
        old_moisture = node.moisture

        result = simulation.long_press_node(node_id)

        self.assertEqual(result.kind, "long_press")
        self.assertGreater(node.density, old_density)
        self.assertGreater(node.moisture, old_moisture)

    def test_drag_moves_node_on_depth_locked_plane(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        node = simulation.state.nodes[node_id]
        projection = project_point(node.position, camera)
        old_depth = camera_depth(node.position, camera)

        result = simulation.drag_node_to_screen(
            node_id,
            projection.screen_x + 28.0,
            projection.screen_y - 16.0,
            camera,
        )

        self.assertEqual(result.kind, "drag")
        self.assertAlmostEqual(camera_depth(node.position, camera), old_depth)

    def test_overstretched_edge_splits_cluster_with_same_lineage(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        projection = project_point(simulation.state.nodes[node_id].position, camera)
        child = simulation.tap_screen(seed_stimulus_x(projection), projection.screen_y, camera)
        child_id = first_spawned_node_id(child)
        child_node = simulation.state.nodes[child_id]
        child_projection = project_point(child_node.position, camera)

        simulation.drag_node_to_screen(
            child_node.id,
            child_projection.screen_x + 150.0,
            child_projection.screen_y,
            camera,
        )

        lineage_ids = {node.lineage_id for node in simulation.state.nodes.values()}
        cluster_ids = {node.cluster_id for node in simulation.state.nodes.values()}
        self.assertEqual(len(lineage_ids), 1)
        self.assertEqual(len(cluster_ids), 2)
        self.assertEqual(len(simulation.state.edges), 0)

    def test_flick_splits_selected_node_from_cluster(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        projection = project_point(simulation.state.nodes[node_id].position, camera)
        child = simulation.tap_screen(seed_stimulus_x(projection), projection.screen_y, camera)
        child_id = first_spawned_node_id(child)

        result = simulation.flick_node(child_id, Vec3(500.0, 0.0, 0.0))

        self.assertEqual(result.kind, "flick")
        self.assertEqual(len(simulation.state.edges), 0)
        self.assertEqual(len({node.cluster_id for node in simulation.state.nodes.values()}), 2)

    def test_near_slow_fragments_merge_and_preserve_lineage_history(self) -> None:
        simulation, camera, node_id = self.seed_cloud()
        projection = project_point(simulation.state.nodes[node_id].position, camera)
        child = simulation.tap_screen(seed_stimulus_x(projection), projection.screen_y, camera)
        child_id = first_spawned_node_id(child)
        lineage = simulation.state.active_lineage()
        self.assertIsNotNone(lineage)
        lineage.scored_event_ids.add("lineage:sample")
        simulation.flick_node(child_id, Vec3(0.0, 0.0, 0.0))

        parent = simulation.state.nodes[node_id]
        child_node = simulation.state.nodes[child_id]
        child_node.previous_position = child_node.position
        child_node.position = parent.position + Vec3(8.0, 0.0, 0.0)
        child_node.velocity = Vec3(0.0, 0.0, 0.0)
        parent.velocity = Vec3(0.0, 0.0, 0.0)

        simulation.update(1.0 / 60.0)

        self.assertEqual(len({node.cluster_id for node in simulation.state.nodes.values()}), 1)
        self.assertGreaterEqual(len(simulation.state.edges), 1)
        self.assertIn("lineage:sample", lineage.scored_event_ids)


if __name__ == "__main__":
    unittest.main()
