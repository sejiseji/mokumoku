from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.app import should_draw_edge_payload
from src.camera.camera import build_camera_basis
from src.camera.projection import project_point
from src.cloud.rendering import BridgePayload, NodePayload, collect_cloud_render_items
from src.cloud.simulation import CloudSimulation
from src.motion.atlas import WeatherMotionAtlas
from src.motion.runtime import WeatherMotionRuntime
from src.rng import RandomSource


@dataclass(frozen=True)
class PrototypeAAcceptanceReport:
    node_count: int
    edge_count: int
    bridge_count: int
    max_camera_tap_error: float
    max_ambient_offset: float
    normal_edges_hidden: bool

    @property
    def passed(self) -> bool:
        return (
            5 <= self.node_count <= 8
            and self.edge_count >= self.node_count - 1
            and self.bridge_count >= 2
            and self.max_camera_tap_error <= config.PROTOTYPE_A_TAP_ALIGNMENT_TOLERANCE_PX
            and self.max_ambient_offset <= 0.0
            and self.normal_edges_hidden
        )


def run_prototype_a_acceptance(seed: int = 12345) -> PrototypeAAcceptanceReport:
    simulation = build_connected_cloud(seed)
    camera = build_camera_basis(config.CAMERA_FRONT_YAW)
    atlas = WeatherMotionAtlas.build(seed=seed)
    runtime = WeatherMotionRuntime()

    items = collect_cloud_render_items(
        simulation.state,
        camera,
        frame=config.FPS * 2,
        motion_atlas=atlas,
        motion_runtime=runtime,
    )
    node_payloads = [item.payload for item in items if isinstance(item.payload, NodePayload)]
    bridge_payloads = [item.payload for item in items if isinstance(item.payload, BridgePayload)]

    return PrototypeAAcceptanceReport(
        node_count=len(node_payloads),
        edge_count=len(simulation.state.edges),
        bridge_count=len(bridge_payloads),
        max_camera_tap_error=max_camera_tap_error(seed),
        max_ambient_offset=max_ambient_offset(simulation, atlas, runtime),
        normal_edges_hidden=not should_draw_edge_payload(debug_enabled=False),
    )


def build_connected_cloud(seed: int = 12345) -> CloudSimulation:
    simulation = CloudSimulation(RandomSource(seed))
    camera = build_camera_basis(config.CAMERA_FRONT_YAW)
    root_result = simulation.tap_screen(160.0, 190.0, camera)
    if root_result.node_id is None:
        raise RuntimeError("failed to create acceptance root cloud node")

    node_id = root_result.node_id
    for offset_x, offset_y in ((26.0, 0.0), (-24.0, 4.0), (8.0, -22.0), (6.0, 24.0), (30.0, -18.0)):
        node = simulation.state.nodes[node_id]
        projection = project_point(node.position, camera)
        result = simulation.tap_screen(
            projection.screen_x + offset_x,
            projection.screen_y + offset_y,
            camera,
        )
        if result.node_id is not None:
            node_id = result.node_id
        simulation.advance_time(0.08)

    simulation.advance_time(0.40)
    return simulation


def max_camera_tap_error(seed: int = 12345) -> float:
    errors: list[float] = []
    for yaw in (config.CAMERA_LEFT_YAW, config.CAMERA_FRONT_YAW, config.CAMERA_RIGHT_YAW):
        camera = build_camera_basis(yaw)
        simulation = CloudSimulation(RandomSource(seed + int(yaw * 10)))
        result = simulation.tap_screen(172.0, 156.0, camera)
        if result.node_id is None:
            errors.append(float("inf"))
            continue
        projection = project_point(simulation.state.nodes[result.node_id].position, camera)
        errors.append(
            ((projection.screen_x - 172.0) ** 2 + (projection.screen_y - 156.0) ** 2)
            ** 0.5
        )
    return max(errors)


def max_ambient_offset(
    simulation: CloudSimulation,
    atlas: WeatherMotionAtlas,
    runtime: WeatherMotionRuntime,
) -> float:
    camera = build_camera_basis(config.CAMERA_FRONT_YAW)
    max_offset = 0.0
    for frame in range(0, config.FPS * 30 + 1, config.FPS * 5):
        items = collect_cloud_render_items(
            simulation.state,
            camera,
            frame=frame,
            motion_atlas=atlas,
            motion_runtime=runtime,
        )
        for item in items:
            if isinstance(item.payload, NodePayload):
                max_offset = max(
                    max_offset,
                    abs(item.payload.offset_x),
                    abs(item.payload.offset_y),
                )
    return max_offset
