from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.camera.camera import CameraController
from src.camera.projection import camera_depth
from src.cloud.rendering import EdgePayload, NodePayload, collect_cloud_render_items
from src.cloud.simulation import CloudSimulation
from src.rng import RandomSource


@dataclass
class AppState:
    seed: int
    frame: int = 0


@dataclass
class ActivePointer:
    start_x: float
    start_y: float
    last_x: float
    last_y: float
    press_frame: int
    selected_node_id: int | None
    dragging: bool = False
    long_press_sent: bool = False


class MokumokuApp:
    """Pyxel app shell.

    Cloud simulation and gameplay are intentionally left for later Prototype A waves.
    """

    def __init__(
        self,
        seed: int,
        headless: bool = False,
        smoke_frames: int | None = None,
    ) -> None:
        import pyxel

        self.pyxel = pyxel
        self.state = AppState(seed=seed)
        self.rng = RandomSource(seed)
        self.camera = CameraController()
        self.cloud = CloudSimulation(self.rng)
        self.pointer: ActivePointer | None = None
        self.previous_selected_id: int | None = None
        self.debug_enabled = True
        self.assets_loaded = False
        self.smoke_frames = smoke_frames

        pyxel.init(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            title=config.WINDOW_TITLE,
            fps=config.FPS,
            headless=headless,
        )
        self.load_assets()
        pyxel.run(self.update, self.draw)

    def load_assets(self) -> None:
        resource_path = Path(config.PYXEL_RESOURCE_PATH)
        if resource_path.exists():
            self.pyxel.load(str(resource_path))
            self.assets_loaded = True

    def update(self) -> None:
        pyxel = self.pyxel
        if pyxel.btnp(pyxel.KEY_ESCAPE):
            pyxel.quit()

        if pyxel.btnp(pyxel.KEY_Q):
            self.camera.request_relative(-1)
            self.cancel_pointer()
        if pyxel.btnp(pyxel.KEY_E):
            self.camera.request_relative(1)
            self.cancel_pointer()
        if pyxel.btnp(pyxel.KEY_C):
            self.camera.cycle()
            self.cancel_pointer()
        if pyxel.btnp(pyxel.KEY_D):
            self.debug_enabled = not self.debug_enabled
        key_f4 = getattr(pyxel, "KEY_F4", None)
        if key_f4 is not None and pyxel.btnp(key_f4):
            self.cloud.advance_time(8.0)

        self.camera.update(1.0 / config.FPS)
        self.update_pointer()
        self.cloud.update(1.0 / config.FPS)
        self.state.frame += 1
        if self.smoke_frames is not None and self.state.frame >= self.smoke_frames:
            pyxel.quit()

    def update_pointer(self) -> None:
        pyxel = self.pyxel
        mouse_button_left = getattr(pyxel, "MOUSE_BUTTON_LEFT", 0)
        pressed = pyxel.btn(mouse_button_left)
        just_pressed = pyxel.btnp(mouse_button_left)
        just_released = pyxel.btnr(mouse_button_left)
        x = float(pyxel.mouse_x)
        y = float(pyxel.mouse_y)

        if not self.camera.can_accept_cloud_input():
            self.cancel_pointer()
            return

        camera = self.camera.basis()
        if just_pressed:
            selected = self.cloud.hit_node(x, y, camera, self.previous_selected_id)
            self.pointer = ActivePointer(
                start_x=x,
                start_y=y,
                last_x=x,
                last_y=y,
                press_frame=self.state.frame,
                selected_node_id=None if selected is None else selected.id,
            )
            self.previous_selected_id = None if selected is None else selected.id
            return

        pointer = self.pointer
        if pointer is None:
            return

        duration = (self.state.frame - pointer.press_frame) / config.FPS
        distance = math.hypot(x - pointer.start_x, y - pointer.start_y)

        if pressed and pointer.selected_node_id is not None:
            if distance >= config.DRAG_START_DISTANCE:
                pointer.dragging = True
                self.cloud.drag_node_to_screen(pointer.selected_node_id, x, y, camera)
            elif (
                duration >= config.LONG_PRESS_SECONDS
                and distance <= config.PRESS_SLOP_PX
                and not pointer.long_press_sent
            ):
                self.cloud.long_press_node(pointer.selected_node_id)
                pointer.long_press_sent = True

        if just_released:
            self.finish_pointer(x, y, duration, distance, camera)
            return

        pointer.last_x = x
        pointer.last_y = y

    def finish_pointer(self, x: float, y: float, duration: float, distance: float, camera) -> None:
        pointer = self.pointer
        if pointer is None:
            return

        speed = distance / max(duration, 1.0 / config.FPS)
        if (
            pointer.selected_node_id is not None
            and pointer.dragging
            and distance >= config.FLICK_MIN_DISTANCE
            and speed >= config.FLICK_MIN_SPEED
        ):
            velocity = camera.right * ((x - pointer.start_x) / max(duration, 0.001))
            velocity -= camera.up * ((y - pointer.start_y) / max(duration, 0.001))
            self.cloud.flick_node(pointer.selected_node_id, velocity)
        elif not pointer.dragging and not pointer.long_press_sent:
            self.cloud.tap_screen(x, y, camera)

        self.pointer = None

    def cancel_pointer(self) -> None:
        self.pointer = None
        self.previous_selected_id = None

    def draw(self) -> None:
        pyxel = self.pyxel
        pyxel.cls(config.COLOR_SKY)
        pyxel.rect(
            0,
            config.GROUND_TOP_Y,
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT - config.GROUND_TOP_Y,
            config.COLOR_GROUND,
        )
        pyxel.line(
            0,
            config.SKY_BOTTOM_Y,
            config.SCREEN_WIDTH,
            config.SKY_BOTTOM_Y,
            config.COLOR_UI,
        )
        self.draw_cloud()
        pyxel.text(8, 8, "MOKUMOKU Prototype A4", config.COLOR_UI)
        pyxel.text(
            8,
            18,
            "Tap cloud/sky, hold, drag, flick.",
            config.COLOR_UI,
        )
        pyxel.text(8, 28, f"yaw {self.camera.current_yaw:5.1f}", config.COLOR_UI)
        pyxel.text(
            8,
            38,
            f"cloud input {'on' if self.camera.can_accept_cloud_input() else 'locked'}",
            config.COLOR_UI,
        )
        if self.debug_enabled:
            self.draw_debug()
        pyxel.text(8, config.SCREEN_HEIGHT - 14, "Q/E/C cam  D debug  F4 age", config.COLOR_UI)

    def draw_cloud(self) -> None:
        camera = self.camera.basis()
        for item in collect_cloud_render_items(self.cloud.state, camera, self.state.frame):
            if isinstance(item.payload, EdgePayload):
                self.draw_edge_payload(item.payload)
            elif isinstance(item.payload, NodePayload):
                self.draw_node_payload(item.payload)

    def draw_edge_payload(self, payload: EdgePayload) -> None:
        pyxel = self.pyxel
        color = 5 if payload.edge.strain < 1.6 else 4
        pyxel.line(
            int(payload.point_a.screen_x),
            int(payload.point_a.screen_y),
            int(payload.point_b.screen_x),
            int(payload.point_b.screen_y),
            color,
        )

    def draw_node_payload(self, payload: NodePayload) -> None:
        pyxel = self.pyxel
        node = payload.node
        projection = payload.projection
        sprite = payload.sprite
        x = int(projection.screen_x + payload.offset_x - sprite.width / 2)
        y = int(projection.screen_y + payload.offset_y - sprite.height / 2)

        if self.assets_loaded:
            pyxel.blt(
                x,
                y,
                sprite.image,
                sprite.u,
                sprite.v,
                sprite.width,
                sprite.height,
                sprite.colkey,
            )
        else:
            radius = max(2, int(node.radius * projection.scale * max(0.25, node.fade)))
            pyxel.circ(x, y, radius, 7)

    def draw_debug(self) -> None:
        pyxel = self.pyxel
        lineage = self.cloud.state.active_lineage()
        cluster_count = 0 if lineage is None else len(lineage.active_cluster_ids)
        selected_text = "-"
        if self.pointer is not None and self.pointer.selected_node_id is not None:
            node = self.cloud.state.nodes[self.pointer.selected_node_id]
            selected_text = f"{node.id} d={camera_depth(node.position, self.camera.basis()):.1f}"
        mature = sum(1 for node in self.cloud.state.nodes.values() if node.incubation > 0.0)
        pruning = sum(1 for node in self.cloud.state.nodes.values() if node.is_pruning)
        counts = (
            f"nodes {len(self.cloud.state.nodes)} "
            f"edges {len(self.cloud.state.edges)} "
            f"clusters {cluster_count}"
        )
        pyxel.text(8, 50, counts, config.COLOR_UI)
        pyxel.text(8, 60, f"selected {selected_text}", config.COLOR_UI)
        pyxel.text(8, 70, f"mature {mature} pruning {pruning}", config.COLOR_UI)


def run(seed: int = 12345, headless: bool = False, smoke_frames: int | None = None) -> None:
    MokumokuApp(seed=seed, headless=headless, smoke_frames=smoke_frames)
