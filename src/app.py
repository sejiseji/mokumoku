from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.camera.camera import CameraController
from src.camera.projection import camera_depth
from src.cloud.incubation import build_adjacency, node_retention_score
from src.cloud.rendering import (
    BridgePayload,
    EdgePayload,
    NodePayload,
    collect_cloud_render_items,
)
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


@dataclass(frozen=True)
class CameraButton:
    direction: int
    label: str
    x: int
    y: int
    width: int
    height: int


def camera_buttons() -> tuple[CameraButton, CameraButton]:
    y = config.CAMERA_BUTTON_Y
    width = config.CAMERA_BUTTON_WIDTH
    height = config.CAMERA_BUTTON_HEIGHT
    margin = config.CAMERA_BUTTON_MARGIN_X
    return (
        CameraButton(-1, "<", margin, y, width, height),
        CameraButton(
            1,
            ">",
            config.SCREEN_WIDTH - margin - width,
            y,
            width,
            height,
        ),
    )


def camera_button_direction_at(screen_x: float, screen_y: float) -> int | None:
    for button in camera_buttons():
        if (
            button.x <= screen_x < button.x + button.width
            and button.y <= screen_y < button.y + button.height
        ):
            return button.direction
    return None


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
        self.debug_enabled = False
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
            self.request_camera_relative(-1)
        if pyxel.btnp(pyxel.KEY_E):
            self.request_camera_relative(1)
        if pyxel.btnp(pyxel.KEY_C):
            self.camera.cycle()
            self.cancel_pointer()
        if pyxel.btnp(pyxel.KEY_D):
            self.debug_enabled = not self.debug_enabled
        key_f4 = getattr(pyxel, "KEY_F4", None)
        if key_f4 is not None and pyxel.btnp(key_f4):
            self.cloud.advance_time(8.0)

        consumed_pointer = self.update_camera_buttons()
        self.camera.update(1.0 / config.FPS)
        if not consumed_pointer:
            self.update_pointer()
        self.cloud.update(1.0 / config.FPS)
        self.state.frame += 1
        if self.smoke_frames is not None and self.state.frame >= self.smoke_frames:
            pyxel.quit()

    def request_camera_relative(self, direction: int) -> None:
        self.camera.request_relative(direction)
        self.cancel_pointer()

    def update_camera_buttons(self) -> bool:
        pyxel = self.pyxel
        mouse_button_left = getattr(pyxel, "MOUSE_BUTTON_LEFT", 0)
        if not pyxel.btnp(mouse_button_left):
            return False

        direction = camera_button_direction_at(float(pyxel.mouse_x), float(pyxel.mouse_y))
        if direction is None:
            return False

        self.cancel_pointer()
        if self.camera.can_accept_cloud_input():
            self.request_camera_relative(direction)
        return True

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
        self.draw_camera_buttons()
        pyxel.text(8, 8, "MOKUMOKU Prototype A4.5", config.COLOR_UI)
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
        pyxel.text(8, config.SCREEN_HEIGHT - 14, "</> cam  D debug  F4 age", config.COLOR_UI)

    def draw_cloud(self) -> None:
        camera = self.camera.basis()
        for item in collect_cloud_render_items(self.cloud.state, camera, self.state.frame):
            if isinstance(item.payload, EdgePayload):
                if self.debug_enabled:
                    self.draw_edge_payload(item.payload)
            elif isinstance(item.payload, BridgePayload):
                self.draw_bridge_payload(item.payload)
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

    def draw_bridge_payload(self, payload: BridgePayload) -> None:
        pyxel = self.pyxel
        projection = payload.point
        sprite = payload.sprite
        x = int(projection.screen_x - sprite.width / 2)
        y = int(projection.screen_y - sprite.height / 2)

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
            radius = max(2, int(sprite.width / 2))
            pyxel.circ(int(projection.screen_x), int(projection.screen_y), radius, 7)

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
            if payload.mesh_intensity > 0.0:
                self.draw_single_cloud_mesh(payload, x, y)
        else:
            radius = max(2, int(node.radius * projection.scale * max(0.25, node.fade)))
            pyxel.circ(x, y, radius, 7)

    def draw_single_cloud_mesh(self, payload: NodePayload, x: int, y: int) -> None:
        pyxel = self.pyxel
        sprite = payload.sprite
        phase = payload.mesh_phase
        intensity = payload.mesh_intensity
        cx = x + sprite.width // 2
        cy = y + sprite.height // 2
        radius = max(4, int(sprite.width * 0.42))
        color = 6 if intensity > 0.45 else 5
        drift = math.sin(phase) * radius * 0.22

        for band in (-0.34, 0.12, 0.45):
            band_y = int(cy + band * radius + drift * (0.5 + abs(band)))
            half = int(radius * (1.0 - abs(band) * 0.42))
            middle_y = int(band_y + math.sin(phase + band * 2.7) * 1.2)
            pyxel.line(cx - half, band_y, cx, middle_y, color)
            pyxel.line(cx, middle_y, cx + half, band_y, color)

        diagonal_shift = int(math.cos(phase * 0.7) * radius * 0.2)
        pyxel.line(
            cx - radius + 2,
            cy - radius // 2 + diagonal_shift,
            cx + radius - 2,
            cy + radius // 2 + diagonal_shift,
            color,
        )

    def draw_camera_buttons(self) -> None:
        pyxel = self.pyxel
        disabled = not self.camera.can_accept_cloud_input()
        border = 5 if disabled else config.COLOR_UI
        text_color = 5 if disabled else config.COLOR_UI
        for button in camera_buttons():
            pyxel.rect(button.x, button.y, button.width, button.height, config.COLOR_GROUND)
            pyxel.rectb(button.x, button.y, button.width, button.height, border)
            pyxel.text(
                button.x + button.width // 2 - 2,
                button.y + button.height // 2 - 3,
                button.label,
                text_color,
            )

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
        adjacency = build_adjacency(self.cloud.state)
        retentions = [
            node_retention_score(node, adjacency.get(node.id, set()), self.cloud.state)
            for node in self.cloud.state.live_nodes()
        ]
        average_retention = sum(retentions) / len(retentions) if retentions else 0.0
        counts = (
            f"nodes {len(self.cloud.state.nodes)} "
            f"edges {len(self.cloud.state.edges)} "
            f"clusters {cluster_count}"
        )
        pyxel.text(8, 50, counts, config.COLOR_UI)
        pyxel.text(8, 60, f"selected {selected_text}", config.COLOR_UI)
        pyxel.text(8, 70, f"mature {mature} pruning {pruning}", config.COLOR_UI)
        pyxel.text(8, 80, f"retain {average_retention:.2f}", config.COLOR_UI)


def run(seed: int = 12345, headless: bool = False, smoke_frames: int | None = None) -> None:
    MokumokuApp(seed=seed, headless=headless, smoke_frames=smoke_frames)
