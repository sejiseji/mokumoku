from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.assets.sprite_map import CloudSpriteFamily
from src.build_info import APP_BUILD_LABEL
from src.camera.camera import CameraController
from src.camera.projection import camera_depth
from src.cloud.incubation import build_adjacency, node_retention_score
from src.cloud.reaction import ReactionEventKind, charge_level, reaction_radius_px
from src.cloud.rendering import (
    BodyPayload,
    BridgePayload,
    CloudDepthLayer,
    EdgePayload,
    NodePayload,
    collect_cloud_render_items,
)
from src.cloud.simulation import CloudOperationResult, CloudSimulation
from src.motion.atlas import WeatherMotionAtlas
from src.motion.runtime import TouchResponseKind, WeatherMotionRuntime
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
    last_drag_hold_frame: int | None = None


@dataclass(frozen=True)
class CloudReactionWave:
    reaction_id: int
    origin_screen_x: float
    origin_screen_y: float
    release_frame: int
    max_radius_px: float
    charge_level: float
    local_density: float


@dataclass(frozen=True)
class CameraButton:
    direction: int
    label: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class CameraDialPointer:
    start_x: float
    start_y: float
    moved: bool = False


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


def dial_x_to_yaw(pointer_x: float) -> float:
    normalized = (pointer_x - config.CAMERA_DIAL_LEFT) / (
        config.CAMERA_DIAL_RIGHT - config.CAMERA_DIAL_LEFT
    )
    normalized = max(0.0, min(1.0, normalized))
    return config.CAMERA_MIN_YAW + (
        config.CAMERA_MAX_YAW - config.CAMERA_MIN_YAW
    ) * normalized


def yaw_to_dial_x(yaw: float) -> float:
    clamped_yaw = max(config.CAMERA_MIN_YAW, min(config.CAMERA_MAX_YAW, yaw))
    normalized = (clamped_yaw - config.CAMERA_MIN_YAW) / (
        config.CAMERA_MAX_YAW - config.CAMERA_MIN_YAW
    )
    return config.CAMERA_DIAL_LEFT + (
        config.CAMERA_DIAL_RIGHT - config.CAMERA_DIAL_LEFT
    ) * normalized


def camera_dial_hit(screen_x: float, screen_y: float) -> bool:
    return (
        config.CAMERA_DIAL_LEFT - config.CAMERA_DIAL_TOUCH_PADDING_X
        <= screen_x
        <= config.CAMERA_DIAL_RIGHT + config.CAMERA_DIAL_TOUCH_PADDING_X
        and config.CAMERA_DIAL_Y - config.CAMERA_DIAL_TOUCH_PADDING_Y
        <= screen_y
        <= config.CAMERA_DIAL_Y + config.CAMERA_DIAL_TOUCH_PADDING_Y
    )


def camera_dial_center_hit(screen_x: float, screen_y: float) -> bool:
    center_x = (config.CAMERA_DIAL_LEFT + config.CAMERA_DIAL_RIGHT) / 2.0
    return camera_dial_hit(screen_x, screen_y) and abs(screen_x - center_x) <= (
        config.CAMERA_DIAL_TOUCH_PADDING_X + 2
    )


def should_draw_edge_payload(debug_enabled: bool) -> bool:
    return debug_enabled


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
        atlas_start = time.perf_counter()
        self.motion_atlas = WeatherMotionAtlas.build(seed=seed)
        self.motion_atlas_build_ms = (time.perf_counter() - atlas_start) * 1000.0
        self.motion_runtime = WeatherMotionRuntime()
        self.pointer: ActivePointer | None = None
        self.camera_dial_pointer: CameraDialPointer | None = None
        self.reaction_waves: list[CloudReactionWave] = []
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
            self.camera_dial_pointer = None
            self.camera.cycle()
            self.cancel_pointer()
        if pyxel.btnp(pyxel.KEY_D):
            self.debug_enabled = not self.debug_enabled
        key_f4 = getattr(pyxel, "KEY_F4", None)
        if key_f4 is not None and pyxel.btnp(key_f4):
            self.cloud.advance_time(8.0)

        consumed_pointer = self.update_camera_controls()
        self.camera.update(1.0 / config.FPS)
        if not consumed_pointer:
            self.update_pointer()
        self.cloud.update(1.0 / config.FPS)
        self.prune_reaction_waves()
        self.state.frame += 1
        if self.smoke_frames is not None and self.state.frame >= self.smoke_frames:
            pyxel.quit()

    def request_camera_relative(self, direction: int) -> None:
        self.camera_dial_pointer = None
        self.camera.request_relative(direction)
        self.cancel_pointer()

    def update_camera_controls(self) -> bool:
        if self.update_camera_dial():
            return True
        return self.update_camera_buttons()

    def update_camera_dial(self) -> bool:
        pyxel = self.pyxel
        mouse_button_left = getattr(pyxel, "MOUSE_BUTTON_LEFT", 0)
        pressed = pyxel.btn(mouse_button_left)
        just_pressed = pyxel.btnp(mouse_button_left)
        just_released = pyxel.btnr(mouse_button_left)
        x = float(pyxel.mouse_x)
        y = float(pyxel.mouse_y)

        dial_pointer = self.camera_dial_pointer
        if dial_pointer is not None:
            if just_released or not pressed:
                yaw = dial_x_to_yaw(x)
                if dial_pointer.moved:
                    self.camera.update_dial_drag(yaw)
                    self.camera.end_dial_drag()
                else:
                    if camera_dial_center_hit(dial_pointer.start_x, dial_pointer.start_y):
                        yaw = config.CAMERA_FRONT_YAW
                    else:
                        yaw = dial_x_to_yaw(dial_pointer.start_x)
                    self.camera.request_yaw(
                        yaw,
                        transition_seconds=config.CAMERA_DIAL_TRACK_TAP_SECONDS,
                    )
                self.camera_dial_pointer = None
                return True

            if pressed:
                if (
                    math.hypot(x - dial_pointer.start_x, y - dial_pointer.start_y)
                    >= config.PRESS_SLOP_PX
                ):
                    dial_pointer.moved = True
                self.camera.update_dial_drag(dial_x_to_yaw(x))
                return True

        if just_pressed and camera_dial_hit(x, y):
            self.cancel_pointer()
            self.camera_dial_pointer = CameraDialPointer(start_x=x, start_y=y)
            self.camera.begin_dial_drag(dial_x_to_yaw(x))
            return True

        return False

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
            if distance >= config.REACTION_DRAG_START_DISTANCE_PX:
                if not pointer.dragging:
                    self.motion_runtime.trigger_response_wave(
                        self.cloud.state,
                        pointer.selected_node_id,
                        self.state.frame,
                        TouchResponseKind.DRAG_START,
                    )
                    pointer.last_drag_hold_frame = self.state.frame
                elif (
                    pointer.last_drag_hold_frame is None
                    or self.state.frame - pointer.last_drag_hold_frame
                    >= config.CLOUD_DRAG_HOLD_REFRESH_FRAMES
                ):
                    self.motion_runtime.trigger_drag_hold(
                        pointer.selected_node_id,
                        self.state.frame,
                    )
                    pointer.last_drag_hold_frame = self.state.frame
                pointer.dragging = True
                self.cloud.drag_node_to_screen(pointer.selected_node_id, x, y, camera)

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
            result = self.cloud.flick_node(pointer.selected_node_id, velocity)
            self.trigger_operation_response(result, TouchResponseKind.RELEASE)
        elif pointer.selected_node_id is not None and pointer.dragging:
            self.motion_runtime.trigger_response_wave(
                self.cloud.state,
                pointer.selected_node_id,
                self.state.frame,
                TouchResponseKind.RELEASE,
            )
        elif not pointer.dragging and not pointer.long_press_sent:
            response_kind = (
                TouchResponseKind.LONG_PRESS
                if duration >= config.LONG_PRESS_SECONDS
                else TouchResponseKind.TAP
            )
            result = self.cloud.radial_reaction_screen(
                pointer.start_x,
                pointer.start_y,
                duration,
                camera,
                pointer.selected_node_id,
            )
            self.trigger_operation_response(result, response_kind)
            self.add_reaction_wave(result, pointer.start_x, pointer.start_y)

        self.pointer = None

    def trigger_operation_response(
        self,
        result: CloudOperationResult,
        response_kind: TouchResponseKind,
    ) -> None:
        if result.reaction_events:
            for event in result.reaction_events:
                offset = max(0, event.execute_frame - result.created_frame)
                strength = max(
                    1,
                    int(
                        round(
                            config.CLOUD_TAP_RESPONSE_STRENGTH
                            * (0.36 + event.energy * 0.64)
                        )
                    ),
                )
                if event.kind is ReactionEventKind.SECONDARY_SPROUT:
                    strength = max(strength, config.CLOUD_TAP_RESPONSE_STRENGTH - 2)
                elif event.kind in (
                    ReactionEventKind.SEED_IGNITION,
                    ReactionEventKind.IGNITE_DORMANT,
                ):
                    strength = max(strength, config.CLOUD_TAP_RESPONSE_STRENGTH - 4)
                elif event.kind is ReactionEventKind.CREATE_SEED:
                    strength = max(strength, config.CLOUD_TAP_RESPONSE_STRENGTH - 3)
                elif event.kind is ReactionEventKind.VISUAL_WAVE_HIT:
                    strength = min(strength, 5)
                elif event.kind is ReactionEventKind.GRAPH_NODE_PULSE:
                    strength = min(strength, config.CLOUD_TAP_RESPONSE_STRENGTH - 5)
                self.motion_runtime.schedule_response(
                    event.target_node_id,
                    self.state.frame + offset,
                    config.REACTION_PULSE_DURATION_FRAMES,
                    strength,
                    graph_distance=event.generation,
                    response_kind=response_kind,
                )
            return
        self.motion_runtime.trigger_response_wave(
            self.cloud.state,
            result.node_id,
            self.state.frame,
            response_kind,
        )

    def add_reaction_wave(
        self,
        result: CloudOperationResult,
        screen_x: float,
        screen_y: float,
    ) -> None:
        summary = result.reaction_summary
        if summary is None or result.reaction_id is None:
            return
        self.reaction_waves.append(
            CloudReactionWave(
                reaction_id=result.reaction_id,
                origin_screen_x=screen_x,
                origin_screen_y=screen_y,
                release_frame=self.state.frame,
                max_radius_px=summary.max_radius_px,
                charge_level=summary.charge_level,
                local_density=summary.local_density,
            )
        )
        self.reaction_waves = self.reaction_waves[-6:]

    def prune_reaction_waves(self) -> None:
        active: list[CloudReactionWave] = []
        for wave in self.reaction_waves:
            age = self.state.frame - wave.release_frame
            radius = age * config.REACTION_WAVE_SPEED_PX_PER_FRAME
            if radius <= wave.max_radius_px + config.REACTION_WAVE_RING_WIDTH_PX:
                active.append(wave)
        self.reaction_waves = active

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
        self.draw_radial_reaction_feedback()
        self.draw_camera_buttons()
        self.draw_camera_dial()
        pyxel.text(8, 8, "MOKUMOKU Prototype A8", config.COLOR_UI)
        pyxel.text(
            8,
            18,
            "Press/release cloud/sky, drag, flick.",
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
        pyxel.text(8, config.SCREEN_HEIGHT - 14, "dial cam  D debug  F4 age", config.COLOR_UI)
        pyxel.text(
            config.SCREEN_WIDTH - 8 - len(APP_BUILD_LABEL) * 4,
            config.SCREEN_HEIGHT - 14,
            APP_BUILD_LABEL,
            config.COLOR_UI,
        )

    def draw_radial_reaction_feedback(self) -> None:
        for wave in self.reaction_waves:
            age = self.state.frame - wave.release_frame
            radius = age * config.REACTION_WAVE_SPEED_PX_PER_FRAME
            if radius <= 0.0:
                continue
            fade = 1.0 - radius / max(1.0, wave.max_radius_px)
            color = 7 if fade > 0.45 else 6
            self.draw_dotted_circle(
                wave.origin_screen_x,
                wave.origin_screen_y,
                min(radius, wave.max_radius_px),
                color,
                step_degrees=18,
                dot_radius=1 if wave.charge_level < 0.68 else 2,
            )

        pointer = self.pointer
        if pointer is None or pointer.dragging or not self.camera.can_accept_cloud_input():
            return
        hold_seconds = (self.state.frame - pointer.press_frame) / config.FPS
        charge = charge_level(hold_seconds)
        radius = reaction_radius_px(charge)
        color = 6 if charge < 0.55 else 7
        self.draw_dotted_circle(
            pointer.start_x,
            pointer.start_y,
            radius,
            color,
            step_degrees=24,
            dot_radius=1,
        )

    def draw_dotted_circle(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        color: int,
        step_degrees: int,
        dot_radius: int,
    ) -> None:
        pyxel = self.pyxel
        if radius < 2.0:
            return
        for degrees in range(0, 360, step_degrees):
            angle = math.radians(degrees)
            x = int(round(center_x + math.cos(angle) * radius))
            y = int(round(center_y + math.sin(angle) * radius))
            if x < 0 or x >= config.SCREEN_WIDTH or y < 0 or y >= config.SCREEN_HEIGHT:
                continue
            if dot_radius <= 1:
                pyxel.pset(x, y, color)
            else:
                pyxel.circ(x, y, dot_radius, color)

    def draw_cloud(self) -> None:
        camera = self.camera.basis()
        for item in collect_cloud_render_items(
            self.cloud.state,
            camera,
            self.state.frame,
            self.motion_atlas,
            self.motion_runtime,
        ):
            if isinstance(item.payload, EdgePayload):
                if should_draw_edge_payload(self.debug_enabled):
                    self.draw_edge_payload(item.payload)
            elif isinstance(item.payload, BodyPayload):
                self.draw_body_payload(item.payload)
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

    def draw_body_payload(self, payload: BodyPayload) -> None:
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
            radius = max(2, int(payload.visual_radius))
            pyxel.circ(int(projection.screen_x), int(projection.screen_y), radius, 7)

    def draw_node_payload(self, payload: NodePayload) -> None:
        pyxel = self.pyxel
        node = payload.node
        projection = payload.projection
        sprite = payload.sprite
        x = int(projection.screen_x + payload.offset_x - sprite.width / 2)
        y = int(projection.screen_y + payload.offset_y - sprite.height / 2)

        if self.assets_loaded:
            self.draw_cloud_sprite(payload, x, y)
            self.draw_cloud_surface_light(payload, x, y)
            if payload.mesh_intensity > 0.0:
                self.draw_single_cloud_mesh(payload, x, y)
            self.draw_cloud_shape_overlay(payload, x, y)
        else:
            radius = max(2, int(node.radius * projection.scale * max(0.25, node.fade)))
            color = 6 if payload.depth_layer is CloudDepthLayer.BACK else 7
            pyxel.circ(
                int(projection.screen_x + payload.offset_x),
                int(projection.screen_y + payload.offset_y),
                radius,
                color,
            )
            self.draw_cloud_surface_light(payload, x, y)
            self.draw_cloud_shape_overlay(payload, x, y)

    def draw_cloud_sprite(self, payload: NodePayload, x: int, y: int) -> None:
        pyxel = self.pyxel
        sprite = payload.sprite
        if payload.depth_layer is CloudDepthLayer.BACK:
            pyxel.pal(7, 6)
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
        if payload.depth_layer is CloudDepthLayer.BACK:
            pyxel.pal()

    def draw_cloud_surface_light(self, payload: NodePayload, x: int, y: int) -> None:
        if payload.family in (CloudSpriteFamily.INTERNAL, CloudSpriteFamily.FADE):
            return
        if payload.surface_exposure < config.CLOUD_SURFACE_INTERNAL_EXPOSURE:
            return

        pyxel = self.pyxel
        sprite = payload.sprite
        cx = x + sprite.width // 2
        cy = y + sprite.height // 2
        radius = max(4, int(sprite.width * 0.43))
        direction_count = max(4, config.CLOUD_SURFACE_DIRECTION_COUNT)
        bright = 6 if payload.depth_layer is CloudDepthLayer.BACK else 7
        middle = 5 if payload.depth_layer is CloudDepthLayer.BACK else 6
        shadow = 5

        highlight_budget = 4 if payload.depth_layer is CloudDepthLayer.FRONT else 2
        shadow_budget = 3
        for index in range(direction_count):
            if payload.exposure_mask and not (payload.exposure_mask & (1 << index)):
                continue
            angle = math.tau * index / direction_count
            dx = math.cos(angle)
            dy = math.sin(angle)
            px = int(round(cx + dx * radius))
            py = int(round(cy + dy * radius * 0.86))
            if dy < -0.28 and highlight_budget > 0:
                pyxel.pset(px, py, bright)
                if payload.surface_exposure > config.CLOUD_SURFACE_STRONG_EXPOSURE:
                    pyxel.pset(px + 1, py, bright)
                highlight_budget -= 1
            elif dy > 0.38 and shadow_budget > 0:
                color = shadow if payload.depth_layer is not CloudDepthLayer.BACK else middle
                pyxel.pset(px, py, color)
                shadow_budget -= 1

    def draw_cloud_shape_overlay(self, payload: NodePayload, x: int, y: int) -> None:
        if (
            payload.shape_level <= 0
            and payload.growth_level <= 0
            and payload.response_kind is None
        ):
            return

        pyxel = self.pyxel
        sprite = payload.sprite
        cx = x + sprite.width // 2
        cy = y + sprite.height // 2
        scale = sprite.width / 16.0
        bright = 6 if payload.depth_layer is CloudDepthLayer.BACK else 7
        middle = 5 if payload.depth_layer is CloudDepthLayer.BACK else 6

        def unit(value: float) -> int:
            return int(round(value * scale))

        def puff(px: float, py: float, radius: float, color: int) -> None:
            pyxel.circ(cx + unit(px), cy + unit(py), max(1, unit(radius)), color)

        if payload.shape_level >= 1:
            puff(-7.0, -1.5, 0.8, bright)
            puff(6.0, 2.5, 0.7, middle)
        if payload.shape_level >= 2:
            puff(-2.0, -7.0, 0.8, bright)
            puff(2.5, 6.5, 0.7, middle)

        self.draw_cloud_response_morph(
            payload,
            cx,
            cy,
            max(4, int(sprite.width * 0.42)),
            bright,
            middle,
        )

    def draw_cloud_response_morph(
        self,
        payload: NodePayload,
        cx: int,
        cy: int,
        radius: int,
        bright: int,
        middle: int,
    ) -> None:
        growth = payload.growth_level
        if growth <= 0 and payload.response_kind is None:
            return

        pyxel = self.pyxel
        response_kind = payload.response_kind or TouchResponseKind.TAP
        expansion = min(3, max(1, growth // 4))
        response_radius = radius + expansion
        if response_kind is TouchResponseKind.DRAG_START:
            pyxel.line(cx - response_radius, cy, cx + response_radius, cy, bright)
            if growth >= 5:
                pyxel.line(
                    cx - response_radius + 2,
                    cy - 2,
                    cx + response_radius - 2,
                    cy - 2,
                    middle,
                )
        elif response_kind is TouchResponseKind.DRAG_HOLD:
            pyxel.line(
                cx - response_radius + 1,
                cy + 2,
                cx + response_radius - 1,
                cy - 2,
                middle,
            )
        elif response_kind is TouchResponseKind.LONG_PRESS:
            pyxel.circb(cx, cy + 1, response_radius, middle)
            if growth >= 8:
                pyxel.line(
                    cx - response_radius // 2,
                    cy + response_radius,
                    cx + response_radius // 2,
                    cy + response_radius,
                    5,
                )
        elif response_kind is TouchResponseKind.RELEASE:
            pyxel.circb(cx, cy, max(3, response_radius - 1), middle)
        else:
            pyxel.circb(cx, cy, response_radius, middle)
            if growth >= 6:
                pyxel.line(
                    cx - response_radius // 3,
                    cy - response_radius,
                    cx + response_radius // 3,
                    cy - response_radius,
                    bright,
                )

    def draw_single_cloud_mesh(self, payload: NodePayload, x: int, y: int) -> None:
        pyxel = self.pyxel
        sprite = payload.sprite
        phase = payload.mesh_phase
        intensity = payload.mesh_intensity
        cx = x + sprite.width // 2
        cy = y + sprite.height // 2
        radius = max(4, int(sprite.width * 0.42))
        color = 6 if intensity > 0.45 else 5
        drift = cyclic_wave(phase) * radius * 0.16

        for band in (-0.34, 0.12, 0.45):
            band_y = int(cy + band * radius + drift * (0.5 + abs(band)))
            half = int(radius * (1.0 - abs(band) * 0.42))
            middle_y = int(band_y + cyclic_wave(phase + band * 2.7) * 0.9)
            pyxel.line(cx - half, band_y, cx, middle_y, color)
            pyxel.line(cx, middle_y, cx + half, band_y, color)

        diagonal_shift = int(cyclic_wave(phase * 0.7 + math.tau * 0.25) * radius * 0.14)
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

    def draw_camera_dial(self) -> None:
        pyxel = self.pyxel
        left = config.CAMERA_DIAL_LEFT
        right = config.CAMERA_DIAL_RIGHT
        y = config.CAMERA_DIAL_Y
        center = (left + right) // 2
        knob_x = int(round(yaw_to_dial_x(self.camera.current_yaw)))
        track_color = 5
        tick_color = config.COLOR_UI
        knob_color = 7 if self.camera.is_dial_active() else config.COLOR_UI

        pyxel.line(left, y, right, y, track_color)
        pyxel.line(left + 8, y - 3, right - 8, y - 3, track_color)
        for tick_x in (left, center, right):
            pyxel.line(tick_x, y - 4, tick_x, y + 4, tick_color)

        half_w = config.CAMERA_DIAL_KNOB_WIDTH // 2
        half_h = config.CAMERA_DIAL_KNOB_HEIGHT // 2
        pyxel.tri(knob_x, y - half_h, knob_x - half_w, y, knob_x + half_w, y, knob_color)
        pyxel.tri(knob_x, y + half_h, knob_x - half_w, y, knob_x + half_w, y, knob_color)

        pyxel.text(left - 3, y - 13, "L", tick_color)
        pyxel.text(center - 2, y - 13, "0", tick_color)
        pyxel.text(right - 2, y - 13, "R", tick_color)

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
        pyxel.text(
            8,
            90,
            f"quiet {int(config.QUIET_CLOUD_MOTION_ENABLED)} "
            f"local {int(config.AMBIENT_LOCAL_POSITION_ENABLED)} "
            f"size {int(config.AMBIENT_SIZE_PULSE_ENABLED)}",
            config.COLOR_UI,
        )
        pyxel.text(
            8,
            100,
            f"cluster offset {int(config.ENABLE_CLUSTER_AMBIENT_OFFSET)}",
            config.COLOR_UI,
        )
        active_morphs = self.motion_runtime.active_morph_node_ids()
        next_morph = self.motion_runtime.next_morph_event_frame()
        next_text = "-" if next_morph is None else str(next_morph)
        pyxel.text(
            8,
            110,
            f"morph {list(active_morphs[:3])} next {next_text}",
            config.COLOR_UI,
        )
        pyxel.text(
            8,
            120,
            f"responses {self.motion_runtime.growth_pulse_count(self.state.frame)}",
            config.COLOR_UI,
        )
        pyxel.text(
            8,
            130,
            f"camera yaw {self.camera.current_yaw:.2f}",
            config.COLOR_UI,
        )


def run(seed: int = 12345, headless: bool = False, smoke_frames: int | None = None) -> None:
    MokumokuApp(seed=seed, headless=headless, smoke_frames=smoke_frames)


def cyclic_wave(phase: float) -> float:
    unit = (phase / math.tau) % 1.0
    if unit < 0.25:
        return unit * 4.0
    if unit < 0.75:
        return 2.0 - unit * 4.0
    return unit * 4.0 - 4.0
