from __future__ import annotations

from dataclasses import dataclass, field

from src import config


@dataclass
class WeatherMotionRuntime:
    cluster_x: dict[int, int] = field(default_factory=dict)
    cluster_y: dict[int, int] = field(default_factory=dict)
    local_x: dict[int, int] = field(default_factory=dict)
    local_y: dict[int, int] = field(default_factory=dict)

    def cluster_offset(self, cluster_key: int, raw_x: int, raw_y: int) -> tuple[int, int]:
        dx = hysteresis_step(
            self.cluster_x.get(cluster_key, 0),
            raw_x,
            config.CLOUD_CLUSTER_ENTER_THRESHOLD,
            config.CLOUD_CLUSTER_EXIT_THRESHOLD,
        )
        dy = hysteresis_step(
            self.cluster_y.get(cluster_key, 0),
            raw_y,
            config.CLOUD_CLUSTER_ENTER_THRESHOLD,
            config.CLOUD_CLUSTER_EXIT_THRESHOLD,
        )
        self.cluster_x[cluster_key] = dx
        self.cluster_y[cluster_key] = dy
        return dx, dy

    def local_offset(self, node_id: int, raw_x: int, raw_y: int) -> tuple[int, int]:
        dx = hysteresis_step(
            self.local_x.get(node_id, 0),
            raw_x,
            config.CLOUD_LOCAL_ENTER_THRESHOLD,
            config.CLOUD_LOCAL_EXIT_THRESHOLD,
        )
        dy = hysteresis_step(
            self.local_y.get(node_id, 0),
            raw_y,
            config.CLOUD_LOCAL_ENTER_THRESHOLD,
            config.CLOUD_LOCAL_EXIT_THRESHOLD,
        )
        self.local_x[node_id] = dx
        self.local_y[node_id] = dy
        return dx, dy


def hysteresis_step(current: int, raw: int, enter: int, exit: int) -> int:
    if current == 0:
        if raw > enter:
            return 1
        if raw < -enter:
            return -1
        return 0
    if current == 1 and raw < exit:
        return 0
    if current == -1 and raw > -exit:
        return 0
    return current
