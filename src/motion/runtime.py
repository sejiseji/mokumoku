from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WeatherMotionRuntime:
    growth_started_at: dict[int, int] = field(default_factory=dict)

    def trigger_growth(self, node_id: int | None, frame: int) -> None:
        if node_id is None:
            return
        self.growth_started_at[node_id] = frame

    def growth_level(self, node_id: int, frame: int, growth_ease: bytes) -> int:
        started_at = self.growth_started_at.get(node_id)
        if started_at is None:
            return 0
        age = frame - started_at
        if age < 0 or age >= len(growth_ease):
            return 0
        return growth_ease[age]


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
