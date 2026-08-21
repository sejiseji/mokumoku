from __future__ import annotations

from math import cos, exp, hypot, log, sin, tau


def sample_weather_motion(
    sample_index: int,
    group_index: int,
    phase01: float,
) -> tuple[float, float, float]:
    """Sample the expensive weather motion field during atlas construction only."""

    u = tau * ((sample_index * 0.6180339887498948) % 1.0)
    t = tau * phase01
    phase = float(group_index % 16) * 13.0

    k = cos(5.0 * u) * sin(u)
    e = cos(3.0 * u) * cos(2.0 * u)
    radial = hypot(k, e)

    d = 1.20 + 0.14 * radial**3 - 0.17 * sin(0.55 * t + phase) ** 3
    d = max(0.70, min(2.20, d))

    pulse = exp(log(d) * sin(d * d - 1.15 * t + phase))
    pulse = max(0.68, min(1.50, pulse))

    c = 0.42 * d - 0.20 * t + phase
    x = sin(c) + 0.34 * k * pulse
    y = 0.42 * sin(4.0 * c) + 0.26 * e * pulse

    return x, y, pulse

