from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prototype_acceptance import run_prototype_a_acceptance  # noqa: E402


def main() -> int:
    report = run_prototype_a_acceptance()
    print(f"nodes {report.node_count}")
    print(f"edges {report.edge_count}")
    print(f"bridges {report.bridge_count}")
    print(f"max camera tap error {report.max_camera_tap_error:.3f}px")
    print(f"max ambient offset {report.max_ambient_offset:.3f}px")
    print(f"normal edges hidden {int(report.normal_edges_hidden)}")
    print(f"passed {int(report.passed)}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
