from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from src import config
from src.rng import RandomSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class A0ProjectTests(unittest.TestCase):
    def test_resolution_matches_spec_initial_values(self) -> None:
        self.assertEqual(config.SCREEN_WIDTH, 320)
        self.assertEqual(config.SCREEN_HEIGHT, 568)
        self.assertEqual(config.SKY_BOTTOM_Y, 378)
        self.assertEqual(config.GROUND_TOP_Y, 378)
        self.assertEqual(config.FPS, 60)

    def test_rng_is_deterministic_for_seed(self) -> None:
        first = RandomSource(12345)
        second = RandomSource(12345)
        self.assertEqual(
            [first.uniform(-1.0, 1.0) for _ in range(5)],
            [second.uniform(-1.0, 1.0) for _ in range(5)],
        )

    def test_web_build_dry_run(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/build_web.py", "--dry-run"],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dry-run ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
