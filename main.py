from __future__ import annotations

import argparse

from src.app import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOKUMOKU Prototype")
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic simulation seed.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Pyxel without opening a window.",
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=None,
        help="Quit automatically after this many frames. Intended for headless smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(seed=args.seed, headless=args.headless, smoke_frames=args.smoke_frames)


if __name__ == "__main__":
    main()
