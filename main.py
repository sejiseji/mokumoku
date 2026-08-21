from __future__ import annotations

import argparse

from src.app import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOKUMOKU Prototype")
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic simulation seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(seed=args.seed)


if __name__ == "__main__":
    main()
