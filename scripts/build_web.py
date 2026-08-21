from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
APP_NAME = "mokumoku"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MOKUMOKU for Pyxel Web.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate build inputs without invoking the Pyxel CLI.",
    )
    return parser.parse_args()


def ensure_inputs() -> None:
    required = [
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "src" / "app.py",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "assets" / "mokumoku.pyxres",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(f"missing build input(s): {joined}")


def run_command(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def disable_virtual_gamepad(html_path: Path) -> None:
    text = html_path.read_text(encoding="utf-8")
    text = text.replace(', gamepad: "enabled"', "")
    text = text.replace('gamepad: "enabled",', "")
    text = text.replace("kitao/pyxel@2.7.0", "kitao/pyxel@2.9.9")
    mobile_head = (
        '<head><meta name="viewport" content="width=device-width, initial-scale=1.0, '
        'viewport-fit=cover, user-scalable=no">'
        "<style>html,body,canvas{touch-action:none;overscroll-behavior:none;}"
        "body{margin:0;background:#111;}</style></head>"
    )
    if "touch-action:none" not in text:
        if "</head>" in text:
            text = text.replace("</head>", mobile_head.removeprefix("<head>"))
        elif "<head>" in text:
            text = text.replace("<head>", mobile_head)
        else:
            text = text.replace("<!doctype html>\n", f"<!doctype html>\n{mobile_head}\n", 1)
    html_path.write_text(text, encoding="utf-8")


def build_web() -> Path:
    ensure_inputs()
    if shutil.which("pyxel") is None:
        raise RuntimeError("pyxel CLI was not found. Install dependencies with `pip install -e .`.")

    DOCS_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mokumoku_web_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        package_dir = temp_dir / APP_NAME
        shutil.copytree(
            PROJECT_ROOT,
            package_dir,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "*.egg-info",
                "docs",
                "scripts",
                "tests",
            ),
        )

        run_command(["pyxel", "package", ".", "main.py"], cwd=package_dir)
        app_file = package_dir / f"{APP_NAME}.pyxapp"
        run_command(["pyxel", "app2html", str(app_file.name)], cwd=package_dir)

        html_path = package_dir / f"{APP_NAME}.html"
        disable_virtual_gamepad(html_path)

        output_path = DOCS_DIR / "index.html"
        shutil.copy2(html_path, output_path)
        (DOCS_DIR / ".nojekyll").touch()
        return output_path


def main() -> int:
    args = parse_args()
    ensure_inputs()
    if args.dry_run:
        print("dry-run ok: build inputs are present")
        return 0

    try:
        output_path = build_web()
    except Exception as exc:
        print(f"web build failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {output_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
