from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
BUILDS_DIR = DOCS_DIR / "builds"
APP_NAME = "mokumoku"
BUILD_ID_LENGTH = 12
VERSIONED_BUILD_RETENTION = 3


@dataclass(frozen=True)
class WebBuildResult:
    root_path: Path
    versioned_path: Path
    build_id: str
    pruned_paths: tuple[Path, ...] = ()


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


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disable_virtual_gamepad(html_path: Path, build_id: str | None = None) -> None:
    text = html_path.read_text(encoding="utf-8")
    text = text.replace(', gamepad: "enabled"', "")
    text = text.replace('gamepad: "enabled",', "")
    text = text.replace("kitao/pyxel@2.7.0", "kitao/pyxel@2.9.9")
    if build_id is not None:
        versioned_name = f'{APP_NAME}-{build_id}.pyxapp'
        text = text.replace(f'name: "{APP_NAME}.pyxapp"', f'name: "{versioned_name}"')
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
    if build_id is not None and 'name="mokumoku-build"' not in text:
        build_meta = f'<meta name="mokumoku-build" content="{build_id}">'
        if "</head>" in text:
            text = text.replace("</head>", f"{build_meta}</head>", 1)
        else:
            text = f"{build_meta}\n{text}"
    html_path.write_text(text, encoding="utf-8")


def prune_versioned_builds(
    builds_dir: Path,
    retain: int = VERSIONED_BUILD_RETENTION,
) -> tuple[Path, ...]:
    if retain < 1:
        raise ValueError("retain must be at least 1")
    if not builds_dir.exists():
        return ()

    versioned_dirs = sorted(
        (path for path in builds_dir.iterdir() if path.is_dir()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    stale_dirs = tuple(versioned_dirs[retain:])
    for path in stale_dirs:
        shutil.rmtree(path)
    return stale_dirs


def build_web() -> WebBuildResult:
    ensure_inputs()
    if shutil.which("pyxel") is None:
        raise RuntimeError("pyxel CLI was not found. Install dependencies with `pip install -e .`.")

    DOCS_DIR.mkdir(exist_ok=True)
    BUILDS_DIR.mkdir(exist_ok=True)
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
        build_id = file_digest(app_file)[:BUILD_ID_LENGTH]
        run_command(["pyxel", "app2html", str(app_file.name)], cwd=package_dir)

        html_path = package_dir / f"{APP_NAME}.html"
        disable_virtual_gamepad(html_path, build_id)

        root_path = DOCS_DIR / "index.html"
        versioned_dir = BUILDS_DIR / build_id
        versioned_dir.mkdir(parents=True, exist_ok=True)
        versioned_path = versioned_dir / "index.html"
        shutil.copy2(html_path, root_path)
        shutil.copy2(html_path, versioned_path)
        os.utime(versioned_dir, None)
        pruned_paths = prune_versioned_builds(BUILDS_DIR)
        (DOCS_DIR / ".nojekyll").touch()
        return WebBuildResult(
            root_path=root_path,
            versioned_path=versioned_path,
            build_id=build_id,
            pruned_paths=pruned_paths,
        )


def main() -> int:
    args = parse_args()
    ensure_inputs()
    if args.dry_run:
        print("dry-run ok: build inputs are present")
        return 0

    try:
        result = build_web()
    except Exception as exc:
        print(f"web build failed: {exc}", file=sys.stderr)
        return 1

    print(f"build id {result.build_id}")
    print(f"wrote {result.root_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote {result.versioned_path.relative_to(PROJECT_ROOT)}")
    for path in result.pruned_paths:
        print(f"removed {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
