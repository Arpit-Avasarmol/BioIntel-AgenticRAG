#!/usr/bin/env python
"""Package the repository into a clean, distributable ``.zip`` archive.

Excludes virtual-envs, caches, VCS metadata, build artifacts and local data so
the archive contains only source, configs, tests, and docs. The archive is
written next to the repo root by default (and the path is printed on success).

Usage
-----
    python scripts/package_zip.py
    python scripts/package_zip.py --output /some/where/biointel-agent.zip
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIX = "biointel-agent"

# Directories excluded anywhere in the tree.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".venv-verify",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "node_modules",
    ".data",
    "data",
    "volumes",
    ".idea",
    ".vscode",
    "htmlcov",
    "dist",
    "build",
    ".eggs",
}

# File glob patterns excluded anywhere in the tree.
EXCLUDE_FILE_GLOBS = [
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    ".env",
    ".coverage",
    "*.egg-info",
    ".DS_Store",
    f"{ARCHIVE_PREFIX}*.zip",
]


def _excluded(path: Path) -> bool:
    parts = set(path.relative_to(REPO_ROOT).parts)
    if parts & EXCLUDE_DIRS:
        return True
    name = path.name
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_FILE_GLOBS)


def collect_files() -> list[Path]:
    files: list[Path] = []
    for p in REPO_ROOT.rglob("*"):
        if p.is_dir():
            continue
        if _excluded(p):
            continue
        files.append(p)
    return sorted(files)


def build_zip(output: Path) -> tuple[int, int]:
    files = collect_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            # Store paths under a top-level <ARCHIVE_PREFIX>/ folder.
            arcname = Path(ARCHIVE_PREFIX) / f.relative_to(REPO_ROOT)
            zf.write(f, arcname)
    size = output.stat().st_size
    return len(files), size


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the repo into a .zip.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / f"{ARCHIVE_PREFIX}.zip",
        help="Output .zip path (default: <repo>/biointel-agent.zip).",
    )
    args = parser.parse_args()

    n_files, size = build_zip(args.output)
    mb = size / (1024 * 1024)
    print(f"Packaged {n_files} files -> {args.output}  ({mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
