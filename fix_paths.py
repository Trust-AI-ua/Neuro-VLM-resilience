#!/usr/bin/env python3
"""Rewrite relative image-path prefixes in split files to absolute paths."""
import argparse
from pathlib import Path

PREFIXES = [
    ("data/coco/train2017/", "coco_dir"),
    ("data/nocaps/images/",  "nocaps_dir"),
    # CC3M paths are self-contained; no substitution needed.
]


def fix_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for rel, absolute in replacements:
        text = text.replace(rel, absolute)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite relative image-path prefixes in split files to absolute paths."
    )
    parser.add_argument(
        "--coco_dir",
        default=None,
        metavar="PATH",
        help="Absolute path to COCO train2017/ directory",
    )
    parser.add_argument(
        "--nocaps_dir",
        default=None,
        metavar="PATH",
        help="Absolute path to NoCaps images/ directory",
    )
    args = parser.parse_args()

    dir_map = {"coco_dir": args.coco_dir, "nocaps_dir": args.nocaps_dir}

    replacements: list[tuple[str, str]] = []
    for rel_prefix, key in PREFIXES:
        abs_dir = dir_map.get(key)
        if abs_dir:
            replacements.append((rel_prefix, abs_dir.rstrip("/") + "/"))

    if not replacements:
        print("No --*_dir flags provided. Nothing to do.")
        return

    runs_root = Path("experiments/runs")
    if not runs_root.exists():
        print(f"[error] {runs_root} not found. Run this script from the repo root.")
        return

    changed, total = 0, 0
    for pattern in ("*.tsv", "*.txt"):
        for fpath in sorted(runs_root.rglob(pattern)):
            total += 1
            if fix_file(fpath, replacements):
                changed += 1
                print(f"  updated: {fpath}")

    print(f"Done. {changed}/{total} file(s) updated.")


if __name__ == "__main__":
    main()
