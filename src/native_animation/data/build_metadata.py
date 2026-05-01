#!/usr/bin/env python3
"""Convert raw Sakugabooru clips into DiffSynth metadata CSV files.

This script targets the project use case:
- keep the format extremely simple
- split by series to reduce leakage
- create prompt text from series + tags
- write DiffSynth-friendly CSVs with at least `video` and `prompt`
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Root of sakugabooru clip directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write train/val/test CSV files")
    parser.add_argument("--seed", type=int, default=42, help="Series split seed")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio by series")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio by series")
    parser.add_argument("--limit", type=int, help="Optional cap on number of metadata entries")
    parser.add_argument("--min-score", type=int, default=0, help="Drop clips below this Sakugabooru score")
    parser.add_argument("--max-tags", type=int, default=20, help="Maximum number of tags to include in prompt text")
    parser.add_argument(
        "--prompt-prefix",
        default="native animation, anime",
        help="Prefix prepended to every generated prompt.",
    )
    return parser.parse_args()


def normalize_tags(tags: str) -> List[str]:
    """Split Sakugabooru's whitespace-separated tag string into a clean list."""
    return [tag.strip() for tag in tags.split() if tag.strip()]


def build_prompt(series: str, tags: List[str], max_tags: int, prompt_prefix: str) -> str:
    """Compose a training prompt: ``"<prefix>, <series>, <tag1>, <tag2>, ..."``."""
    if tags:
        return f"{prompt_prefix}, {series}, " + ", ".join(tags[:max_tags])
    return f"{prompt_prefix}, {series}"


def find_video_for_json(metadata_path: Path) -> Optional[Path]:
    """Locate the .mp4 that corresponds to a Sakugabooru JSON metadata file.

    Tries an exact-stem match first, then ``<stem>_s*.mp4`` (split clips), and
    finally any prefix match. Returns ``None`` if nothing plausible is found.
    """
    stem = metadata_path.stem
    parent = metadata_path.parent
    direct = parent / f"{stem}.mp4"
    if direct.exists():
        return direct
    suffix_matches = sorted(parent.glob(f"{stem}_s*.mp4"))
    if suffix_matches:
        return suffix_matches[0]
    fuzzy_matches = sorted(parent.glob(f"{stem}*.mp4"))
    if fuzzy_matches:
        return fuzzy_matches[0]
    return None


def build_series_split(series_names: Iterable[str], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, str]:
    """Assign each unique series to train/val/test.

    Splitting *by series* (not by clip) prevents leakage where the train and
    eval sets share frames from the same show. Small bumps guarantee at least
    one series in val/test when the requested ratio would otherwise round to 0.
    """
    names = sorted(set(series_names))
    rng = random.Random(seed)
    rng.shuffle(names)

    total = len(names)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)

    # Guarantee at least one series in val/test when feasible.
    if total >= 3 and test_ratio > 0 and test_count == 0:
        test_count = 1
    if total - test_count >= 2 and val_ratio > 0 and val_count == 0:
        val_count = 1
    if test_count + val_count > total:
        val_count = max(0, total - test_count)

    split_map: Dict[str, str] = {}
    for idx, name in enumerate(names):
        if idx < test_count:
            split_map[name] = "test"
        elif idx < test_count + val_count:
            split_map[name] = "val"
        else:
            split_map[name] = "train"
    return split_map


def load_json(path: Path) -> Dict:
    """Read a single Sakugabooru JSON sidecar file."""
    with path.open() as handle:
        return json.load(handle)


def gather_json_paths(input_root: Path, limit: Optional[int]) -> List[Path]:
    """Collect every Sakugabooru JSON under ``input_root`` (optionally capped)."""
    paths = sorted(input_root.rglob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def write_csv(path: Path, rows: List[Dict]) -> None:
    """Write ``rows`` to ``path`` with the canonical DiffSynth-friendly columns."""
    fieldnames = [
        "video",
        "prompt",
        "series",
        "tags",
        "score",
        "clip_id",
        "width",
        "height",
        "source",
        "split",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = gather_json_paths(input_root, args.limit)
    if not json_paths:
        raise SystemExit(f"No JSON metadata found under {input_root}")

    # Series name is taken from the parent directory; the split is computed
    # once up front so every clip from a given series lands in the same split.
    split_map = build_series_split((path.parent.name for path in json_paths), args.val_ratio, args.test_ratio, args.seed)

    rows: List[Dict] = []
    skipped_missing_video = 0
    skipped_low_score = 0

    for json_path in json_paths:
        info = load_json(json_path)
        score = int(info.get("score", 0) or 0)
        if score < args.min_score:
            skipped_low_score += 1
            continue

        video_path = find_video_for_json(json_path)
        if video_path is None:
            skipped_missing_video += 1
            continue

        series = json_path.parent.name
        tags = info.get("tags", "")
        tag_list = normalize_tags(tags)
        rows.append(
            {
                # Stored relative to dataset root so the CSV stays portable.
                "video": str(video_path.relative_to(input_root)),
                "prompt": build_prompt(series, tag_list, args.max_tags, args.prompt_prefix),
                "series": series,
                "tags": tags,
                "score": score,
                "clip_id": info.get("id", json_path.stem),
                "width": info.get("width"),
                "height": info.get("height"),
                "source": info.get("source"),
                "split": split_map[series],
            }
        )

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]

    # Always write all four CSVs so downstream jobs can pick whichever split
    # they need without rerunning the build.
    write_csv(output_dir / "metadata_all.csv", rows)
    write_csv(output_dir / "metadata_train.csv", train_rows)
    write_csv(output_dir / "metadata_val.csv", val_rows)
    write_csv(output_dir / "metadata_test.csv", test_rows)

    summary = {
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "num_series": len(set(row["series"] for row in rows)),
        "top_series": Counter(row["series"] for row in rows).most_common(20),
        "skipped_missing_video": skipped_missing_video,
        "skipped_low_score": skipped_low_score,
        "input_root": str(input_root),
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
