#!/usr/bin/env python3
"""Extract first-frame keyframes from selected Sakugabooru clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from native_animation.data.sampling import (
    extract_first_frame,
    parse_row_indices,
    read_metadata_rows,
    resolve_video_path,
    sanitize_name,
    select_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Metadata CSV with at least video/prompt columns")
    parser.add_argument("--dataset-base-path", type=Path, required=True, help="Root directory that contains the video files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for extracted keyframes")
    parser.add_argument("--limit", type=int, default=2, help="Number of samples to extract")
    parser.add_argument("--row-indices", type=str, default=None, help="Comma-separated explicit CSV row indices")
    parser.add_argument("--unique-series", action="store_true", help="Keep at most one sample per series")
    parser.add_argument("--height", type=int, default=480, help="Saved keyframe height")
    parser.add_argument("--width", type=int, default=832, help="Saved keyframe width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_metadata_rows(args.input_csv)
    # Default ordering picks the highest-scored Sakuga clips first so the
    # extracted keyframes are decent demo material.
    selected = select_rows(
        rows=rows,
        limit=args.limit,
        row_indices=parse_row_indices(args.row_indices),
        sort_by_score_desc=True,
        unique_series=args.unique_series,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for row_index, row in selected:
        video_path = resolve_video_path(args.dataset_base_path, row)
        file_stem = sanitize_name(Path(row["video"]).stem)
        # Prefix with the original CSV row index so outputs stay traceable.
        keyframe_path = args.output_dir / f"{row_index:05d}_{file_stem}.png"
        extract_first_frame(video_path, keyframe_path, resize=(args.width, args.height))
        summary.append(
            {
                "row_index": row_index,
                "series": row.get("series"),
                "score": row.get("score"),
                "video": row.get("video"),
                "prompt": row.get("prompt"),
                "keyframe": str(keyframe_path),
            }
        )
        print(f"[OK] {video_path} -> {keyframe_path}")

    # ``summary.json`` is consumed downstream by the inference / eval scripts
    # to recover which clip every keyframe came from.
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
