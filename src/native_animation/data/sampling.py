"""Helpers for selecting Sakuga samples and extracting keyframes.

Shared utilities used by the keyframe extraction script, the baseline
inference runner, and any other tool that needs to pick rows from the
DiffSynth-style metadata CSV and grab their first frame.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg
from PIL import Image


def read_metadata_rows(csv_path: Path) -> list[dict]:
    """Read the metadata CSV into a list of dict rows."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_row_indices(raw: str | None) -> list[int] | None:
    """Parse a comma-separated row-index string from the CLI into ints."""
    if raw is None or raw.strip() == "":
        return None
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def select_rows(
    rows: list[dict],
    limit: int = 2,
    row_indices: Iterable[int] | None = None,
    sort_by_score_desc: bool = True,
    unique_series: bool = False,
) -> list[tuple[int, dict]]:
    """Pick which CSV rows to process and return ``(original_index, row)`` pairs.

    Selection precedence: explicit ``row_indices`` overrides everything;
    otherwise rows are optionally sorted by Sakugabooru score (highest first)
    and ``unique_series`` enforces at most one clip per series so demo runs
    stay visually diverse.
    """
    indexed_rows = list(enumerate(rows))

    if row_indices is not None:
        return [(idx, rows[idx]) for idx in row_indices]

    if sort_by_score_desc:
        indexed_rows.sort(key=lambda item: int(item[1].get("score") or 0), reverse=True)

    if not unique_series:
        return indexed_rows[:limit]

    chosen: list[tuple[int, dict]] = []
    seen_series: set[str] = set()
    for idx, row in indexed_rows:
        series = row.get("series", "")
        if series in seen_series:
            continue
        chosen.append((idx, row))
        seen_series.add(series)
        if len(chosen) >= limit:
            break
    return chosen


def sanitize_name(name: str) -> str:
    """Make ``name`` safe to use as a filename component (alnum / -/_ only)."""
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)
    return safe[:120] if safe else "sample"


def resolve_video_path(dataset_base_path: Path, row: dict) -> Path:
    """Resolve a row's relative ``video`` path against the dataset root."""
    return (dataset_base_path / row["video"]).resolve()


def extract_first_frame(video_path: Path, output_path: Path, resize: tuple[int, int] | None = None) -> Path:
    """Save the first frame of ``video_path`` as an image, optionally resized.

    Uses the imageio-ffmpeg binary so this works without a system ffmpeg, and
    pins ``-threads 1`` to keep behavior deterministic on the cluster.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-y",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )

    # Re-open through PIL so the saved file is a clean RGB PNG at the target size.
    image = Image.open(output_path).convert("RGB")
    if resize is not None:
        image = image.resize(resize)
    image.save(output_path)
    return output_path
