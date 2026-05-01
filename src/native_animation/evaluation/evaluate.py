#!/usr/bin/env python3
"""Headless evaluation for native-animation video generation outputs.

For each (ground-truth, generated) pair we sample frames, score per-frame
fidelity with CLIP cosine similarity (penalized by optical-flow disagreement),
and aggregate into four headline metrics:

- ``CFS``: continuation fidelity = mean per-frame score
- ``TCS``: temporal consistency = mean - std (high mean, low jitter)
- ``WorstSegment``: worst-case windowed score (catches localized failures)
- ``DFS``: diffusion-failure score (mid-clip dip + high frame-to-frame jitter)

The final score combines these with hand-tuned weights. The script can be
fed by a generation summary JSON, a directory of paired mp4 files, or an
explicit single pair.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# CLIP backends are loaded lazily; either OpenAI's reference repo or the
# open_clip drop-in is acceptable, picked by whichever import succeeds.
try:
    import clip as openai_clip
except ImportError:
    openai_clip = None

try:
    import open_clip
except ImportError:
    open_clip = None


# Matches our Sakuga clip naming convention (e.g. ``clip_42_3``) to pair
# ground-truth clips with their generated counterpart in directory mode.
PAIR_KEY_RE = re.compile(r"(clip_\d+_\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--summary-json", type=Path, help="Summary JSON produced by a generation script.")
    source_group.add_argument("--pairs-dir", type=Path, help="Directory that contains ground-truth/generated mp4 files.")
    source_group.add_argument("--ground-truth-video", type=Path, help="Single ground-truth video.")
    parser.add_argument("--generated-video", type=Path, help="Single generated video for explicit-pair mode.")
    parser.add_argument("--dataset-base-path", type=Path, help="Dataset root used to resolve relative source videos from summary JSON.")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames sampled per video.")
    parser.add_argument("--window", type=int, default=4, help="Local temporal matching window.")
    parser.add_argument("--normalized-points", type=int, default=50, help="Length of the normalized score curve.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of pairs to evaluate.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON file for full results.")
    parser.add_argument("--plots-dir", type=Path, default=None, help="Optional directory to save score-curve plots.")
    return parser.parse_args()


def sample_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    """Sample ``num_frames`` evenly spaced BGR frames from ``video_path``."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return []

    target_count = min(num_frames, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, target_count, dtype=int)

    frames = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    return frames


def resize_pair(gt_frames: list[np.ndarray], gen_frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bring both clips to the smaller of their resolutions for fair comparison."""
    target_width = min(gt_frames[0].shape[1], gen_frames[0].shape[1])
    target_height = min(gt_frames[0].shape[0], gen_frames[0].shape[0])
    target_size = (target_width, target_height)
    return (
        [cv2.resize(frame, target_size) for frame in gt_frames],
        [cv2.resize(frame, target_size) for frame in gen_frames],
    )


@lru_cache(maxsize=1)
def load_clip_backbone() -> tuple[str, object, object, str]:
    """Load CLIP once and cache it for the life of the process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if openai_clip is not None:
        model, preprocess = openai_clip.load("ViT-B/32", device=device)
        return "openai", model, preprocess, device
    if open_clip is not None:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
        return "open_clip", model, preprocess, device
    raise ImportError(
        "No CLIP backend is available. Install `clip` or `open_clip_torch` to use the evaluator."
    )


def clip_feature(frame: np.ndarray) -> np.ndarray:
    """Encode a single BGR frame to an L2-normalized CLIP image embedding."""
    backend_name, model, preprocess, device = load_clip_backbone()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(input_tensor)
    # L2 normalize so dot product == cosine similarity.
    feature = feature / feature.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return feature.detach().cpu().numpy()[0].astype(np.float32)


def extract_features(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Encode every frame in a list to a CLIP feature vector."""
    return [clip_feature(frame) for frame in frames]


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Cosine similarity for already-L2-normalized vectors."""
    return float(np.dot(lhs, rhs))


def compute_frame_scores(
    gt_frames: list[np.ndarray],
    gen_frames: list[np.ndarray],
    window: int,
) -> np.ndarray:
    """Per-frame fidelity score combining CLIP similarity and flow agreement.

    For each ground-truth frame we search a +/- ``window`` neighborhood of
    generated frames for the best CLIP cosine match (this absorbs small
    timing offsets between the two clips). We then penalize that score by
    the optical-flow disagreement between the two frame pairs so a static
    generation can't game CLIP similarity.
    """
    gt_features = extract_features(gt_frames)
    gen_features = extract_features(gen_frames)
    scores = []

    for frame_idx in range(len(gt_frames) - 1):
        # Local search range around the GT frame's index in the generated clip.
        gen_start = max(0, frame_idx - window)
        gen_end = min(len(gen_features), frame_idx + window + 1)

        # Pick the most similar generated frame within the window.
        best_similarity = -1.0
        best_gen_idx = gen_start
        for gen_idx in range(gen_start, gen_end):
            similarity = cosine_similarity(gt_features[frame_idx], gen_features[gen_idx])
            if similarity > best_similarity:
                best_similarity = similarity
                best_gen_idx = gen_idx

        # Optical flow on grayscale is enough; we only care about motion
        # magnitude, not color-space details.
        gt_prev = cv2.cvtColor(gt_frames[frame_idx], cv2.COLOR_BGR2GRAY)
        gt_next = cv2.cvtColor(gt_frames[frame_idx + 1], cv2.COLOR_BGR2GRAY)
        gen_prev = cv2.cvtColor(gen_frames[best_gen_idx], cv2.COLOR_BGR2GRAY)
        gen_next = cv2.cvtColor(
            gen_frames[min(best_gen_idx + 1, len(gen_frames) - 1)],
            cv2.COLOR_BGR2GRAY,
        )

        flow_gt = cv2.calcOpticalFlowFarneback(gt_prev, gt_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_gen = cv2.calcOpticalFlowFarneback(gen_prev, gen_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_diff = np.mean(np.abs(flow_gt - flow_gen))
        # Squash to [0, 1) so the penalty stays bounded regardless of magnitude.
        flow_norm = np.tanh(flow_diff / 10.0)

        scores.append(best_similarity - 0.5 * float(flow_norm))

    return np.array(scores, dtype=np.float32)


def normalize_scores(scores: np.ndarray, num_points: int) -> np.ndarray:
    """Resample the score curve to a fixed length so plots/aggregates align."""
    if len(scores) == 0:
        return scores
    if len(scores) == 1:
        return np.repeat(scores, num_points).astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(scores))
    x_new = np.linspace(0.0, 1.0, num_points)
    return np.interp(x_new, x_old, scores).astype(np.float32)


def diffusion_failure_score(scores: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Detect the classic mid-clip diffusion-collapse signature.

    Splits the curve into thirds: failures usually show a healthy start/end
    but a slumped middle, plus high frame-to-frame jitter. DFS combines a
    mid-drop term (70%) with a smoothness term (30%), each clipped against
    an empirically tuned 0.08 cutoff so DFS lives in [0, 1].
    """
    if len(scores) < 3:
        average = float(np.mean(scores)) if len(scores) else 0.0
        return 0.0, 0.0, 0.0, average, average, average

    third = max(len(scores) // 3, 1)
    start = float(np.mean(scores[:third]))
    middle = float(np.mean(scores[third : 2 * third]))
    end = float(np.mean(scores[2 * third :]))
    mid_drop = ((start + end) / 2.0) - middle
    smoothness = float(np.mean(np.abs(np.diff(scores)))) if len(scores) > 1 else 0.0
    mid_norm = float(np.clip(mid_drop / 0.08, 0.0, 1.0))
    smooth_norm = float(np.clip(smoothness / 0.08, 0.0, 1.0))
    dfs = 0.7 * mid_norm + 0.3 * smooth_norm
    return float(dfs), float(mid_drop), float(smoothness), start, middle, end


def temporal_consistency(scores: np.ndarray) -> float:
    """Reward high mean and low jitter together (mean - std)."""
    return float(np.mean(scores) - np.std(scores))


def worst_segment(scores: np.ndarray, window: int = 5) -> float:
    """Min over rolling windows so a localized failure isn't averaged away."""
    if len(scores) == 0:
        return 0.0
    window = min(window, len(scores))
    return float(min(np.mean(scores[idx : idx + window]) for idx in range(len(scores) - window + 1)))


def continuation_fidelity(scores: np.ndarray) -> float:
    """Plain mean per-frame score across the clip."""
    return float(np.mean(scores)) if len(scores) else 0.0


def final_score(cfs: float, tcs: float, worst: float, dfs: float) -> float:
    """Headline score: weighted positives minus a strong DFS penalty."""
    return float(0.4 * cfs + 0.25 * tcs + 0.2 * worst - 0.5 * dfs)


def classify_result(result: dict) -> str:
    """Translate the four metrics into a human-readable verdict bucket."""
    cfs = result["CFS"]
    tcs = result["TCS"]
    worst = result["WorstSegment"]
    dfs = result["DFS"]

    # Thresholds were calibrated by eye against labeled good / bad clips.
    if cfs > 0.93 and tcs > 0.90 and worst > 0.90 and dfs < 0.10:
        return "[OK] Good continuation"
    if dfs > 0.7:
        return "[FAIL] Diffusion failure"
    if cfs < 0.7 or worst < 0.6:
        return "[FAIL] Severe breakdown"
    if tcs < 0.7:
        return "[FAIL] Temporal instability"
    if worst < 0.85:
        return "[WARN] Weak segments"
    if tcs < 0.85:
        return "[WARN] Slight instability"
    return "[OK] Acceptable continuation"


def slugify(name: str) -> str:
    """Make ``name`` filesystem-safe for plot output paths."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe[:120] if safe else "result"


def save_plot(scores: np.ndarray, output_path: Path) -> None:
    """Save the per-frame score curve as a PNG (Agg backend, no display needed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(0.0, 1.0, len(scores))
    plt.figure(figsize=(6, 3.5))
    plt.plot(xs, scores, marker="o")
    plt.title("Quality Curve")
    plt.xlabel("Normalized time")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_pair(
    ground_truth_path: Path,
    generated_path: Path,
    num_frames: int,
    window: int,
    normalized_points: int,
) -> dict:
    """Score a single (ground-truth, generated) clip pair end to end."""
    gt_frames = sample_frames(ground_truth_path, num_frames=num_frames)
    gen_frames = sample_frames(generated_path, num_frames=num_frames)
    if len(gt_frames) < 2 or len(gen_frames) < 2:
        return {"error": "Frame extraction failed or returned too few frames."}

    # Truncate to a common length and resolution before scoring.
    usable_frames = min(len(gt_frames), len(gen_frames))
    gt_frames, gen_frames = gt_frames[:usable_frames], gen_frames[:usable_frames]
    gt_frames, gen_frames = resize_pair(gt_frames, gen_frames)

    raw_scores = compute_frame_scores(gt_frames, gen_frames, window=window)
    normalized_scores = normalize_scores(raw_scores, num_points=normalized_points)

    cfs = continuation_fidelity(normalized_scores)
    tcs = temporal_consistency(normalized_scores)
    worst = worst_segment(normalized_scores)
    dfs, mid_drop, smoothness, start, middle, end = diffusion_failure_score(normalized_scores)
    result = {
        "ground_truth": str(ground_truth_path),
        "generated": str(generated_path),
        "CFS": cfs,
        "TCS": tcs,
        "WorstSegment": worst,
        "MidDrop": mid_drop,
        "Smoothness": smoothness,
        "DFS": dfs,
        "Start": start,
        "Middle": middle,
        "End": end,
        "FinalScore": final_score(cfs, tcs, worst, dfs),
        "scores": normalized_scores.tolist(),
    }
    result["label"] = classify_result(result)
    return result


def load_pairs_from_summary(summary_path: Path, dataset_base_path: Path | None) -> list[tuple[str, Path, Path]]:
    """Load pairs from a generation script's ``summary.json`` output."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pairs = []
    for item in summary:
        source_video = Path(item["video"])
        # Summaries usually carry relative source paths -- resolve against dataset root.
        if not source_video.is_absolute():
            if dataset_base_path is None:
                raise SystemExit(
                    "Summary mode requires --dataset-base-path when `video` paths are relative."
                )
            source_video = (dataset_base_path / source_video).resolve()
        generated_video = Path(item["generated_video"]).resolve()
        pair_name = f"row_{item.get('row_index', len(pairs))}_{generated_video.stem}"
        pairs.append((pair_name, source_video, generated_video))
    return pairs


def extract_pair_key(filename: str) -> str | None:
    """Pull the ``clip_<series>_<index>`` token out of a filename, if present."""
    match = PAIR_KEY_RE.search(filename)
    return match.group(1) if match else None


def load_pairs_from_directory(folder: Path) -> list[tuple[str, Path, Path]]:
    """Pair ``clip_*.mp4`` ground-truth files with their generated counterparts.

    Filenames that start with ``clip_`` are treated as ground truth; everything
    else sharing the same ``clip_<a>_<b>`` key counts as a generated variant
    (so multiple generations against one source are all evaluated).
    """
    ground_truth = {}
    generated = {}
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() != ".mp4":
            continue
        pair_key = extract_pair_key(file_path.name)
        if pair_key is None:
            continue
        if file_path.name.startswith("clip_"):
            ground_truth[pair_key] = file_path
        else:
            generated.setdefault(pair_key, []).append(file_path)

    pairs = []
    for pair_key, gt_path in ground_truth.items():
        for gen_path in generated.get(pair_key, []):
            pairs.append((pair_key, gt_path, gen_path))
    return pairs


def summarize_results(results: list[dict]) -> dict:
    """Aggregate per-pair metrics into a single run-level summary."""
    if not results:
        return {"num_results": 0}
    return {
        "num_results": len(results),
        "average_final_score": float(np.mean([result["FinalScore"] for result in results])),
        "average_dfs": float(np.mean([result["DFS"] for result in results])),
        "average_cfs": float(np.mean([result["CFS"] for result in results])),
        "average_tcs": float(np.mean([result["TCS"] for result in results])),
    }


def resolve_pairs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    """Dispatch to the right pair loader based on which CLI source was given."""
    if args.summary_json is not None:
        return load_pairs_from_summary(args.summary_json.resolve(), args.dataset_base_path.resolve() if args.dataset_base_path else None)
    if args.pairs_dir is not None:
        return load_pairs_from_directory(args.pairs_dir.resolve())
    if args.ground_truth_video is not None:
        if args.generated_video is None:
            raise SystemExit("Explicit-pair mode requires --generated-video.")
        return [("single_pair", args.ground_truth_video.resolve(), args.generated_video.resolve())]
    raise SystemExit("No evaluation source was provided.")


def main() -> None:
    args = parse_args()
    pairs = resolve_pairs(args)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No evaluable video pairs were found.")

    if args.plots_dir is not None:
        args.plots_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for pair_name, ground_truth_path, generated_path in pairs:
        print(f"[EVAL] {pair_name}: {generated_path.name} vs {ground_truth_path.name}")
        result = evaluate_pair(
            ground_truth_path=ground_truth_path,
            generated_path=generated_path,
            num_frames=args.num_frames,
            window=args.window,
            normalized_points=args.normalized_points,
        )
        if "error" in result:
            print(f"[SKIP] {pair_name}: {result['error']}")
            continue
        result["pair_name"] = pair_name
        results.append(result)
        print(
            f"{result['label']} | Final={result['FinalScore']:.3f} "
            f"CFS={result['CFS']:.3f} TCS={result['TCS']:.3f} "
            f"Worst={result['WorstSegment']:.3f} DFS={result['DFS']:.3f}"
        )

        if args.plots_dir is not None:
            plot_path = args.plots_dir / f"{slugify(pair_name)}.png"
            save_plot(np.array(result["scores"], dtype=np.float32), plot_path)

    summary = summarize_results(results)
    output = {"summary": summary, "results": results}

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        print(f"[OK] wrote {args.output_json}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
