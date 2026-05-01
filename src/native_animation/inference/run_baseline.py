#!/usr/bin/env python3
"""Run a small base Wan Flow Matching demo on selected Sakuga clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

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
    parser.add_argument("--input-csv", type=Path, required=True, help="Metadata CSV")
    parser.add_argument("--dataset-base-path", type=Path, required=True, help="Root directory for video files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for keyframes and generated videos")
    parser.add_argument("--limit", type=int, default=2, help="Number of samples to run")
    parser.add_argument("--row-indices", type=str, default=None, help="Comma-separated explicit CSV row indices")
    parser.add_argument("--unique-series", action="store_true", help="Choose at most one sample per series")
    parser.add_argument("--height", type=int, default=480, help="Generation height")
    parser.add_argument("--width", type=int, default=832, help="Generation width")
    parser.add_argument("--num-frames", type=int, default=49, help="Generated video length")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Number of Flow Matching steps")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed-base", type=int, default=1234, help="Base random seed")
    parser.add_argument("--fps", type=int, default=15, help="Saved video fps")
    parser.add_argument("--quality", type=int, default=5, help="Saved video quality")
    parser.add_argument("--negative-prompt", default="low quality, blurry, static, distorted, deformed, ugly", help="Negative prompt")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled inference")
    parser.add_argument("--low-vram", action="store_true", help="Use DiffSynth's low-VRAM offload config")
    return parser.parse_args()


def build_pipe(low_vram: bool) -> WanVideoPipeline:
    """Construct a vanilla Wan2.2 TI2V-5B pipeline for the baseline comparison.

    When ``low_vram`` is set we use DiffSynth's disk-offload config so the
    pipeline fits on a single small GPU at the cost of throughput.
    """
    model_configs = []
    if low_vram:
        # Stream weights from disk -> CPU -> GPU on demand.
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    else:
        vram_config = {}

    # Wan ships the text encoder, DiT weights, and VAE under separate file
    # patterns; load all three from the same model id.
    for pattern in (
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model*.safetensors",
        "Wan2.2_VAE.pth",
    ):
        model_configs.append(
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern=pattern,
                **vram_config,
            )
        )

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device": "cuda",
        "model_configs": model_configs,
        # The 5B checkpoint reuses the umt5 tokenizer published with the 1.3B model.
        "tokenizer_config": ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
    }
    if low_vram:
        # Reserve ~2 GB of headroom for activations.
        kwargs["vram_limit"] = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2

    return WanVideoPipeline.from_pretrained(**kwargs)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo inference script.")

    rows = read_metadata_rows(args.input_csv)
    selected = select_rows(
        rows=rows,
        limit=args.limit,
        row_indices=parse_row_indices(args.row_indices),
        sort_by_score_desc=True,
        unique_series=args.unique_series,
    )

    pipe = build_pipe(low_vram=args.low_vram)
    output_dir = args.output_dir.resolve()
    keyframe_dir = output_dir / "keyframes"
    generated_dir = output_dir / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for order, (row_index, row) in enumerate(selected):
        video_path = resolve_video_path(args.dataset_base_path, row)
        file_stem = sanitize_name(Path(row["video"]).stem)
        keyframe_path = keyframe_dir / f"{row_index:05d}_{file_stem}.png"
        output_video_path = generated_dir / f"{row_index:05d}_{file_stem}.mp4"
        prompt = row["prompt"]
        # Vary the seed across rows so different clips don't share noise.
        seed = args.seed_base + order

        # Pull the first frame from the source clip and feed it as the I2V keyframe.
        extract_first_frame(video_path, keyframe_path, resize=(args.width, args.height))
        input_image = Image.open(keyframe_path).convert("RGB")

        print(f"[RUN] row={row_index} series={row.get('series')} score={row.get('score')} video={row.get('video')}")
        print(f"[RUN] prompt={prompt}")

        video = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            input_image=input_image,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            seed=seed,
            tiled=args.tiled,
        )
        save_video(video, str(output_video_path), fps=args.fps, quality=args.quality)
        print(f"[OK] wrote {output_video_path}")

        summary.append(
            {
                "row_index": row_index,
                "series": row.get("series"),
                "score": row.get("score"),
                "video": row.get("video"),
                "prompt": prompt,
                "keyframe": str(keyframe_path),
                "generated_video": str(output_video_path),
                "seed": seed,
            }
        )
        # Free fragmented allocations between clips so long demo runs don't OOM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ``summary.json`` is what the evaluator consumes to pair generated videos
    # against their source clips.
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
