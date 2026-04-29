#!/usr/bin/env python3
"""Minimal keyframe-to-video inference for the native-animation FM project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

from native_animation.modeling.native_flowmatch import NativeAnimationFlowMatchScheduler


MODEL_PRESETS = {
    "wan21_i2v_14b_480p": [
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    "wan22_ti2v_5b": [
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", required=True, help="Path to the keyframe image")
    parser.add_argument("--prompt", required=True, help="Positive prompt")
    parser.add_argument("--negative-prompt", default="low quality, blurry, static, distorted", help="Negative prompt")
    parser.add_argument("--lora-path", default=None, help="Optional trained LoRA checkpoint")
    parser.add_argument("--output", default="native_animation_flowmatch.mp4", help="Output video path")
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="wan22_ti2v_5b", help="Backbone preset")
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--width", type=int, default=832, help="Output width")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of video frames to generate")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--native-scheduler-shift", type=float, default=3.0, help="Scheduler shift used at inference")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="Saved video fps")
    parser.add_argument("--quality", type=int, default=5, help="Saved video quality")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=MODEL_PRESETS[args.model_preset],
    )
    pipe.scheduler = NativeAnimationFlowMatchScheduler(shift=args.native_scheduler_shift)

    if args.lora_path is not None:
        pipe.load_lora(pipe.dit, args.lora_path, alpha=1.0)

    image = Image.open(args.input_image).convert("RGB").resize((args.width, args.height))
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=args.tiled,
    )
    save_video(video, args.output, fps=args.fps, quality=args.quality)


if __name__ == "__main__":
    main()
