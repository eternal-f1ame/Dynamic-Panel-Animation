#!/usr/bin/env python3
"""Minimal native-animation Flow Matching training entrypoint.

Wraps DiffSynth's training infrastructure: builds a Wan video pipeline,
swaps in the project scheduler, and routes the SFT path through
``NativeAnimationFlowMatchLoss`` so the motion-aware + delta-consistency
loss replaces the stock velocity loss.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import accelerate

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize, LoadAudio, LoadVideo, ToAbsolutePath
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    DirectDistillLoss,
    ModelLogger,
    add_general_config,
    add_video_size_config,
    launch_data_process_task,
    launch_training_task,
)
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

from native_animation.modeling.native_flowmatch import (
    NativeAnimationFlowMatchLoss,
    NativeAnimationFlowMatchScheduler,
)

# Quiet HuggingFace's tokenizer-parallelism warning under multi-process accelerate.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NativeAnimationWanTrainingModule(DiffusionTrainingModule):
    """DiffSynth training module wired to the native-animation FM loss."""

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        native_scheduler_shift=3.0,
        motion_weighting_scale=1.0,
        delta_loss_weight=0.25,
    ):
        super().__init__()
        # Force gradient checkpointing on: video Flow Matching activations are
        # large enough that disabling it almost always OOMs on the cluster.
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is disabled. Enabling it to reduce the chance of running out of memory."
            )
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.default_dtype(),
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        # Replace Wan's stock scheduler with the project's keyframe-preserving variant.
        self.pipe.scheduler = NativeAnimationFlowMatchScheduler(shift=native_scheduler_shift)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Toggle the requested submodules between train/eval and attach LoRAs.
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.task = task
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.motion_weighting_scale = motion_weighting_scale
        self.delta_loss_weight = delta_loss_weight
        # Maps DiffSynth task names to the appropriate loss callable. The
        # ``data_process`` variants short-circuit (return inputs unchanged) so
        # the launcher can preprocess data without computing a loss.
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": self.compute_sft_loss,
            "sft:train": self.compute_sft_loss,
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe,
                **inputs_shared,
                **inputs_posi,
            ),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe,
                **inputs_shared,
                **inputs_posi,
            ),
        }

    @staticmethod
    def default_dtype():
        import torch

        return torch.bfloat16

    def compute_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        """SFT loss: invoke the project's native-animation Flow Matching loss."""
        return NativeAnimationFlowMatchLoss(
            pipe,
            motion_weighting_scale=self.motion_weighting_scale,
            delta_loss_weight=self.delta_loss_weight,
            **inputs_shared,
            **inputs_posi,
        )

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        """Materialize optional conditioning fields (keyframe, end image, etc).

        The keyframe-conditioned setup populates ``input_image`` from the
        first decoded video frame; other branches are kept for compatibility
        with adjacent DiffSynth tasks (audio, VACE references, ...).
        """
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input in ("reference_image", "vace_reference_image"):
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        # WanToDance variant: it decodes one latent slice per chunk of 4 frames,
        # so reconstruct the implied frame count for the pipeline.
        if inputs_shared.get("framewise_decoding", False):
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared

    def get_pipeline_inputs(self, data):
        """Pack a dataset sample into the (shared, positive, negative) input triple."""
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # CFG is disabled during training; only the conditional path is run.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        """Standard DiffSynth forward: run pipeline units, then dispatch the loss."""
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        # Each unit (text encode, VAE encode, ...) transforms the input tuple
        # in place before the loss head consumes the final latents.
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)


def native_animation_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.set_defaults(data_file_keys="video")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument(
        "--audio_processor_path",
        type=str,
        default=None,
        help="Path to the audio processor. Only needed for speech-to-video variants.",
    )
    parser.add_argument(
        "--max_timestep_boundary",
        type=float,
        default=1.0,
        help="Upper training-timestep boundary as a fraction of the scheduler range.",
    )
    parser.add_argument(
        "--min_timestep_boundary",
        type=float,
        default=0.0,
        help="Lower training-timestep boundary as a fraction of the scheduler range.",
    )
    parser.add_argument(
        "--initialize_model_on_cpu",
        default=False,
        action="store_true",
        help="Initialize models on CPU before accelerate moves them.",
    )
    parser.add_argument(
        "--framewise_decoding",
        default=False,
        action="store_true",
        help="Enable this only for WanToDance global models.",
    )
    parser.add_argument(
        "--native_scheduler_shift",
        type=float,
        default=3.0,
        help="Shift used by the native-animation scheduler. Lower than Wan's default to preserve keyframe structure.",
    )
    parser.add_argument(
        "--motion_weighting_scale",
        type=float,
        default=1.0,
        help="Additional weight placed on frames with larger latent motion.",
    )
    parser.add_argument(
        "--delta_loss_weight",
        type=float,
        default=0.25,
        help="Weight for the latent frame-difference consistency regularizer.",
    )
    return parser


def main():
    parser = native_animation_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    # Build the video dataset. Resolutions are forced to multiples of 16 to
    # match the VAE's spatial stride, and the temporal stride (4 vs 1) tracks
    # whether the model uses framewise decoding.
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        # Per-key operator overrides for non-default fields (face video crop,
        # 16 kHz audio loading, raw music paths). Unused outside their tasks.
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )

    model = NativeAnimationWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        native_scheduler_shift=args.native_scheduler_shift,
        motion_weighting_scale=args.motion_weighting_scale,
        delta_loss_weight=args.delta_loss_weight,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    # ``data_process`` tasks pre-cache encoded latents/embeddings; ``train``
    # tasks run the actual gradient loop. Same module, different launcher.
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
