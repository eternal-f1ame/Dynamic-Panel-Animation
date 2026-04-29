"""Project-owned Flow Matching utilities for native-animation I2V."""

from __future__ import annotations

import torch

from diffsynth.diffusion.base_pipeline import BasePipeline
from diffsynth.diffusion.flow_match import FlowMatchScheduler


class NativeAnimationFlowMatchScheduler(FlowMatchScheduler):
    """A slightly gentler Wan schedule for keyframe-preserving animation."""

    def __init__(self, shift: float = 3.0):
        super().__init__("Wan")
        self.shift = shift

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        kwargs.setdefault("shift", self.shift)
        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            training=training,
            **kwargs,
        )


def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = (prediction.float() - target.float()).pow(2)
    if weights is None:
        return loss.mean()
    weights = weights.to(device=loss.device, dtype=loss.dtype)
    normalizer = max(loss.numel() / weights.numel(), 1.0)
    return (loss * weights).sum() / (weights.sum().clamp(min=1e-6) * normalizer)


def _motion_frame_weights(
    input_latents: torch.Tensor,
    anchor_frames: int,
    motion_weighting_scale: float,
) -> torch.Tensor | None:
    if motion_weighting_scale <= 0 or input_latents.shape[2] <= 1:
        return None

    motion = (input_latents[:, :, 1:] - input_latents[:, :, :-1]).abs().float().mean(dim=(1, 3, 4))
    motion = motion / motion.amax(dim=1, keepdim=True).clamp(min=1e-6)
    weights = 1.0 + motion_weighting_scale * motion

    if anchor_frames <= 0:
        ones = torch.ones(weights.shape[0], 1, dtype=weights.dtype, device=weights.device)
        weights = torch.cat([ones, weights], dim=1)

    return weights[:, None, :, None, None]


def NativeAnimationFlowMatchLoss(
    pipe: BasePipeline,
    motion_weighting_scale: float = 1.0,
    delta_loss_weight: float = 0.25,
    **inputs,
) -> torch.Tensor:
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device).reshape(1, 1, 1, 1, 1)

    input_latents = inputs["input_latents"]
    noise = torch.randn_like(input_latents)
    noisy_latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    first_frame_latents = inputs.get("first_frame_latents")
    anchor_frames = 0
    if first_frame_latents is not None:
        noisy_latents[:, :, 0:1] = first_frame_latents
        anchor_frames = first_frame_latents.shape[2]

    model_inputs = dict(inputs)
    model_inputs["latents"] = noisy_latents

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **model_inputs, timestep=timestep)

    if anchor_frames > 0:
        noise_pred_main = noise_pred[:, :, anchor_frames:]
        training_target_main = training_target[:, :, anchor_frames:]
        noisy_latents_main = noisy_latents[:, :, anchor_frames:]
    else:
        noise_pred_main = noise_pred
        training_target_main = training_target
        noisy_latents_main = noisy_latents

    motion_weights = _motion_frame_weights(
        input_latents=input_latents,
        anchor_frames=anchor_frames,
        motion_weighting_scale=motion_weighting_scale,
    )

    timestep_weight = pipe.scheduler.training_weight(timestep)
    velocity_loss = _weighted_mse(noise_pred_main, training_target_main, motion_weights)
    total_loss = velocity_loss

    if delta_loss_weight > 0 and input_latents.shape[2] > 1:
        predicted_clean = noisy_latents_main - sigma * noise_pred_main
        if anchor_frames > 0:
            predicted_sequence = torch.cat([first_frame_latents, predicted_clean], dim=2)
        else:
            predicted_sequence = predicted_clean

        predicted_deltas = predicted_sequence[:, :, 1:] - predicted_sequence[:, :, :-1]
        target_deltas = input_latents[:, :, 1:] - input_latents[:, :, :-1]
        delta_loss = _weighted_mse(predicted_deltas, target_deltas, motion_weights)
        total_loss = total_loss + delta_loss_weight * delta_loss

    return total_loss * timestep_weight
