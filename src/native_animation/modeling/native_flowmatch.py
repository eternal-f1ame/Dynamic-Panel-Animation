"""Project-owned Flow Matching utilities for native-animation I2V.

Defines the custom scheduler and training loss that adapt the Wan Flow Matching
backbone to keyframe-conditioned native animation. The two project-specific
ideas are (1) a gentler scheduler shift that preserves the keyframe and
(2) a composite loss that adds motion-aware frame weighting and a latent
temporal-difference consistency term on top of the standard velocity loss.
"""

from __future__ import annotations

import torch

from diffsynth.diffusion.base_pipeline import BasePipeline
from diffsynth.diffusion.flow_match import FlowMatchScheduler


class NativeAnimationFlowMatchScheduler(FlowMatchScheduler):
    """A slightly gentler Wan schedule for keyframe-preserving animation.

    Wan's default shift biases the schedule toward heavy noise, which tends to
    erase the conditioning keyframe. We default to ``shift=3.0`` so early
    timesteps stay closer to the clean signal and the keyframe survives.
    """

    def __init__(self, shift: float = 3.0):
        super().__init__("Wan")
        self.shift = shift

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        # Forward the project-specific shift unless the caller overrides it.
        kwargs.setdefault("shift", self.shift)
        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            training=training,
            **kwargs,
        )


def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """MSE with an optional per-frame weight tensor.

    ``weights`` is broadcast across the non-frame axes, so its element count is
    smaller than ``loss``. ``normalizer`` rescales the weighted sum so the
    overall loss magnitude matches the unweighted ``loss.mean()`` baseline and
    ``motion_weighting_scale`` only changes *relative* per-frame importance.
    """
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
    """Build per-frame loss weights proportional to latent motion magnitude.

    Frames whose latents change a lot relative to their neighbour carry more
    weight (``1 + scale * normalized_motion``). This nudges the model to
    spend capacity on the visually active parts of a clip rather than long
    static stretches that dominate Sakuga clips.
    """
    if motion_weighting_scale <= 0 or input_latents.shape[2] <= 1:
        return None

    # Mean absolute frame-to-frame latent delta -> per-(batch, frame) score.
    motion = (input_latents[:, :, 1:] - input_latents[:, :, :-1]).abs().float().mean(dim=(1, 3, 4))
    # Normalize per-clip so weights are scale-invariant across batches.
    motion = motion / motion.amax(dim=1, keepdim=True).clamp(min=1e-6)
    weights = 1.0 + motion_weighting_scale * motion

    # When no anchor is consumed, the diff above produced T-1 weights for T
    # frames; pad the leading slot with 1.0 so shapes line up with the latents.
    if anchor_frames <= 0:
        ones = torch.ones(weights.shape[0], 1, dtype=weights.dtype, device=weights.device)
        weights = torch.cat([ones, weights], dim=1)

    # Reshape to (B, 1, T, 1, 1) for broadcasting across channel/spatial axes.
    return weights[:, None, :, None, None]


def NativeAnimationFlowMatchLoss(
    pipe: BasePipeline,
    motion_weighting_scale: float = 1.0,
    delta_loss_weight: float = 0.25,
    **inputs,
) -> torch.Tensor:
    """Native-animation Flow Matching training loss.

    Combines (a) the standard velocity-prediction MSE with motion-aware frame
    weighting and (b) a latent temporal-difference consistency term that asks
    the predicted clean sequence to reproduce the ground-truth frame-to-frame
    deltas. Anchor (keyframe) latents are kept clean during noising and
    excluded from the loss so the keyframe stays pinned.
    """
    # Sample a training timestep inside the configured boundary window.
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device).reshape(1, 1, 1, 1, 1)

    # Forward Flow Matching noising: x_t = (1 - sigma) * x_0 + sigma * noise.
    input_latents = inputs["input_latents"]
    noise = torch.randn_like(input_latents)
    noisy_latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    # Replace the leading frames with the clean keyframe latents so the model
    # always sees a noise-free anchor on input.
    first_frame_latents = inputs.get("first_frame_latents")
    anchor_frames = 0
    if first_frame_latents is not None:
        noisy_latents[:, :, 0:1] = first_frame_latents
        anchor_frames = first_frame_latents.shape[2]

    model_inputs = dict(inputs)
    model_inputs["latents"] = noisy_latents

    # Predict the velocity field with the in-pipeline DiT (and any extras).
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **model_inputs, timestep=timestep)

    # The anchor frames are supervised by being clamped clean, not by the loss
    # itself, so slice them off before computing supervision.
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

    # Standard motion-weighted velocity loss.
    timestep_weight = pipe.scheduler.training_weight(timestep)
    velocity_loss = _weighted_mse(noise_pred_main, training_target_main, motion_weights)
    total_loss = velocity_loss

    # Latent-delta consistency: regress predicted clean x_0 differences onto
    # the ground-truth latent differences. This penalizes flicker / collapse
    # on top of the per-frame velocity error.
    if delta_loss_weight > 0 and input_latents.shape[2] > 1:
        # x_0 estimate from the velocity prediction in Flow Matching form.
        predicted_clean = noisy_latents_main - sigma * noise_pred_main
        if anchor_frames > 0:
            predicted_sequence = torch.cat([first_frame_latents, predicted_clean], dim=2)
        else:
            predicted_sequence = predicted_clean

        predicted_deltas = predicted_sequence[:, :, 1:] - predicted_sequence[:, :, :-1]
        target_deltas = input_latents[:, :, 1:] - input_latents[:, :, :-1]
        delta_loss = _weighted_mse(predicted_deltas, target_deltas, motion_weights)
        total_loss = total_loss + delta_loss_weight * delta_loss

    # ``training_weight`` rescales by the schedule's per-timestep importance.
    return total_loss * timestep_weight
