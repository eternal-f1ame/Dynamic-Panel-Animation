# Method

This document describes the technical method behind Native Animation Flow Matching (Native FM). For background and motivation, see the project [report](../Final_Project_Report/main.tex).

## Problem statement

Given a single anime keyframe `x_key`, generate a sequence of `T+1` frames `x_{0..T}` that

1. preserves the keyframe's artistic style at `t=0`,
2. produces temporally coherent motion (no flicker, no mid-clip collapse),
3. respects the non-physical motion logic of native animation (smears, impact frames, morphing).

The model is a Wan2.2-TI2V backbone with first-frame conditioning. We do not modify the architecture; the contribution is on the training objective and noise schedule.

## Background: Flow Matching

Flow Matching (Lipman et al., 2023; Liu et al., 2023) replaces a diffusion model's stochastic denoising trajectory with a deterministic ODE. Given clean latents `z_{0..T}` and Gaussian noise `eps`, the noisy interpolant at time `sigma` is

```
z_sigma = (1 - sigma) * z + sigma * eps,    sigma in [0, 1]
```

A velocity network `v_theta(z_sigma, sigma, c)` is trained to regress the straight-line target

```
v* = eps - z
```

with conditioning `c` (here: prompt and keyframe). Sampling integrates the learned velocity backwards from `sigma=1` to `sigma=0`.

## Native FM contributions

Native FM modifies three components of the standard Flow Matching recipe to target the failure modes that off-the-shelf models exhibit on stylized animation. All three live in `src/native_animation/modeling/native_flowmatch.py`.

### 1. Keyframe-preserving scheduler shift

Wan's default schedule places too much density at high `sigma`, which erases the keyframe before the velocity prediction can recover it. We adopt the shift-parameterized schedule of Esser et al. (2024) with a smaller shift (`shift=3.0` vs. Wan's ~5):

```python
NativeAnimationFlowMatchScheduler(shift=3.0)
```

Smaller shift keeps the early timesteps closer to the data, giving the keyframe-anchoring step (below) a usable signal.

### 2. Anchor-frame clamping

At every training step, the noisy slot at the keyframe position is overwritten with its clean latent before the forward pass:

```python
noisy_latents[:, :, 0:1] = first_frame_latents
```

The model therefore always sees a noise-free anchor at the front of the sequence. The anchor's velocity error is **excluded** from the supervised loss because its supervision is provided by the clamp itself, not by gradient descent.

### 3. Motion-aware frame weighting

Sakuga clips are dominated by long, near-static stretches punctuated by short bursts of motion -- exactly the frames whose collapse is most visible. We derive per-frame weights from latent motion magnitude:

```
m_t      = mean(|z_t - z_{t-1}|)            # mean abs latent delta
m_bar_t  = m_t / max_{t'} m_{t'}            # per-clip normalized
w_t      = 1 + alpha * m_bar_t              # alpha = motion_weighting_scale
```

The motion-weighted velocity loss is

```
L_motion = (1/Z) * sum_t  w_t * || v_theta(z_sigma)_t - v*_t ||^2
```

with `Z` chosen so that `alpha=0` recovers the unweighted MSE. `alpha` rebalances frames without changing the gradient norm. We use `alpha=1.0` throughout.

### 4. Latent temporal-difference consistency

A motion-weighted velocity loss still supervises each frame independently, leaving frame-to-frame jitter unpenalized. We recover an `x_0` estimate from the velocity head and regress its temporal differences against the ground truth:

```
z_hat_t = z_sigma_t - sigma * v_theta_t      # x_0 estimate
L_delta = (1/Z) * sum_t  w_t * || (z_hat_t - z_hat_{t-1}) - (z_t - z_{t-1}) ||^2
```

The Delta form leaves slow, smooth motion almost untouched and concentrates penalty on flicker and mid-clip discontinuities -- the exact failure pattern that DFS detects at evaluation time.

### Final objective

The full training loss is

```
L = L_motion + lambda * L_delta
```

with `lambda = 0.25` (`delta_loss_weight`). We deliberately keep `L_delta` smaller than `L_motion` at the per-frame scale: `L_delta` is a regularizer for stability, not a competing reconstruction objective.

## Hyperparameters used in the reported runs

| Hyperparameter            | Value | Notes                                                    |
| ------------------------- | ----- | -------------------------------------------------------- |
| `native_scheduler_shift`  | 3.0   | Lower than Wan default to preserve the keyframe          |
| `motion_weighting_scale`  | 1.0   | `alpha` in the weighting formula                          |
| `delta_loss_weight`       | 0.25  | `lambda` in the final objective                          |
| LoRA rank                 | 32    | Applied to `q,k,v,o,ffn.0,ffn.2` of the DiT              |
| Resolution                | 480x832 | Matches Wan2.2-TI2V-5B's native I2V resolution          |
| Frame count               | 49    | ~1.6s at 30fps; we also run 81 frames (~2.7s)            |
| Dataset                   | ~1.2M | Sliding-window pairs from 11.8k Sakugabooru clips         |
| Optimizer                 | AdamW | DiffSynth default                                        |

## Evaluation metrics

Native FM is evaluated against held-out clips on four complementary axes (`src/native_animation/evaluation/evaluate.py`):

- **CFS (Continuation Fidelity Score)**: per-frame CLIP cosine similarity between generated and ground-truth frames, penalized by Farneback optical-flow disagreement so static generations cannot exploit pure CLIP similarity.
- **TCS (Temporal Consistency Score)**: `mean - std` of the per-frame score curve. Rewards high mean and low jitter together.
- **WorstSegment**: minimum mean score across length-5 rolling windows. Captures the segment a viewer is most likely to notice.
- **DFS (Diffusion Failure Score)**: a two-component detector for the canonical "healthy ends, slumped middle" collapse signature.

The aggregate score is

```
FinalScore = 0.4 * CFS + 0.25 * TCS + 0.2 * WorstSeg - 0.5 * DFS
```

The DFS coefficient is intentionally larger in magnitude than any positive coefficient: a single severe collapse is perceptually catastrophic, and the aggregate must reflect that.

## Repository structure

```
src/native_animation/
  data/        # metadata building, sampling, keyframe extraction
  modeling/    # NativeAnimationFlowMatchScheduler, NativeAnimationFlowMatchLoss
  training/    # accelerate-based fine-tuning entrypoint
  inference/   # baseline and trained-model generation entrypoints
  evaluation/  # CLIP+flow-derived headless evaluator
src/diffsynth/ # vendored runtime subset (not the contribution)
scripts/       # shell + SLURM entrypoints
docs/          # this file
```

## References

The core ideas are documented in the project report at `Final_Project_Report/main.tex` with a complete bibliography. The most directly relevant prior work is:

- Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.
- Liu et al., "Flow Straight and Fast: Rectified Flow," ICLR 2023.
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," ICML 2024.
- Wan Team, "Wan2.x: Open and Advanced Large-Scale Video Generative Models," 2025.
