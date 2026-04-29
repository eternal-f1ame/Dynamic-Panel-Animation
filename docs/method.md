# Method Overview

The project trains a keyframe-conditioned video model for native-animation continuation.

## Contribution Boundary

The project-owned contribution lives in `src/native_animation/`.

In particular, the main method contribution is:

- `src/native_animation/modeling/native_flowmatch.py`

The vendored `src/diffsynth/` tree is supporting runtime code, not the primary contribution.

## Core Idea

The baseline DiffSynth Wan training path already uses the first frame of each training clip as the conditioning image. The project contribution stays intentionally small and focused:

- keep the standard Flow Matching target
- add motion-aware frame weighting
- add a latent temporal-difference consistency term

## Loss Sketch

Let `x_0` be the clean latent video, `eps ~ N(0, I)`, and:

```text
x_t = (1 - sigma_t) x_0 + sigma_t eps
```

The baseline target is:

```text
v*(x_t, t) = eps - x_0
```

The project variant optimizes:

```text
L = L_fm^motion + lambda_delta L_delta
```

where:

- `L_fm^motion` increases the contribution of frames with larger latent motion
- `L_delta` matches frame-to-frame latent differences between the prediction and the target sequence

## Why This Structure Exists

The exported repository separates:

- project-owned code in `src/native_animation/`
- vendored runtime code in `src/diffsynth/`
- operational cluster scripts in `scripts/`

That keeps the submission readable and makes it obvious which parts are the team contribution.
