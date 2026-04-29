# Native Animation Flow Matching

Standalone repository for keyframe-conditioned native-animation video generation.

This repository is the submission-facing version of the project. It keeps the code that the team actually built for the native-animation task, plus the minimum vendored DiffSynth runtime needed to run it.

## Project Goal

The goal is to generate short native-animation continuations from a single keyframe.

Concretely, the project focuses on:

- using the first frame of a clip as the conditioning keyframe
- fine-tuning a Wan-based Flow Matching video model on Sakugabooru clips
- comparing baseline generations against a project-owned native-animation Flow Matching variant
- evaluating results with a repeatable headless scoring workflow

## Team Contributions

The main team-owned work in this repository lives under `src/native_animation/`.

The contribution is not just one loss file. It includes:

- dataset collection and curation
  - sourcing native-animation clips from Sakugabooru for the project dataset
  - filtering, organizing, and maintaining the raw clip collection outside version control
- `src/native_animation/data/`
  - dataset metadata construction from the collected Sakugabooru clip set
  - sample selection and keyframe extraction utilities
- `src/native_animation/modeling/native_flowmatch.py`
  - a custom native-animation Flow Matching scheduler and loss
  - motion-aware frame weighting
  - latent temporal-difference consistency regularization
- `src/native_animation/training/train.py`
  - a focused fine-tuning entrypoint that plugs the custom method into DiffSynth
- `src/native_animation/inference/run_baseline.py`
  - baseline Wan inference on held-out clips for comparison
- `src/native_animation/inference/generate.py`
  - inference for the trained native-animation variant
- `src/native_animation/evaluation/evaluate.py`
  - a headless evaluator adapted from the shared prototype
  - CLIP-based fidelity scoring plus temporal and collapse-sensitive metrics
- `scripts/slurm/`
  - cluster-ready job templates for metadata build, smoke test, baseline inference, and training

## Repository Layout

- `src/native_animation/`
  - project-owned code
  - `data/` metadata building, sampling, and keyframe extraction
  - `modeling/` custom Flow Matching scheduler and loss
  - `training/` fine-tuning entrypoint
  - `inference/` baseline and trained-model generation entrypoints
  - `evaluation/` headless video evaluation utilities
- `src/diffsynth/`
  - vendored DiffSynth runtime subset required by the project code
  - included for execution, but not the main project contribution
- `scripts/`
  - reusable shell and SLURM entrypoints
- `docs/`
  - method notes and provenance

## Typical Workflow

1. Build metadata CSVs from the raw clip collection.
2. Run a small baseline inference pass on held-out clips.
3. Fine-tune the native-animation Flow Matching variant.
4. Generate videos from selected keyframes.
5. Evaluate baseline and trained outputs with the headless scorer.

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Available console entrypoints:

- `native-animation-build-metadata`
- `native-animation-extract-keyframes`
- `native-animation-run-baseline`
- `native-animation-generate`
- `native-animation-train`
- `native-animation-evaluate`

SLURM templates live in `scripts/slurm/`.

For the method details, see `docs/method.md`.

## What Is Not Bundled

This repository does not ship:

- model weights or caches
- raw datasets
- generated outputs
- checkpoints

Those artifacts are environment-specific and should stay outside version control.

## Third-Party Boundary

The `src/diffsynth/` directory is a vendored runtime subset carved out of the larger DiffSynth tree. It is included so the standalone repository remains runnable, but it should not be confused with the team's project-owned method layer in `src/native_animation/`.

See `THIRD_PARTY.md` for provenance notes.
