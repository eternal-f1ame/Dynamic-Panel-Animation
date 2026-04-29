#!/bin/bash
#SBATCH --job-name=na-train-fm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --constraint=gmem48|gmem80
#SBATCH --export=ALL
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

mkdir -p artifacts/logs artifacts/checkpoints data/metadata
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V

export DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/../Anime/sakugabooru_clips}
export METADATA_PATH=${METADATA_PATH:-data/metadata/metadata_train.csv}
export OUTPUT_PATH=${OUTPUT_PATH:-artifacts/checkpoints/native_animation_flowmatch_lora}
export HEIGHT=${HEIGHT:-480}
export WIDTH=${WIDTH:-832}
export NUM_FRAMES=${NUM_FRAMES:-49}
export NUM_EPOCHS=${NUM_EPOCHS:-2}
export LEARNING_RATE=${LEARNING_RATE:-1e-4}
export DATASET_REPEAT=${DATASET_REPEAT:-20}
export LORA_RANK=${LORA_RANK:-32}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export NATIVE_SCHEDULER_SHIFT=${NATIVE_SCHEDULER_SHIFT:-3.0}
export MOTION_WEIGHTING_SCALE=${MOTION_WEIGHTING_SCALE:-1.0}
export DELTA_LOSS_WEIGHT=${DELTA_LOSS_WEIGHT:-0.25}
export MODEL_ID_WITH_ORIGIN_PATHS=${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth}

bash scripts/train_native_animation.sh
