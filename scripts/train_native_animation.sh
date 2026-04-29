#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/../Anime/sakugabooru_clips}
METADATA_PATH=${METADATA_PATH:-data/metadata/metadata_train.csv}
OUTPUT_PATH=${OUTPUT_PATH:-artifacts/checkpoints/native_animation_flowmatch_lora}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-49}
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DATASET_REPEAT=${DATASET_REPEAT:-20}
LORA_RANK=${LORA_RANK:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1000}
NATIVE_SCHEDULER_SHIFT=${NATIVE_SCHEDULER_SHIFT:-3.0}
MOTION_WEIGHTING_SCALE=${MOTION_WEIGHTING_SCALE:-1.0}
DELTA_LOSS_WEIGHT=${DELTA_LOSS_WEIGHT:-0.25}
MODEL_ID_WITH_ORIGIN_PATHS=${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth}
ACCELERATE_EXTRA_ARGS=${ACCELERATE_EXTRA_ARGS:-}

accelerate launch ${ACCELERATE_EXTRA_ARGS} src/native_animation/training/train.py \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --data_file_keys "video" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "${LORA_RANK}" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --native_scheduler_shift "${NATIVE_SCHEDULER_SHIFT}" \
  --motion_weighting_scale "${MOTION_WEIGHTING_SCALE}" \
  --delta_loss_weight "${DELTA_LOSS_WEIGHT}"
