#!/bin/bash
#SBATCH --job-name=na-base-demo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --constraint=gmem48|gmem80
#SBATCH --export=ALL
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

mkdir -p artifacts/logs artifacts/base_demo
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/../Anime/sakugabooru_clips}

echo "HOSTNAME=$(hostname)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V
nvidia-smi

python -m native_animation.inference.run_baseline \
  --input-csv data/metadata/metadata_test.csv \
  --dataset-base-path "$DATA_ROOT" \
  --output-dir artifacts/base_demo \
  --limit 2 \
  --unique-series \
  --height 480 \
  --width 832 \
  --num-frames 49 \
  --num-inference-steps 30 \
  --cfg-scale 5.0 \
  --seed-base 1234 \
  --fps 15 \
  --quality 5 \
  --tiled
