#!/bin/bash
#SBATCH --job-name=na-build-meta
#SBATCH --partition=short
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

mkdir -p artifacts/logs data/metadata
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/../Anime/sakugabooru_clips}

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V

python -m native_animation.data.build_metadata \
  --input-root "$DATA_ROOT" \
  --output-dir data/metadata \
  --seed 42 \
  --val-ratio 0.1 \
  --test-ratio 0.1
