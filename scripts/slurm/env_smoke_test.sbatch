#!/bin/bash
#SBATCH --job-name=na-env-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --export=ALL
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

mkdir -p artifacts/logs
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "HOSTNAME=$(hostname)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V
nvidia-smi

timeout 120s python -u -c "import sys; print('exe=', sys.executable, flush=True); import torch; print('torch=', torch.__version__, flush=True); print('cuda=', torch.cuda.is_available(), flush=True); import accelerate; print('accelerate=', accelerate.__version__, flush=True); import native_animation; print('native_animation=', native_animation.__version__, flush=True); import diffsynth; print('diffsynth=ok', flush=True)"
