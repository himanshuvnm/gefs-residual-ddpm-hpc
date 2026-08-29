#!/bin/bash
#SBATCH --job-name=ddpm50m_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=<gpu_partition>
#SBATCH --output=logs/ddpm50m_smoke_%j.out
#SBATCH --error=logs/ddpm50m_smoke_%j.err

set -euo pipefail

module purge
module load python/3.12
module load cuda

source /path/to/venv/bin/activate

export PROJECT_ROOT=/path/to/gefs-residual-ddpm-hpc
export DATA_ROOT=/path/to/atmospheric_ensemble_data
export OUTDIR=$PROJECT_ROOT/outputs/ddpm50m_smoke

mkdir -p logs "$OUTDIR"

python "$PROJECT_ROOT/scripts/train_residual_ddpm_50m.py" \
  --config "$PROJECT_ROOT/configs/ddpm_hgt500_example.yaml" \
  --data-root "$DATA_ROOT" \
  --outdir "$OUTDIR" \
  --epochs 2 \
  --batch-size 4
