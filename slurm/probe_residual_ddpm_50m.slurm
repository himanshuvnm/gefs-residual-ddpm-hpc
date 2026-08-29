#!/bin/bash
#SBATCH --job-name=ddpm50m_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=<gpu_partition>
#SBATCH --output=logs/ddpm50m_probe_%j.out
#SBATCH --error=logs/ddpm50m_probe_%j.err

set -euo pipefail

module purge
module load python/3.12
module load cuda

source /path/to/venv/bin/activate

export PROJECT_ROOT=/path/to/gefs-residual-ddpm-hpc
export CHECKPOINT=/path/to/checkpoint.pt
export DATA_ROOT=/path/to/atmospheric_ensemble_data
export OUTDIR=$PROJECT_ROOT/outputs/ddpm50m_probe

mkdir -p logs "$OUTDIR"

python "$PROJECT_ROOT/scripts/probe_residual_ddpm_50m.py" \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --outdir "$OUTDIR"
