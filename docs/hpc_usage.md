# HPC Usage

The `slurm/` directory contains portable templates for running the workflow on
an HPC cluster.

Typical usage:

```bash
sbatch slurm/train_residual_ddpm_50m_smoke.slurm
sbatch slurm/train_residual_ddpm_50m_full.slurm
sbatch slurm/probe_residual_ddpm_50m.slurm
```

Before running, edit:

- `<your_account>`
- `<gpu_partition>`
- `/path/to/venv`
- `/path/to/gefs-residual-ddpm-hpc`
- `/path/to/atmospheric_ensemble_data`
- `/path/to/checkpoint.pt`
