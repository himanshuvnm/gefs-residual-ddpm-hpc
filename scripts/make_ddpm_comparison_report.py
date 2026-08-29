"""
Template script for a conditional residual DDPM atmospheric-ensemble workflow.

This file is a cleaned portfolio template. Users should adapt paths, metadata,
and data-loading logic for their own HPC system and dataset.
"""

import os
import json
from pathlib import Path

PROJ = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()

v1_path = PROJ / "figures/ddpm_v1_diagnostics/ddpm_diagnostic_summary.json"
v2_path = PROJ / "results/cond_ddpm_v2_gpu/gefs_cond_ddpm_v2_rescued_summary.json"
v3_path = PROJ / "results/swin_cond_ddpm_v3/gefs_swin_cond_ddpm_v3_summary.json"
v3_multi_path = PROJ / "results/swin_cond_ddpm_v3_multicycle/swin_v3_multicycle_summary.json"

missing = [str(p) for p in [v1_path, v2_path, v3_path, v3_multi_path] if not p.exists()]
if missing:
    print("Missing required files:")
    for m in missing:
        print("  ", m)
    raise SystemExit(1)

v1 = json.loads(v1_path.read_text())
v2 = json.loads(v2_path.read_text())
v3 = json.loads(v3_path.read_text())
v3_multi = json.loads(v3_multi_path.read_text())

outdir = PROJ / "results/final_swin_gefs_diffusion_report"
outdir.mkdir(parents=True, exist_ok=True)

v3m = v3_multi["aggregate"]

lines = []
lines.append("GEFS diffusion model comparison")
lines.append("")
lines.append("v1: unconditional CNN DDPM")
lines.append("v2: descriptor-vector conditioned CNN DDPM")
lines.append("v3: Swin-style spatially conditioned DDPM")
lines.append("")
lines.append("Single-cycle comparison")
lines.append(f"{'model':35s} {'spread_RMSE':>14s} {'cov_resp_err':>14s} {'local_err':>14s}")
lines.append("-" * 82)

lines.append(
    f"{'v1 unconditional CNN':35s} "
    f"{v1['spread_rmse']:14.6f} "
    f"{v1['same_variable_covariance_response_relative_error']:14.6f} "
    f"{'NA':>14s}"
)

lines.append(
    f"{'v2 descriptor CNN':35s} "
    f"{v2['spread_rmse']:14.6f} "
    f"{v2['same_variable_covariance_response_relative_error']:14.6f} "
    f"{v2['localized_same_variable_covariance_response_relative_error']:14.6f}"
)

lines.append(
    f"{'v3 Swin spatial cond.':35s} "
    f"{v3['metrics']['spread_rmse']:14.6f} "
    f"{v3['metrics']['same_variable_covariance_response_relative_error']:14.6f} "
    f"{v3['metrics']['localized_same_variable_covariance_response_relative_error']:14.6f}"
)

lines.append("")
lines.append("Multi-cycle Swin-v3 validation")
lines.append(f"cycles evaluated: {v3_multi['num_cycles_evaluated']}")
lines.append("")

for k, stats in v3m.items():
    lines.append(k)
    lines.append(f"  mean   : {stats['mean']:.6f}")
    lines.append(f"  median : {stats['median']:.6f}")
    lines.append(f"  std    : {stats['std']:.6f}")
    lines.append(f"  min    : {stats['min']:.6f}")
    lines.append(f"  max    : {stats['max']:.6f}")
    lines.append("")

# useful derived comparisons
v1_cov = v1["same_variable_covariance_response_relative_error"]
v2_cov = v2["same_variable_covariance_response_relative_error"]
v3_cov = v3["metrics"]["same_variable_covariance_response_relative_error"]
v3_local_mean = v3m["localized_same_variable_covariance_response_relative_error"]["mean"]

lines.append("Derived comparison")
lines.append(f"v1 -> v2 cov-response change: {100*(v2_cov-v1_cov)/abs(v1_cov):+.2f}%")
lines.append(f"v1 -> v3 cov-response change: {100*(v3_cov-v1_cov)/abs(v1_cov):+.2f}%")
lines.append(f"v2 -> v3 cov-response change: {100*(v3_cov-v2_cov)/abs(v2_cov):+.2f}%")
lines.append(f"Swin-v3 multi-cycle mean localized error: {v3_local_mean:.6f}")
lines.append("")

lines.append("Interpretation")
lines.append("- The unconditional DDPM generated perturbation samples but had poor covariance response.")
lines.append("- Descriptor-vector conditioning gave only a small covariance-response improvement and worsened spread.")
lines.append("- Swin-style spatial conditioning improved covariance response and produced consistent localized error below 1.0 across validation cycles.")
lines.append("- This is not yet an operational DA ensemble generator.")
lines.append("- It is the first result supporting the direction: spatially conditioned attention/diffusion is more appropriate than simple vector-conditioned CNN diffusion.")
lines.append("- Next step: add covariance-response / DA-response auxiliary loss to Swin-v3 training.")

txt = "\n".join(lines)
print(txt)

out = outdir / "gefs_ddpm_v1_v2_swin_v3_comparison.txt"
out.write_text(txt)
print("\nSaved:", out)
