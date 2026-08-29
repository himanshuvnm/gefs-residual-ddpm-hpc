"""
Template script for a conditional residual DDPM atmospheric-ensemble workflow.

This file is a cleaned portfolio template. Users should adapt paths, metadata,
and data-loading logic for their own HPC system and dataset.
"""

import os
import re
import json
import glob
import math
import time
import random
import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VAR_NAMES = ["HGT500", "TMP850", "UGRD850", "VGRD850"]


# -----------------------------
# Utilities
# -----------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cycle_name_from_path(p):
    b = os.path.basename(p)
    b = b.replace("X_", "").replace(".npy", "")
    return b


def load_cycle_var(path, var_idx=0):
    """
    Robust loader for GEFS X-factor files.

    Expected useful shapes:
      (31, 4, 361, 720)
      (4, 31, 361, 720)
      (31, 4*361*720)
      (4*361*720, 31)
      (31, 361, 720)
    Returns:
      array float32, shape (members, 360, 720)
    """
    arr = np.load(path, mmap_mode="r")

    if arr.ndim == 4:
        if arr.shape[0] == 31 and arr.shape[1] >= 4:
            out = arr[:, var_idx, :360, :]
        elif arr.shape[0] >= 4 and arr.shape[1] == 31:
            out = arr[var_idx, :, :360, :]
        else:
            raise ValueError(f"Unsupported 4D shape {arr.shape} for {path}")

    elif arr.ndim == 3:
        if arr.shape[0] == 31:
            out = arr[:, :360, :]
        else:
            raise ValueError(f"Unsupported 3D shape {arr.shape} for {path}")

    elif arr.ndim == 2:
        K = 4 * 361 * 720
        if arr.shape[0] == 31 and arr.shape[1] == K:
            tmp = arr.reshape(31, 4, 361, 720)
            out = tmp[:, var_idx, :360, :]
        elif arr.shape[1] == 31 and arr.shape[0] == K:
            tmp = arr.T.reshape(31, 4, 361, 720)
            out = tmp[:, var_idx, :360, :]
        else:
            raise ValueError(f"Unsupported 2D shape {arr.shape} for {path}")
    else:
        raise ValueError(f"Unsupported ndim={arr.ndim}, shape={arr.shape} for {path}")

    return np.asarray(out, dtype=np.float32)


def avgpool_baseline(patch, scale=6):
    """
    patch: torch tensor (B,1,H,W)
    returns:
      coarse: (B,1,H/scale,W/scale)
      base:   (B,1,H,W)
    """
    coarse = F.avg_pool2d(patch, kernel_size=scale, stride=scale)
    base = F.interpolate(coarse, size=patch.shape[-2:], mode="bilinear", align_corners=False)
    return coarse, base


def estimate_stats(paths, var_idx, patch_h, patch_w, n_files=10, samples_per_file=16, scale=6):
    print("Estimating field/residual statistics...", flush=True)

    rng = np.random.default_rng(123)
    chosen = list(paths)
    rng.shuffle(chosen)
    chosen = chosen[:min(n_files, len(chosen))]

    field_vals = []
    resid_vals = []

    for p in chosen:
        X = load_cycle_var(p, var_idx=var_idx)
        M, H, W = X.shape

        for _ in range(samples_per_file):
            m = int(rng.integers(0, M))
            y0 = int(rng.integers(0, H - patch_h + 1))
            x0 = int(rng.integers(0, W - patch_w + 1))

            patch_np = X[m, y0:y0 + patch_h, x0:x0 + patch_w].astype(np.float32)
            patch = torch.from_numpy(patch_np)[None, None]

            _, base = avgpool_baseline(patch, scale=scale)
            resid = patch - base

            field_vals.append(patch_np.reshape(-1))
            resid_vals.append(resid.numpy().reshape(-1))

    field_vals = np.concatenate(field_vals)
    resid_vals = np.concatenate(resid_vals)

    stats = {
        "field_mean": float(field_vals.mean()),
        "field_std": float(field_vals.std() + 1e-6),
        "resid_mean": float(resid_vals.mean()),
        "resid_std": float(resid_vals.std() + 1e-6),
    }

    print("Stats:", stats, flush=True)
    return stats


def make_beta_schedule(T, device):
    """
    Cosine DDPM schedule.
    """
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 1e-5, 0.999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def extract(a, t, x_shape):
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# -----------------------------
# Swin-style shifted-window blocks
# -----------------------------
def window_partition(x, ws):
    # x: B,H,W,C
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, ws * ws, C)


def window_reverse(windows, ws, B, H, W, C):
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, C)


def calculate_mask(H, W, ws, shift, device):
    if shift == 0:
        return None

    img_mask = torch.zeros((1, H, W, 1), device=device)

    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, ws)
    mask_windows = mask_windows.view(-1, ws * ws)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class WindowAttention2D(nn.Module):
    def __init__(self, dim=512, heads=8, ws=8):
        super().__init__()
        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.ws = ws
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        # Relative position bias, Swin-style
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), heads)
        )

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x_windows, mask=None):
        # x_windows: nW*B, N, C
        Bn, N, C = x_windows.shape

        qkv = self.qkv(x_windows)
        qkv = qkv.reshape(Bn, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(N, N, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            B = attn.shape[0] // nW
            attn = attn.view(B, nW, self.heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.heads, N, N)

        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(Bn, N, C)
        out = self.proj(out)
        return out


class SwinBlock2D(nn.Module):
    def __init__(self, dim=512, heads=8, ws=8, shift=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.ws = ws
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention2D(dim=dim, heads=heads, ws=ws)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        # x: B,C,H,W
        B, C, H, W = x.shape
        assert H % self.ws == 0 and W % self.ws == 0, f"H,W must divide ws={self.ws}"

        shortcut = x
        h = x.permute(0, 2, 3, 1).contiguous()
        h = self.norm1(h)

        if self.shift > 0:
            shifted = torch.roll(h, shifts=(-self.shift, -self.shift), dims=(1, 2))
            mask = calculate_mask(H, W, self.ws, self.shift, h.device)
        else:
            shifted = h
            mask = None

        windows = window_partition(shifted, self.ws)
        attn_windows = self.attn(windows, mask=mask)
        shifted_back = window_reverse(attn_windows, self.ws, B, H, W, C)

        if self.shift > 0:
            h = torch.roll(shifted_back, shifts=(self.shift, self.shift), dims=(1, 2))
        else:
            h = shifted_back

        h = h.permute(0, 3, 1, 2).contiguous()
        x = shortcut + h

        h2 = x.permute(0, 2, 3, 1).contiguous()
        h2 = h2 + self.mlp(self.norm2(h2))
        x = h2.permute(0, 3, 1, 2).contiguous()
        return x


class V9SwinResidualDDPM(nn.Module):
    """
    53M-parameter shifted-window Swin-style conditional residual DDPM.

    Input channels:
      0: noisy normalized residual x_t
      1: normalized upsampled coarse baseline
      2: latitude coordinate channel

    Output:
      predicted DDPM noise in residual space
    """
    def __init__(self, in_chans=3, embed_dim=512, depth=16, heads=8, ws=8):
        super().__init__()
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.SiLU(),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        blocks = []
        for i in range(depth):
            shift = 0 if i % 2 == 0 else ws // 2
            blocks.append(SwinBlock2D(dim=embed_dim, heads=heads, ws=ws, shift=shift))
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.GroupNorm(32, embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        h = self.stem(x)

        temb = timestep_embedding(t, self.embed_dim)
        temb = self.time_mlp(temb)[:, :, None, None]
        h = h + temb

        for blk in self.blocks:
            h = blk(h)

        return self.head(h)


# -----------------------------
# Data sampling
# -----------------------------
class CycleCache:
    def __init__(self, max_items=3, var_idx=0):
        self.max_items = max_items
        self.var_idx = var_idx
        self.cache = OrderedDict()

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        arr = load_cycle_var(path, var_idx=self.var_idx)

        self.cache[path] = arr
        self.cache.move_to_end(path)

        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

        return arr


def make_lat_coord(y0, patch_h, H_full, device):
    yy = torch.arange(y0, y0 + patch_h, device=device).float()
    yy = 2.0 * yy / max(H_full - 1, 1) - 1.0
    return yy[None, None, :, None].expand(1, 1, patch_h, 1)


def sample_training_patch(cache, train_files, args, stats, device):
    p = random.choice(train_files)
    X = cache.get(p)
    M, H, W = X.shape

    m = random.randrange(M)
    y0 = random.randrange(0, H - args.patch_h + 1)
    x0 = random.randrange(0, W - args.patch_w + 1)

    patch_np = X[m, y0:y0 + args.patch_h, x0:x0 + args.patch_w].copy()
    patch = torch.from_numpy(patch_np).to(device=device, dtype=torch.float32)[None, None]

    _, base = avgpool_baseline(patch, scale=args.scale)
    resid = patch - base

    x0_resid = (resid - stats["resid_mean"]) / stats["resid_std"]
    base_norm = (base - stats["field_mean"]) / stats["field_std"]

    lat = make_lat_coord(y0, args.patch_h, H, device)
    lat = lat.expand(1, 1, args.patch_h, args.patch_w)

    return x0_resid, base_norm, lat


# -----------------------------
# Sampling and evaluation
# -----------------------------
@torch.no_grad()
def sample_residual_patch(model, base_patch, lat_patch, stats, alphas_cumprod, args, device):
    model.eval()

    B, _, H, W = base_patch.shape
    x = torch.randn(B, 1, H, W, device=device)

    timesteps = torch.linspace(args.T - 1, 0, args.sample_steps, device=device).long()
    timesteps = timesteps.tolist()

    base_norm = (base_patch - stats["field_mean"]) / stats["field_std"]

    for i, t_int in enumerate(timesteps):
        t = torch.full((B,), int(t_int), device=device, dtype=torch.long)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            eps = model(torch.cat([x, base_norm, lat_patch], dim=1), t)

        a_t = alphas_cumprod[int(t_int)]
        sqrt_a_t = torch.sqrt(a_t)
        sqrt_oma_t = torch.sqrt(1.0 - a_t)

        x0_pred = (x - sqrt_oma_t * eps.float()) / sqrt_a_t
        x0_pred = torch.clamp(x0_pred, -6.0, 6.0)

        if i == len(timesteps) - 1:
            x = x0_pred
        else:
            t_prev = int(timesteps[i + 1])
            a_prev = alphas_cumprod[t_prev]
            x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1.0 - a_prev) * eps.float()

    resid = x * stats["resid_std"] + stats["resid_mean"]
    return resid


@torch.no_grad()
def generate_full_cycle(model, eval_file, stats, alphas_cumprod, args, device):
    X = load_cycle_var(eval_file, var_idx=args.var_idx)
    M, H, W = X.shape

    M_eval = min(args.eval_members, M)
    ref = X[:M_eval].astype(np.float32)
    gen = np.zeros_like(ref)

    for m in range(M_eval):
        target = torch.from_numpy(ref[m]).to(device=device, dtype=torch.float32)[None, None]
        _, full_base = avgpool_baseline(target, scale=args.scale)

        out = torch.zeros_like(target)

        for y0 in range(0, H, args.patch_h):
            for x0 in range(0, W, args.patch_w):
                base_patch = full_base[:, :, y0:y0 + args.patch_h, x0:x0 + args.patch_w]
                lat_patch = make_lat_coord(y0, args.patch_h, H, device)
                lat_patch = lat_patch.expand(1, 1, args.patch_h, args.patch_w)

                resid_patch = sample_residual_patch(
                    model, base_patch, lat_patch, stats, alphas_cumprod, args, device
                )
                out[:, :, y0:y0 + args.patch_h, x0:x0 + args.patch_w] = base_patch + resid_patch

        gen[m] = out[0, 0].detach().cpu().numpy()

        print(f"generated eval member {m+1}/{M_eval}", flush=True)

    return ref, gen


def covariance_response(A, obs_y, obs_x):
    X = A - A.mean(axis=0, keepdims=True)
    a = X[:, obs_y, obs_x]
    cov = np.tensordot(a, X, axes=(0, 0)) / max(A.shape[0] - 1, 1)
    return cov.astype(np.float32)


def local_mask(H, W, obs_y, obs_x, radius=55):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d2 = (yy - obs_y) ** 2 + (xx - obs_x) ** 2
    return np.exp(-0.5 * d2 / (radius ** 2)).astype(np.float32)


def rel_err(a, b, mask=None):
    if mask is not None:
        a = a * mask
        b = b * mask
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))


def make_figures(ref, gen, args, out_dir, fig_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    M, H, W = ref.shape
    sel = min(args.member_plot, M - 1)

    target = ref[sel]
    generated = gen[sel]

    # Baseline and residual for the selected member
    with torch.no_grad():
        t = torch.from_numpy(target)[None, None].float()
        _, base = avgpool_baseline(t, scale=args.scale)
        base_np = base[0, 0].numpy()

    resid_gen = generated - base_np
    corr = float(np.corrcoef(target.reshape(-1), generated.reshape(-1))[0, 1])

    ref_spread = ref.std(axis=0)
    gen_spread = gen.std(axis=0)
    spread_rmse = float(np.sqrt(np.mean((gen_spread - ref_spread) ** 2)))

    obs_y = min(args.obs_y, H - 1)
    obs_x = min(args.obs_x, W - 1)

    cov_ref = covariance_response(ref, obs_y, obs_x)
    cov_gen = covariance_response(gen, obs_y, obs_x)

    global_cov_err = rel_err(cov_gen, cov_ref)
    mask = local_mask(H, W, obs_y, obs_x, radius=args.local_radius)
    local_cov_err = rel_err(cov_gen, cov_ref, mask=mask)

    summary = {
        "model": "GEFS v9 50M shifted-window Swin-style conditional residual DDPM",
        "variable": VAR_NAMES[args.var_idx],
        "eval_file": os.path.basename(args.eval_file),
        "n_members_eval": int(M),
        "member_plot": int(sel),
        "member_correlation": corr,
        "spread_rmse": spread_rmse,
        "global_covariance_response_relative_error": global_cov_err,
        "localized_covariance_response_relative_error": local_cov_err,
        "obs_y": int(obs_y),
        "obs_x": int(obs_x),
        "local_radius": float(args.local_radius),
    }

    with open(os.path.join(out_dir, "gefs_v9_50m_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.save(os.path.join(out_dir, "gefs_v9_reference_cycle.npy"), ref.astype(np.float32))
    np.save(os.path.join(out_dir, "gefs_v9_generated_cycle.npy"), gen.astype(np.float32))

    with open(os.path.join(out_dir, "gefs_v9_50m_report.txt"), "w") as f:
        f.write("GEFS v9 50M shifted-window Swin-style conditional residual DDPM\n")
        f.write("=" * 72 + "\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # Decoder-style panel
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)

    panels = [
        (base_np, "Upsampled coarse baseline"),
        (resid_gen, "Sampled DDPM residual"),
        (generated, f"Generated full field\ncorr={corr:.3f}"),
        (target, "Reference GEFS full field"),
        (generated - target, "Generated - reference"),
    ]

    for ax, (img, title) in zip(axes, panels):
        if "residual" in title.lower() or "Generated - reference" in title:
            vmax = np.nanpercentile(np.abs(img), 99)
            im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(img, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"GEFS v9 50M Swin-DDPM | {VAR_NAMES[args.var_idx]} | {os.path.basename(args.eval_file)} | member {sel}",
        fontsize=14,
    )
    fig.savefig(os.path.join(fig_dir, "gefs_v9_50m_decoder_panel.png"), dpi=180)
    plt.close(fig)

    # Covariance panel
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)

    panels = [
        (ref_spread, "Reference spread", "viridis"),
        (gen_spread, f"Generated spread\nRMSE={spread_rmse:.3f}", "viridis"),
        (cov_ref, "Reference covariance response", "RdBu_r"),
        (cov_gen, f"Generated covariance response\nlocal err={local_cov_err:.3f}", "RdBu_r"),
        (cov_gen - cov_ref, "Covariance response error", "RdBu_r"),
    ]

    for ax, (img, title, cmap) in zip(axes, panels):
        if cmap == "RdBu_r":
            vmax = np.nanpercentile(np.abs(img), 99)
            im = ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(img, cmap=cmap)
        ax.scatter([obs_x], [obs_y], marker="*", s=80)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"GEFS v9 50M Swin-DDPM covariance-response | global err={global_cov_err:.3f}, local err={local_cov_err:.3f}",
        fontsize=14,
    )
    fig.savefig(os.path.join(fig_dir, "gefs_v9_50m_covariance_panel.png"), dpi=180)
    plt.close(fig)

    return summary


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--xroot", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--fig-dir", type=str, required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)

    p.add_argument("--var-idx", type=int, default=0)
    p.add_argument("--eval-cycle-substring", type=str, default="20260525_06")
    p.add_argument("--eval-file", type=str, default="")

    p.add_argument("--patch-h", type=int, default=120)
    p.add_argument("--patch-w", type=int, default=240)
    p.add_argument("--scale", type=int, default=6)

    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--window-size", type=int, default=8)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--stats-files", type=int, default=10)
    p.add_argument("--stats-samples-per-file", type=int, default=16)

    p.add_argument("--eval-members", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=25)
    p.add_argument("--member-plot", type=int, default=17)
    p.add_argument("--obs-y", type=int, default=304)
    p.add_argument("--obs-x", type=int, default=230)
    p.add_argument("--local-radius", type=float, default=55.0)

    p.add_argument("--seed", type=int, default=123)

    args = p.parse_args()

    seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    files = sorted(glob.glob(os.path.join(args.xroot, "X_*.npy")))
    if len(files) == 0:
        raise RuntimeError(f"No X_*.npy files found under {args.xroot}")

    if args.eval_file:
        eval_file = args.eval_file
    else:
        matches = [f for f in files if args.eval_cycle_substring in os.path.basename(f)]
        eval_file = matches[0] if matches else files[-1]

    train_files = [f for f in files if f != eval_file]
    if len(train_files) == 0:
        raise RuntimeError("No train files after excluding eval file.")

    args.eval_file = eval_file

    print("=" * 80, flush=True)
    print("GEFS v9 50M shifted-window Swin-style conditional residual DDPM", flush=True)
    print("=" * 80, flush=True)
    print(f"xroot      : {args.xroot}", flush=True)
    print(f"n files    : {len(files)}", flush=True)
    print(f"n train    : {len(train_files)}", flush=True)
    print(f"eval file  : {eval_file}", flush=True)
    print(f"variable   : {VAR_NAMES[args.var_idx]}", flush=True)
    print(f"device     : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU GB     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}", flush=True)

    stats_path = os.path.join(args.out_dir, "gefs_v9_50m_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        print(f"Loaded existing stats: {stats_path}", flush=True)
    else:
        stats = estimate_stats(
            train_files,
            var_idx=args.var_idx,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            n_files=args.stats_files,
            samples_per_file=args.stats_samples_per_file,
            scale=args.scale,
        )
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    model = V9SwinResidualDDPM(
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        ws=args.window_size,
    ).to(device)

    nparams = count_params(model)
    print(f"trainable parameters: {nparams:,}", flush=True)

    betas, alphas, alphas_cumprod = make_beta_schedule(args.T, device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    cache = CycleCache(max_items=3, var_idx=args.var_idx)

    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for step in range(1, args.steps_per_epoch + 1):
            x0, base_norm, lat = sample_training_patch(cache, train_files, args, stats, device)

            B = x0.shape[0]
            t = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)

            sqrt_ac = torch.sqrt(extract(alphas_cumprod, t, x0.shape))
            sqrt_om = torch.sqrt(1.0 - extract(alphas_cumprod, t, x0.shape))
            x_t = sqrt_ac * x0 + sqrt_om * noise

            model_in = torch.cat([x_t, base_norm, lat], dim=1)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_noise = model(model_in, t)
                loss_noise = F.mse_loss(pred_noise, noise)

                # small x0 consistency term stabilizes residual texture
                x0_hat = (x_t - sqrt_om * pred_noise.float()) / sqrt_ac
                loss_x0 = F.smooth_l1_loss(x0_hat, x0)

                loss = loss_noise + 0.05 * loss_x0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            losses.append(float(loss.item()))
            global_step += 1

            if step % 25 == 0:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(
                    f"epoch {epoch:03d} step {step:05d}/{args.steps_per_epoch} "
                    f"global {global_step:07d} loss {np.mean(losses[-25:]):.6f} "
                    f"peakGB {peak:.2f}",
                    flush=True,
                )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "args": vars(args),
            "stats": stats,
            "epoch": epoch,
            "global_step": global_step,
            "nparams": nparams,
        }
        # -------------------------------------------------------------
        # Storage-aware checkpoint policy
        #
        # Always maintain one rolling latest checkpoint.
        # Permanent epoch snapshots are retained only every 10 epochs
        # and at the final epoch.
        #
        # This reduces a 60-epoch run from ~37 GB to ~4 GB while still
        # preserving useful recovery milestones.
        # -------------------------------------------------------------

        latest_path = os.path.join(
            args.ckpt_dir,
            "gefs_v9_50m_latest.pt",
        )

        torch.save(
            ckpt,
            latest_path,
        )

        save_milestone = (
            epoch % 10 == 0
            or epoch == args.epochs
        )

        if save_milestone:

            ckpt_path = os.path.join(
                args.ckpt_dir,
                f"gefs_v9_50m_epoch{epoch:03d}.pt",
            )

            torch.save(
                ckpt,
                ckpt_path,
            )

            print(
                f"saved milestone checkpoint: {ckpt_path}",
                flush=True,
            )

        print(
            f"updated latest checkpoint: {latest_path}",
            flush=True,
        )

    print("Training completed.", flush=True)
    print(f"walltime min: {(time.time() - t0)/60:.2f}", flush=True)

    print("Running validation generation...", flush=True)
    ref, gen = generate_full_cycle(model, eval_file, stats, alphas_cumprod, args, device)

    summary = make_figures(ref, gen, args, args.out_dir, args.fig_dir)

    print("Final summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
