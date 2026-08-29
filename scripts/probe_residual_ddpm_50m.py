"""
Template script for a conditional residual DDPM atmospheric-ensemble workflow.

This file is a cleaned portfolio template. Users should adapt paths, metadata,
and data-loading logic for their own HPC system and dataset.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    def forward(self, x):
        # x: B,H,W,C
        B, H, W, C = x.shape
        w = window_partition(x, self.ws)  # nW*B, ws*ws, C
        qkv = self.qkv(w)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(w.shape[0], w.shape[1], C)
        out = self.proj(out)
        out = window_reverse(out, self.ws, B, H, W, C)
        return out


class SwinLikeBlock(nn.Module):
    def __init__(self, dim=512, heads=8, ws=8, mlp_ratio=4):
        super().__init__()
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
        h = x.permute(0, 2, 3, 1).contiguous()
        h = h + self.attn(self.norm1(h))
        h = h + self.mlp(self.norm2(h))
        return h.permute(0, 3, 1, 2).contiguous()


class ResidualDDPM50M(nn.Module):
    """
    Conditional residual DDPM backbone.

    Input channels:
      0: noisy residual x_t
      1: upsampled coarse baseline
      2: optional conditioning field / coarse repeated field

    Output:
      predicted noise in residual space.
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

        self.blocks = nn.ModuleList([
            SwinLikeBlock(dim=embed_dim, heads=heads, ws=ws, mlp_ratio=4)
            for _ in range(depth)
        ])

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


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = ResidualDDPM50M(
        in_chans=3,
        embed_dim=512,
        depth=16,
        heads=8,
        ws=8,
    ).to(device)

    nparams = count_params(model)
    print(f"trainable parameters: {nparams:,}")
    print(f"target: about 50M")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"total GPU GB: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")

    B, H, W = 1, 120, 240
    x = torch.randn(B, 3, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.reset_peak_memory_stats()

    model.train()
    opt.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = model(x, t)
        target = torch.randn_like(y)
        loss = F.mse_loss(y, target)

    loss.backward()
    opt.step()

    torch.cuda.synchronize()

    print(f"forward/backward loss: {loss.item():.6f}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"peak allocated GB: {torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"peak reserved GB:   {torch.cuda.max_memory_reserved() / 1024**3:.2f}")
    print("V9-50M conditional residual DDPM probe: SUCCESS")


if __name__ == "__main__":
    main()
