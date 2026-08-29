"""
Template script for a conditional residual DDPM atmospheric-ensemble workflow.

This file is a cleaned portfolio template. Users should adapt paths, metadata,
and data-loading logic for their own HPC system and dataset.
"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def try_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        return ccrs, cfeature, LongitudeFormatter, LatitudeFormatter
    except Exception:
        return None, None, None, None


def roll_lon(x):
    return np.roll(x, x.shape[-1] // 2, axis=-1)


def covariance_response(A, obs_y, obs_x):
    """
    A: ensemble array, shape (Ne,H,W)
    response at all grid points to the selected observation grid point.
    """
    X = A - A.mean(axis=0, keepdims=True)
    a = X[:, obs_y, obs_x]
    return np.tensordot(a, X, axes=(0, 0)) / max(A.shape[0] - 1, 1)


def gaspari_cohn(r):
    """
    Compactly supported Gaspari-Cohn localization.
    r = distance / radius.
    Support ends at r = 2.
    """
    r = np.asarray(r)
    out = np.zeros_like(r, dtype=np.float64)

    m1 = (r >= 0) & (r <= 1)
    x = r[m1]
    out[m1] = (
        1
        - 5/3*x**2
        + 5/8*x**3
        + 1/2*x**4
        - 1/4*x**5
    )

    m2 = (r > 1) & (r <= 2)
    x = r[m2]
    out[m2] = (
        4
        - 5*x
        + 5/3*x**2
        + 5/8*x**3
        - 1/2*x**4
        + 1/12*x**5
        - 2/(3*x)
    )

    out[r > 2] = 0.0
    return np.clip(out, 0.0, 1.0)


def localization_taper(H, W, obs_y, obs_x, radius):
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]

    dy = yy - obs_y
    dx0 = np.abs(xx - obs_x)

    # periodic longitude distance
    dx = np.minimum(dx0, W - dx0)

    dist = np.sqrt(dy**2 + dx**2)
    taper = gaspari_cohn(dist / radius)
    return taper.astype(np.float32)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def setup_axis(ax, ccrs, cfeature, LongitudeFormatter, LatitudeFormatter):
    if ccrs is not None:
        ax.set_global()
        ax.coastlines(linewidth=0.65)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, alpha=0.5)

        xticks = [-180, -120, -60, 0, 60, 120, 180]
        yticks = [-90, -60, -30, 0, 30, 60, 90]
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.20, linewidth=0.4)
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax.grid(alpha=0.25, linewidth=0.4)


def plot_map(ax, field, title, cmap, symmetric, obs_lon, obs_lat,
             ccrs=None, cfeature=None, LongitudeFormatter=None, LatitudeFormatter=None):
    img = roll_lon(field)
    extent = [-180, 180, -90, 90]

    if symmetric:
        vmax = np.nanpercentile(np.abs(img), 99)
        vmin = -vmax
    else:
        vmin = np.nanpercentile(img, 1)
        vmax = np.nanpercentile(img, 99)

    if ccrs is not None:
        im = ax.imshow(
            img,
            extent=extent,
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        setup_axis(ax, ccrs, cfeature, LongitudeFormatter, LatitudeFormatter)
        ax.scatter(
            [obs_lon], [obs_lat],
            marker="*", s=95, color="black",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
    else:
        im = ax.imshow(
            img,
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        setup_axis(ax, ccrs, cfeature, LongitudeFormatter, LatitudeFormatter)
        ax.scatter([obs_lon], [obs_lat], marker="*", s=95, color="black", zorder=10)

    ax.set_title(title, fontsize=9)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    p.add_argument("--fig-dir", required=True)
    p.add_argument("--tag", default="gefs")
    p.add_argument("--obs-y", type=int, default=304)
    p.add_argument("--obs-x", type=int, default=230)
    p.add_argument("--local-radius", type=float, default=55.0)
    args = p.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    ref = np.load(os.path.join(args.result_dir, "gefs_v9_reference_cycle.npy"))
    gen = np.load(os.path.join(args.result_dir, "gefs_v9_generated_cycle.npy"))

    M, H, W = ref.shape

    obs_y = min(args.obs_y, H - 1)
    obs_x = min(args.obs_x, W - 1)

    # For unrolled 0..360 longitude grid converted to -180..180 plot.
    obs_lon = -180 + 360 * (obs_x / W)
    obs_lat = 90 - 180 * (obs_y / H)

    cov_ref = covariance_response(ref, obs_y, obs_x)
    cov_gen = covariance_response(gen, obs_y, obs_x)

    taper = localization_taper(H, W, obs_y, obs_x, args.local_radius)

    cov_ref_loc = cov_ref * taper
    cov_gen_loc = cov_gen * taper
    loc_err_map = cov_gen_loc - cov_ref_loc

    unloc_err = rel_l2(cov_gen, cov_ref)
    loc_err = rel_l2(cov_gen_loc, cov_ref_loc)

    ccrs, cfeature, LongitudeFormatter, LatitudeFormatter = try_cartopy()
    projection = ccrs.PlateCarree() if ccrs is not None else None

    print("Cartopy:", "available" if ccrs is not None else "not available")
    print("ref shape:", ref.shape)
    print("gen shape:", gen.shape)
    print("obs_y, obs_x:", obs_y, obs_x)
    print("obs_lat, obs_lon:", obs_lat, obs_lon)
    print("unlocalized relative covariance-response error:", unloc_err)
    print("localized relative covariance-response error:", loc_err)

    # 2 x 3 panel
    fig = plt.figure(figsize=(20, 9))

    axes = []
    for i in range(6):
        if ccrs is not None:
            axes.append(fig.add_subplot(2, 3, i + 1, projection=projection))
        else:
            axes.append(fig.add_subplot(2, 3, i + 1))

    panels = [
        (cov_ref, "Reference unlocalized covariance-action map", "RdBu_r", True),
        (cov_gen, "Generated unlocalized covariance-action map", "RdBu_r", True),
        (cov_gen - cov_ref, f"Unlocalized error | rel.={unloc_err:.3f}", "RdBu_r", True),
        (cov_ref_loc, "Reference localized covariance-action map", "RdBu_r", True),
        (cov_gen_loc, "Generated localized covariance-action map", "RdBu_r", True),
        (loc_err_map, f"Localized error | rel.={loc_err:.3f}", "RdBu_r", True),
    ]

    for ax, (img, title, cmap, sym) in zip(axes, panels):
        im = plot_map(
            ax, img, title, cmap, sym,
            obs_lon=obs_lon,
            obs_lat=obs_lat,
            ccrs=ccrs,
            cfeature=cfeature,
            LongitudeFormatter=LongitudeFormatter,
            LatitudeFormatter=LatitudeFormatter,
        )
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.043, pad=0.065)

    fig.suptitle(
        f"GEFS HGT500 Swin-DDPM ensemble: localized DA covariance-action diagnostic\n"
        f"black star = covariance reference location | localization radius = {args.local_radius:g} grid points",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(args.fig_dir, f"{args.tag}_localized_covariance_action_panel.png")
    fig.savefig(out, dpi=230, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_members": int(M),
        "obs_y": int(obs_y),
        "obs_x": int(obs_x),
        "obs_lat": float(obs_lat),
        "obs_lon": float(obs_lon),
        "local_radius_gridpoints": float(args.local_radius),
        "unlocalized_relative_covariance_response_error": float(unloc_err),
        "localized_relative_covariance_response_error": float(loc_err),
        "figure": out,
    }

    with open(os.path.join(args.fig_dir, f"{args.tag}_localized_covariance_action_summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)

    print("WROTE:")
    print(out)
    print(os.path.join(args.fig_dir, f"{args.tag}_localized_covariance_action_summary.json"))


if __name__ == "__main__":
    main()
