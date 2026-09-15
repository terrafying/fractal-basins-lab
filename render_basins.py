#!/usr/bin/env python3
"""Render basin maps as smooth dark-field visuals (infinitySI aesthetic).

Inputs : runs/exp01/<pid>/slice_seed<k>.npz  (settling-time fields)
Outputs: runs/figs/<pid>_seed<k>.png          stills (300 dpi, void bg)
         runs/figs/drift_<pid>_seed<k>.mp4    slow pan/zoom drift loop (if ffmpeg)

Aesthetic: void #050508, cyan->violet field, mono text, hairline frame.
The settling field is rendered at native resolution and smoothed by the
display interpolation only (spline36) so fractal boundaries stay crisp.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "exp01"
FIGS = ROOT / "runs" / "figs"

VOID = "#050508"
INK = "#c9d4e0"
CYAN = "#2fd8e8"
VIOLET = "#4b2fe0"
DEEP = "#0a0620"

CMAP = LinearSegmentedColormap.from_list(
    "basin", [VOID, DEEP, "#1a1e5e", VIOLET, "#7a4fe8", CYAN, "#eaffff"])
plt.rcParams.update({
    "font.family": "Menlo",
    "text.color": INK,
    "figure.facecolor": VOID,
    "axes.facecolor": VOID,
    "savefig.facecolor": VOID,
})


def render_field(field, title, out_png, dpi=300, sigma=1.5):
    import numpy as _np
    if sigma:
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(field.astype(float), sigma)
    tmax = max(field.max(), 1)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    im = ax.imshow(field, cmap=CMAP, interpolation="spline36",
                   vmin=0, vmax=tmax, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430"); s.set_linewidth(0.6)
    ax.set_title(title, fontsize=9, pad=12, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=6, colors=INK)
    cbar.outline.set_edgecolor("#1c2430")
    cbar.set_label("loops to settle", fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print("wrote", out_png)


def render_drift(field, title, out_mp4, frames=240, zoom=(1.0, 1.6), fps=30, sigma=1.5):
    """Slow push-in with slight lateral drift; loops by easing in and out."""
    import subprocess
    import tempfile
    if sigma:
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(field.astype(float), sigma)
    tmax = max(field.max(), 1)
    tmp = Path(tempfile.mkdtemp())
    phases = np.concatenate([np.linspace(0, 1, frames // 2),
                             np.linspace(1, 0, frames // 2)])
    for i, p in enumerate(phases):
        z = zoom[0] + p * (zoom[1] - zoom[0])
        cx = 0.5 + 0.06 * np.sin(2 * np.pi * p)
        cy = 0.5 + 0.06 * np.cos(2 * np.pi * p)
        h = field.shape[0] / z
        w = field.shape[1] / z
        x0 = np.clip(cx * field.shape[1] - w / 2, 0, field.shape[1] - w)
        y0 = np.clip(cy * field.shape[0] - h / 2, 0, field.shape[0] - h)
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=110)
        ax.imshow(field, cmap=CMAP, interpolation="spline36", vmin=0, vmax=tmax,
                  origin="lower", extent=[0, field.shape[1], 0, field.shape[0]])
        ax.set_xlim(x0, x0 + w); ax.set_ylim(y0, y0 + h)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#1c2430")
        ax.set_title(title, fontsize=8, loc="left")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        fig.savefig(tmp / f"f{i:04d}.png")
        plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", str(fps),
                    "-i", str(tmp / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(out_mp4)], check=True)
    print("wrote", out_mp4)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    drift = "--drift" in sys.argv
    for npz in sorted(RUNS.glob("*/*.npz")):
        pid = npz.parent.name
        seed = npz.stem.replace("slice_seed", "")
        field = np.load(npz)["settling"]
        be = npz.with_name("metrics.json")
        title = f"{pid}  slice seed {seed}  settling field"
        render_field(field, title, FIGS / f"{pid}_seed{seed}.png")
        if drift:
            render_drift(field, title, FIGS / f"drift_{pid}_seed{seed}.mp4")


if __name__ == "__main__":
    main()
