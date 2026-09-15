#!/usr/bin/env python3
"""Animate the crystallization of a solution over reasoning loops.

For each loop t, every pixel = one initial condition on the 2D slice; its
color = fraction of the 81 cells where the decoded grid at loop t still
disagrees with that trajectory's FINAL decoded grid. Loop 1 is mostly noise;
the fractal basin geometry emerges as the field cools. infinitySI palette,
void background, slow render at 6 fps (loops are the natural clock here).

Input : runs/exp02/<pid>/traces_seed<k>.npz
Output: runs/figs/crystal_<pid>_seed<k>.mp4  (+ a frame of the final loop)
"""
import sys, tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
VOID, INK = "#050508", "#c9d4e0"
CMAP = LinearSegmentedColormap.from_list(
    "err", [VOID, "#10143a", "#33207a", "#7a3fe0", "#e05fd8", "#ffd9f5"])


def render(traces, title, out_mp4, fps=6):
    final = traces[:, -1]                                   # (B, 9, 9)
    err = (traces != final[:, None]).mean(-1).mean(-1)      # (B, steps) in [0,1]
    RES = int(np.sqrt(traces.shape[0]))
    fields = err.reshape(-1, RES, RES)                      # (steps, RES, RES)
    from scipy.ndimage import gaussian_filter
    fields = gaussian_filter(fields, sigma=(0, 2, 2))       # smooth per frame
    vmax = float(np.percentile(fields, 99))                 # outlier-robust scale

    tmp = Path(tempfile.mkdtemp())
    n = len(fields)
    for t in range(n):
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=110)
        im = ax.imshow(fields[t], cmap=CMAP, interpolation="spline36",
                       vmin=0, vmax=max(vmax, 1e-4), origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#1c2430")
        ax.set_title(f"{title}   loop {t+1}/{n}   mean err {fields[t].mean():.3f}",
                     fontsize=8, loc="left", color=INK)
        fig.patch.set_facecolor(VOID); ax.set_facecolor(VOID)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
        fig.savefig(tmp / f"f{t:04d}.png", facecolor=VOID)
        plt.close(fig)
    # hold the last frame so the crystallized state lingers
    import shutil
    for i in range(fps * 2):
        shutil.copy(tmp / f"f{n-1:04d}.png", tmp / f"f{n+i:04d}.png")
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(tmp / "f%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(out_mp4)], check=True)
    print("wrote", out_mp4)


def main():
    figs = ROOT / "runs" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    for npz in sorted((ROOT / "runs" / "exp02").glob("*/*.npz")):
        pid = npz.parent.name
        seed = npz.stem.replace("traces_seed", "")
        traces = np.load(npz)["traces"]
        render(traces, f"{pid} slice seed {seed}", figs / f"crystal_{pid}_seed{seed}.mp4")


if __name__ == "__main__":
    main()
