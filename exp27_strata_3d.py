#!/usr/bin/env python3
"""exp27 — the atlas as geological strata: a 3D cross-section visual.

Stacks the four response-field maps (Paris L8 / L14 / L26 + Tokyo L14) as
translucent slabs in 3D — depth = network layer, color = semantic topic of
what the model says at that (plane-position, depth). Rendered in the
infinitySI aesthetic, then a slow turntable animation.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Hiragino Sans GB"]
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp27"
OUT.mkdir(parents=True, exist_ok=True)

SLABS = [
    ("runs/exp20/response_field_paris_L8.json", "Paris  L8", 8),
    ("runs/exp20/response_field.json", "Qwen3-1.7B  Paris  L14", 14),
    ("runs/exp20/response_field_tokyo_L14.json", "Tokyo  L14", 15.2),
    ("runs/exp20/response_field_paris_L26.json", "Paris  L26", 26),
]
VOID, INK = "#050508", "#c9d4e0"
PALETTE = ["#2a3a5c", "#7fd4c8", "#e07a5f", "#c9a227", "#8f6fd4",
           "#4f8f5f", "#b0527a", "#5aa0c8", "#d49a6a", "#9a9ab0"]

def load_grid(path):
    rows = json.load(open(ROOT / path))
    n = int(max(r["i"] for r in rows)) + 1
    topics = sorted({r["topic"] for r in rows})
    tmap = {t: i for i, t in enumerate(topics)}
    grid = np.full((n, n), -1)
    for r in rows:
        grid[r["i"], r["j"]] = tmap[r["topic"]]
    return grid, topics

def topic_rgb(grid, topics):
    cmap = ListedColormap(PALETTE[:len(topics)])
    rgba = cmap((grid + 0.5) / len(topics))
    rgba[grid < 0] = [0, 0, 0, 0]
    return rgba

slabs = []
for path, label, layer in SLABS:
    grid, topics = load_grid(path)
    slabs.append((topic_rgb(grid, topics), label, layer))
    print(f"loaded {label}: {len(topics)} topics")

# normalize depth axis to relative depth in the network (28 layers)
fig = plt.figure(figsize=(13, 10), dpi=110)
fig.patch.set_facecolor(VOID)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(VOID)
ax.xaxis.set_pane_color((0, 0, 0, 0))
ax.yaxis.set_pane_color((0, 0, 0, 0))
ax.zaxis.set_pane_color((0.03, 0.03, 0.06, 1.0))
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis._axinfo["grid"]["color"] = (0.15, 0.18, 0.25, 0.4)

n = slabs[0][0].shape[0]
xs, ys = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
depths = [s[2] / 27.5 for s in slabs]
for k, (rgba, label, layer) in enumerate(slabs):
    z = np.full_like(xs, depths[k])
    ax.plot_surface(xs, ys, z, facecolors=rgba, rstride=1, cstride=1,
                    shade=False, alpha=0.92)
    # pillar from base to slab
    for (x0, y0) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        ax.plot([x0, x0], [y0, y0], [0, depths[k]], color="#1c2430", lw=0.6)
    ax.text2D(0.02, 0.06 + k * 0.05, f"{label}  (relative depth "
              f"{depths[k]:.2f})", transform=ax.transAxes, fontsize=9,
              color=INK, family="monospace")
# legend: topic colors (shared palette, from the first slab)
import matplotlib.patches as mpatches
_, ref_topics = load_grid(SLABS[0][0])
handles = [mpatches.Patch(facecolor=PALETTE[i % len(PALETTE)], label=t)
           for i, t in enumerate(ref_topics)]
leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.42),
                frameon=False, fontsize=8, labelcolor=INK, title="topic",
                title_fontproperties={"family": MONO if (MONO := ["DejaVu Sans", "Hiragino Sans GB"]) else None})
leg.get_title().set_color(INK)
# base plane: shared topic palette legend of slab 0
ax.text2D(0.02, 0.96, "the atlas as strata — semantic response fields stacked "
          "by depth", transform=ax.transAxes, fontsize=12, color=INK,
          family="monospace")
ax.set_zlim(0, 1.0)
ax.set_axis_off()
ax.view_init(elev=22, azim=-58)
fig.savefig(OUT / "atlas_strata.png", facecolor=VOID, bbox_inches="tight")
print("wrote", OUT / "atlas_strata.png")

# turntable animation
import subprocess
frames = OUT / "frames"
frames.mkdir(exist_ok=True)
for k, azim in enumerate(np.linspace(-58, 302, 72)):
    ax.view_init(elev=22, azim=azim)
    fig.savefig(frames / f"f{k:03d}.png", facecolor=VOID)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12",
                "-i", str(frames / "f%03d.png"), "-c:v", "libx264",
                "-crf", "20", "-pix_fmt", "yuv420p",
                str(OUT / "atlas_strata_turntable.mp4")], check=True)
print("wrote", OUT / "atlas_strata_turntable.mp4")
