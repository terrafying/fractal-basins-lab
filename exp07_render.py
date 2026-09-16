#!/usr/bin/env python3
"""Render exp07 prompt-basin results as a figure: per-variant answer identity
matrix + token-count trace, for each puzzle."""
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp07"

results = json.loads((OUT / "prompt_basins.json").read_text())
puzzles = sorted({r["puzzle"] for r in results})
n_v = len({r["variant"] for r in results})

fig, axes = plt.subplots(len(puzzles), 1, figsize=(14, 3.1 * len(puzzles)), dpi=120)
fig.patch.set_facecolor("#050508")
if len(puzzles) == 1:
    axes = [axes]

for ax, pid in zip(axes, puzzles):
    rs = [r for r in results if r["puzzle"] == pid]
    grids = [tuple(r["grid"]) for r in rs]
    lut = {}
    lab = []
    for g in grids:
        if g not in lut:
            lut[g] = len(lut)
        lab.append(lut[g])
    lab = np.array(lab)
    toks = np.array([r["tokens"] for r in rs])
    valid = np.array([r["valid"] for r in rs])
    n = len(rs)
    im = ax.imshow(np.vstack([lab, valid.astype(int) * (max(lut.values()) + 1),
                              np.clip(toks, 0, 1500) / 100.0]),
                   aspect="auto", cmap="turbo", interpolation="nearest")
    top = Counter(grids).most_common(2)
    ax.set_title(f"{pid}: {n} variants, {len(lut)} distinct answers, "
                 f"{sum(valid)} valid | top: {top[0][1]}x, {top[1][1]}x | "
                 f"rows: answer-id / valid / tokens(/100)",
                 fontsize=8, color="#c9d4e0", loc="left")
    ax.set_yticks([0, 1, 2], ["id", "ok", "tok"], color="#c9d4e0")
    ax.set_facecolor("#050508")
    for s in ax.spines.values():
        s.set_color("#1c2430")

fig.savefig(OUT / "prompt_basins.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "prompt_basins.png")
