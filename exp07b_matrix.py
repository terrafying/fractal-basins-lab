#!/usr/bin/env python3
"""exp07b — prompt-space chart: PHRASING x FORMAT matrix, answer identity as color.

From exp07's data (Qwen3.6-Flash, 4x4 sudoku p4_med): rows = 4 instruction
phrasings, cols = 5 grid formats. Cell color = answer identity (valid answers
grouped; invalid/unparsable get their own colors). Cell text = token count.
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp07"
results = json.loads((OUT / "prompt_basins.json").read_text())
rows = [r for r in results if r["puzzle"] == "p4_med"]
print(f"{len(rows)} variants for p4_med")

def split_variant(r):
    p = r["prompt"]
    phr = p.split("\n\n")[0]
    body = p.split("\n\n", 1)[1]
    # classify format by delimiter used in first grid line
    line = body.strip().splitlines()[0]
    if "row " in line:
        return phr, "labeled rows"
    if "," in line:
        return phr, "comma"
    if " " in line.strip():
        return phr, "space"
    if line.startswith("["):
        return phr, "bracket"
    return phr, "plain"

phr_list, fmt_list = [], []
for r in rows:
    ph, fm = split_variant(r)
    if ph not in phr_list:
        phr_list.append(ph)
    if fm not in fmt_list:
        fmt_list.append(fm)

grids = [tuple(r["grid"]) for r in rows]
freq = Counter(grids)
order = [g for g, _ in freq.most_common()]
g2k = {g: k for k, g in enumerate(order)}
n_answers = len(order)

R, C = len(phr_list), len(fmt_list)
Amat = np.full((R, C), -1)
Vmat = np.zeros((R, C), bool)
Tmat = np.zeros((R, C), int)
ann = [[""] * C for _ in range(R)]
for r in rows:
    ph, fm = split_variant(r)
    i, j = phr_list.index(ph), fmt_list.index(fm)
    Amat[i, j] = g2k[tuple(r["grid"])]
    Vmat[i, j] = r["valid"]
    Tmat[i, j] = r["tokens"]
    ann[i][j] = str(r["grid"][0]) if r["valid"] else ("x" if r["grid"][0] != "__UNPARSED__" else "u")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=130,
                              gridspec_kw={"width_ratios": [1.1, 1]})
fig.patch.set_facecolor("#050508")
# answer identity panel
cmap = plt.get_cmap("turbo", n_answers)
im = ax.imshow(Amat, cmap=ListedColormap(plt.get_cmap("turbo", n_ans := n_answers).colors),
               interpolation="nearest", vmin=0, vmax=max(n_answers - 1, 1))
ax.set_title("Qwen3.6-Flash, 4x4 sudoku: prompt-space chart\n"
             "cell = answer identity (color), text = first row / u=unparsed",
             fontsize=9, color="#c9d4e0", loc="left")
ax.set_xticks(range(C), [f.replace("row ", "r") for f in fmt_list], rotation=30,
              ha="right", fontsize=7, color="#c9d4e0")
ax.set_yticks(range(R), [p[:30] + "..." for p in phr_list], fontsize=6,
              color="#c9d4e0")
for i in range(R):
    for j in range(C):
        if Amat[i, j] >= 0:
            ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=6,
                    color="white")
for s in ax.spines.values():
    s.set_color("#1c2430")

# token-count panel (overthinking visible)
im2 = ax2.imshow(Tmat, cmap="inferno", interpolation="nearest")
ax2.set_title("completion tokens per variant\n(dark = instant, bright = overthink spiral)",
              fontsize=9, color="#c9d4e0", loc="left")
fig.colorbar(im2, ax=ax2, fraction=0.04)
ax2.set_xticks(range(C), [f for f in fmt_list], rotation=30, ha="right",
               fontsize=7, color="#c9d4e0")
ax2.set_yticks(range(R), [""] * R)
ax2.set_facecolor("#050508")
for s in ax2.spines.values():
    s.set_color("#1c2430")

fig.savefig(OUT / "prompt_space_matrix.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "prompt_space_matrix.png")
