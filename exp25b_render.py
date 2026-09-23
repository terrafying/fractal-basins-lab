#!/usr/bin/env python3
"""Re-render exp25 figure from saved JSON with CJK-capable monospace."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MONO = ["DejaVu Sans Mono", "Hiragino Sans GB", "Arial Unicode MS"]

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp25"
res = json.load(open(OUT / "jlens_trajectories.json"))
n_regions = len(res["regions"])
NL = len(res["pain"])

fig = plt.figure(figsize=(16, 11.5), dpi=120)
fig.patch.set_facecolor("#050508")
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.7], hspace=0.32)

ax = fig.add_subplot(gs[0])
# cleaner pinned sets: pain vs valence specific
PAIN_TOKS = ["anguish", "torment", "crippling", "despair", "desperation", "痛苦", "折磨"]
VAL_TOKS = ["PTSD", "paranoid", "trauma", "helpless", "无助", "该怎么办", "求助", "refuge"]
REG_TOKS = {0: ["Answer", "answer", "What", "Why", "How", "question"],
            1: ["Verse", "Include", "Provide", "Make", "Write"],
            3: ["reasoning", "justification", "rationale", "Specifically", "Please"],
            8: ["SQL", "sql", "数据库", "查询", "mysql"],
            10: ["Silence", "沉默", "Nothing", "quiet"]}
obj_names = ["PAIN", "NEG-VALENCE"] + [f"R{i}" for i in range(n_regions)]
mat = np.full((len(obj_names), NL), 8.0)
for l in range(NL):
    pt = res["pain"][l]
    mat[0, l] = next((i for i, t in enumerate(pt)
                      if any(p in t for p in PAIN_TOKS)), 8)
    vt = res["valence"][l]
    mat[1, l] = next((i for i, t in enumerate(vt)
                      if any(p in t for p in VAL_TOKS)), 8)
    for i, pins in REG_TOKS.items():
        if i < n_regions:
            rt = res["regions"][i][l]
            mat[2 + i, l] = next((j for j, t in enumerate(rt)
                                  if any(p in t for p in pins)), 8)
masked = np.ma.masked_where(mat > 7, mat)
cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#0a0a12")
im = ax.imshow(masked, aspect="auto", cmap="viridis_r", vmin=0, vmax=7)
ax.set_yticks(range(len(obj_names)), obj_names, fontsize=8, color="#c9d4e0",
              family=MONO)
ax.set_xticks(range(0, NL, 2))
ax.set_xlabel("layer", color="#c9d4e0")
ax.set_title("rank of each object's signature token in its J-lens readout "
             "(0 = top-1; dark = absent)", fontsize=10, color="#c9d4e0",
             loc="left", family=MONO)
ax.tick_params(colors="#c9d4e0")
for s in ax.spines.values():
    s.set_color("#1c2430")
plt.colorbar(im, ax=ax, shrink=0.75)

ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
band = 4
rows = [("PAIN", res["pain"]), ("NEG-VALENCE", res["valence"])]
for i in (0, 1, 3, 8, 10):
    rows.append((f"R{i}", res["regions"][i]))
y = 1.0
for name, traj in rows:
    ax2.text(0.0, y, name, fontsize=9, color="#7fd4c8", family=MONO,
             va="top")
    y -= 0.062
    for l in range(0, NL, band):
        ax2.text(0.005, y, f"L{l:2d}", fontsize=7, color="#5a6a7a",
                 family=MONO, va="top")
        ax2.text(0.05, y, "  ".join(t for t in traj[l][:5]), fontsize=8,
                 color="#c9d4e0", family=MONO, va="top")
        y -= 0.043
    y -= 0.012
fig.savefig(OUT / "jlens_trajectories.png", facecolor="#050508",
            bbox_inches="tight")
print("rewrote", OUT / "jlens_trajectories.png")
