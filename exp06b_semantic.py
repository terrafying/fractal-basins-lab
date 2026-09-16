#!/usr/bin/env python3
"""exp06b — deeper analysis of the FPRM identity-basin slice (CPU).

1. Semantic-distance map: color each initial condition by how far ITS basin's
   answer is from the majority answer (Hamming /81). Continuous semantic
   field -> where in latent space does the model end up 'slightly wrong'
   vs 'completely wrong'?
2. Basin-size spectrum: log-log histogram of basin sizes (power law?).
3. Wrong-answer anatomy: which CELLS differ from the majority answer, across
   basins — is the disagreement concentrated on specific cells (the puzzle's
   hard constraints) or spread evenly?
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp06"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(ROOT / "runs/exp04/hard_a/fprm_slice_seed0.npz")
finals = d["finals"]                                   # (B, 9, 9)
B = len(finals)
RES = int(np.sqrt(B))
flat = finals.reshape(B, -1)

# majority answer
vals, counts = np.unique(flat, axis=0, return_counts=True)
majority = vals[np.argmax(counts)].reshape(9, 9)
print(f"majority answer: {counts.max()}/{B} ({counts.max()/B:.1%})")

# hamming-to-majority per condition
hdist = np.array([int((f.reshape(9, 9) != majority).sum()) for f in finals])
hmap = hdist.reshape(RES, RES)
print(f"hamming-to-majority: min {hdist.min()} med {int(np.median(hdist))} "
      f"max {hdist.max()}  mean {hdist.mean():.1f}")

# basin-size spectrum
sizes = np.sort(counts)[::-1]
nz = sizes[sizes > 0]
print(f"basin sizes: top10 {nz[:10]}")
print(f"singleton basins: {(counts == 1).sum()} / {len(counts)}")

# disagreement anatomy: per-cell fraction of basins' answers differing from majority
diff_cells = (finals.reshape(B, 9, 9) != majority).mean(0)
top_cells = np.dstack(np.unravel_index(np.argsort(diff_cells.ravel())[::-1],
                                       diff_cells.shape))[0][:8]
print("cells most often wrong (r,c,fraction):",
      [(int(r), int(c), round(float(diff_cells[r, c]), 2)) for r, c in top_cells])

# ---- figures ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), dpi=130)
fig.patch.set_facecolor("#050508")

im0 = axes[0].imshow(hmap, cmap="inferno", interpolation="spline36", origin="lower")
axes[0].set_title("semantic distance to majority answer (Hamming/81)",
                  fontsize=9, color="#c9d4e0", loc="left")
fig.colorbar(im0, ax=axes[0], fraction=0.04)

axes[1].loglog(np.arange(1, len(nz) + 1), nz, ".", color="#2fd8e8", ms=4)
axes[1].set_title(f"basin-size spectrum ({len(counts)} basins)", fontsize=9,
                  color="#c9d4e0", loc="left")
axes[1].set_xlabel("rank", color="#c9d4e0"); axes[1].set_ylabel("size", color="#c9d4e0")
axes[1].set_facecolor("#050508")
for s in axes[1].spines.values():
    s.set_color("#1c2430")
axes[1].tick_params(colors="#c9d4e0")

im2 = axes[2].imshow(diff_cells, cmap="hot", interpolation="nearest", origin="lower")
axes[2].set_title("disagreement anatomy: per-cell wrongness fraction",
                  fontsize=9, color="#c9d4e0", loc="left")
fig.colorbar(im2, ax=axes[2], fraction=0.04)

for ax in axes:
    ax.set_facecolor("#050508")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430")
fig.savefig(OUT / "fprm_semantic_structure.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "fprm_semantic_structure.png")

# relationship: settling time vs semantic distance (do slow starts end up
# 'more wrong'? -> correlation of basin geometry with error severity)
from scipy.stats import spearmanr
rho, p = spearmanr(settling := d["settling"].ravel(), hdist)
print(f"settling vs hamming-to-majority: rho={rho:.3f} (p={p:.1e})")
