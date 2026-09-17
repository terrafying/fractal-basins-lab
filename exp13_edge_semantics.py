#!/usr/bin/env python3
"""exp13 — edge semantics of the error atlas (CPU, exp06 data).

Question: in the FPRM answer-basin map, are SPATIALLY ADJACENT basins
semantically closer (smaller Hamming distance between their answers) than
random basin pairs? If yes, the atlas is a smooth semantic manifold —
neighboring starts err in neighboring ways. If no, errors are interleaved
chaotically (the Wada intuition, in semantic rather than topological form).

Also: which CELLS of the answer differ across each adjacency edge — the
error-atlas edge annotation (which constraint the boundary encodes).
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp13"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(ROOT / "runs/exp04/hard_a/fprm_slice_seed0.npz")
finals = d["finals"]                                   # (B, 9, 9)
B = len(finals)
RES = int(np.sqrt(B))
flat = finals.reshape(B, -1)
uniq, inv = np.unique(flat, axis=0, return_inverse=True)
lab = inv.reshape(RES, RES)
C = len(uniq)
answers = uniq.reshape(C, 9, 9)

def ham(a, b):
    return int((a != b).sum())

# spatial adjacency between classes (4-conn), with edge counts
edge_ham = Counter()
edge_count = Counter()
for r in range(RES - 1):
    for c in range(RES):
        a, b = lab[r, c], lab[r + 1, c]
        if a != b:
            h = ham(answers[a], answers[b])
            key = (min(a, b), max(a, b))
            edge_ham[key] = min(edge_ham.get(key, 10**9), h)  # keep min per pair
            edge_count[key] += 1
for r in range(RES):
    for c in range(RES - 1):
        a, b = lab[r, c], lab[r, c + 1]
        if a != b:
            h = ham(answers[a], answers[b])
            key = (min(a, b), max(a, b))
            edge_ham[key] = min(edge_ham.get(key, 10**9), h)
            edge_count[key] += 1

adj_pairs = np.array(list(edge_ham.values()))
adj_weights = np.array(list(edge_count.values()))
print(f"adjacent class pairs: {len(adj_pairs)} (edges {adj_weights.sum()})")

# random class pairs (same count), excluding identical
rng = np.random.default_rng(0)
rand_ham = []
while len(rand_ham) < len(adj_pairs) * 3:
    i, j = rng.integers(0, C, 2)
    if i != j:
        rand_ham.append(ham(answers[i], answers[j]))
rand_ham = np.array(rand_ham)

print(f"adjacent-pair hamming:  mean {adj_pairs.mean():.2f} med {np.median(adj_pairs):.0f}")
print(f"random-pair   hamming:  mean {rand_ham.mean():.2f} med {np.median(rand_ham):.0f}")
rho, p = spearmanr(adj_weights, adj_pairs)
print(f"edge weight vs hamming: rho={rho:.3f} (p={p:.1e})")
print(f"adjacent pairs with hamming<=2 (near-identical answers): "
      f"{(adj_pairs <= 2).sum()} ({(adj_pairs <= 2).mean():.1%})")
print(f"random    pairs with hamming<=2: {(rand_ham <= 2).mean():.1%}")

# figure: distributions + the strongest edges annotated by differing cells
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=120)
fig.patch.set_facecolor("#050508")
axes[0].hist(rand_ham, bins=40, alpha=0.5, density=True, color="#7a4fe8",
             label=f"random pairs (mean {rand_ham.mean():.1f})")
axes[0].hist(adj_pairs, bins=40, alpha=0.7, density=True, color="#2fd8e8",
             label=f"adjacent pairs (mean {adj_pairs.mean():.1f})")
axes[0].set_title("semantic distance: adjacent vs random basin pairs",
                  fontsize=9, color="#c9d4e0", loc="left")
axes[0].legend(fontsize=7, facecolor="#0a0a12", labelcolor="#c9d4e0")
axes[0].set_facecolor("#0a0a12")
axes[0].tick_params(colors="#c9d4e0")
for s in axes[0].spines.values():
    s.set_color("#1c2430")

# map: min-hamming-to-neighbor per pixel (semantic smoothness field)
smooth = np.full((RES, RES), np.nan)
for (i, j), h in edge_ham.items():
    for pix in (i, j):
        pass
# simpler: per-pixel min hamming to 4 neighbors' answers
sm = np.full((RES, RES), np.nan)
for r in range(RES):
    for c in range(RES):
        vals = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < RES and 0 <= c2 < RES and lab[r2, c2] != lab[r, c]:
                vals.append(ham(answers[lab[r, c]], answers[lab[r2, c2]]))
        if vals:
            sm[r, c] = min(vals)
im = axes[1].imshow(sm, cmap="viridis_r", interpolation="nearest", origin="lower")
axes[1].set_title("per-pixel semantic smoothness:\nmin Hamming to neighboring answers",
                  fontsize=9, color="#c9d4e0", loc="left")
fig.colorbar(im, ax=axes[1], fraction=0.04)
for ax in axes:
    ax.set_facecolor("#0a0a12")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430")
fig.savefig(OUT / "edge_semantics.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "edge_semantics.png")
np.savez_compressed(OUT / "edge_stats.npz", adj_pairs=adj_pairs,
                    adj_weights=adj_weights, rand_ham=rand_ham, smooth=sm)
