#!/usr/bin/env python3
"""exp09d — depth stack figure + boundary-adjacency graph of the number manifold.

A. 2x2 grid: answer-value chart at layers 8/14/20/26 (common clipped scale).
B. Adjacency graph per layer: nodes = distinct answers (positioned by value on
   a vertical number line), edges = two answers are spatially adjacent on the
   chart. Shows which semantic regions connect at each depth — the manifold's
   neighborhood structure as computation proceeds.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp09c"
LAYERS = [8, 14, 20, 26]
TRUTH = 65
VMAX = 95  # common clipped scale

charts = {}
for L in LAYERS:
    p = OUT / f"number_chart_L{L}.json"
    results = json.loads(p.read_text())
    RES = int(np.sqrt(len(results)))
    amap = np.array([r["answer"] if r["answer"] is not None else np.nan
                     for r in results], dtype=float).reshape(RES, RES)
    charts[L] = amap

# ---- A: depth stack ----
fig, axes = plt.subplots(2, 2, figsize=(13, 12), dpi=120)
fig.patch.set_facecolor("#050508")
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#2a2a33")
for ax, L in zip(axes.ravel(), LAYERS):
    amap = charts[L]
    clipped = np.clip(amap, 0, VMAX)
    im = ax.imshow(np.ma.masked_invalid(clipped), cmap=cmap,
                   interpolation="nearest", origin="lower", vmin=0, vmax=VMAX)
    n_ans = len(set(a for a in amap.ravel() if not np.isnan(a)))
    ax.set_title(f"layer {L}: {n_ans} distinct answers "
                 f"(values clipped to 0..{VMAX}, gray=unparsed)",
                 fontsize=9, color="#c9d4e0", loc="left")
    ax.set_facecolor("#050508"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430")
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
fig.savefig(OUT / "depth_stack.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "depth_stack.png")

# ---- B: adjacency graphs ----
fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), dpi=120)
fig.patch.set_facecolor("#050508")
for ax, L in zip(axes, LAYERS):
    amap = charts[L]
    adj = defaultdict(set)
    freq = Counter()
    R, C = amap.shape
    for r in range(R - 1):
        for c in range(C - 1):
            a, b_ = amap[r, c], amap[r, c + 1]
            if not np.isnan(a) and not np.isnan(b_) and a != b_:
                adj[a].add(b_); adj[b_].add(a)
            a, b_ = amap[r, c], amap[r + 1, c]
            if not np.isnan(a) and not np.isnan(b_) and a != b_:
                adj[a].add(b_); adj[b_].add(a)
    for r in range(R):
        for c in range(C):
            if not np.isnan(amap[r, c]):
                freq[amap[r, c]] += 1
    # nodes = answers with freq >= 5; y = value, x spread to reduce overlap
    nodes = sorted(v for v, f in freq.items() if f >= 5)
    pos = {}
    for i, v in enumerate(nodes):
        pos[v] = (0.15 * np.sin(2.7 * v), v)
    truth_col = "#e05f2f" if TRUTH in nodes else "#888888"
    for v in nodes:
        col = "#2fd8e8" if v == TRUTH else "#7a4fe8"
        ax.scatter(*pos[v], s=40 + 3 * min(freq[v], 200), color=col, zorder=3)
        ax.text(pos[v][0] + 0.05, pos[v][1] + 1.5, str(int(v)), fontsize=6,
                color="#c9d4e0")
    for a in nodes:
        for b_ in adj.get(a, []):
            if b_ in pos and a < b_:
                x = [pos[a][0], pos[b_][0]]
                y = [pos[a][1], pos[b_][1]]
                mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
                ax.plot(x + [mx], y + [my], color="#3a4a5a", lw=0.8, zorder=1)
                pass
    ax.set_title(f"layer {L}: semantic adjacency\n"
                 f"({len(nodes)} answers, edges = shared border)",
                 fontsize=8, color="#c9d4e0", loc="left")
    ax.set_facecolor("#050508")
    ax.set_ylim(-5, 100); ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430")
fig.savefig(OUT / "adjacency_graphs.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "adjacency_graphs.png")
