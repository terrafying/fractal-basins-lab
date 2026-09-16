#!/usr/bin/env python3
"""Exp 05 — semantic structure of the basin landscape (CPU, existing data).

From exp02's full decoded trajectories (48^2 slice, hard_a, 24 loops):

1. Saddle maps: every maximal run of >=3 consecutive loops where a
   trajectory's decoded grid is constant AND differs from its final grid is a
   "saddle episode" — the model holding a nearly-correct wrong answer. Each
   distinct wrong grid gets a semantic label (its own cells!). Output: which
   saddle each initial condition visited first, as a categorical map of the
   2D latent slice, plus per-saddle stats (Hamming distance to final, visit
   counts, spatial clustering).

2. Settling predictability: can EARLY behavior (loops 1-3, before any
   settling info) predict total settling time? If yes, the geometric
   initialization speedup (pick z0 in fast regions after a 3-loop probe) is
   feasible. Spearman correlation between early change-rate features and
   final settling, overall and on the slow tail.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp05"
OUT.mkdir(parents=True, exist_ok=True)


def hamming(a, b):
    return int((a != b).sum())


def main():
    d = np.load(ROOT / "runs/exp02/hard_a/traces_seed0.npz")
    tr = d["traces"]                                  # (B, 24, 9, 9)
    B, S = tr.shape[0], tr.shape[1]
    RES = int(np.sqrt(B))
    final = tr[:, -1]

    # ---- 1. saddle episodes ----
    labels = np.full(B, -1, dtype=int)                # -1 = no saddle visited
    saddle_registry = {}                              # grid bytes -> saddle id
    saddle_stats = []                                 # (id, hamming, visits)
    for b in range(B):
        g = tr[b]
        run_start = None
        for t in range(1, S):
            same = (g[t] == g[t - 1]).all()
            if same and run_start is None:
                run_start = t - 1
            elif not same and run_start is not None:
                runlen = t - run_start
                grid = g[run_start]
                if runlen >= 3 and hamming(grid, final[b]) > 0:
                    key = grid.tobytes()
                    if key not in saddle_registry:
                        saddle_registry[key] = len(saddle_registry)
                        saddle_stats.append([saddle_registry[key],
                                             hamming(grid, final[b]), 0])
                    sid = saddle_registry[key]
                    saddle_stats[sid][2] += 1
                    if labels[b] == -1:
                        labels[b] = sid
                run_start = None
        # run touching the cap
        if run_start is not None and (S - run_start) >= 3:
            grid = g[run_start]
            if hamming(grid, final[b]) > 0:
                key = grid.tobytes()
                if key not in saddle_registry:
                    saddle_registry[key] = len(saddle_registry)
                    saddle_stats.append([saddle_registry[key], hamming(grid, final[b]), 0])
                saddle_stats[saddle_registry[key]][2] += 1
                if labels[b] == -1:
                    labels[b] = saddle_registry[key]

    n_visited = int((labels >= 0).sum())
    print(f"conditions visiting a saddle (>=3-loop hold of a wrong grid): "
          f"{n_visited}/{B}")
    print(f"distinct saddle grids: {len(saddle_registry)}")
    for sid, ham, vis in sorted(saddle_stats, key=lambda s: -s[2])[:8]:
        print(f"  saddle {sid}: hamming-to-final {ham}/81, visited {vis}x")

    # spatial clustering of saddle visits: lag-1 correlation of the indicator
    ind = (labels >= 0).astype(float).reshape(RES, RES)
    def lagcorr(a, lag=1):
        x, y = a[:, :-lag].ravel(), a[:, lag:].ravel()
        return np.corrcoef(x, y)[0, 1]
    print(f"saddle-visit spatial autocorr (lag1): rows {lagcorr(ind):.3f}, "
          f"cols {lagcorr(ind.T):.3f}  (0 = spatially random)")

    # categorical map render (one color per visited saddle, void for none)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("turbo", max(len(saddle_registry), 1))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    fig.patch.set_facecolor("#050508"); ax.set_facecolor("#050508")
    masked = np.ma.masked_less(labels.reshape(RES, RES), 0)
    ax.imshow(masked, cmap=cmap, interpolation="nearest", origin="lower")
    ax.set_title(f"hard_a: first saddle visited (semantic label)  "
                 f"{n_visited}/{B} conds", fontsize=9, color="#c9d4e0", loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430")
    fig.savefig(OUT / "saddle_map_hard_a.png", facecolor="#050508",
                bbox_inches="tight")
    print("wrote", OUT / "saddle_map_hard_a.png")

    # ---- 2. settling predictability from early loops ----
    conv = (tr == final[:, None]).all(-1).all(-1)
    settling = S - conv.sum(1)
    # early features: how much the decode moved in loops 1-3 (no final used)
    d12 = np.array([hamming(tr[b, 1], tr[b, 0]) for b in range(B)])
    d23 = np.array([hamming(tr[b, 2], tr[b, 1]) for b in range(B)])
    early_move = d12 + d23
    # early FLI proxy: divergence from spatial neighbors by loop 3 — the
    # paper's lambda_F computed on decoded grids
    g3 = tr[:, 2].reshape(RES, RES, 9, 9)
    fli = np.zeros(B)
    for r in range(RES):
        for c in range(RES):
            b = r * RES + c
            dists = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < RES and 0 <= c2 < RES:
                    dists.append(hamming(g3[r, c], g3[r2, c2]))
            fli[b] = np.mean(dists)
    rho_fli, p_fli = spearmanr(fli, settling)
    rho_all, p_all = spearmanr(early_move, settling)
    slow = settling > 6
    rho_fli_slow = spearmanr(fli[slow], settling[slow])[0] if slow.sum() > 10 else float("nan")
    rho_slow = spearmanr(early_move[slow], settling[slow])[0] if slow.sum() > 10 else float("nan")
    print(f"\nsettling predictability from loops 1-3 (final not used):")
    print(f"  change-rate  Spearman rho (all): {rho_all:.3f} (p={p_all:.1e})")
    print(f"  change-rate  Spearman rho (slow tail, n={slow.sum()}): {rho_slow:.3f}")
    print(f"  early-FLI    Spearman rho (all): {rho_fli:.3f} (p={p_fli:.1e})")
    print(f"  early-FLI    Spearman rho (slow tail): {rho_fli_slow:.3f}")
    np.savez_compressed(OUT / "saddle_labels.npz", labels=labels,
                        settling=settling, early_move=early_move, fli=fli)


if __name__ == "__main__":
    sys.exit(main())
