#!/usr/bin/env python3
"""Exp 06 — FPRM sanity check + identity-basin analysis (the multistability harvest).

Part A (GPU, small): native-init FPRM solves (no latent injection) on hard_a
and easy_b. If native solves succeed where injected starts never reach the
true solution, the injected-slice multistability is real basin structure
rather than an artifact of our initialization scale.

Part B (CPU, existing data): identity-basin map of the exp04 FPRM slice —
color each initial condition by WHICH final grid it reached (clustered by
Hamming distance), then an approximate Wada test: for boundary pixels
adjacent to basin A and B, how often is a third basin within 2 pixels?
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "exp06"
OUT.mkdir(parents=True, exist_ok=True)


def part_a():
    from loopscape import get_solver
    from exp01_sudoku_slices import PUZZLES
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    solver = get_solver("fprm", device=device)
    for pid in ("hard_a", "easy_b"):
        puzzle = PUZZLES.get(pid)
        if not puzzle:
            continue
        solved = 0
        N = 8
        for trial in range(N):
            r = solver.solve(puzzle, fp_thresh=0.02, max_iters=200, stepsize=1.0)
            solved += bool(r.solved)
        print(f"[native-init] {pid}: solved {solved}/{N}", flush=True)


def part_b():
    d = np.load(ROOT / "runs/exp04/hard_a/fprm_slice_seed0.npz")
    finals = d["finals"]                                  # (B, 9, 9)
    settling = d["settling"]
    B = len(finals)
    RES = int(np.sqrt(B))

    flat = finals.reshape(B, -1)
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    print(f"\n[identity] distinct final grids: {len(uniq)}")
    order = np.argsort(counts)[::-1]
    for k in order[:6]:
        print(f"  identity {k}: {counts[k]} conds ({counts[k]/B:.1%})")
    # cluster near-identical finals: anything within hamming<=2 of a bigger
    # cluster joins it (decode jitter vs genuine different answers)
    merged = {}
    reps = []
    assign = np.empty(B, dtype=int)
    for k in order:
        g = uniq[k].reshape(9, 9)
        cid = None
        for ri, rg in enumerate(reps):
            if int((g != rg).sum()) <= 2:
                cid = ri
                break
        if cid is None:
            cid = len(reps)
            reps.append(g)
        idx = np.where((flat == uniq[k]).all(1))[0]
        assign[idx] = cid
        merged[cid] = merged.get(cid, 0) + counts[k]
    print(f"[identity] clusters after hamming<=2 merge: {len(reps)}")
    for cid, n in sorted(merged.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  basin {cid}: {n} conds ({n/B:.1%})")

    # identity-basin map render
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = assign.reshape(RES, RES)
    nb = len(reps)
    cmap = plt.get_cmap("turbo", nb)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), dpi=130)
    for ax, (im, title) in zip(axes, [(lab, "final-answer identity"),
                                      (settling, "settling iter (0-200)")]):
        fig.patch.set_facecolor("#050508")
        if title.startswith("final"):
            ax.imshow(im, cmap=cmap, interpolation="nearest", origin="lower")
        else:
            ax.imshow(im, cmap="magma", interpolation="spline36", origin="lower")
        ax.set_title(f"FPRM hard_a 64^2: {title}", fontsize=9,
                     color="#c9d4e0", loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#1c2430")
    fig.savefig(OUT / "fprm_identity_basins.png", facecolor="#050508",
                bbox_inches="tight")
    print("wrote", OUT / "fprm_identity_basins.png")

    # approximate Wada test on the label map: boundary pixels (adjacent to a
    # different label) — how often does a 5x5 neighborhood contain >=3 labels?
    triple = 0
    boundary = 0
    for r in range(1, RES - 1):
        for c in range(1, RES - 1):
            l0 = lab[r, c]
            neigh = {lab[r + dr, c + dc]
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if (dr, dc) != (0, 0)}
            if any(x != l0 for x in neigh):
                boundary += 1
                win = lab[max(0, r - 2):r + 3, max(0, c - 2):c + 3]
                if len(set(win.ravel().tolist())) >= 3:
                    triple += 1
    print(f"[wada-approx] boundary px: {boundary}; with >=3 basins in 5x5: "
          f"{triple} ({triple/max(boundary,1):.1%})")


if __name__ == "__main__":
    part_a()
    part_b()
