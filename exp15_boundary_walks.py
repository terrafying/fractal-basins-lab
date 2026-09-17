#!/usr/bin/env python3
"""exp15 — boundary-tracing walks on the FPRM atlas (the road network).

Method (DeepSeek-guided):
- Pick adjacent basin pairs from the 64^2 identity map.
- Bisect along the segment between a point in each basin to find a boundary
  point (oracle = live FPRM run from that (a,b) slice coordinate).
- Predictor-corrector walk: step tangentially, bisect back to the boundary.
- Record: curve, adjacent answers, hamming between them; junction detection
  (does a third basin appear nearby?); overlay on the 64^2 map.

This converts the static atlas into a mapped road network: fault lines where
the model's answer changes, with their semantic edges.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "exp15"
OUT.mkdir(parents=True, exist_ok=True)
from loopscape import get_solver
from exp01_sudoku_slices import plane, PUZZLES
from loopscape.puzzle import parse_puzzle

RES = 200
LAYER_SCALE = 24.0 * 500.0 / 3.0  # exp04 scaling: z0 normalized to rms 500, dirs = plane cols * 500
device = "mps"

d = np.load(ROOT / "runs/exp04/hard_a/fprm_slice_seed0.npz")
finals = d["finals"].reshape(RES, RES, 9, 9)
settling = d["settling"].reshape(RES, RES)
puzzle = PUZZLES["hard_a"]

solver = get_solver("fprm", device=device)

def oracle(a, b):
    """run FPRM from slice point (a,b); return (final grid, settling)."""
    u, v = plane((97, 512), 0)
    z = (a * u + b * v)
    z = z / z.square().mean().add(1e-5).sqrt() * 500.0
    r = solver.solve_batch(puzzle, initial_latents=z[None], fp_thresh=0.02,
                           max_iters=200, stepsize=1.0)
    g = np.array(r.grids[0], dtype=np.int8)
    return g, r.steps

# oracle consistency check against the 64^2 map at a few known points
# (map coords i,j correspond to a,b on the exp04 grid: linspace(-1,1,64))
lin = np.linspace(-1, 1, RES)
match = 0
trials = 6
rng = np.random.default_rng(3)
test_pts = [(int(rng.integers(RES)), int(rng.integers(RES))) for _ in range(trials)]
for i, j in test_pts:
    g_o, _ = oracle(lin[j], lin[i])          # careful: chart[i,j] = a=lin[i](row), b=lin[j](col)? exp04 used mesh rows=a
    g_m = finals[i, j]
    match += int((g_o == g_m).all())
print(f"oracle consistency: {match}/{trials} match the 64^2 map", flush=True)


def find_boundary(p1, p2, iters=14):
    """bisect between two chart points until answers differ; return boundary pt."""
    g1 = finals[p1[0], p1[1]]
    g2 = finals[p2[0], p2[1]]
    lo, hi = np.array(p1, float), np.array(p2, float)
    if (g1 == g2).all():
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        i, j = int(mid[0]), int(mid[1])
        g = finals[i, j]
        if (g == g1).all():
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tangent_walk(start, direction, steps=30, step_len=2.0):
    """walk along the boundary starting near 'start', heading 'direction'."""
    cur = np.array(start, float)
    dirv = np.array(direction, float)
    dirv /= np.linalg.norm(dirv)
    path = [cur.copy()]
    answers = []
    g_ref = None
    for _ in range(steps):
        probe = cur + dirv * step_len
        if not (0 <= probe[0] < RES and 0 <= probe[1] < RES):
            break
        # find the boundary near probe: bisect between cur and probe answers
        i, j = int(probe[0]), int(probe[1])
        g_probe = finals[i, j]
        i0, j0 = int(cur[0]), int(cur[1])
        g_cur = finals[i0, j0]
        if (g_probe == g_cur).all():
            # stepped off the boundary into a third basin or same; try small kicks
            found = False
            for kick in (0.5, 1.0, -0.5, -1.0):
                perp = np.array([-dirv[1], dirv[0]]) * kick
                pt = probe + perp
                if 0 <= pt[0] < RES and 0 <= pt[1] < RES:
                    if not (finals[int(pt[0]), int(pt[1])] == g_cur).all():
                        probe = pt
                        g_probe = finals[int(pt[0]), int(pt[1])]
                        found = True
                        break
            if not found:
                break
        bpt = find_boundary(tuple(map(int, np.floor(cur))), tuple(map(int, np.floor(probe))))
        if bpt is None:
            break
        cur = bpt
        path.append(cur.copy())
        # rotate heading slightly toward the local boundary direction
        dirv = (path[-1] - path[-2])
        n = np.linalg.norm(dirv)
        if n > 0:
            dirv = dirv / n
        else:
            break
        answers.append((g_cur.copy(), g_probe.copy()))
    return path


# pick 3 adjacent basin pairs from the map: dominant class vs its neighbors
flat = finals.reshape(-1, 9, 9)
uniq, counts = np.unique(flat.reshape(len(flat), -1), axis=0, return_counts=True)
maj_grid = uniq[np.argmax(counts)].reshape(9, 9)
maj_mask = (finals == maj_grid).all(-1).all(-1)
print(f"majority basin: {maj_mask.sum()} px", flush=True)

# find boundary seed: a majority pixel adjacent to a different-class pixel
seeds = []
done_pairs = set()
for i in range(1, RES - 1):
    for j in range(RES - 1):
        if maj_mask[i, j]:
            for di, dj in ((0, 1), (1, 0)):
                g_n = finals[i + di, j + dj]
                if not (g_n == maj_grid).all():
                    key = tuple(sorted([0, 1]))  # dummy pair id
                    seeds.append(((i, j), (i + di, j + dj)))
                    break
        if len(seeds) >= 3:
            break
    if len(seeds) >= 3:
        break
print(f"boundary seeds: {seeds}", flush=True)

paths = []
t0 = time.time()
for k, (p_maj, p_other) in enumerate(seeds):
    bpt = find_boundary(p_maj, p_other)
    if bpt is None:
        continue
    for direction in ((1.0, 0.3), (-1.0, -0.7)):
        path = tangent_walk(bpt, direction, steps=40)
        paths.append(dict(seed_pair=k, path=[list(map(float, p)) for p in path],
                          n=len(path)))
        print(f"walk {k}.{direction}: {len(path)} boundary points ({time.time()-t0:.0f}s)",
              flush=True)

# ---- figure: map + traced roads ----
fig, ax = plt.subplots(figsize=(9, 9), dpi=130)
fig.patch.set_facecolor("#050508")
ax.imshow(settling, cmap="magma", interpolation="nearest", origin="lower", alpha=0.85)
for pi, p in enumerate(paths):
    pts = np.array(p["path"])
    ax.plot(pts[:, 1], pts[:, 0], "-", lw=1.6, color="#2fd8e8", alpha=0.9)
    ax.scatter(pts[0, 1], pts[0, 0], color="#2fd8e8", s=30, zorder=5)
ax.set_title("FPRM hard_a 64^2: traced semantic boundaries (cyan) over settling field",
             fontsize=9, color="#c9d4e0", loc="left")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#1c2430")
fig.savefig(OUT / "boundary_roads.png", facecolor="#050508", bbox_inches="tight")
json.dump(paths, open(OUT / "traced_boundaries.json", "w"))
print("wrote", OUT / "boundary_roads.png")
