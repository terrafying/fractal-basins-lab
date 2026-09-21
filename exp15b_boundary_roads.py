#!/usr/bin/env python3
"""exp15b — boundary-tracing walks, MAP-ONLY (no live oracle).

Uses the 200^2 FPRM identity map as its own oracle. Walk algorithm:
- find a majority-basin boundary seed (majority px adjacent to other class)
- at each step: propose next point along current heading in a small fan of
  angles; pick the candidate whose class differs from current class but keeps
  the OTHER class within 2 px (stays on the boundary); update heading.
- record path, adjacent class pair, per-step hamming between the answers.

Deliverable: traced boundary curves over the settling field, plus per-walk
statistics (length, class changes, hamming profile along the road).
"""
import json, sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp15"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(ROOT / "runs/exp04/hard_a/fprm_slice_seed0.npz")
finals = d["finals"].reshape(200, 200, 9, 9)
settling = d["settling"].reshape(200, 200)
RES = 200

flat = finals.reshape(-1, 9, 9)
uniq, counts = np.unique(flat.reshape(len(flat), -1), axis=0, return_counts=True)
maj_grid = uniq[np.argmax(counts)].reshape(9, 9)
maj = (finals == maj_grid).all(-1).all(-1)   # bool map: majority basin

def ham(a, b):
    return int((a != b).sum())

def class_at(p):
    i, j = int(round(p[0])), int(round(p[1]))
    if not (0 <= i < RES and 0 <= j < RES):
        return None, None
    return finals[i, j], (i, j)

def walk(start, heading, n_steps=200, step=2.0):
    """walk with heading correction to stay on the majority/non-majority edge."""
    p = np.array(start, float)
    h = np.array(heading, float)
    h /= np.linalg.norm(h)
    path = [p.copy()]
    pairs = []
    side_ref = None  # the non-majority class we started against
    for _ in range(n_steps):
        best = None
        # try fan of headings
        for dth in np.linspace(-0.9, 0.9, 13):
            h_try = np.array([np.cos(dth), np.sin(dth)]) @ np.array([[h[0], h[1]], [-h[1], h[0]]]) if False else None
        for dth in np.linspace(-0.9, 0.9, 13):
            rot = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]])
            h_try = rot @ h
            cand = p + h_try * step
            i, j = int(round(cand[0])), int(round(cand[1]))
            if not (1 <= i < RES - 1 and 1 <= j < RES - 1):
                continue
            is_maj = maj[i, j]
            # want: we are ON the edge -> candidate flips side vs current point side,
            # OR stays with the other class present nearby
            cur_is_maj = maj[int(round(p[0])), int(round(p[1]))]
            # check the OTHER class within 2px on the non-current side
            other_close = False
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    ii, jj = i + dr, j + dc
                    if 0 <= ii < RES and 0 <= jj < RES:
                        if is_maj and not maj[ii, jj]:
                            other_close = True
                        if (not is_maj) and maj[ii, jj]:
                            other_close = True
            if not other_close:
                continue
            score = abs(dth)  # prefer small rotations
            if best is None or score < best[0]:
                best = (score, cand, h_try)
        if best is None:
            break
        _, p, h = best
        path.append(p.copy())
        g_a, g_b = finals[int(round(p[0])), int(round(p[1]))], None
        i, j = int(round(p[0])), int(round(p[1]))
        # the two classes at this boundary point: majority answer + nearest other
        others = []
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                ii, jj = i + dr, j + dc
                if 0 <= ii < RES and 0 <= jj < RES and not maj[ii, jj]:
                    others.append(finals[ii, jj])
        if others:
            pairs.append((maj_grid.copy(), others[0]))
    return path, pairs

# seeds: majority pixels with a non-majority neighbor, spread over the map
seeds = []
done = set()
for i in range(5, RES - 5, 7):
    for j in range(5, RES - 5, 7):
        if maj[i, j] and any(not maj[i + di, j + dj]
                             for di in (-1, 1) for dj in (-1, 1)):
            key = (i // 40, j // 40)
            if key not in done:
                seeds.append(((i, j), (1.0, 0.4)))
                done.add(key)
        if len(seeds) >= 6:
            break
    if len(seeds) >= 6:
        break
print(f"seeds: {seeds}", flush=True)

paths = []
for si, (start, heading) in enumerate(seeds):
    # ensure start is on the edge: nudge to nearest edge point
    path, pairs = walk(start, heading)
    paths.append(dict(seed=si, start=list(map(int, start)),
                      path=[list(map(float, p)) for p in path],
                      n=len(path)))
    print(f"walk {si}: {len(path)} points", flush=True)

# ---- figure ----
fig, ax = plt.subplots(figsize=(9, 9), dpi=130)
fig.patch.set_facecolor("#050508")
ax.imshow(settling, cmap="magma", interpolation="nearest", origin="lower", alpha=0.9)
for pi, p in enumerate(paths):
    pts = np.array(p["path"])
    ax.plot(pts[:, 1], pts[:, 0], "-", lw=1.8, color="#2fd8e8", alpha=0.9)
    ax.scatter(pts[0, 1], pts[0, 0], color="#7fdc9f", s=36, zorder=5, marker="s")
ax.set_title("FPRM hard_a 200^2: boundary roads (cyan) over settling field\n"
             "green squares = walk starts", fontsize=9, color="#c9d4e0", loc="left")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#1c2430")
fig.savefig(OUT / "boundary_roads_maponly.png", facecolor="#050508",
            bbox_inches="tight")
json.dump(paths, open(OUT / "roads_maponly.json", "w"))
print("wrote", OUT / "boundary_roads_maponly.png")

# stats: settling along roads vs map average
road_settling = []
for p in paths:
    pts = np.array(p["path"])
    for y, x in pts:
        i, j = int(round(y)), int(round(x))
        if 0 <= i < RES and 0 <= j < RES:
            road_settling.append(settling[i, j])
print(f"settling along roads: mean {np.mean(road_settling):.1f} vs map mean "
      f"{settling.mean():.1f} (roads should sit on high-settling ridges if "
      f"temporal and semantic boundaries align)")
