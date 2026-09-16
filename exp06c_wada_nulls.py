#!/usr/bin/env python3
"""exp06c — Wada proxy vs DeepSeek-prescribed nulls (CPU, existing exp06 map).

q(r,k) = fraction of boundary pixels with >=k distinct labels in Chebyshev
radius r. Compare observed q against:
  N1  CSR label permutation       (floor; expected high with 1489 classes)
  N2  patch-permutation           (labels permuted among observed connected
       components: preserves patch shapes + class sizes, destroys label
       arrangement)  <-- the decisive null per DeepSeek v4-pro
  N3  random Voronoi              (same #patches, random cell labels)
If observed q does not exceed N2, the 98.1% is fragmentation, not Wada.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
d = np.load(ROOT / "runs/exp04/hard_a/fprm_slice_seed0.npz")
finals = d["finals"]
B = len(finals)
RES = int(np.sqrt(B))
lab = np.zeros((B,), dtype=np.int64)
# compress labels to contiguous ids
uniq, inv = np.unique(finals.reshape(B, -1), axis=0, return_inverse=True)
lab = inv.astype(np.int64)
grid = lab.reshape(RES, RES)
C = len(uniq)
print(f"map {RES}x{RES}, {C} classes")

def boundary_mask(g):
    b = np.zeros_like(g, dtype=bool)
    b[:-1, :] |= g[:-1, :] != g[1:, :]
    b[1:, :] |= g[:-1, :] != g[1:, :]
    b[:, :-1] |= g[:, :-1] != g[:, 1:]
    b[:, 1:] |= g[:, :-1] != g[:, 1:]
    return b

def q_wada(g, r=2, k=3):
    """fraction of boundary pixels with >=k distinct labels in Chebyshev-r box."""
    bm = boundary_mask(g)
    out = 0
    total = int(bm.sum())
    idx = np.argwhere(bm)
    for y, x in idx:
        y0, y1 = max(0, y - r), min(RES, y + r + 1)
        x0, x1 = max(0, x - r), min(RES, x + r + 1)
        if len(np.unique(g[y0:y1, x0:x1])) >= k:
            out += 1
    return out / max(total, 1), total

def connected_components(maskgrid):
    """label connected components (4-conn) of same-class pixels; return list of (class, pixels)."""
    from scipy import ndimage
    comps = []
    for c in np.unique(maskgrid):
        m = maskgrid == c
        cc, n = ndimage.label(m)
        for i in range(1, n + 1):
            px = np.argwhere(cc == i)
            if len(px):
                comps.append((c, px))
    return comps

rng = np.random.default_rng(0)

def patch_permutation(g, n_shuffle=20):
    """permute labels among connected components, preserving shapes and sizes."""
    comps = connected_components(g)
    q_list = []
    for _ in range(n_shuffle):
        labels = rng.permutation([c for c, _ in comps])
        g2 = g.copy()
        for (c, px), newlab in zip(comps, labels):
            ys, xs = px[:, 0], px[:, 1]
            g2[ys, xs] = newlab
        q_list.append(q_wada(g2)[0])
    return np.mean(q_list), np.std(q_list)

def voronoi_null(n_cells=None, n_shuffle=10):
    """random Voronoi: seed n_cells points, assign each pixel nearest seed,
    labels = permutation of classes over cells (approx via unique seeds)."""
    from scipy.spatial import cKDTree
    n_cells = n_cells or C
    q_list = []
    yy, xx = np.mgrid[0:RES, 0:RES]
    pts = np.stack([yy.ravel(), xx.ravel()], 1)
    for _ in range(n_shuffle):
        seeds = rng.integers(0, RES, size=(n_cells, 2))
        t = cKDTree(seeds)
        _, nearest = t.query(pts, workers=-1)
        g2 = nearest.reshape(RES, RES)
        # relabel cells by random distinct class labels (abundance ignored;
        # matched to observed #classes)
        perm = rng.permutation(np.arange(C))[:len(np.unique(g2))]
        relab = {c: p for c, p in zip(np.unique(g2), perm)}
        g2 = np.vectorize(relab.get)(g2)
        q_list.append(q_wada(g2)[0])
    return np.mean(q_list), np.std(q_list)

q_obs, n_bnd = q_wada(grid)
print(f"\nobserved q(r=2,k=3) = {q_obs:.4f}  (boundary pixels: {n_bnd}/{B} = {n_bnd/B:.1%})")

perm = rng.permutation(lab)
q_csr, _ = q_wada(perm.reshape(RES, RES))
print(f"N1 CSR permutation:            q = {q_csr:.4f}")

q_patch, sd_patch = patch_permutation(grid, n_shuffle=10)
print(f"N2 patch-permutation (n=10):   q = {q_patch:.4f} ± {sd_patch:.4f}   <-- decisive")

q_vor, sd_vor = voronoi_null()
print(f"N3 Voronoi ({C} cells):        q = {q_vor:.4f} ± {sd_vor:.4f}")

verdict = "WADA-LIKE (beats patch null)" if q_obs > q_patch + 2 * sd_patch else \
          "FRAGMENTATION ARTIFACT (does not beat patch null)"
print(f"\nverdict: {verdict}")
