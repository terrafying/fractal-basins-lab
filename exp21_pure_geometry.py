#!/usr/bin/env python3
"""exp21 — extract and model the response-field geometry as pure geometry.

From the Paris 32x32 topic map:
1. province masks (per topic, largest connected components)
2. boundary statistics per topic-pair: length, box-counting dimension,
   mean radius from the field center, angular span
3. pure model: the topic field as a function of angle+radius around the
   base point (native prompt position) — fit an angular Fourier +
   radial polynomial classifier; report how few terms reproduce the map
   (compressibility of the semantic geometry)
4. model-map vs actual-map comparison (visual + accuracy)
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp21"
OUT.mkdir(parents=True, exist_ok=True)

results = json.loads((ROOT / "runs/exp20/response_field.json").read_text())
RES = int(np.sqrt(len(results)))
topics = sorted({r["topic"] for r in results})
t2i = {t: i for i, t in enumerate(topics)}
tmap = np.array([t2i[r["topic"]] for r in results]).reshape(RES, RES)

# base point = native prompt position = grid center
c0 = (RES - 1) / 2
yy, xx = np.mgrid[0:RES, 0:RES]
rad = np.hypot(yy - c0, xx - c0)
ang = np.arctan2(yy - c0, xx - c0)

# ---- 1. province masks + largest components ----
masks = {}
for t in topics:
    m = tmap == t2i[t]
    cc, n = ndimage.label(m)
    if n == 0:
        continue
    sizes = ndimage.sum(m, cc, range(1, n + 1))
    biggest = np.argmax(sizes) + 1
    masks[t] = cc == biggest
masks = {t: m for t, m in masks.items() if m.sum() >= 12}
print("provinces (largest component >=12 px):",
      {t: int(m.sum()) for t, m in masks.items()})

# ---- 2. boundary statistics per topic pair ----
def boxcount_dim(mask_pair):
    """dimension of the boundary between two masks."""
    er = mask_pair[0] & ~ndimage.binary_erosion(mask_pair[0])
    bnd = er & mask_pair[1] & ~ndimage.binary_erosion(mask_pair[1])
    bnd |= mask_pair[0] & mask_pair[1]
    if bnd.sum() < 4:
        return None, bnd
    dims = []
    for k in range(1, 5):
        s = 2 ** k
        blk = bnd[::s, ::s]
        dims.append((1 / s, max(blk.sum(), 1)))
    xs = np.log([d[0] for d in dims]); ys = np.log([d[1] for d in dims])
    return np.polyfit(xs, ys, 1)[0], bnd

print("\nboundary statistics (topicA vs topicB): dim, boundary px, mean radius, angular span")
boundaries = []
for t1 in masks:
    for t2 in masks:
        if t1 >= t2:
            continue
        both = masks[t1] | masks[t2]
        # touching = dilate each by 1, overlap
        touch = ndimage.binary_dilation(masks[t1]) & masks[t2]
        if touch.sum() < 8:
            continue
        dim, _ = boxcount_dim((masks[t1], masks[t2]))
        r_touch = rad[touch].mean()
        a_touch = np.degrees(ang[touch])
        span = np.ptp(np.sort(a_touch)) if len(a_touch) > 1 else 0
        # angular span accounting for wraparound
        hist, edges = np.histogram(a_touch, bins=36, range=(-180, 180))
        occupied = np.where(hist > 0)[0]
        span = len(occupied) * 10
        print(f"  {t1[:18]} vs {t2[:18]}: D={dim if dim is None else round(dim,2)}, "
              f"touch px={int(touch.sum())}, mean r={r_touch:.0f}, ang span~{span}deg")
        boundaries.append((t1, t2, dim, int(touch.sum()), r_touch, span))

# ---- 3. pure model: angular Fourier + radial power fit ----
# features: for each pixel, [cos(k*ang), sin(k*ang) for k<=3] + [rad^j for j<=2]
def design(ang, rad, K=3):
    f = []
    for k in range(1, K + 1):
        f += [np.cos(k * ang), np.sin(k * ang)]
    f += [rad, rad ** 2]
    return np.stack([f_i.ravel() for f_i in f], -1)

Xf = design(ang, rad, K=3)
y = tmap.ravel()
Xtr = np.column_stack([np.ones_like(y), Xf])
# multinomial least squares via one-vs-all ridge
lam = 1e-2
W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]),
                    Xtr.T @ np.eye(len(y), dtype=float)[y])
pred_scores = Xtr @ W
pred = np.argmax(pred_scores, axis=1)
acc = (pred == y).mean()
# balanced accuracy
bal = np.mean([ (pred[y==c]==c).mean() for c in np.unique(y)])
print(f"\npure model: angular Fourier K=3 + radial^2 -> map accuracy {acc:.1%}, "
      f"balanced {bal:.1%} (chance ~{1/len(topics):.0%})")
model_map = pred.reshape(RES, RES)

# ---- 4. comparison figure ----
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), dpi=120)
fig.patch.set_facecolor("#050508")
cmap = plt.get_cmap("turbo", len(topics))
axes[0].imshow(tmap, cmap=cmap, interpolation="nearest", origin="lower",
               vmin=0, vmax=len(topics) - 1)
axes[0].set_title("actual topic field", fontsize=9, color="#c9d4e0", loc="left")
axes[1].imshow(model_map, cmap=cmap, interpolation="nearest", origin="lower",
               vmin=0, vmax=len(topics) - 1)
axes[1].set_title(f"pure model: angular Fourier(3) + radial^2\n"
                  f"accuracy {acc:.0%}, balanced {bal:.0%}", fontsize=9,
                  color="#c9d4e0", loc="left")
diff = (model_map != y.reshape(RES, RES))
axes[2].imshow(diff, cmap="gray", interpolation="nearest", origin="lower")
axes[2].set_title(f"disagreement ({diff.mean():.0%} of px)", fontsize=9,
                  color="#c9d4e0", loc="left")
for ax in axes:
    ax.set_facecolor("#050508"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430")
fig.savefig(OUT / "pure_geometry_model.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "pure_geometry_model.png")
json.dump(dict(boundaries=[(a, b, d, n, r, s) for a, b, d, n, r, s in boundaries],
               model_accuracy=float(acc), balanced=float(bal)),
          open(OUT / "pure_geometry.json", "w"), indent=1)
