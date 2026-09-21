#!/usr/bin/env python3
"""exp17 — the decision star: hub geometry + semantic wedge signatures.

The 200^2 FPRM settling field showed a radial hub-and-wedge structure. Now
measure it and ask the semantic question: are the wedges SEMANTICALLY
DISTINCT, i.e. do wrong answers in different angular sectors differ in
different CELLS (different constraint violations), or are all sectors just
random noise around the truth?

1. Locate the hub: the point that best explains the radial structure
   (scan candidate centers; score = how well settling class is predicted
   by angle alone, i.e. wedges are angular sectors).
2. Wedge census at radius r: angular intervals of fast/slow, widths, count.
3. Semantic wedge signatures: per angular sector, the mean deviation pattern
   of wrong answers (which cells differ from truth); correlation between
   sector and cell-signature (chi2 / permutation test).
"""
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp17"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(ROOT / "runs/exp04/hard_a/fprm_slice_seed0.npz")
finals = d["finals"].reshape(200, 200, 9, 9)
settling = d["settling"].reshape(200, 200)
RES = 200

# true solution
import re
PUZZLE = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."
g = np.array([int(ch) if ch.isdigit() else 0 for ch in PUZZLE]).reshape(9, 9)
sols = []
def rec(pos=0):
    if len(sols) >= 1: return
    while pos < 81 and g[pos//9, pos%9] != 0: pos += 1
    if pos == 81: sols.append(g.copy()); return
    r, c = pos//9, pos%9
    for v in range(1, 10):
        used = set(g[r]) | set(g[:, c]) | set(g[3*(r//3):3*(r//3)+3, 3*(c//3):3*(c//3)+3].ravel())
        if v not in used:
            g[r, c] = v; rec(pos+1); g[r, c] = 0
rec()
truth = sols[0]
correct = (finals == truth).all(-1).all(-1)
print(f"correct share: {correct.mean():.3f}", flush=True)

# ---- 1. locate hub by angle-predictability scan ----
yy, xx = np.mgrid[0:RES, 0:RES]
best = None
for cy in range(60, 160, 10):
    for cx in range(50, 150, 10):
        ang = np.arctan2(yy - cy, xx - cx)
        rad = np.hypot(yy - cy, xx - cx)
        m = (rad > 30) & (rad < 95)
        # score: mutual information between angular bin and fast/slow class
        angbin = ((ang[m] + np.pi) / (2 * np.pi) * 24).astype(int) % 24
        slow = (settling[m] > np.quantile(settling[m], 0.5)).astype(int)
        # MI via contingency
        score = 0.0
        for b in range(24):
            sel = angbin == b
            if sel.sum() < 10: continue
            p1 = slow[sel].mean()
            ptot = slow.mean()
            # KL-ish contribution
            if 0 < p1 < 1:
                score += sel.sum() / len(slow) * abs(p1 - ptot)
        if best is None or score > best[0]:
            best = (score, cy, cx)
print(f"hub estimate: row {best[1]}, col {best[2]} (wedge score {best[0]:.3f})",
      flush=True)
cy, cx = best[1], best[2]

# ---- 2. wedge census vs radius ----
ang = np.arctan2(yy - cy, xx - cx)
rad = np.hypot(yy - cy, xx - cx)
fast = settling < np.quantile(settling, 0.33)
wedge_counts = []
for r0 in range(30, 100, 10):
    m = (rad >= r0) & (rad < r0 + 10)
    a = ang[m]; f = fast[m]
    # count angular transitions between fast/slow going around the circle
    order = np.argsort(a)
    seq = f[order]
    trans = int((seq[1:] != seq[:-1]).sum()) + int((seq[0] != seq[-1]))
    wedge_counts.append((r0, trans, f.mean()))
print("radius -> wedge transitions (fast fraction):")
for r0, t, ff in wedge_counts:
    print(f"  r={r0}-{r0+10}: {t} wedges, fast {ff:.0%}")

# ---- 3. semantic wedge signatures ----
# deviate pattern per condition: which cells differ from truth
dev = (finals != truth).astype(np.int8)          # (200,200,9,9) 0/1
m_ring = (rad >= 40) & (rad < 95)
sector_count = 12
sec_dev = []
for s in range(sector_count):
    lo, hi = -np.pi + s * 2 * np.pi / sector_count, -np.pi + (s + 1) * 2 * np.pi / sector_count
    m = m_ring & (ang >= lo) & (ang < hi) & (~correct)
    if m.sum() < 30:
        sec_dev.append(None); continue
    sec_dev.append(dev[m].mean(0))               # mean deviation pattern (9,9)
sec_dev_valid = [s for s in sec_dev if s is not None]
print(f"sectors with enough wrong answers: {len(sec_dev_valid)}/12", flush=True)

# pairwise correlation between sector signatures (same cells wrong?)
sims = []
for i in range(len(sec_dev_valid)):
    for j in range(i + 1, len(sec_dev_valid)):
        a, b = sec_dev_valid[i].ravel(), sec_dev_valid[j].ravel()
        if a.std() > 0 and b.std() > 0:
            sims.append(np.corrcoef(a, b)[0, 1])
sims = np.array(sims)
print(f"cross-sector signature correlation: mean {sims.mean():.3f} "
      f"(0 = sectors fail in different cells, 1 = all sectors fail the same way)")

# null: shuffle sector assignment within ring
rng = np.random.default_rng(0)
null = []
m_wrong = m_ring & (~correct)
wrong_ang = ang[m_wrong]; wrong_dev = dev[m_wrong]
for _ in range(30):
    sec_n = ((wrong_ang + np.pi) / (2 * np.pi) * sector_count).astype(int) % sector_count
    rng.shuffle(sec_n)
    sd = []
    for s in range(sector_count):
        sel = sec_n == s
        if sel.sum() >= 30:
            sd.append(wrong_dev[sel].mean(0))
    if len(sd) < 2: continue
    for i in range(len(sd)):
        for j in range(i + 1, len(sd)):
            a, b = sd[i].ravel(), sd[j].ravel()
            if a.std() > 0 and b.std() > 0:
                null.append(np.corrcoef(a, b)[0, 1])
null = np.array(null)
print(f"null (shuffled sectors): mean {null.mean():.3f} +- {null.std():.3f}")
z = (sims.mean() - null.mean()) / max(null.std(), 1e-9)
print(f"z = {z:.1f} -> sectors are "
      f"{'SEMANTICALLY DISTINCT (different cells wrong per wedge)' if z < -3 else 'SEMANTICALLY ALIKE' if z > 3 else 'not distinguishable from shuffled'}")

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), dpi=120)
fig.patch.set_facecolor("#050508")
im = axes[0].imshow(settling, cmap="magma", interpolation="nearest", origin="lower")
axes[0].scatter([cx], [cy], color="#7fdc9f", s=60, marker="*", zorder=5)
th = np.linspace(-np.pi, np.pi, 100)
for rr in (40, 60, 80):
    axes[0].plot(cx + rr * np.cos(th), cy + rr * np.sin(th), "--", color="#2fd8e8",
                 lw=0.8, alpha=0.6)
axes[0].set_title(f"hub at (row {cy}, col {cx}) — star centered on it",
                  fontsize=9, color="#c9d4e0", loc="left")
im2 = axes[1].imshow(np.mean([s for s in sec_dev_valid if s is not None], 0)
                     if sec_dev_valid else np.zeros((9, 9)), cmap="hot",
                     interpolation="nearest")
axes[1].set_title("mean deviation cells across sectors (wrongness anatomy)",
                  fontsize=9, color="#c9d4e0", loc="left")
for ax in axes:
    ax.set_facecolor("#050508"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430")
fig.savefig(OUT / "decision_star.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "decision_star.png")
np.savez_compressed(OUT / "hub_wedges.npz", cy=cy, cx=cx, sims=sims, null=null)
