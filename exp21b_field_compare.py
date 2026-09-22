#!/usr/bin/env python3
"""exp21b — compare the pure-geometry model across the field stack:
Paris L8 / L14 / L26 + Tokyo L14. For each: harmonic model accuracy, balanced
accuracy, fast-fraction radial decay. Does the star sharpen with depth, and
does the polar structure replicate across prompts?
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp21"
OUT.mkdir(parents=True, exist_ok=True)

def analyze(tag):
    results = json.loads((ROOT / f"runs/exp20/response_field{tag}.json").read_text())
    RES = int(np.sqrt(len(results)))
    topics = sorted({r["topic"] for r in results})
    t2i = {t: i for i, t in enumerate(topics)}
    tmap = np.array([t2i[r["topic"]] for r in results]).reshape(RES, RES)
    c0 = (RES - 1) / 2
    yy, xx = np.mgrid[0:RES, 0:RES]
    rad = np.hypot(yy - c0, xx - c0)
    ang = np.arctan2(yy - c0, xx - c0)
    def design(ang, rad, K=3):
        f = []
        for k in range(1, K + 1):
            f += [np.cos(k * ang), np.sin(k * ang)]
        f += [rad, rad ** 2]
        return np.stack([f_i.ravel() for f_i in f], -1)
    Xf = design(ang, rad, 3)
    y = tmap.ravel()
    Xtr = np.column_stack([np.ones_like(y), Xf])
    lam = 1e-2
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]),
                        Xtr.T @ np.eye(len(y), dtype=float)[y])
    pred = np.argmax(Xtr @ W, axis=1)
    acc = (pred == y).mean()
    bal = np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])
    # radial decay of modal-agreement: how well does the CENTER (radius<5) topic
    # extend outward? proxy: correlation between radius and entropy of topics
    ent = []
    for r0 in range(0, int(c0), 8):
        m = (rad >= r0) & (rad < r0 + 8)
        cnt = Counter(tmap[m].ravel())
        ps = np.array([c for _, c in cnt.most_common()]) / m.sum()
        ent.append(-(ps * np.log(ps + 1e-12)).sum())
    return acc, bal, ent, len(topics)

rows = []
for tag in ["", "_paris_L8", "_paris_L26", "_tokyo_L14"]:
    acc, bal, ent, nt = analyze(tag)
    rows.append((tag or "paris_L14", acc, bal, ent))
    print(f"{tag or 'paris_L14':12s}: model acc {acc:.1%}, balanced {bal:.1%}, "
          f"radial entropy {['%.2f' % e for e in ent]}")

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
fig.patch.set_facecolor("#050508")
for tag, acc, bal, ent in rows:
    ax.plot(range(len(ent)), ent, "o-", lw=1.5,
            label=f"{tag} (harmonic-fit acc {acc:.0%})")
ax.set_xlabel("radius ring from base point (8px bands)", color="#c9d4e0")
ax.set_ylabel("topic entropy", color="#c9d4e0")
ax.set_title("radial entropy decay: how fast the field diversifies from the "
             "prompt's native point", fontsize=9, color="#c9d4e0", loc="left")
ax.legend(fontsize=7, facecolor="#0a0a12", labelcolor="#c9d4e0")
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors="#c9d4e0")
fig.savefig(ROOT / "runs/exp21/field_stack_comparison.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", ROOT / "runs/exp21/field_stack_comparison.png")
