#!/usr/bin/env python3
"""The figure that carries the paper's claim: basin metrics vs task difficulty.

Reads runs/exp01/summary.json, plots mean settling time, basin entropy Sb, and
uncertainty exponent alpha against the measured difficulty axis (MRV
backtracking guesses). One point per (puzzle, seed); puzzle families colored.
This replaces the drift/zoom videos as the primary visual: it shows the
difficulty dependence directly, which no zoom animation can.

usage: .venv/bin/python scripts/plot_difficulty.py [--out runs/figs/difficulty_axis.png]
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "runs" / "exp01" / "summary.json"

FAMILY_COLORS = {"easy_b": "#2a9d8f", "hard_a": "#e9c46a", "easy_a": "#e76f51"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "figs" / "difficulty_axis.png"))
    a = ap.parse_args()

    s = json.loads(SUMMARY.read_text())
    rows = []
    for pid, recs in s.get("puzzles", {}).items():
        for r in recs:
            if not r.get("valid_slice", True):
                continue
            rows.append((pid, r["backtracking_guesses"], r["seed"],
                         r["mean_settling"], r["basin_entropy"], r["alpha"]))
    if not rows:
        print("no valid slices in summary.json yet", file=sys.stderr)
        return 1
    rows.sort(key=lambda x: (x[1], x[0], x[2]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [(3, "mean settling time (loops)", (0, 24)),
               (4, "basin entropy $S_b$", None),
               (5, "uncertainty exponent $\\alpha$", (0, 1))]
    for ax, (idx, label, ylim) in zip(axes, metrics):
        for pid in sorted({r[0] for r in rows}):
            pts = [(r[1], r[idx]) for r in rows if r[0] == pid]
            ax.scatter(*zip(*pts), s=42, color=FAMILY_COLORS.get(pid, "gray"),
                       label=pid, zorder=3)
        # family means to guide the eye across the difficulty axis
        for pid in sorted({r[0] for r in rows}):
            pts = [(r[1], r[idx]) for r in rows if r[0] == pid]
            xs = [p[0] for p in pts]
            ax.scatter([sum(xs) / len(xs)], [sum(p[1] for p in pts) / len(pts)],
                       marker="_", s=600, color=FAMILY_COLORS.get(pid, "gray"),
                       zorder=2, linewidths=2.5)
        ax.set_xlabel("classical difficulty (MRV backtracking guesses)")
        ax.set_ylabel(label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    axes[0].legend(title="puzzle", loc="best", fontsize=8)
    n = len(rows)
    fig.suptitle(f"Fractal basins vs task difficulty - EqR sudoku, res 128, "
                 f"{n} valid slices", fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out, f"({n} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
