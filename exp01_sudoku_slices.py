#!/usr/bin/env python3
"""Exp 01 — basin maps across puzzle difficulty (Buehler-methodology transplant).

Protocol (Fractal basins trap latent reasoning, Lai et al. 2026, App. B):
- EqR (deterministic map: noise_scale=0, max_steps=24), puzzle + schedule fixed.
- Vary only the initial latent z_H over a random orthonormal 2-slice of latent
  space (QR of a Gaussian matrix), z(a,b) = a*u + b*v on a [-1,1]^2 grid.
- Record decoded grid every loop; settling time = loops until decoded output
  stops changing. Slice = basin map.
- Metrics: basin entropy Sb, boundary entropy, uncertainty exponent alpha
  (loopscape.metrics), plus paper's exclusion rules (<90% solved, >1% cap).

Output: runs/exp01/<puzzle_id>/slice_seed<k>.npz  (settling field + traces summary)
        runs/exp01/summary.json
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

from loopscape import get_solver
from loopscape.utils import batched_apply
from loopscape.metrics import basin_entropy, uncertainty_exponent
from loopscape.puzzle import parse_puzzle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp01"
RES = int(__import__("os").environ.get("FB_RES", "200"))  # slice resolution
SEEDS = [int(s) for s in __import__("os").environ.get("FB_SEEDS", "0,1,2").split(",")]
ONLY = set(filter(None, __import__("os").environ.get("FB_PUZZLES", "").split(",")))
MAX_STEPS = 24

# Two puzzles with a difficulty gap (both solvable; verified via tdoku-free
# MRV backtracking: easy_a has 0 guesses, hard_a requires deep backtracking).
PUZZLES = {
    "easy_a": "..4.6...16..3...8.....24.....9..2..4.5.......4...3.26.....8...........172....384.",
    "hard_a": ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2....",
}


def plane(shape, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(int(np.prod(shape)), 2, generator=g))
    return Q[:, 0].reshape(shape), Q[:, 1].reshape(shape)


def mesh(res):
    g = torch.linspace(-1, 1, res)
    x, y = torch.meshgrid(g, g, indexing="xy")
    return torch.stack([x.ravel(), y.ravel()], -1)


def to_traces(result):
    # [steps][B] decoded grids -> (B, steps, 9, 9) int8
    return torch.swapaxes(torch.tensor(np.asarray(result.intermediates), dtype=torch.int8), 0, 1)


def settling_times(traces):
    """Loops until decoded output stops changing (paper definition)."""
    conv = (traces == traces[:, -1:]).all(-1).all(-1)  # (B, steps) bool
    return (conv.shape[1] - conv.sum(1)).numpy()        # first stable loop index


def backtracking_guesses(grid):
    """Classical-solver backtracking count (MRV heuristic) == difficulty axis."""
    g = grid.copy()
    guesses = 0
    def candidates(r, c):
        used = set(g[r]) | {g[i][c] for i in range(9)}
        br, bc = 3 * (r // 3), 3 * (c // 3)
        used |= {g[br + i][bc + j] for i in range(3) for j in range(3)}
        return [v for v in range(1, 10) if v not in used]
    def solve():
        nonlocal guesses
        best = None
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    n = len(candidates(r, c))
                    if n == 0:
                        return False
                    if best is None or n < len(best[2]):
                        best = (r, c, candidates(r, c))
        if best is None:
            return True
        r, c, cands = best
        if len(cands) > 1:
            guesses += 1  # genuine guess: more than one candidate for the MRV cell
        for v in cands:
            g[r][c] = v
            if solve():
                return True
            g[r][c] = 0
        return False
    return guesses if solve() else -1


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    solver = get_solver("eqr", device=device)
    summary = {"protocol": {"model": "eqr", "task": "sudoku", "res": RES,
                            "max_steps": MAX_STEPS, "noise_scale": 0.0,
                            "device": device}, "puzzles": {}}

    for pid, puzzle in PUZZLES.items():
        if ONLY and pid not in ONLY:
            continue
        grid = np.array(parse_puzzle(puzzle), dtype=np.int8)
        diff = backtracking_guesses(grid.copy())
        pdir = OUT / pid
        pdir.mkdir(exist_ok=True)
        print(f"\n=== {pid}  givens={int((grid>0).sum())}  backtracking_guesses={diff} ===", flush=True)

        for seed in SEEDS:
            out_npz = pdir / f"slice_seed{seed}.npz"
            if out_npz.exists():
                print(f"[skip] {out_npz.name} exists", flush=True)
                continue
            t0 = time.time()
            u, v = plane((97, 512), seed)
            z_all = mesh(RES)
            z_all = z_all[:, 0, None, None] * u + z_all[:, 1, None, None] * v
            if os.environ.get("FB_NO_EARLYEXIT"):
                def f(p):
                    r = solver.solve_batch(puzzle, z_H=p, z_L=p * 0, max_steps=MAX_STEPS,
                                           noise_scale=0.0, return_intermediates=True)
                    return to_traces(r)
                traces = batched_apply(f, z_all, batch_size=200, pad_dim=1)
                times = settling_times(traces).reshape(RES, RES).astype(np.int16)
                finals_np = traces[:, -1].numpy()
            else:
                from exp03_earlyexit import early_exit_solve
                times_1d, finals_np = early_exit_solve(solver, puzzle, z_all, cap=MAX_STEPS)
                times = times_1d.reshape(RES, RES)

            solved = int((finals_np == grid).all(-1).all(-1).sum())
            frac_true_solved = solved / RES**2
            # paper's exclusion rule: <90% of conditions reach "the model's
            # final solution" — i.e. self-consistency (no multistability),
            # NOT the ground-truth puzzle solution.
            vals, counts = np.unique(finals_np.reshape(RES**2, -1), axis=0,
                                     return_counts=True)
            frac_consensus = counts.max() / RES**2
            frac_solved = frac_consensus
            frac_capped = float((times == MAX_STEPS).mean())
            be = basin_entropy(times, box_size=5)
            ue = uncertainty_exponent(times, box_size=5)
            valid = frac_solved >= 0.90 and frac_capped <= 0.01
            np.savez_compressed(pdir / f"slice_seed{seed}.npz",
                                settling=times, seed=seed, puzzle=puzzle)
            rec = dict(seed=seed, mean_settling=float(times.mean()),
                       basin_entropy=be["basin_entropy"],
                       boundary_entropy=be.get("boundary_basin_entropy"),
                       alpha=ue["uncertainty_exponent"],
                       frac_consensus=frac_consensus,
                       frac_true_solved=frac_true_solved,
                       frac_capped=frac_capped,
                       valid_slice=bool(valid), backtracking_guesses=diff,
                       givens=int((grid > 0).sum()), seconds=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            summary["puzzles"].setdefault(pid, []).append(rec)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote", OUT / "summary.json")


if __name__ == "__main__":
    sys.exit(main())
