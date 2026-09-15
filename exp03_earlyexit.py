#!/usr/bin/env python3
"""Exp 03 — early-exit probe optimization.

Most trajectories settle within ~3-7 loops but we pay for 24. This experiment:
1. verifies that chunked solving with latent-carry continuation is EXACTLY
   equivalent to the continuous 24-loop map (settling times + final grids),
2. measures the actual speedup.

Stopping rule: a condition is "settled" when its decoded grid is stable for 2
consecutive loops (verified safe: 0/2304 violations in exp02 traces).
"""
import time
import numpy as np
import torch

from loopscape import get_solver
from loopscape.utils import batched_apply
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from exp01_sudoku_slices import plane, mesh, to_traces

PUZZLE = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."
N = 256
CHUNK = 8
CAP = 24


def settling_from_traces(tr):
    conv = (tr == tr[:, -1:]).all(-1).all(-1)
    return tr.shape[1] - conv.sum(1)


def early_exit_solve(solver, puzzle, z0, cap=24, chunk=8, batch=4096):
    """Chunked solving with latent-carry continuation + early exit.

    Verified exactly equivalent to a continuous cap-loop run (exp03: settling
    100%, final grids 100%, on EqR sudoku hard_a). Returns per-condition
    settling times and final grids.
    """
    N = z0.shape[0]
    settling = np.empty(N, dtype=np.int16)
    finals = np.zeros((N, 9, 9), dtype=np.int8)
    for lo in range(0, N, batch):
        hi = min(lo + batch, N)
        zB = z0[lo:hi]
        n = hi - lo
        carry_H, carry_L = zB.clone(), zB * 0
        alive = np.arange(n)
        pieces = {k: [] for k in range(n)}
        start = 0
        while alive.size and start < cap:
            steps = min(chunk, cap - start)
            r = solver.solve_batch(puzzle, z_H=carry_H, z_L=carry_L, max_steps=steps,
                                   noise_scale=0.0, return_intermediates=True,
                                   return_latents=True)
            tr = to_traces(r).numpy()
            for k, i in enumerate(alive):
                pieces[i].append(tr[k])
            stable = ((tr[:, -1] == tr[:, -2]).all(-1).all(-1)
                      if steps > 1 else np.zeros(len(alive), bool))
            keep = ~stable
            hist_H = r.extra["z_H_history"]
            hist_L = r.extra["z_L_history"]
            if isinstance(hist_H, list):
                hist_H = torch.stack(hist_H)
                hist_L = torch.stack(hist_L)
            carry_H = hist_H[-1][torch.tensor(keep)].clone()
            carry_L = hist_L[-1][torch.tensor(keep)].clone()
            alive = alive[keep]
            start += steps
        for i in range(n):
            g = np.concatenate(pieces[i], axis=0)
            conv = (g == g[-1:]).all(-1).all(-1)
            settling[lo + i] = len(g) - int(conv.sum())
            finals[lo + i] = g[-1]
    return settling, finals


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    solver = get_solver("eqr", device=device)
    u, v = plane((97, 512), 0)
    pts = mesh(16)[:N]  # 16^2 = 256
    z0 = pts[:, 0, None, None] * u + pts[:, 1, None, None] * v

    # --- continuous reference ---
    t0 = time.time()
    r = solver.solve_batch(PUZZLE, z_H=z0, z_L=z0 * 0, max_steps=CAP,
                           noise_scale=0.0, return_intermediates=True)
    tr_c = to_traces(r)
    t_cont = settling_from_traces(tr_c.numpy())
    grids_c = tr_c[:, -1].numpy()
    dt_cont = time.time() - t0
    print(f"continuous: {dt_cont:.1f}s  loop-units {N*CAP}")

    # --- chunked with continuation ---
    t0 = time.time()
    carry_H, carry_L = z0.clone(), z0 * 0
    alive = np.arange(N)
    pieces = {i: [] for i in range(N)}                # per-condition decoded grids
    total_loops = 0
    chunk = 0
    while alive.size and chunk * CHUNK < CAP:
        chunk += 1
        base = (chunk - 1) * CHUNK
        steps = min(CHUNK, CAP - base)
        r = solver.solve_batch(PUZZLE, z_H=carry_H, z_L=carry_L, max_steps=steps,
                               noise_scale=0.0, return_intermediates=True,
                               return_latents=True)
        tr = to_traces(r).numpy()                       # (alive_B, steps, 9, 9)
        total_loops += alive.size * steps
        for k, i in enumerate(alive):
            pieces[i].append(tr[k])
        # drop conditions with a stable pair (grids never regress afterwards;
        # validated 0/2304 in exp02)
        stable = (tr[:, -1] == tr[:, -2]).all(-1).all(-1) if steps > 1 else np.zeros(len(alive), bool)
        keep = ~stable
        hist_H = r.extra["z_H_history"]
        hist_L = r.extra["z_L_history"]
        if isinstance(hist_H, list):                    # per-seed mode shape
            hist_H = torch.stack(hist_H)
            hist_L = torch.stack(hist_L)
        carry_H = hist_H[-1][torch.tensor(keep)].clone()
        carry_L = hist_L[-1][torch.tensor(keep)].clone()
        alive = alive[keep]
    for i in alive:  # capped: never stabilized
        pass
    # exact settling from each condition's own concatenated trajectory:
    # first loop of the final constant run (== continuous definition)
    settling = np.empty(N, dtype=int)
    finals = np.zeros((N, 9, 9), dtype=np.int8)
    for i in range(N):
        g = np.concatenate(pieces[i], axis=0)           # (loops_i, 9, 9)
        conv = (g == g[-1:]).all(-1).all(-1)
        settling[i] = len(g) - int(conv.sum())
        finals[i] = g[-1]
    dt_chunk = time.time() - t0

    # --- compare ---
    match_t = (settling == t_cont).mean()
    match_g = (finals == grids_c).all(-1).all(-1).mean()
    print(f"chunked:    {dt_chunk:.1f}s  loop-units {total_loops} "
          f"({total_loops/(N*CAP):.0%} of continuous)")
    print(f"equivalence: settling match {match_t:.1%}, final-grid match {match_g:.1%}")
    print(f"wall speedup: {dt_cont/dt_chunk:.2f}x")


if __name__ == "__main__":
    main()
