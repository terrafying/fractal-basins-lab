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


def early_exit_solve(solver, puzzle, z0, cap=24, chunk=8, batch=512,
                     progress_path=None):
    """Chunked solving with latent-carry continuation + early exit.

    Verified exactly equivalent to a continuous cap-loop run (exp03: settling
    100%, final grids 100%, on EqR sudoku hard_a). Returns per-condition
    settling times and final grids.

    Resumable: with progress_path set, per-chunk state (carries, alive set,
    accumulated decoded grids) is saved after every chunk and reloaded on a
    later call matching the same puzzle+z0 fingerprint. A killed job loses at
    most one chunk; the state file is a plain portable npz.
    """
    import hashlib
    N = z0.shape[0]
    fp = hashlib.sha1()
    fp.update(puzzle.encode())
    fp.update(z0.numpy().tobytes())
    fingerprint = fp.hexdigest()[:16]

    settling = np.empty(N, dtype=np.int16)
    finals = np.zeros((N, 9, 9), dtype=np.int8)
    start = 0
    alive = np.arange(N)
    carry_H, carry_L = z0.clone(), z0 * 0
    pieces = {k: [] for k in range(N)}
    resumed = False

    if progress_path and Path(progress_path).exists():
        try:
            st = np.load(progress_path, allow_pickle=True)
            if str(st["fingerprint"]) == fingerprint and int(st["cap"]) == cap:
                settling = st["settling"]
                finals = st["finals"]
                start = int(st["start"])
                alive = st["alive"]
                carry_H = torch.from_numpy(st["carry_H"])
                carry_L = torch.from_numpy(st["carry_L"])
                grids = st["pieces"]  # (alive_n, loops_so_far, 9, 9) int8
                for k, i in enumerate(alive):
                    pieces[i] = [g for g in grids[k]]
                resumed = True
                print(f"[resume] {Path(progress_path).name} at loop {start}/"
                      f"{cap}, {len(alive)} alive", flush=True)
        except (KeyError, ValueError) as e:
            print(f"[progress-stale] {e} - starting fresh", flush=True)
    if progress_path and not resumed:
        Path(progress_path).unlink(missing_ok=True)

    total_loops = 0
    while alive.size and start < cap:
        steps = min(chunk, cap - start)
        r = solver.solve_batch(puzzle, z_H=carry_H, z_L=carry_L, max_steps=steps,
                               noise_scale=0.0, return_intermediates=True,
                               return_latents=True)
        tr = to_traces(r).numpy()
        total_loops += alive.size * steps
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
        if progress_path:
            alive_grids = (np.stack([np.concatenate(pieces[i], axis=0)
                                     for i in alive]) if alive.size else
                           np.zeros((0, 0, 9, 9), dtype=np.int8))
            np.savez_compressed(progress_path,
                                fingerprint=fingerprint, cap=cap, start=start,
                                alive=alive, carry_H=carry_H.numpy(),
                                carry_L=carry_L.numpy(), settling=settling,
                                finals=finals, pieces=alive_grids.astype(np.int8))

    for i in range(N):
        g = np.concatenate(pieces[i], axis=0) if pieces[i] else \
            np.zeros((1, 9, 9), dtype=np.int8)
        conv = (g == g[-1:]).all(-1).all(-1)
        settling[i] = len(g) - int(conv.sum())
        finals[i] = g[-1]
    if progress_path:
        Path(progress_path).unlink(missing_ok=True)  # complete: state not needed
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
