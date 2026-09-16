#!/usr/bin/env python3
"""Exp 04 — cross-model replication (FPRM vs EqR) on identical slices.

Same slice machinery (identical plane seeds => same z0 grid) on the FPRM
fixed-point reasoner: latent (97,512), fp_thresh=0.02, max_iters=200 (paper's
App. B settings). Comparing basin geometry between independently trained
architectures on the same puzzles tests whether basin landscapes are
task-structure (isomorphic across models) or architecture-specific.

Output: runs/exp04/<pid>/fprm_slice_seed<k>.npz (settling + finals, self-describing)
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

from loopscape import get_solver
from loopscape.utils import batched_apply
from loopscape.metrics import basin_entropy, uncertainty_exponent

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from exp01_sudoku_slices import plane, mesh, to_traces, PUZZLES, backtracking_guesses
from loopscape.puzzle import parse_puzzle

OUT = ROOT / "runs" / "exp04"
RES = int(os.environ.get("FB_RES", "64"))
SEEDS = [int(s) for s in os.environ.get("FB_SEEDS", "0,1").split(",")]
ONLY = set(filter(None, os.environ.get("FB_PUZZLES", "hard_a").split(",")))
FP_THRESH = float(os.environ.get("FB_TAU", "0.02"))
MAX_ITERS = int(os.environ.get("FB_ITERS", "200"))


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    solver = get_solver("fprm", device=device)
    records = []

    for pid, puzzle in PUZZLES.items():
        if ONLY and pid not in ONLY:
            continue
        grid = np.array(parse_puzzle(puzzle), dtype=np.int8)
        diff = backtracking_guesses(grid.copy())
        pdir = OUT / pid
        pdir.mkdir(exist_ok=True)

        for seed in SEEDS:
            out_npz = pdir / f"fprm_slice_seed{seed}.npz"
            if out_npz.exists():
                print(f"[skip] {out_npz.name} exists", flush=True)
                continue
            t0 = time.time()
            u, v = plane((97, 512), seed)   # identical planes to exp01/eqr
            z0 = mesh(RES)
            z0 = z0[:, 0, None, None] * u + z0[:, 1, None, None] * v
            torch.manual_seed(0)
            z0 = z0 + torch.randn(97, 512) * 0  # keep deterministic; scale via norm below
            z0 = z0 / z0.square().mean(-1, keepdim=True).add(1e-5).sqrt() * 500.0  # paper's FPRM scale

            def f(p):
                r = solver.solve_batch(puzzle, initial_latents=p, fp_thresh=FP_THRESH,
                                       max_iters=MAX_ITERS, stepsize=1.0,
                                       return_intermediates=True)
                return to_traces(r)
            from loopscape.utils import batched_apply
            traces = batched_apply(f, z0, batch_size=128, pad_dim=1)
            conv = (traces == traces[:, -1:]).all(-1).all(-1)
            times = (traces.shape[1] - conv.sum(1)).numpy().reshape(RES, RES).astype(np.int16)
            finals = traces[:, -1].numpy()

            vals, counts = np.unique(finals.reshape(RES**2, -1), axis=0, return_counts=True)
            frac_consensus = counts.max() / RES**2
            frac_true = int((finals == grid).all(-1).all(-1).sum()) / RES**2
            frac_capped = float((times == MAX_ITERS).mean())
            be = basin_entropy(times, box_size=5)
            ue = uncertainty_exponent(times, box_size=5)
            rec = dict(model="fprm", seed=seed, res=RES,
                       mean_settling=float(times.mean()),
                       basin_entropy=be["basin_entropy"],
                       boundary_entropy=be.get("boundary_basin_entropy"),
                       alpha=ue["uncertainty_exponent"],
                       frac_consensus=frac_consensus, frac_true_solved=frac_true,
                       frac_capped=frac_capped, fp_thresh=FP_THRESH,
                       backtracking_guesses=diff,
                       seconds=round(time.time() - t0, 1))
            np.savez_compressed(out_npz, settling=times, finals=finals, res=RES,
                                seed=seed, puzzle=puzzle, model="fprm",
                                backtracking_guesses=diff)
            print(json.dumps(rec), flush=True)
            records.append(rec)

    (OUT / "summary_fprm.json").write_text(json.dumps({"records": records}, indent=2))
    print("wrote", OUT / "summary_fprm.json")


if __name__ == "__main__":
    sys.exit(main())
