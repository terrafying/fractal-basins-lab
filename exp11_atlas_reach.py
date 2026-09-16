#!/usr/bin/env python3
"""exp11 — is the injected atlas functional, or peripheral?

GLM-5.3's objection: our injected-slice basins may be artifacts of an
off-manifold probing regime (frac_true_solved = 0 there, while native init
solves the puzzle). Decisive test:
1. Build FPRM atlases (64^2 slices) for each puzzle.
2. Run FPRM NATIVE (its own init) on each puzzle.
3. Ask: (a) is the native answer one of the atlas classes? (b) how far is it
   from the atlas majority (semantic distance)? (c) when native FAILS, does
   its error sit inside the atlas's answer distribution (i.e. does the atlas
   predict natural error modes), or is it outside everything the slice saw?
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from exp04_crossmodel import OUT as EXP04, FP_THRESH, MAX_ITERS
from exp01_sudoku_slices import PUZZLES, plane, mesh, to_traces
from loopscape import get_solver
from loopscape.puzzle import parse_puzzle

OUT = ROOT / "runs" / "exp11"
OUT.mkdir(parents=True, exist_ok=True)
RES = 64
SEED = 0


def hamming(a, b):
    return int((a.reshape(9, 9) != b.reshape(9, 9)).sum())


def main():
    device = ("mps" if torch.backends.mps.is_available() else "cpu")
    solver = get_solver("fprm", device=device)
    report = {}

    for pid, puzzle in PUZZLES.items():
        if not puzzle:
            continue
        true_grid = np.array(parse_puzzle(puzzle), dtype=np.int8)
        atlas_path = EXP04 / pid / f"fprm_slice_seed{SEED}.npz"
        if not atlas_path.exists() or int(np.load(atlas_path)["res"]) != RES:
            print(f"[atlas] building {pid} 64^2...", flush=True)
            u, v = plane((97, 512), SEED)
            z0 = mesh(RES)
            z0 = z0[:, 0, None, None] * u + z0[:, 1, None, None] * v
            z0 = z0 / z0.square().mean(-1, keepdim=True).add(1e-5).sqrt() * 500.0
            from loopscape.utils import batched_apply
            def f(p):
                r = solver.solve_batch(puzzle, initial_latents=p, fp_thresh=FP_THRESH,
                                       max_iters=MAX_ITERS, stepsize=1.0,
                                       return_intermediates=True)
                return to_traces(r)
            traces = batched_apply(f, z0, batch_size=128, pad_dim=1)
            finals = traces[:, -1].numpy()
            pdir = EXP04 / pid
            pdir.mkdir(exist_ok=True)
            np.savez_compressed(pdir / f"fprm_slice_seed{SEED}.npz",
                                finals=finals, res=RES, seed=SEED, puzzle=puzzle,
                                model="fprm")
        else:
            finals = np.load(atlas_path)["finals"]

        # native run (deterministic -> one trajectory; a few starts for safety)
        native = []
        for trial in range(3):
            r = solver.solve(puzzle, fp_thresh=FP_THRESH, max_iters=MAX_ITERS,
                             stepsize=1.0)
            g = np.array(r.grid, dtype=np.int8)
            native.append(dict(grid=g, solved=bool(r.solved),
                               correct=not (g.reshape(9, 9) != true_grid).any()))

        # analysis
        atlas_flat = finals.reshape(len(finals), -1)
        uniq, counts = np.unique(atlas_flat, axis=0, return_counts=True)
        majority = uniq[np.argmax(counts)].reshape(9, 9)
        rep = dict(puzzle=pid, atlas_classes=int(len(uniq)),
                   atlas_modal_share=float(counts.max() / len(finals)),
                   native=[])
        for nv in native:
            in_atlas = bool((atlas_flat == nv["grid"].reshape(1, -1)).all(1).any())
            h_maj = hamming(nv["grid"], majority)
            # distance to nearest atlas class
            d_near = min(hamming(nv["grid"], u.reshape(9, 9)) for u in uniq)
            rep["native"].append(dict(solved=nv["solved"], correct=nv["correct"],
                                      in_atlas=in_atlas, hamming_to_majority=h_maj,
                                      hamming_to_nearest_atlas=d_near))
        report[pid] = rep
        r0 = rep["native"][0]
        print(f"{pid}: atlas {rep['atlas_classes']} classes "
              f"(modal {rep['atlas_modal_share']:.0%}) | native solved={r0['solved']} "
              f"correct={r0['correct']} in_atlas={r0['in_atlas']} "
              f"h_majority={r0['hamming_to_majority']} h_nearest={r0['hamming_to_nearest_atlas']}",
              flush=True)

    (OUT / "atlas_reach.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "atlas_reach.json")


if __name__ == "__main__":
    main()
