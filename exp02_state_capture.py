#!/usr/bin/env python3
"""Exp 02 — capture full state evolution for animation.

Same deterministic-map protocol as exp01, but saves the per-loop decoded grids
(traces) for one slice, so we can animate how the solution crystallizes over
reasoning loops. Small grid (48^2) to stay cheap.

Output: runs/exp02/hard_a/traces_seed0.npz  (traces int8 [B, steps, 9, 9])
"""
import os, sys
from pathlib import Path

import numpy as np
import torch

from loopscape import get_solver
from loopscape.utils import batched_apply

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp02" / "hard_a"
RES = int(os.environ.get("FB_RES", "48"))
SEED = int(os.environ.get("FB_SEED", "0"))
MAX_STEPS = 24
PUZZLE = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."

sys.path.insert(0, str(ROOT))
from exp01_sudoku_slices import plane, mesh, to_traces


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"traces_seed{SEED}.npz"
    if out.exists():
        print("[skip]", out)
        return
    solver = get_solver("eqr", device=device)
    u, v = plane((97, 512), SEED)
    def f(p):
        z = p[:, 0, None, None] * u + p[:, 1, None, None] * v
        r = solver.solve_batch(PUZZLE, z_H=z, z_L=z * 0, max_steps=MAX_STEPS,
                               noise_scale=0.0, return_intermediates=True)
        return to_traces(r)
    traces = batched_apply(f, mesh(RES), batch_size=96, pad_dim=1)
    np.savez_compressed(out, traces=traces.numpy().astype(np.int8), puzzle=PUZZLE,
                        res=RES, seed=SEED)
    print("wrote", out, traces.shape)


if __name__ == "__main__":
    main()
