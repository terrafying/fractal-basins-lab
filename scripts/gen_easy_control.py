#!/usr/bin/env python3
"""Generate a singles-solvable easy sudoku control (0 MRV backtracking guesses).

Fixes the exp01 difficulty axis: the embedded easy_a string measures 1778
backtracking guesses (harder than hard_a's 951), so there was no easy control.
This script backtracks a full grid from a fixed seed, then carves clues away
while the puzzle stays solvable by pure constraint propagation (every MRV cell
has exactly one candidate -> 0 guesses). Output is deterministic per seed.

usage: .venv/bin/python scripts/gen_easy_control.py [--seed 7] [--givens 38]
writes runs/easy_b_puzzle.txt (81-char string, 0=given-empty) on success.
"""
import argparse, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from exp01_sudoku_slices import backtracking_guesses  # noqa: E402


def full_grid(rng):
    g = np.zeros((9, 9), dtype=np.int8)

    def ok(r, c, v):
        return (v not in g[r]) and (v not in g[:, c]) and \
               (v not in g[3 * (r // 3):3 * (r // 3) + 3, 3 * (c // 3):3 * (c // 3) + 3])

    def fill(pos=0):
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        vals = [v for v in range(1, 10) if ok(r, c, v)]
        if not vals:
            return False
        rng.shuffle(vals)
        for v in vals:
            g[r, c] = v
            if fill(pos + 1):
                return True
            g[r, c] = 0
        return False

    fill()
    return g


def carve(full, target_givens, rng):
    """Remove clues (random order, one pass) while remaining singles-solvable."""
    g = full.copy()
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    givens = 81
    for r, c in cells:
        if givens <= target_givens:
            break
        keep = g[r, c]
        g[r, c] = 0
        if backtracking_guesses(g.copy()) != 0:
            g[r, c] = keep          # removal breaks singles-solvability: restore
        else:
            givens -= 1
    return g


def to_string(g):
    return "".join(str(v) if v else "." for v in g.ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--givens", type=int, default=38)
    ap.add_argument("--out", default=str(ROOT / "runs" / "easy_b_puzzle.txt"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    full = full_grid(rng)
    puzzle = carve(full, a.givens, rng)
    guesses = backtracking_guesses(puzzle.copy())
    s = to_string(puzzle)
    print(f"seed={a.seed} givens={int((puzzle > 0).sum())} backtracking_guesses={guesses}")
    print(s)
    if guesses != 0:
        print("FAIL: not singles-solvable", file=sys.stderr)
        return 1
    Path(a.out).write_text(s + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
