# Reproducing "Fractal basins trap latent reasoning" on a 27M-param recurrent solver

Independent reproduction of Lai, Bao, Quinn, Gilpin (2026),
[arXiv:2609.04963](https://arxiv.org/abs/2609.04963), on their open-weights EqR
sudoku solver. Everything here runs free on Apple-silicon MPS (or a Colab T4);
no frontier API in any measurement loop. Repo:
[github.com/terrafying/fractal-basins-lab](https://github.com/terrafying/fractal-basins-lab)

STATUS: draft — numbers land as the run matrix fills in (see TODOs).

## What the paper claims, and what we test

The paper shows looped latent reasoners don't always converge to one answer.
The decoded output depends sensitively on the initial latent state, and the
boundaries between attractor basins are fractal: zoom in and the
solution/answer boundary keeps structure at every scale. They quantify this
with basin entropy (Sb) and the uncertainty exponent (alpha), and show the
fractal transition strengthens with task difficulty.

Our test, following their Appendix B protocol:

- EqR (~27M params), sudoku, deterministic map (noise_scale=0, max_steps=24).
- Sample initial latents on a random orthonormal 2-slice of the (97, 512)
  latent space (QR of a Gaussian; slice seeds 0/1/2), grid [-1,1]^2 at res 128
  (16,384 conditions per slice).
- Settling time = number of loops until the decoded grid stops changing.
  The settling-time field over the slice IS the basin map.
- Metrics via the authors' own `loopscape` package: basin_entropy,
  uncertainty_exponent (box size 5).
- Exclusion rules from the paper: drop a slice if <90% of conditions reach the
  model's final (self-consistent) solution or >1% hit the loop cap.

Difficulty axis (measured, MRV backtracking guesses): easy_b = 0 guesses
(40 givens, generated singles-solvable control, `scripts/gen_easy_control.py`),
hard_a = 951, easy_a = 1778. Reproduction note: the lab's original setup
assumed easy_a was the 0-guess control per a stale comment — measurement showed
1778, i.e. the two original puzzles were both hard and the "easy/hard" labels
were wrong. easy_b was generated to restore the axis; all slices record their
puzzle string and measured difficulty, so earlier easy_a/hard_a runs remain
usable data points on a 3-point axis (0 / 951 / 1778).

## Results

TODO(auto): paste the summary.json table once seeds complete
(mean settling, Sb, alpha, frac_consensus per puzzle/seed; validity flags).

Early signals from the filled slices (interpret with care until all seeds land):

- Difficulty ordering holds: easy_b (0 guesses) settles immediately
  (mean settling ~0, no boundary structure, alpha undefined), hard_a (951)
  settles in ~3-5 loops, easy_a (1778, the hardest) settles near the 24-loop
  cap with heavy multistability (seed 0: 6,069 distinct final grids across
  16,384 conditions, consensus only 52%). Rougher classical difficulty buys
  longer settling and more fragmented basins - the paper's core claim, on a
  measured (not assumed) difficulty axis.
- alpha at box size 5 for hard_a seed 0 is 0.236, well under 0.5: boundary
  roughness consistent with the paper's fractal finding.

### Figures (and why the videos are mostly gone)

The primary visual is `runs/figs/difficulty_axis.png`
(`scripts/plot_difficulty.py`): settling time, basin entropy, and alpha
against measured difficulty, one point per seed. That figure states the
claim; no animation adds information to it.

Video policy learned the hard way: the drift/zoom basin animations
(`render_basins.py --drift`) look cinematic but reveal nothing a still
doesn't - zooming a fractal shows more of the same, by definition, and there
is no time axis in a settled field. The one video we keep is
`anim_crystallization.py`: the decoded grid evolving loop by loop IS new
information per frame (the actual solving dynamic), which stills can't show.

## The optimization: exact early exit

Paying all 24 loops for every condition wastes most of the compute — most
trajectories settle in 3-7. `exp03_earlyexit.py` runs chunked solves with
latent-carry continuation and drops conditions that already show a stable
decoded pair (grids never regress afterwards; validated 0/2304 regressions on
exp02 traces). Result: bit-exact equivalence with the continuous cap-24 run
(100% settling match, 100% final-grid match on hard_a) at 2.75x wall speedup,
and ~60% fewer loop-units at res 128. On a 5 cond/s MPS device this is the
difference between a half-day and an hour per slice.

This matters for anyone replicating on free compute: the protocol is cheap
once you stop paying for loops that already settled.

## Deviations and unknowns

- The paper's figure grid resolution is higher than res 128; our alpha values
  are therefore expected to sit near (not exactly on) theirs. Same-direction
  comparison only.
- We use the paper's exclusion rules but score "solved" as self-consistency of
  the decoded grid (their App. B criterion), not ground-truth match. We record
  both; frac_true_solved for hard_a is low (the model genuinely fails the hard
  puzzle) while consensus stays high — multistability, not correctness, drives
  the basin structure.
- FPRM cross-architecture replication not yet run (the paper uses two
  solvers). Tracked in the repo.

## Reproduce it

```bash
git clone https://github.com/terrafying/fractal-basins-lab
cd fractal-basins-lab
pip install "loopscape @ git+https://github.com/GilpinLab/loopscape"
FB_RES=64 FB_SEEDS=0 python exp01_sudoku_slices.py    # ~10 min on M-Pro class
python render_basins.py                               # basin map PNG
python render_basins.py --drift                       # animated drift
python anim_crystallization.py                        # solution crystallization
```

Colab T4 notebook included (`colab/fractal_basins_colab.ipynb`).

## What's next

- Fill the 3-seed x 2-puzzle matrix (running now on a two-Mac mini-cluster).
- FPRM cross-check.
- Boundary-boxing curves (alpha via box-counting sweep, not just box=5).
- The interesting question the paper leaves open: does the fractal boundary
  predict WHERE the model is wrong, not just that it wavers?

## Credits

Buehler's agent-lab methodology (autonomous computational labs) inspired the
lab structure: every claim needs >=3 slice seeds and pre-registered
predictions before the holdout runs. The `loopscape` package and EqR weights
are the paper authors'; the experiments and this reproduction are mine.
