# Fractal Basins Lab — Autonomous Research Protocol

Transplant of Buehler's "AI agents autonomously build computational laboratories"
methodology onto the Lai et al. 2026 "Fractal basins trap latent reasoning" system.

## The analogy that drives everything

| Metamaterial lab (Buehler 2026) | This lab |
|---|---|
| specimen geometry | puzzle instance (difficulty axis) |
| stiffness / peak load / work-to-failure | settling time, basin entropy, uncertainty exponent |
| abrupt vs progressive fracture transition | smooth vs fractal basin transition |
| load path redistribution under member failure | saddle redirection of latent trajectories |
| design → simulate → predict → holdout-test | (difficulty, slice) → map → predict Sb/alpha → test on held-out slice |
| 3 autonomous runs from same prompt | ≥3 slice seeds + ≥2 solver architectures |

## Key numbers (verify these constants before trusting any run)

- EqR sudoku: latent (97, 512), max_steps=24, noise_scale=0.0 (deterministic map).
- Slice: random orthonormal 2-plane via QR of Gaussian (97*512, 2), grid [-1,1]^2.
- Settling time: loops until decoded grid stops changing.
- Exclusions (paper App B): drop slice if <90% conditions reach final solution or
  >1% hit loop cap.
- Metrics: `loopscape.metrics.basin_entropy`, `uncertainty_exponent` (box 5),
  `fli_discrete` for saddle localization.

## Cost discipline (the whole point)

- All probe compute is local MPS, free: EqR (~27M) and FPRM are tiny.
- Frontier/Astra usage: ONLY for (a) conceptual interpretation of a finished
  result, (b) writing/rewriting hypothesis text, (c) one-shot design of NEW
  experiment scripts. Never inside measurement loops. Budget: a few calls/run.
- Cheap-tier models for everything else.

## Run book

1. `python exp01_sudoku_slices.py` — basin slices at 2 difficulty levels.
2. `python render_basins.py` — stills; add `--drift` for animated loops.
3. Add experiments as expNN_*.py following exp01's shape:
   - one question, one difficulty axis or one architecture change per experiment
   - write npz per slice + summary.json
   - always record solver, seeds, protocol in summary.json
4. Holdout discipline (Buehler): before running a new (difficulty, seed) pair,
   write the predicted Sb and alpha into the summary stub. Compare after.
5. Replication: any claim needs >=3 slice seeds and, ideally, EqR AND FPRM.

## Findings log

See runs/exp01/summary.json for live numbers.
