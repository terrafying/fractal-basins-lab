---
type: paper
authors: Lai, Bao, Quinn, Gilpin
year: 2026
arxiv: 2609.04963
status: read (core)
tags: [fractal-basins, source-paper]
---

# Paper — Fractal basins trap latent reasoning (Lai 2026)

The source paper. Reasoning models (looped: EqR, FPRM, TRM, Parcae, Huginn)
are discrete-time dynamical systems over latent state. On hard tasks they show
transient chaos: fractal basins of settling time. Basin entropy correlates
with task difficulty. Settling slowdowns are caused by trajectories scattering
off weakly-unstable saddle points that decode as nearly-correct attempts
("Plinko" model). Code: GilpinLab/loopscape.

## What we verified / extended
- Replicated difficulty -> fractality on our stack (EqR hard_a alpha ~0.28,
  easy control Sb=0) — [[Atlas - our own results]]
- Retracted our own claim: answer-identity Wada at 1489 basins was a
  fragmentation artifact (patch-permutation null). Wada-like structure lives
  in the SETTLING (temporal) geometry: z=8.5 at 200^2 vs patch null.
- Found metric artifact in their lineage of probes: parse_puzzle returns
  givens; any "true solution" comparison needs a real solver.

## Open threads
- Saddle points decode as nearly-correct attempts -> semantic labeling of
  boundaries (which cell was wrong) — our exp05 found EqR flickers too fast;
  FPRM's 200-iter regime is the right venue.
