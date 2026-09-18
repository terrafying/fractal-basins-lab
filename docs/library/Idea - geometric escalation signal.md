---
type: idea
project: fractal-basins-atlas
status: designed, needs production test
tags: [application, escalation, calibration]
---

# Idea — geometric escalation signal

Near a basin center, answers are robust; near interlocked settling boundaries,
tiny perturbations change resolution time and reliability. Operational: probe
the local neighborhood cheaply (neighbor divergence at loop 3 = early-FLI,
rho 0.58 with settling), measure boundary distance, and escalate to a bigger
model when the query sits near the interlocked region.

## Why geometric > self-consistency
Self-consistency costs k full generations. Early-FLI costs one batch of
truncated probes. Calm-half starts had 0.1% slow conditions vs 8% elsewhere.
Best regime: batch inference (wall-time = slowest condition; ~2.5x cut) and
expensive-loop models (FPRM 200 iters).

## Test to run
On a task suite: compare escalation-decision quality (error caught per
compute) for early-FLI gate vs self-consistency vs semantic entropy at equal
compute. Related: [[Paper - Semantic entropy hallucination detection (Farquhar 2024)]].
