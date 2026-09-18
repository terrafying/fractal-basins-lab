---
type: idea
project: fractal-basins-atlas
status: hypothesis
tags: [scaling, levin, capability]
---

# Idea — light-cone scaling law

Levin: collective intelligence grows by enlarging the cognitive light cone.
Operationalized for models: does basin structure change with capability?
Concretely, across the Qwen family (0.6B / 1.7B / 4B / 8B) on identical
activation-basin charts:
- number of distinct answer basins per unit perturbation volume
- settling/CoT-length variance across the chart
- boundary box-counting dimension (orderly vs interlocked)
- semantic smoothness of the error atlas (exp13 metric)

Hypothesis space: bigger models either (a) enlarge the correct basin (more
robust, smoother), or (b) enrich the wrong-answer atlas (more diverse
competent errors). Either outcome is a result; (a)+(b) together would say
capability reorganizes the atlas rather than shrinking it.

Also applies across training checkpoints of one model: the source paper's
bifurcation-during-training figure (basins form by cascade) — our metrics as
a grokking/phase-transition monitor.
