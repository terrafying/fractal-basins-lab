---
type: paper
authors: Farquhar, Kossen, Kuhn, Gal
year: 2024
journal: Nature 630, 625-630
status: read (core-adjacent)
tags: [hallucination, uncertainty, semantic-entropy]
---

# Paper — Detecting hallucinations using semantic entropy (Farquhar 2024)

Confabulations = wrong AND arbitrary generations (sensitive to random seed).
Detect via semantic entropy: sample answers, cluster by meaning, compute
entropy over meaning-clusters. High entropy flags confabulation.

## Relation to our program
Semantic entropy is a SAMPLE-space statistic. Our basin geometry is the
ACTIVATION-space version: the model's answer space has literal geographic
structure, and our early-FLI (rho 0.58) is a one-forward-pass geometric
predictor of the same instability. Follow-up: Semantic Entropy Probes
(arXiv:2406.15927) read SE from hidden states in a single pass — the bridge
between their sampling method and our geometric one. Our atlas gives the
geometric explanation of WHY semantic entropy works: high-SE prompts sit near
interlocked settling boundaries.
