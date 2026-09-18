---
type: idea
project: fractal-basins-atlas
status: sketched
tags: [application, safety, red-teaming]
---

# Idea — boundary-walk red-teaming

Wada-like boundaries mean: near any boundary point, infinitesimal nudges
change the answer qualitatively. So instead of random jailbreak fuzzing, WALK
along the boundary: start at a flip point (prompt/activation perturbation that
changes the answer), step tangentially to stay on the boundary, and enumerate
the diverse answers reachable within a tiny radius. The boundary is a
generator of adversarial variants with maximum semantic diversity per
perturbation budget.

## Implementation hooks
exp15's tangent-walk machinery (map-based, oracle-optional). On autoregressive
models: walk in embedding space at the last prompt token (exp09 machinery) or
in prompt-space (exp07 variants).

## Ethics
Local-lab only, own models. The atlas is also defensive: it tells you WHERE
the model is structurally least reliable, which is calibration information.
