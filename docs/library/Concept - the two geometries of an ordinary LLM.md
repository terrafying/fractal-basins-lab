---
type: results
tags: [our-own, ordinary-llm]
---

# Concept — the two geometries of an ordinary LLM (exp07 vs exp09)

Qwen3-1.7B / Qwen3.6-Flash dissociation:
- Activation space: ORDERLY. Smooth basin boundary (box-count D = 0.944 vs
  shuffled null 1.76); number-manifold chart shows coherent provinces and
  diagonal bands; failures (0-echo, unparsed) are provinces, not noise.
- Prompt space: CHAOTIC. 30% answer consensus across semantically identical
  paraphrases; bimodal overthinking (19-token answers vs 1500-token loops).

Reading: the instability of ordinary LLMs lives in the LANGUAGE INTERFACE
(which words realize the thought), not in the internal computation. The atlas
is stable; the map from language onto the atlas is not. That is the empirical
bridge to the Levin framing: the pattern is stable, the ingress is noisy.

Artifacts: runs/exp07/prompt_space_matrix.png,
runs/exp09c/depth_stack.png, runs/exp09c/adjacency_graphs.png,
runs/exp09c/number_manifold_chart_L*.png.
