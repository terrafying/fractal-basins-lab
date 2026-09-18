---
type: paper
authors: Hao et al. (Meta)
year: 2024
arxiv: 2412.06769
status: to-read
tags: [latent-reasoning, coconut]
---

# Paper — Coconut: Training LLMs to Reason in a Continuous Latent Space (Hao 2024)

Chain of Continuous Thought: feed the last hidden state back as the next input
embedding instead of decoding. Continuous thought encodes SUPERPOSITIONS of
multiple candidate next steps (BFS-like exploration instead of premature
commitment). Outperforms CoT on planning tasks needing backtracking.

## Relation
The ordinary-LLM counterpart of our looped models. Coconut's latent "BFS
superposition" is exactly what our basin maps visualize in looped models —
the atlas picture predicts that Coconut's continuous thought explores several
basins simultaneously. Also the natural target for our activation-basin
methodology on a mainstream architecture.
