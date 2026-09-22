---
type: paper
authors: Rumbelow, Jessica
year: 2026
arxiv: 2605.14347
url: https://github.com/jessicarumbelow/exemplar-partitioning
status: integrating
tags: [interpretability, sae-alternative, voronoi, atlas-method]
---

# Paper — Exemplar Partitioning for Mechanistic Interpretability (Rumbelow 2026)

Unsupervised Voronoi partition of centered unit-norm activation space via
leader-clustering on cosine distance. Each region anchored by an OBSERVED
exemplar (traceable to a prompt+token). ~10^3x cheaper than SAEs; matches
SAE-A AUROC on AxBench (0.937 vs 0.911 at Gemma-2-2B-it L20 p1). Because
exemplars are observed rather than learned, dictionaries are directly
comparable across layers, models, and checkpoints (shared-prompt round-trip
correspondence: 19% base -> 44% after instruction tuning, Gemma vs Llama).
Prebuilt dictionaries on HF (Gemma-2-2B(-it), layers 4/12/20). MIT, pip
installable.

## Why this matters to the atlas program (integration hooks)

1. ATLAS AS DICTIONARY: our basin maps slice 2D planes by hand. EP gives
   principled cells over the FULL activation space at any layer. Build an EP
   dictionary over FPRM/EqR last-token activations across many puzzles ->
   label each region by which answer-basin its assignments land in -> the
   atlas gains an interpretable FEATURE dimension (which cell = which error
   mode), upgrading our z=-37.6 wedge result from geometric to semantic.
2. CROSS-MODEL ISOMORPHISM: EP's shared-exemplar correspondence metric is
   exactly our isomorphism test, already engineered. EqR vs FPRM region
   correspondence would quantify "same task structure" directly.
3. SADDLE-KICK MECHANISM: exemplar steering + ablation hooks (README recipe)
   give the causal intervention machinery our saddle-kick design lacked:
   steer along a region's exemplar, or project a region out, and watch which
   answers flip. Causality via interpretable directions.
4. SEMANTIC UNCERTAINTY LINK: regions with high answer-entropy across our
   probe starts = the "confabulation" regions semantic-entropy work detects
   from outside; EP gives the inside view.

## Method notes
- Cluster in centered unit-norm space; exemplar = first-arrival activation;
  region mean = better detector, exemplar = traceable + steerable.
- Same extract_fn for calibration and discovery (else silent garbage).
- Threshold = percentile of pairwise cosine distances (p10 default).

## Our integration plan
- exp18: build EP dictionary on FPRM activations (last prompt token, layer 20,
  500+ puzzle prompts), assign the 4096 slice starts to regions, then
  region -> answer-basin contingency map. Test: does EP region structure
  predict basin identity better than raw position?
- Cross-model: identical prompt set -> EqR dictionary vs FPRM dictionary ->
  correspondence metric. This is the isomorphism experiment with an
  off-the-shelf metric.
