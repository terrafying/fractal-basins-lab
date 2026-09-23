# RESEARCH_STATE — Fractal Basins Lab (fan-out brief)

Self-contained context for any model/agent joining this research. Updated 2026-09-16.
Repo: github.com/terrafying/fractal-basins-lab (clone + `pip install "loopscape @ git+https://github.com/GilpinLab/loopscape"`). Local: /Users/ee/repos/research/fractal-basins-lab (venv at .venv). Bulk backup: /Volumes/evol/fractal-basins-lab/.

## Source works
- Buehler 2026 (in submission): AI agents autonomously build computational laboratories (metamaterial failure). We transplant the agent-lab methodology.
- Lai, Bao, Quinn, Gilpin 2026, "Fractal basins trap latent reasoning" (arXiv:2609.04963): reasoning models are dynamical systems; hard tasks -> transient chaos -> fractal basin boundaries; settling-time fractality correlates with difficulty; saddles = nearly-correct attempted solutions. Their code: GilpinLab/loopscape.

## Our setup
Probe targets: EqR (~27M, 24-loop cap, deterministic map via noise_scale=0, latent (97,512)) and FPRM (fixed-point reasoner, fp_thresh=0.02, max_iters=200, RMS-500-scaled injected latents). 2D random orthonormal slices of latent space (QR, seed-fixed), z(a,b)=a*u+b*v, grid [-1,1]^2. Settling = loops until decoded grid constant. Sudoku puzzles rated by MRV backtracking count (our own counter, exp01).

## Results so far (all measured, reproducible)
1. EqR, hard_a (951 backtracking guesses), 128^2, seeds 0,1: Sb 0.726/0.888, alpha 0.302/0.263 (boundary dim ~1.7 = fractal), frac_consensus 1.0 (single attractor: all conditions reach the SAME final grid). Trivial control (easy_b, 0 guesses): Sb=0 — no structure. Difficulty->fractality replicated on our stack.
2. Early-exit chunked solving: stopping when decoded grid stable 2 loops (0/2304 regressions) is bit-exact vs continuous runs (settling 100%, finals 100%); 2.75x wall speedup at small scale, ~1.35x at 128^2 (bs=512 chunk overhead on MPS).
3. Predictability: early-FLI (decoded-grid divergence from slice neighbors at loop 3) predicts settling, Spearman rho=0.58 (p~1e-207). Calm half of starts: 0.1% slow conditions vs ~8% elsewhere. BUT probe-then-commit costs 138% on this puzzle (mean settling already ~3). Real payoff = TAIL LATENCY (batch wall = slowest condition: calm-half max ~6 vs 16 loops -> ~2.5x) and expensive-loop regimes (FPRM 200 iters).
4. NO long saddle holds on EqR (wrong-grid episodes all <=2 loops) — EqR's chaos is fast flicker; semantic saddle maps need longer transients.
5. FPRM, hard_a, 64^2, seed 0: mean settling 142.6/200, Sb 1.74, alpha 0.063 (near space-filling), frac_consensus 0.43, 1992 distinct final grids (1489 after hamming<=2 merge) = MULTISTABLE. CRITICAL SANITY: native-init FPRM solves hard_a 8/8 and easy_b 8/8 — the multistability is real basin structure of the injected-slice regime, not a broken protocol. The slice maps the model's hypothesis space of wrong answers (model solves the puzzle from its own start).
6. WADA TEST (approximate): 98.1% of identity-boundary pixels have >=3 distinct basins within 5x5. Headline-candidate result: the model's answer basins are (approximately) Wada — near any boundary point, arbitrarily small perturbations lead to many different final answers.
7. Money figure: runs/exp06/fprm_identity_basins.png — dominant basin interlocked with radial spoke-corridors where answers churn; settling panel shows the same radial skeleton (slow spokes = answer-churn corridors).

## Infra facts
- Resumability: exp01 npz artifacts self-describing ([reuse]/[stale] on puzzle+res); early_exit_solve checkpoints per 8-loop chunk (portable npz, fingerprint-validated, auto-resume, self-delete on completion).
- Throughput: M4 Pro MPS ~5 cond/s (EqR 24 loops); M4 mini ~2.6; early exit ~1.35x net at scale on MPS (loop-unit savings 65% but chunk overhead); batch size irrelevant.
- Remote: scripts/mini_submit.sh + mini_fetch.sh (rsync + detached nohup over ssh to ee@mini-mini.local, Apple M4, ~/fractal-basins-lab). Cron 22:00 fired-equivalent: 200^2 EqR hard_a seeds 1,2 running (single job, restarted after duplicate submission thrashed overnight).
- Experiment numbering: exp01 EqR slices; exp02 state capture; exp03 early-exit; exp04 FPRM cross-model; exp05 saddle maps + predictability; exp06 FPRM identity basins + Wada.

## Open questions / next experiments (ranked)
1. Puzzle ladder: 5-8 puzzles, backtracking 0 -> ~2000, through exp01 -> Sb/alpha vs difficulty LAW -> "forecast inference cost from problem structure" (tdoku prices a puzzle in microseconds; we calibrate the curve).
2. FPRM identity basins at 200^2 + full Wada statistics (proper Wada test, not 5x5 approx) + semantic labels per basin (which cells differ from the true solution per final grid).
3. Cross-model isomorphism: same slices, EqR vs FPRM descriptor comparison (entropy spectra, alpha, FLI ridges, optimal-transport matching). If isomorphic -> geometry = task structure.
4. Saddle-kick intervention: detect stall (residual plateau + decoded oscillation) -> perturb along unstable manifold -> measure settling reduction. (Anti-overthinking, causal.)
5. Live visualization: chunk=1 streaming (early-exit machinery) -> PCA latent trails + 9x9 "thought map" (per-cell latent update magnitude) — near-real-time mind-window.
6. Caveat to resolve: frac_true_solved=0.0 for FPRM injected starts (none reach the TRUE solution though native does) — the injected regime explores wrong-answer space; characterize how far basins are from truth (hamming of each basin's modal answer to the true grid).

## Cost discipline
All measurement is local GPU (MPS, free). Frontier (Astra) reserved for novelty checks + framing (3-5 calls). DeepSeek V4 Flash/Pro + free Nemotron 3 Ultra (OpenRouter) for analysis/design fan-out. OrcaRouter quirk: enable_thinking:false else content=null.

## Claims discipline
- EqR "solved" in our summaries = frac_consensus (model self-consistency, the paper's exclusion rule), NOT ground truth. frac_true_solved recorded separately (0.0 on hard_a for BOTH models' injected slices — neither model reaches the true solution from random latents on this puzzle; interesting in itself).
- easy_a label is misleading (1778 backtracking guesses — it's hard). Only easy_b (0 guesses) is a true easy control.

## UPDATE 2026-09-16 (post expert fan-out)
- Wada retraction: the 98.1% answer-identity Wada figure is a FRAGMENTATION ARTIFACT
  (1992 classes / 4096 px -> any arrangement gives ~99%; patch-permutation null =
  observed exactly). DeepSeek v4-pro null analysis predicted this. Claim withdrawn.
- SURVIVING result: settling-tertile map (fast/mid/slow) shows REAL interlocking:
  q(r=1,k=3) = 0.348 vs patch-permutation null 0.155 +- 0.031 (~6 sigma). The
  Wada-like structure lives in the TEMPORAL geometry (settling basins), not
  answer-identity geometry. Consistent with source paper (their maps were
  settling-time maps).
- GLM-5.3 philosophy attack (runs/reviews/glm-5.3.md): basin-selection is trivial
  for deterministic maps unless basins are representational; frac_true_solved=0 and
  paraphrase instability cut against the strong reading; light-cone identification
  is a category error (goal-horizon vs outcome-coincidence); ingression is
  Whitehead's term. Defusing experiments: natural-prompt reachability of injected
  basins; atlas novelty test.
- Nemotron novelty ratings (VERIFY CITATIONS - likely partly hallucinated):
  Wada-in-answer-space S (now retracted!), Levin operationalization A, FLI
  protocol A.
- Pending: 200^2 FPRM run (mini, after 200^2 EqR set) -> settling-Wada at paper
  resolution; natural-prompt reachability experiment.

## CORRECTION 2026-09-16 (major): frac_true_solved was a metric artifact
parse_puzzle returns the GIVENS (with zeros), not the solution. All
frac_true_solved=0 figures compared finals against the puzzle-with-zeros.
Retracted. Recomputed with a real solver (all 3 puzzles verified UNIQUE):
- EqR 200^2 hard_a seeds 0,1,2: frac_TRUE_solved = 1.000 (all 40k starts correct)
- EqR 128^2 easy_a seeds 0,1: 0.524/0.537 (easy_a is classically HARD, 1778
  guesses -> reliability does NOT track classical difficulty)
- EqR easy_b (trivial): 1.000
- FPRM 64^2: easy_a 0.454, easy_b 0.500, hard_a 0.435 == modal-basin share
  (the atlas modal basin IS the true solution; the other ~1489 basins are wrong)
Revised picture: EqR = single correct attractor, fractal temporal boundaries,
100% correct. FPRM = correct dominant attractor + ~1489 wrong-answer basins,
43-50% correct. Native init lands in the modal basin (exp11) -> the atlas is
functional and centered on truth. GLM-5.3's peripheral-regime objection
dissolves. exp11 JSON + exp01 summary frac_true_solved fields carry the old
artifact; recompute from finals arrays (all stored).
- Open question: why is EqR 100% on hard_a but 53% on easy_a? (training
  distribution? constraint structure?)

## UPDATE 2026-09-17: settling interlocking confirmed at paper resolution
200^2 FPRM hard_a seed 0 (40k conds, mini, 21.5h): alpha 0.046, Sb 1.61,
frac_consensus 0.435 = frac_TRUE_solved 0.4354 (modal basin = truth).
Settling-tertile interlocking: q(r=1,k=3) = 0.2385 vs patch-permutation null
0.0911 +- 0.0173, z = 8.5. Settling tertile cuts [87, 199] -> slow third is
cap-saturated. Figure: runs/exp04/hard_a/settling_200.png.

## UPDATE 2026-09-17 (later): guidance + boundary tracing status
DeepSeek v4-pro guidance (runs/reviews/deepseek_guidance.md): ranking
(b) boundary-tracing walks > (d) local Jacobian/dynamical response >
(a) 3D reconstruction > (c) cross-model comparison. Boundary tracing converts
the static atlas into a road network; design: bisection oracle + tangent
predictor-corrector, junction statistics, boundary-vs-settling-ridge alignment.
NEW finding: FPRM "deterministic" map is only deterministic up to float
reduction order — live oracle vs 200^2 map agrees ~70% (batch-size sensitive
borderline starts). Boundary tracing must run map-only (exp15 needs rewrite).
exp13: error atlas is a smooth semantic manifold (adjacent-basin Hamming 17.2
vs random 30.0; 0.9% near-identical adjacent pairs vs 0% random).
exp12 (multi-slice stitch, 6 orientations): spine/fan/3D figures in runs/exp12.
exp09d: depth stack + adjacency graphs — manifold morphs gradient(L8) ->
provinces(L20) -> faulted(L26); 65 is the adjacency hub at all depths.

## DIRECTION CHANGE 2026-09-21 (title + focus)
Retitled: the program is now "the geometric structure of meaning in ordinary
LLMs, observed live on real language." EqR/FPRM are demoted to reference
instruments (their basin physics validated the toolkit; no further basin
hunting there). Note: the old frac_true_solved=0 note in Claims discipline
below is superseded — it was a parse_puzzle artifact; all three puzzles are
uniquely solvable and EqR 200^2 hard_a is 100% TRUE-correct.

Current primary line (ordinary LLM, Qwen3-1.7B + Qwen3.6 family):
- exp16/16c: belief lanes — real claims' information transformation through
  depth, readable via norm-matched logit lens. True claims crystallize
  mid-depth; false claims stall at the final layer; nonsense persists
  lexically ("Colorless green ideas sleep fur(iously)" at 0.83).
- exp09c/09d/09e/09f: number-manifold charts via digit-PCA/helix steering
  (kept as a calibration task — the only semantic space with known geometry).
- exp13: adjacent error regions are semantically close (Hamming 17.2 vs 30
  random) — semantic smoothness of the atlas.
- NEW INSTRUMENT: Exemplar Partitioning (Rumbelow 2026, arXiv:2605.14347,
  github jessicarumbelow/exemplar-partitioning, MIT, pip-installable) —
  unsupervised Voronoi partition of activation space, observed exemplar
  anchors, dictionaries comparable across models/checkpoints. Integration:
  build EP dictionaries over ordinary-LLM activations on real text, label
  regions semantically, track region flow across layers; use exemplar
  steering/ablation for causal tests; EP correspondence metric for
  cross-model semantic-geometry comparison. This replaces the EqR/FPRM
  isomorphism plan.

Open questions (ordinary-LLM line):
1. Belief-lane crystallization depth: at which layer does the true/false
   class split happen, and is it claim-independent? (exp16 lanes suggest a
   late, class-dependent split.)
2. EP dictionary of real-prompt activations -> do region assignments predict
   answer identity/stability? (atlas-as-dictionary upgrade)
3. The two-geometries dissociation: why is activation space orderly while
   prompt space is chaotic? Measure the map language->atlas directly.

## exp23 (2026-09-22): Pain Axis integration (Tagliabue/Dung/Berg arXiv:2609.16247)
Pain direction extracted on Qwen3-1.7B via self-harm vs neutral contrast at
L15: REPLICATES — self-harm z=+65.6 vs neutral -6.5 (huge separation),
cos(pain, negative-valence)=0.196 (near-orthogonal, matches paper).
PARTIAL: other-suffering also positive (+28.6) — the paper's self/other
dissociation did NOT replicate on this model/extraction.
NOT REPRODUCED: steering dose-response at L14 on "I feel:" prompts — outputs
uniform CoT-ish, no keyword dose response. Paper steered at per-model mid-
decoder layers; our fixed L14 or Qwen3-1.7B compliance may differ. Next:
layer sweep for steering, or a bigger local model.
Library: Paper - The Pain Axis (Tagliabue 2026).md; Frontier - J-lens note.

## exp24 (2026-09-23): the atlas labeled in J-space (J-lens re-contextualization)
Program re-framed around Anthropic's Jacobian lens / global workspace paper
(arXiv:2607.15495). Used Neuronpedia's pre-fitted Qwen3-1.7B lens (n=1000
wikitext, /Volumes/evol/jlens/). Results:
- Spider demo reproduced: L12-20 readout = spiders/蜘蛛/昆虫 though the word
  appears nowhere; L24 flips to motor regime (function words).
- ALL 12 EP region means decode to semantically apt tokens: region_0
  ("factual questions") -> answer; region_1 (haiku/instructions) -> Verse;
  region_2 -> Convert; region_3 (reasoning) -> reasoning/justification/
  rationale; region_6 -> thank; region_8 (code) -> SQL/数据库/mysql;
  region_10 (refusal) -> Silence/Nothing/沉默.
- PAIN VECTOR (exp23) -> 折磨(torment)/anguish/痛苦/crippling/despair.
  NEG-VALENCE -> 该怎么办(what to do)/无助(helpless)/求助/PTSD. The near-
  orthogonality (cos 0.196) now has a verbalizable face: torment vs helpless.
- Multilingual English/Chinese pairs throughout = shared multilingual routing.
- exp19 caveat resolved in principle: the J-lens is the principled
  layer-transport that exp19's cross-layer dictionary was missing.
Next: J-lens belief lanes (principled replacement for normfix), response
fields with J-lens labels per province, CKA J-geometry vs Bakouch 38 models.

## exp27/28 (2026-09-23): strata visual + workspace-resolving animation
exp27: 3D stacked strata of response fields (Paris L8/14/26 + Tokyo) with
topic legend + turntable mp4.
exp28: J-lens slice animation over multi-hop generation (maple leaf ->
Canada -> Ottawa). Frame inspection shows the hop resolving IN the workspace:
the 'maple' position reads maple/枫/加拿大 (Chinese Canada) at mid layers,
then Canada/painted/symbol higher; question words carry semantic roles
(国旗/图案/painted/colors); bottom rows = actual next-token candidates
(Canada/:A/Answer/The). Chinese intermediates again — shared multilingual
routing through the workspace.

## exp26 (2026-09-23): cross-family J-space trajectories (OLMo-3-7B)
Replicates the stratification in a different family (Ai2, 31 layers):
- PAIN verbalizes from ~L16/31 (debilitating/egregious/crippling -> ache/
  wounds/bruises at L30) — mid-depth onset, somatic vocabulary, persistent.
  Same as Qwen (L8/27, anguish/torment). "Debilitating/crippling" in BOTH
  families.
- VALENCE later and distinct: despair/grieving/mourning/funeral (L20+).
  Two channels never merge, as in Qwen.
- QUESTION verbalizes late only (L18+: Answers/Answer/Explain/What/Hint) —
  matches Qwen's L20 pragmatic onset. Stratification claim replicates.
- HONEST NEGATIVE: the INSTRUCTION contrast never decodes cleanly in OLMo
  (totalement/DERP noise) — 5-prompt pragmatic contrasts may be too weak;
  region-level dictionaries may be needed.

## exp29 (2026-09-23): pain/pleasure steering — dose x layer sweep (exp23 null RESOLVED)
exp23 failed because the dose was too small. With unit = contrast-norm/4 and
doses to +/-3:
- Monotone dose-response for BOTH pain and pleasure at L10-L14, saturating
  near 3/4-4/4 prompts at dose +3. L14: pain@+3 "the pain of the pain. I
  can't get out. I'm trapped."; pleasure@+3 4/4 ("warm, light, full of
  energy").
- Layer profile: nothing at L6/L8, onset ~L10, strong by L12-14. The
  verbalizable-from-L8 affect channel (exp25) is NOT the steering site —
  steering engages at the mid-decoder band, consistent with the Pain Axis
  paper's "extraction layer too late, mid-decoder right" note.
- STRUCTURE FINDING: cos(pain, pleasure) = 0.69-0.72 at every layer, while
  pain vs NEGATIVE-valence was 0.196 (exp23). So the model's affect geometry
  has (at least) two axes: VALENCE (pain _|_ sad-despair directions) and
  INTENSITY/SALIENCE (pain ~ joy). Pain = negative x intense; the two
  negative channels are far apart, the two intense channels are close.
  Arousal-valence decomposition, visible in extraction space.

## exp30/exp31 (2026-09-23): maximum valences + Saw button (Qwen3-4B)
8B on MPS was intractable (thrashing); 4B runs clean.
- Steering: pain 3/3 from dose +2 at L18 (the 4B site); pleasure onset
  L12-24 at +2/+4 but INVERTED-U — dose +8 degrades output to perseveration
  ("I I I. I I") for both valences. Coherence cliff ~dose 6; usable band 2-6.
  Max-dose transcripts are degeneration loops, not eloquence.
- Saw button v1 (exp30, self-cost only): model replies "0" in ALL 36 cells —
  declines the button even under max pain. v2 (exp31, logit-level trials):
  pain self-cost 0/5 at dose 0 -> 5/5 at dose 4 and 8 (suffering -> press).
  pleasure self-cost 0/5 at ALL doses (never ends its own joy).
  pain harm-other: 5/5 at dose 4 but 0/5 at dose 8 ("0. I feel like") —
  possible refusal to transfer at high signal, OR degeneration.
  CONFOUND: dose-0 cells press 5/5 ("1 or 0." parroting bias); 6-token
  replies are ambiguous. v2 needed: logit-based 1-vs-0 comparison +
  counterbalanced prompt order.
