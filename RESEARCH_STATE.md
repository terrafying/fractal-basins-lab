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
