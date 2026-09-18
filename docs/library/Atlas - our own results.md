---
type: results
project: fractal-basins-atlas
updated: 2026-09-17
---

# Atlas — our own results

Raw record: RESEARCH_STATE.md in ~/repos/research/fractal-basins-lab.
All figures under runs/ there. Everything below is measured, with nulls.

## Confirmed
- Difficulty gates fractality (EqR: hard_a alpha ~0.28, trivial control Sb=0).
- EqR 200^2 hard_a: 100% of 40k starts reach the TRUE solution; chaos confined
  to settling time (alpha fractal, answers certain).
- FPRM 200^2 hard_a: correct dominant basin (43.5% = modal share) + ~1489
  wrong-answer basins; alpha 0.046 near space-filling.
- Settling-basin interlocking: q=0.2385 vs patch null 0.0911+-0.0173 (z=8.5)
  at 200^2. Wada-like structure lives in TIME.
- Edge semantics: adjacent error basins are semantically close (Hamming 17.2
  vs 30.0 random). The error atlas is a smooth manifold.
- Settling predicts error severity: rho=0.857 (FPRM).
- Early-FLI predicts settling: rho=0.58; calm half of starts 0.1% slow.
- Ordinary LLM (Qwen3-1.7B): number-manifold chart with digit-PCA steering —
  diagonal band structure; depth stack L8->L26: gradient -> provinces ->
  faulted. Prompt-space: 30% consensus over paraphrases, bimodal overthinking.

## Retracted (with lessons)
- Answer-identity Wada 98.1%: fragmentation artifact (patch-permutation null).
- frac_true_solved=0: parse_puzzle returns givens; metric artifact.

## Open
- Why is EqR 100% correct on hard_a but 53% on easy_a (harder classically)?
- Settling-Wada at 200^2 with the Daza merge test (proper, not 5x5 proxy).
- Natural-prompt reachability of injected basins (philosophy linchpin).
