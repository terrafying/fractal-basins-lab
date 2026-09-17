#!/usr/bin/env python3
"""exp14 — guidance check-in: DeepSeek v4-pro reviews the cartography program."""
import json, os, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "reviews"
OUT.mkdir(parents=True, exist_ok=True)

BRIEF = """# Cartography program check-in (results since your last review)

Your Wada-null critique was decisive: the 98.1% answer-identity Wada figure was a
fragmentation artifact (patch-permutation null reproduces it exactly). Retracted.
What survived and what's new since:

1. Metric correction: parse_puzzle returns givens not solutions. Recomputed:
   EqR 200^2 hard_a: 100% of 40k injected starts reach the TRUE solution (single
   correct attractor, fractal temporal boundaries only). FPRM 200^2: 43.5% true
   == modal-basin share; the atlas modal basin IS the truth; ~1489 wrong-answer
   basins around it. Native init lands in the modal basin -> atlas is functional,
   centered on truth (your 'peripheral regime' objection dissolves).
2. Settling-basin interlocking (your temporal suggestion): q(r=1,k=3) = 0.2385 vs
   patch-permutation null 0.0911 +- 0.0173, z = 8.5 at 200^2. The Wada-like
   structure lives in TIME, not answer identity.
3. Ordinary LLM (Qwen3-1.7B, arithmetic 37+28): number-manifold chart via
   digit-PCA steering directions. Depth stack L8/14/20/26: smooth monotonic
   gradient (L8) -> coarser provinces (L14) -> banded (L20) -> faulted with
   diagonal fracture (L26). 65 is the adjacency hub at all depths; low-value
   error provinces form a connected community.
4. NEW (exp13, edge semantics): adjacent FPRM error basins are semantically
   closer than random pairs (Hamming 17.2 vs 30.0/81; 0.9% near-identical
   adjacent pairs vs 0.0% random). The error atlas is a SMOOTH semantic
   manifold: neighboring starts err in neighboring ways.

# Question
Goal: probe and observe the geometry/shape of latent space and its connections,
forming insights into the structure of reason. Current tools: slice charts,
settling fields, adjacency graphs, edge semantics, multi-slice stitching
(6 orientations through a common point, running now).

Given these results, rank what to build next for maximal structural insight:
(a) full 3D reconstruction from many slices (how many, which orientations,
    registration method)?
(b) boundary-tracing walks (follow semantic boundaries in activation space to
    map the atlas's road network)?
(c) cross-task/cross-model adjacency-graph comparison (universal motifs?)
(d) something else we haven't considered?
Be specific about experimental design and what each would reveal about the
structure of reason that the current artifacts cannot."""

body = json.dumps({
    "model": "deepseek/deepseek-v4-pro-0813",
    "messages": [{"role": "user", "content": BRIEF}],
    "max_tokens": 4000, "temperature": 0.4}).encode()
req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
    data=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                        "Content-Type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=900))
msg = r["choices"][0]["message"]
text = msg.get("content") or msg.get("reasoning") or ""
(OUT / "deepseek_guidance.md").write_text(text)
print(f"done {r.get('usage', {}).get('completion_tokens', 0)} tokens in {time.time()-t0:.0f}s")
print(text[:600])
