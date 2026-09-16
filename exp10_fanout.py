#!/usr/bin/env python3
"""Fan-out: three expert reviews of the fractal-basins research program via OpenRouter.
1. GLM-5.3       — attack the Levin-ingression philosophy
2. deepseek-v4-pro-0813 — rigorous Wada-test protocol design
3. nemotron-3-ultra-550b — novelty referee (prior work)
Responses saved to runs/reviews/.
"""
import json, os, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "reviews"
OUT.mkdir(parents=True, exist_ok=True)
KEY = os.environ["OPENROUTER_API_KEY"]

def call(model, system, user, max_tokens=4000):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.4}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=body, headers={"Authorization": f"Bearer {KEY}",
                            "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    usage = r.get("usage", {})
    text = (msg.get("content") or msg.get("reasoning") or "")
    return text, (usage.get("completion_tokens", 0), dt)

PHIL = (ROOT / "PHILOSOPHY.md").read_text()
STATE = (ROOT / "RESEARCH_STATE.md").read_text()
CONTEXT = f"# RESEARCH STATE (measured results)\n\n{STATE}\n\n# PHILOSOPHY DRAFT (the text to review)\n\n{PHIL}"

REVIEWS = [
    ("glm-5.3",
     "You are a sharp philosopher of mind and cognitive science, in the tradition of critics of panpsychist and Leibnizian arguments. Be adversarial but fair.",
     f"""The draft below claims that LLM answer-basins (measured: 1489 distinct answer basins under injected latent starts, one dominant at 43%, approximate Wada boundaries 98.1% at 5x5 resolution) support a Levin-style 'ingression' account: weights are a Platonic space of stable thoughts, prompts are boundary conditions, inference selects rather than generates.

Attack it. Specifically:
1. Is 'basin-selection = ingression' anything more than Leibnizian preformationism relabeled? What would distinguish the two empirically?
2. The cognitive-light-cone = basin identification: is this a legitimate operationalization of Levin's concept or a category error (his cones are about goal-space horizons of agents, not basins of dynamical maps)?
3. The frozen-landscape idealization for autoregressive models: how much of the philosophical weight survives?
4. For each objection you raise, classify it FATAL / SERIOUS / COSMETIC, and for SERIOUS ones propose what experiment would defuse it.

{CONTEXT}""", 4000),

    ("deepseek/deepseek-v4-pro-0813",
     "You are a rigorous applied-mathematician/statistician specializing in dynamical systems and uncertainty quantification.",
     f"""We measured 'answer-identity basins' of a reasoning model on a 2D slice of latent space (64x64 grid, categorical labels, 1489 classes, dominant class 43.5%). A crude Wada proxy: 98.1% of boundary pixels (pixels adjacent to a different label) have >=3 distinct labels within their 5x5 neighborhood. The standard Wada property requires EVERY boundary point to be on the boundary of ALL basins (infinitely many in the continuum); with 1489 basins even approximate Wada needs careful formulation.

Design a rigorous, statistically defensible test for 'approximate Wada structure' on finite-resolution categorical maps:
1. Formalize 'p-Wada' or 'k-Wada at resolution r' as a testable property (k basins within radius r of a boundary point).
2. Null models: what null distributions must we compare against (spatial shuffles, random Voronoi partitions, cluster-grown nulls, Markov random fields)? Which nulls distinguish Wada structure from merely many-basins-with-wiggly-boundaries?
3. Finite-size effects: pixel dependence, boundary-pixel vs area ratio, resolution scaling (we can rerun at 32/64/128). What scaling analysis makes the claim resolution-robust?
4. Multiple-testing and the 98.1% figure: how should uncertainty be quantified (bootstrap over map regions? block bootstrap?).
5. Give concrete pseudocode for the test we should implement, and state exactly what claim would be defensible in a paper ('near-Wada with k=..., at resolution ..., under null ...').

{CONTEXT}""", 6000),

    ("nvidia/nemotron-3-ultra-550b-a55b",
     "You are a meticulous research librarian and novelty assessor for ML/neuroscience/complex-systems venues. Cite only work you are confident exists; mark uncertain items UNCERTAIN.",
     f"""Novelty check for these claims:
A. Answer-space basin maps of LLMs/looped reasoners exhibit (approximately) Wada basin structure — interlocked basins where boundary points touch many basins.
B. Cognitive light cones (Levin) operationalized as basins of attraction in LLM latent/answer space; ingression-style reading of LLM inference (atlas pre-exists query; prompt = boundary condition).
C. Early-trajectory divergence (neighbor FLI at loop 3) predicts both settling time (rho 0.58) and final answer wrongness (rho 0.86 with settling) in reasoning models.

For each: list the closest prior work you know (title, venue/year, one-line relation), state whether the exact claim appears published, and give a novelty rating (S = would surprise the field / A = new combination of known pieces / B = incremental). Flag anything from 2025-2026 preprints we should read first.

{CONTEXT}""", 4000),
]

for model, system, user, mt in REVIEWS:
    name = model.split("/")[-1].replace(":", "_")
    print(f"=== {model} ===", flush=True)
    try:
        text, (ct, dt) = call(model, system, user, mt)
        (OUT / f"{name}.md").write_text(text)
        print(f"done: {ct} tokens in {dt:.0f}s -> {OUT / f'{name}.md'}", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        (OUT / f"{name}.FAILED").write_text(str(e))
