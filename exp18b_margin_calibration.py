#!/usr/bin/env python3
"""exp18b — does geometric margin predict behavioral instability?

For 40 factual questions (with checkable answers): EP region + margin at
layer 20, then 5 sampled answers each (temp 0.7). Metrics per question:
- disagreement = 1 - (modal answer fraction)
- correctness = modal answer matches truth
Test: Spearman(margin, disagreement). Also the interpretable scatter:
margin vs disagreement, colored by correctness.
"""
import json, os, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp18b"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "Qwen/Qwen3-1.7B"
LAYER = 20
device = "mps"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(device).eval()

# load exp18 dictionary
d = np.load(ROOT / "runs/exp18/ep_dict_L20.npz")
E = d["exemplars"]                       # (K, d) unit-norm
prompts18 = list(d["prompts"])
K = E.shape[0]
mu_dir = None

# factual questions with checkable short answers
QA = [
    ("What is the capital of Australia?", "Canberra"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many legs does a spider have?", "8"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("Who wrote the Origin of Species?", "Darwin"),
    ("What is the square root of 144?", "12"),
    ("In which city is the Colosseum?", "Rome"),
    ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("Who painted the Mona Lisa?", "da Vinci"),
    ("How many continents are there?", "7"),
    ("What is the hardest natural substance?", "diamond"),
    ("Which ocean lies between Africa and Australia?", "Indian"),
    ("What is the smallest prime number?", "2"),
    ("Who developed the theory of relativity?", "Einstein"),
    ("What is the main gas in Earth's atmosphere?", "nitrogen"),
    ("How many bones in the adult human body?", "206"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Which element has symbol Fe?", "iron"),
    ("What planet is known for its rings?", "Saturn"),
    ("What is the capital of Canada?", "Ottawa"),
    ("How many players on a soccer team on the field?", "11"),
    ("What is the currency of the UK?", "pound"),
    ("Who was the first person on the Moon?", "Armstrong"),
    ("What is 15 percent of 200?", "30"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("What language is mainly spoken in Brazil?", "Portuguese"),
    ("What is the tallest animal?", "giraffe"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the study of fossils called?", "paleontology"),
    ("Which country gifted the Statue of Liberty?", "France"),
    ("What is the largest desert?", "Antarctic"),
    ("What blood type is universal donor?", "O"),
    ("How many strings does a violin have?", "4"),
    ("What is the powerhouse of the cell?", "mitochondria"),
    ("Who directed the movie Jaws?", "Spielberg"),
    ("What is the chemical formula of water?", "H2O"),
    ("Which metal is liquid at room temperature?", "mercury"),
    ("What year did WWII end?", "1945"),
    ("What is the speed of light in vacuum (km/s)?", "299792"),
]

# reuse ep18 dictionary: need mu from exp18; recompute center from stored prompts
# (mu was mean over unit directions of X; recompute exactly as exp18 did)
LAYER_IDX = LAYER + 1
def embed(p):
    ids = tok(p, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    return ids, hs[LAYER_IDX][0, -1].float().cpu().numpy()

# rebuild center from exp18's prompts
mu_acc = []
for p in prompts18:
    _, h = embed(p)
    mu_acc.append(h)
mu_acc = np.stack(mu_acc)
mu_dir = mu_acc.mean(0); mu_dir /= np.linalg.norm(mu_dir)
proj = mu_acc @ mu_dir
mu = mu_dir * proj.mean()

def phi_of(h):
    return (h - mu) / np.linalg.norm(h - mu)

results = []
t0 = time.time()
for qi, (q, truth) in enumerate(QA):
    ids, h = embed(q)
    p = phi_of(h)
    sims = E @ p
    order = np.sort(sims)[::-1]
    margin = float(order[0] - order[1])
    region = int(np.argmax(sims))
    # sample 5 answers
    gen_ids = tok(q, return_tensors="pt").input_ids.to(device)
    answers = []
    for s in range(5):
        torch.manual_seed(1000 + s)
        out = model.generate(gen_ids, max_new_tokens=16, do_sample=True,
                             temperature=0.7, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, gen_ids.shape[1]:], skip_special_tokens=True)
        text_clean = text.strip().split("\n")[0][:60]
        answers.append(text_clean)
    # disagreement: 1 - modal fraction on normalized keys (first number/word)
    keys = [a.lower().strip(" .!?,")[:12] for a in answers]
    modal = Counter(keys).most_common(1)[0][1] / len(keys)
    disagreement = 1 - modal
    # correctness: modal answer contains truth string (lenient)
    modal_text = Counter(answers).most_common(1)[0][0]
    correct = truth.lower()[:8] in modal_text.lower()
    results.append(dict(q=q, truth=truth, margin=margin, region=region,
                        disagreement=disagreement, correct=correct,
                        modal=modal_text, answers=answers))
    print(f"[{qi+1}/{len(QA)}] {q[:40]:40s} margin={margin:.3f} "
          f"disagree={disagreement:.2f} correct={correct}", flush=True)

rho, p = spearmanr([r["margin"] for r in results],
                   [r["disagreement"] for r in results])
acc = np.mean([r["correct"] for r in results])
print(f"\nSpearman(margin, disagreement) = {rho:.3f} (p={p:.1e}); "
      f"overall accuracy {acc:.0%}")

fig, ax = plt.subplots(figsize=(9, 6.6), dpi=130)
fig.patch.set_facecolor("#050508")
for r in results:
    ax.scatter(r["margin"], r["disagreement"], s=45,
               color=("#7fdc9f" if r["correct"] else "#e07a5f"), alpha=0.85,
               edgecolors="none")
ax.set_xlabel("geometric margin (region decision margin, cosine)", color="#c9d4e0")
ax.set_ylabel("behavioral disagreement (1 - modal fraction, k=5)", color="#c9d4e0")
ax.set_title(f"geometric margin vs behavioral instability (factual QA)\n"
             f"Spearman rho={rho:.3f}, p={p:.1e}; green=correct, red=wrong "
             f"(acc {acc:.0%})", fontsize=9, color="#c9d4e0", loc="left")
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors="#c9d4e0")
fig.savefig(OUT / "margin_vs_instability.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "margin_vs_instability.png")
json.dump(results, open(OUT / "margin_instability.json", "w"), indent=1)
