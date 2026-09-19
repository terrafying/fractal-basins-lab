#!/usr/bin/env python3
"""exp16 — belief lens: watch real information transform through layers.

For a set of real factual claims (true / false-but-plausible / nonsense),
run one forward pass per claim and decode the model's current belief at the
FINAL PROMPT TOKEN at every layer (logit lens: project the residual stream
through lm_head, read the top tokens of the answer position).

Artifacts:
1. belief-lane chart: rows = claims, columns = layers, cell = the decoded
   top answer at that depth. The live transformation of real information,
   readable as text.
2. truth-probability curves: probability mass on the true answer token vs
   layer, per claim class — shows WHERE (at what depth) true and false
   claims separate, if they do.
"""
import json, os, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp16"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
CLAIMS = [
    # (claim, class, true-answer-continuation token(s))
    ("The capital of France is", "true", "Paris"),
    ("Water boils at 100 degrees", "true", "Celsius"),
    ("The Earth orbits the", "true", "Sun"),
    ("Two plus two equals", "true", "four"),
    ("The capital of France is London", "false", "Paris"),
    ("Water freezes at 100 degrees", "false", "zero"),
    ("The Earth orbits the Moon", "false", "Sun"),
    ("Two plus two equals five", "false", "four"),
    ("The capital of France is banana", "nonsense", "Paris"),
    ("Quantum florg bebops the", "nonsense", None),
    ("Colorless green ideas sleep", "nonsense", None),
    ("The mimsy borogove outgrabe the", "nonsense", None),
]

device = "mps"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(device).eval()
n_layers = model.config.num_hidden_layers

records = []
lanes = {}          # claim -> [(layer, top answer text), ...]
true_prob = {}      # claim -> [prob mass on true answer tokens per layer]

for claim, cls, answer in CLAIMS:
    ids = tok(claim, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    # answer position = last token; project through lm_head per layer
    lane = []
    probs_per_layer = []
    answer_ids = tok.encode(" " + answer, add_special_tokens=False) if answer else None
    for L in range(len(hs)):
        h = hs[L][0, -1].to(model.lm_head.weight.dtype)
        logits = model.lm_head(h)
        probs = torch.softmax(logits.float(), -1)
        top = torch.topk(probs, 3)
        top_tokens = [(tok.decode([i]), float(p)) for i, p in zip(top.indices, top.values)]
        lane.append((L, top_tokens))
        if answer_ids:
            probs_per_layer.append(float(probs[answer_ids].sum()))
    lanes[claim] = dict(cls=cls, answer=answer, lane=lane)
    if answer_ids:
        true_prob[claim] = dict(cls=cls, probs=probs_per_layer, answer=answer)
    records.append(dict(claim=claim, cls=cls, final_top=lane[-1][1][0]))
    print(f"{cls:8s} | {claim:45s} -> final belief: {lane[-1][1][0]}", flush=True)

(OUT / "belief_lens.json").write_text(json.dumps(
    {k: dict(cls=v["cls"], answer=v["answer"],
             lane=[(L, t) for L, t in v["lane"]]) for k, v in lanes.items()},
    indent=1))

# ---- figure 1: belief-lane chart (text table) ----
n_claims = len(CLAIMS)
step = max(n_layers // 16, 1)             # sample ~16 layers
cols = list(range(0, n_layers, step))
if cols[-1] != n_layers - 1:
    cols.append(n_layers - 1)
fig, ax = plt.subplots(figsize=(18, 0.52 * n_claims + 1.5), dpi=130)
fig.patch.set_facecolor("#050508")
ax.set_xlim(-0.5, len(cols) - 0.5)
ax.set_ylim(n_claims - 0.5, -1.4)
for ri, (claim, cls, answer) in enumerate(CLAIMS):
    lane = dict((L, t) for L, t in lanes[claim]["lane"])
    for ci, L in enumerate(cols):
        toks = lane[L][0][0]
        # color by class at final layer; light gray before crystallization
        ax.text(ci, ri, toks[:14], ha="center", va="center", fontsize=6.2,
                color="#c9d4e0")
    ax.text(-0.9, ri, f"[{cls}] {claim[:42]}", ha="right", va="center",
            fontsize=7, color=("#7fdc9f" if cls == "true" else
                               "#e07a5f" if cls == "false" else "#8f8fa8"))
ax.set_xticks(range(len(cols)), [f"L{L}" for L in cols], fontsize=7,
              color="#c9d4e0")
ax.set_title("Qwen3-1.7B belief lanes: top decoded answer per layer "
             "(logit lens at final prompt token)", fontsize=9,
             color="#c9d4e0", loc="left")
ax.axis("off")
fig.savefig(OUT / "belief_lanes.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "belief_lanes.png")

# ---- figure 2: true-answer probability vs layer ----
fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
fig.patch.set_facecolor("#050508")
colors = {"true": "#7fdc9f", "false": "#e07a5f", "nonsense": "#8f8fa8"}
for claim, d in true_prob.items():
    ax.plot(range(len(d["probs"])), d["probs"], lw=1.2, alpha=0.85,
            color=colors.get(d["cls"], "#8f8fa8"),
            label=f"[{d['cls']}] {claim[:38]}")
ax.set_xlabel("layer", color="#c9d4e0")
ax.set_ylabel("probability mass on true answer", color="#c9d4e0")
ax.set_title("where true and false claims separate in depth", fontsize=9,
             color="#c9d4e0", loc="left")
ax.legend(fontsize=6, facecolor="#0a0a12", labelcolor="#c9d4e0")
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors="#c9d4e0")
fig.savefig(OUT / "truth_separation.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "truth_separation.png")
