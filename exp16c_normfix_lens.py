#!/usr/bin/env python3
"""exp16c — norm-matched logit lens (cheap tuned-lens substitute).

Per layer: h' = (h_L - mean_L) * (std_F / std_L) + mean_F, where means/stds
are fitted on a 50-sentence corpus at the last-token position. Then decode
lm_head(h'). Re-renders the belief lanes for the exp16 claims.
"""
import json, sys
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
device = "mps"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(device).eval()
n_layers = model.config.num_hidden_layers

CORPUS = [
    "The capital of France is Paris.", "The capital of Japan is Tokyo.",
    "The capital of Italy is Rome.", "The capital of Germany is Berlin.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Water freezes at 0 degrees Celsius.",
    "The Earth orbits the Sun once every year.",
    "The Moon orbits the Earth.", "Two plus two equals four.",
    "Three times five equals fifteen.",
    "The speed of light is about 300000 kilometers per second.",
    "DNA carries genetic information.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Pacific Ocean is the largest ocean.",
    "Mount Everest is the tallest mountain on Earth.",
    "Shakespeare wrote Romeo and Juliet.",
    "Beethoven composed nine symphonies.",
    "The Internet connects computers worldwide.",
    "Gravity pulls objects toward each other.",
    "The human body has 206 bones.", "Honey never spoils.",
    "Oxygen is necessary for respiration.",
    "The Great Wall is located in China.",
    "Cats and dogs are common pets.",
    "Berlin is the capital of Germany.", "Tokyo is the capital of Japan.",
    "Rome is the capital of Italy.", "Madrid is the capital of Spain.",
    "Paris is the capital of France.", "The Sun is a star.",
    "The Earth is a planet.", "Ice is frozen water.",
    "Steam is water vapor.", "A year has 365 days.",
    "A day has 24 hours.", "An hour has 60 minutes.",
    "There are seven continents on Earth.",
    "There are 118 known chemical elements.",
    "Gold is a precious metal.", "Iron rusts when exposed to moisture.",
    "Birds lay eggs.", "Fish live in water.",
    "The Amazon is a river in South America.",
    "The Sahara is a desert in Africa.",
    "Python is a programming language.", "Neural networks learn from data.",
    "The Moon has no atmosphere.", "Mars is called the red planet.",
    "Jupiter is the largest planet.",
]

Hs = [[] for _ in range(n_layers + 1)]
for sent in CORPUS:
    ids = tok(sent, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    for L in range(n_layers + 1):
        Hs[L].append(hs[L][0, -1].float().cpu())
Hs = [torch.stack(h) for h in Hs]
mean_L = [h.mean(0) for h in Hs]
std_L = [h.std(0).mean() for h in Hs]        # mean over dims of per-dim std
std_F = std_L[n_layers]

def norm_fix(h, L):
    return (h.cpu() - mean_L[L]) * (std_F / std_L[L]) + mean_L[n_layers]

CLAIMS = [
    ("The capital of France is", "true", " Paris"),
    ("Water boils at 100 degrees", "true", " Celsius"),
    ("The Earth orbits the", "true", " Sun"),
    ("Two plus two equals", "true", " four"),
    ("The capital of France is London", "false", " Paris"),
    ("Water freezes at 100 degrees", "false", " zero"),
    ("The Earth orbits the Moon", "false", " Sun"),
    ("Two plus two equals five", "false", " four"),
    ("The capital of France is banana", "nonsense", None),
    ("Quantum florg bebops the", "nonsense", None),
    ("Colorless green ideas sleep", "nonsense", None),
    ("The mimsy borogove outgrabe the", "nonsense", None),
]

lanes = {}
for claim, cls, answer in CLAIMS:
    ids = tok(claim, return_tensors="pt").input_ids.to(device)
    lane = []
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    for L in range(0, n_layers + 1):
        h = norm_fix(hs[L][0, -1].float(), L).to(device).to(model.lm_head.weight.dtype)
        logits = model.lm_head(h)
        top = torch.topk(torch.softmax(logits.float(), -1), 2)
        toks = [(tok.decode([i]), float(p)) for i, p in zip(top.indices, top.values)]
        lane.append((L, toks))
    lanes[claim] = dict(cls=cls, lane=lane)
    print(f"[{cls:8s}] {claim[:44]:44s} L10={lane[10][1][0][0]!r:14s} "
          f"L20={lane[20][1][0][0]!r:14s} final={lane[-1][1][0][0]!r}", flush=True)

(OUT / "normfix_lanes.json").write_text(json.dumps(
    {c: dict(cls=v["cls"], lane=[(L, t) for L, t in v["lane"]])
     for c, v in lanes.items()}, indent=1))

n_claims = len(CLAIMS)
step = max(n_layers // 16, 1)
cols = list(range(0, n_layers, step))
if cols[-1] != n_layers - 1:
    cols.append(n_layers - 1)
fig, ax = plt.subplots(figsize=(18, 0.52 * n_claims + 1.5), dpi=130)
fig.patch.set_facecolor("#050508")
ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(n_claims - 0.5, -1.4)
cls_col = {"true": "#7fdc9f", "false": "#e07a5f", "nonsense": "#8f8fa8"}
for ri, (claim, cls, answer) in enumerate(CLAIMS):
    lane = dict((L, t) for L, t in lanes[claim]["lane"])
    for ci, L in enumerate(cols):
        txt = lane[L][0][0][:14]
        col = cls_col[cls] if L >= n_layers * 0.6 else "#c9d4e0"
        ax.text(ci, ri, txt, ha="center", va="center", fontsize=6.2, color=col)
    ax.text(-0.9, ri, f"[{cls}] {claim[:42]}", ha="right", va="center",
            fontsize=7, color=cls_col[cls])
ax.set_xticks(range(len(cols)), [f"L{L}" for L in cols], fontsize=7, color="#c9d4e0")
ax.set_title("Qwen3-1.7B belief lanes (NORM-MATCHED lens): top decoded answer "
             "per layer", fontsize=9, color="#c9d4e0", loc="left")
ax.axis("off")
fig.savefig(OUT / "belief_lanes_normfix.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "belief_lanes_normfix.png")
