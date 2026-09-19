#!/usr/bin/env python3
"""exp16b — tuned lens for Qwen3-1.7B (translation probes).

The naive logit lens produces garbage at mid layers (representation space is
rotated per layer). Fix (Belrose et al. 2023, cheap variant): fit a per-layer
linear map W_L: hidden_L -> hidden_final on a corpus, then decode
lm_head(W_L h_L). Closed-form ridge, no SGD.

Corpus: a few hundred short factual/varied sentences written inline (real
information, no download). Then re-render the belief lanes from exp16 with
the calibrated lens.
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

CORPUS = [
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
    "The capital of Italy is Rome.",
    "The capital of Spain is Madrid.",
    "The capital of Germany is Berlin.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Water freezes at 0 degrees Celsius.",
    "The Earth orbits the Sun once every year.",
    "The Moon orbits the Earth.",
    "Two plus two equals four.",
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
    "The human body has 206 bones.",
    "Honey never spoils.",
    "Oxygen is necessary for respiration.",
    "The Great Wall is located in China.",
    "Cats and dogs are common pets.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan.",
    "Rome is the capital of Italy.",
    "Madrid is the capital of Spain.",
    "Paris is the capital of France.",
    "The Sun is a star.",
    "The Earth is a planet.",
    "Ice is frozen water.",
    "Steam is water vapor.",
    "A year has 365 days.",
    "A day has 24 hours.",
    "An hour has 60 minutes.",
    "There are seven continents on Earth.",
    "There are 118 known chemical elements.",
    "Gold is a precious metal.",
    "Iron rusts when exposed to moisture.",
    "Birds lay eggs.",
    "Fish live in water.",
    "The Amazon is a river in South America.",
    "The Sahara is a desert in Africa.",
    "Python is a programming language.",
    "Neural networks learn from data.",
    "The Moon has no atmosphere.",
    "Mars is called the red planet.",
    "Jupiter is the largest planet.",
]

def last_hidden_and_final(ids):
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states, out.logits[0, -1].float()

# ---- collect (hidden_L, hidden_final, logits_final) at last token ----
H = {L: [] for L in range(len(CORPUS[0:1]))}  # placeholder
n_layers = model.config.num_hidden_layers
Hs = [[] for _ in range(n_layers + 1)]
F = []
for sent in CORPUS:
    ids = tok(sent, return_tensors="pt").input_ids.to(device)
    hs, logits = last_hidden_and_final(ids)
    for L in range(n_layers + 1):
        Hs[L].append(hs[L][0, -1].float().cpu())
    F.append(logits.cpu())
print(f"corpus: {len(CORPUS)} sentences", flush=True)

# ---- fit per-layer ridge translation to final hidden ----
F_mat = torch.stack(F)                                     # (N, d)
probes = {}
for L in range(n_layers):                                  # skip final layer
    X = torch.stack(Hs[L])                                 # (N, d)
    Y = torch.stack(Hs[n_layers])                          # (N, d)
    # ridge closed form: W = (X^T X + lam I)^-1 X^T Y ; plus intercept
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    lam = 1e-2 * torch.eye(X.shape[1], device="cpu") * X.shape[0]
    W = torch.linalg.solve(Xc.T @ Xc + lam, Xc.T @ Yc)
    b = (Y.mean(0) - X.mean(0) @ W).float()
    probes[L] = (W.to(device), b.to(device))
    # in-sample quality
    pred = X.to(device) @ W.to(device) + b.to(device)
    r2 = 1 - ((pred.cpu() - Yc) ** 2).sum() / ((Yc) ** 2).sum()
    print(f"probe L{L:2d}: R^2 = {float(r2):.3f}", flush=True)

json.dump({f"L{L}": float(1 - ((torch.stack(Hs[L]).to(device) @ probes[L][0].to(device)
            + probes[L][1].to(device)).cpu() - torch.stack(Hs[n_layers])) ** 2).sum()
           / ((torch.stack(Hs[n_layers]) - torch.stack(Hs[n_layers]).mean(0)) ** 2).sum()
           for L in range(n_layers)}, open(OUT / "probe_r2.json", "w"))

# ---- save probes for exp16 lanes ----
torch.save({L: tuple(t.cpu() for t in probes[L]) for L in probes},
           OUT / "tuned_probes.pt")
print("wrote", OUT / "tuned_probes.pt")
