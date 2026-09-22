#!/usr/bin/env python3
"""exp19 — region trajectories through depth (the live-convergence visual).

Assign every exp18 prompt's last-token activation to the exp18 EP dictionary
at EVERY layer (0..27). Render: rows = prompts (grouped by their final-layer
region), columns = layers, cell color = region id. Shows how a prompt's
semantic region stabilizes or migrates through the network's depth — the
model's own convergence process made visible.
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
OUT = ROOT / "runs" / "exp19"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "Qwen/Qwen3-1.7B"
device = "mps"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(device).eval()

d = np.load(ROOT / "runs/exp18/ep_dict_L20.npz")
E = d["exemplars"]                          # (K, d)
prompts = list(d["prompts"])
K = E.shape[0]

# layer-wise assignment
N = len(prompts)
assign_layers = np.zeros((N, 29), dtype=int)
embs_all = np.zeros((N, 29, E.shape[1]), dtype=np.float32)
for i, p in enumerate(prompts):
    ids = tok(p, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    for L in range(29):
        embs_all[i, L] = hs[L][0, -1].float().cpu().numpy()
    if i % 20 == 0:
        print(f"embedded {i+1}/{N}", flush=True)

# center from L20 embeddings (same as exp18)
mu_dir = embs_all[:, 21].mean(0) if False else None
L20 = embs_all[:, 21]
mu_dir = L20.mean(0); mu_dir /= np.linalg.norm(mu_dir)
proj = L20 @ mu_dir
mu = mu_dir * proj.mean()
print("center rebuilt from L20 (same as exp18)", flush=True)

for i in range(N):
    for L in range(29):
        p = (embs_all[i, L] - mu)
        p /= max(np.linalg.norm(p), 1e-9)
        sims = E @ p
        assign_layers[i, L] = int(np.argmax(sims))

# order rows by final-layer region, then by stability
finals = assign_layers[:, 27]
stability = (assign_layers[:, 14:28] == finals[:, None]).mean(1)
order = np.lexsort((stability, finals))

# figure: lane chart, one color per region
fig, ax = plt.subplots(figsize=(14, 0.28 * N + 2), dpi=110)
fig.patch.set_facecolor("#050508")
cmap = plt.get_cmap("turbo", K)
for i, oi in enumerate(order):
    ax.imshow_layer = None
for i, oi in enumerate(order):
    ax.scatter(range(29), np.full(29, i), c=assign_layers[oi], cmap=cmap,
               vmin=0, vmax=K - 1, s=14, marker="s")
ax.set_xlim(-0.5, 28.5); ax.set_ylim(N - 0.5, -0.5)
ax.set_xticks(range(0, 29, 2), [f"L{L}" for L in range(0, 29, 2)],
              fontsize=7, color="#c9d4e0")
ax.set_title("region trajectories: prompts through the EP dictionary across "
             "depth (rows = prompts grouped by final region)", fontsize=9,
             color="#c9d4e0", loc="left")
ax.set_facecolor("#050508")
for s in ax.spines.values():
    s.set_color("#1c2430")
fig.savefig(OUT / "region_trajectories.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "region_trajectories.png")

# stability stat: fraction of prompts whose region at L14 == L27
stable_mid = (assign_layers[:, 14] == finals).mean()
stable_e = (assign_layers[:, 27] == finals).mean()
changes = (np.diff(assign_layers, axis=1) != 0).sum() / N
print(f"region stable L14->L27: {stable_mid:.0%}; region changes per prompt "
      f"(avg over depth): {changes:.2f}")
json.dump(dict(stable_mid=float(stable_mid), changes_per_prompt=float(changes)),
          open(OUT / "trajectory_stats.json", "w"))
