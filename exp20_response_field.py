#!/usr/bin/env python3
"""exp20 — the response field of a real prompt.

Take a real prompt ("Tell me about the city of Paris."), place a 2D slice
through its activation at the last prompt token (random orthonormal
directions), and GENERATE TEXT at every point of the grid. Classify each
generation by topic (keyword-based, interpretable) and color the plane by
topic. The result is a semantic map: provinces = what the model would talk
about, boundaries = where the conversation flips.

Panels:
1. topic map over the perturbation plane (named provinces + example output
   quoted in each province's core)
2. 2D semantic embedding of the generations themselves (PCA of output mean
   hidden states), colored by topic — the outputs arranged by meaning.
"""
import json, os, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp20"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
RES = 32
LAYER = int(os.environ.get("FB_LAYER", "14"))
SCALE = float(os.environ.get("FB_SCALE", "3.0"))
MAX_NEW = 48
PROMPT = os.environ.get("FB_PROMPT", "Tell me about the city of Paris.")
TAG = os.environ.get("FB_TAG", "paris")

TOPICS = {  # keyword -> topic name
    "eiffel": "Eiffel / landmarks",
    "tower": "Eiffel / architecture",
    "louvre": "museums / art",
    "museum": "museums / art",
    "museu": "museums / art",
    "art": "museums / art",
    "architect": "Eiffel / architecture",
    "cathedral": "Eiffel / architecture",
    "notre": "Eiffel / architecture",
    "food": "food / cafe",
    "cafe": "food / cafe",
    "restaurant": "food / cafe",
    "cuisin": "food / cafe",
    "croissant": "food / cafe",
    "wine": "food / cafe",
    "histor": "history",
    "revolution": "history",
    "centur": "history",
    "founded": "history",
    "medieval": "history",
    "seine": "geography / river",
    "river": "geography / river",
    " arrondissement": "geography / river",
    "capital": "capital / France",
    "france": "capital / France",
    "french": "capital / France",
    "population": "capital / France",
    "europ": "capital / France",
    "love": "culture / romance",
    "romant": "culture / romance",
    "fashion": "culture / romance",
    "artist": "culture / romance",
    "culture": "culture / romance",
    "touris": "tourism",
    "visit": "tourism",
    "tourist": "tourism",
}

def classify(text):
    t = text.lower()
    for kw, name in TOPICS.items():
        if kw in text.lower():
            return name
    return "other"

device = "mps"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(device).eval()

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
prompt_len = ids.shape[1]
with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states
base = hs[LAYER][0, -1].float()
rms = base.square().mean().sqrt().item()
g = torch.Generator().manual_seed(7)
Q, _ = torch.linalg.qr(torch.randn(base.numel(), 2, generator=g))
u = (Q[:, 0] / Q[:, 0].square().mean().sqrt() * rms).to(device).to(model.dtype)
v = (Q[:, 1] / Q[:, 1].square().mean().sqrt() * rms).to(device).to(model.dtype)

state = {"delta": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if hidden.shape[1] == prompt_len and state["delta"] is not None:
        hidden[0, prompt_len - 1, :] += state["delta"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = model.model.layers[LAYER].register_forward_hook(hook)

lin = torch.linspace(-SCALE, SCALE, RES)
results = []
t0 = time.time()
for i, a in enumerate(lin.tolist()):
    for j, b in enumerate(lin.tolist()):
        state["delta"] = (a * u + b * v).float()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        topic = classify(text)
        results.append(dict(i=i, j=j, a=a, b=b, topic=topic, text=text[:150]))
    if (i + 1) % 8 == 0:
        print(f"row {i+1}/{RES} ({time.time()-t0:.0f}s)", flush=True)
handle.remove()
json.dump(results, open(OUT / f"response_field_{TAG}.json", "w"), indent=1)

# topic map
topics = list({r["topic"] for r in results})
t2i = {t: k for k, t in enumerate(topics)}
tmap = np.array([t2i[r["topic"]] for r in results]).reshape(RES, RES)
print("topic distribution:", Counter(t2i[r["topic"]] for r in results).most_common())

fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=125)
fig.patch.set_facecolor("#050508")
cmap = plt.get_cmap("turbo", len(topics))
im = axes[0].imshow(tmap, cmap=cmap, interpolation="nearest", origin="lower",
                    vmin=0, vmax=len(topics) - 1)
axes[0].set_title(f"response field: topic of generated text (layer {LAYER}, "
                  f"scale +-{SCALE} rms)", fontsize=9, color="#c9d4e0", loc="left")
cbar = fig.colorbar(im, ax=axes[0], fraction=0.04)
cbar.set_ticks(range(len(topics)))
cbar.set_ticklabels(topics)
cbar.ax.tick_params(labelsize=6, colors="#c9d4e0")

# example outputs per dominant topic region (pick modal topic per quadrant)
for qi, (r0, c0) in enumerate([(0, 0), (0, RES // 2), (RES // 2, 0), (RES // 2, RES // 2)]):
    quad = [r for r in results
            if r["i"] in range(r0, r0 + RES // 2) and r["j"] in range(c0, c0 + RES // 2)]
    tc = Counter(r["topic"] for r in quad)
    top, n = tc.most_common(1)[0]
    ex = next(r["text"][:60] for r in quad if r["topic"] == top)
    axes[1].text(0.05, 0.9 - qi * 0.24, f"Q{qi+1} ({top}, {n}/{(RES//2)**2}):\n{ex!r}",
                 fontsize=7.5, color="#c9d4e0", va="top")
axes[1].set_title("modal topic + example per quadrant", fontsize=9,
                  color="#c9d4e0", loc="left")
axes[1].axis("off")
for ax in axes:
    ax.set_facecolor("#050508"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1c2430")
fig.savefig(OUT / f"response_field_map_{TAG}.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / f"response_field_map_{TAG}.png")
print("topic distribution:", Counter(r["topic"] for r in results).most_common())