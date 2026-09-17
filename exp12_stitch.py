#!/usr/bin/env python3
"""exp12 — multi-slice stitching: 6 chart planes through a common point.

Every slice passes through the same base point (the last-prompt-token
residual), so N slices form a 'book' sharing a spine. Artifacts:
1. spine comparison: answer along the a-axis (b=0) for each orientation —
   anisotropy of the number manifold.
2. fan view: all slices laid out by orientation angle (x = angle, y = radial
   coordinate, color = answer) — 2D stitching of the book.
3. 3D height-field mesh per slice (answer as height over in-plane coords).
"""
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp12"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
RES = 32
LAYER = 20
SCALE = 24.0
A, B_, TRUTH = 37, 28, 65
PROMPT = f"Q: What is {A} + {B_}?\nA:"
SEEDS = [0, 1, 2, 3, 4, 5]

device = "mps"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(device).eval()

def parse_num(text):
    m = re.findall(r"-?\d+", text.split("=")[-1] if "=" in text else text)
    return int(m[0]) if m else None

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
prompt_len = ids.shape[1]
with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states
base = hs[LAYER][0, -1].float()
rms = base.square().mean().sqrt().item()

state = {"delta": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if hidden.shape[1] == prompt_len and state["delta"] is not None:
        hidden[0, prompt_len - 1, :] += state["delta"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = model.model.layers[LAYER].register_forward_hook(hook)

lin = torch.linspace(-SCALE, SCALE, RES)
charts = {}
t0 = time.time()
for seed in SEEDS:
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(base.numel(), 2, generator=g))
    u = (Q[:, 0] / Q[:, 0].square().mean().sqrt() * rms).to(device).to(model.dtype)
    v = (Q[:, 1] / Q[:, 1].square().mean().sqrt() * rms).to(device).to(model.dtype)
    amap = np.full((RES, RES), np.nan)
    for i, a in enumerate(lin.tolist()):
        for j, b in enumerate(lin.tolist()):
            state["delta"] = (a * u + b * v).float()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=12, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            amap[i, j] = parse_num(tok.decode(
                out[0, prompt_len:], skip_special_tokens=True))
    charts[seed] = amap
    print(f"slice seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(OUT / f"chart_seed{seed}.npz", amap=amap, seed=seed,
                        layer=LAYER, scale=SCALE)
handle.remove()

# ---- 1. spine comparison (b=0 row) ----
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
fig.patch.set_facecolor("#050508")
cmap = plt.get_cmap("turbo")
for seed in SEEDS:
    spine = charts[seed][RES // 2, :]  # row nearest b=0? grid rows are a; use middle col?
    # b=0 -> column index RES//2 (columns are b)
    spine = charts[seed][:, RES // 2]
    ax.plot(lin.numpy(), spine, ".", ms=5, label=f"orientation {seed}")
ax.axhline(TRUTH, color="#e05f2f", lw=1, ls="--", label=f"truth {TRUTH}")
ax.set_xlabel("a (along u)", color="#c9d4e0")
ax.set_ylabel("answer value", color="#c9d4e0")
ax.set_title(f"spine comparison: answer along b=0 through the base point, "
             f"6 orientations (layer {LAYER})", fontsize=9, color="#c9d4e0", loc="left")
ax.legend(fontsize=7, facecolor="#0a0a12", labelcolor="#c9d4e0")
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors="#c9d4e0")
fig.savefig(OUT / "spine_comparison.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "spine_comparison.png")

# ---- 2. fan view: angle x radial ----
fig, ax = plt.subplots(figsize=(13, 5), dpi=120)
fig.patch.set_facecolor("#050508")
for si, seed in enumerate(SEEDS):
    amap = charts[seed]
    # radial = along a from center row; use the middle column as before
    col = amap[:, RES // 2]
    rs = lin.numpy()
    ax.scatter(rs + si * 0 * 0, col, s=6, alpha=0.7, label=f"orient {seed}")
ax.axhline(TRUTH, color="#e05f2f", lw=1, ls="--")
ax.set_xlabel("a (radial coordinate through base point)", color="#c9d4e0")
ax.set_ylabel("answer", color="#c9d4e0")
ax.set_title("fan view: answer vs radial coordinate for each slice orientation",
             fontsize=9, color="#c9d4e0", loc="left")
ax.legend(fontsize=7, facecolor="#0a0a12", labelcolor="#c9d4e0", ncol=3)
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors="#c9d4e0")
fig.savefig(OUT / "fan_view.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "fan_view.png")

# ---- 3. 3D height fields ----
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(16, 9), dpi=110)
fig.patch.set_facecolor("#050508")
AA, BB = np.meshgrid(lin.numpy(), lin.numpy(), indexing="ij")
for si, seed in enumerate(SEEDS[:4]):
    amap = charts[seed]
    ax = fig.add_subplot(2, 2, si + 1, projection="3d")
    m = ~np.isnan(amap)
    ax.plot_trisurf(AA[m], BB[m], amap[m], cmap="viridis", edgecolor="none",
                    alpha=0.9)
    ax.set_title(f"orientation {seed}", fontsize=8, color="#c9d4e0")
    ax.set_zlim(-10, 100)
    ax.tick_params(colors="#c9d4e0", labelsize=6)
fig.savefig(OUT / "height_fields_3d.png", facecolor="#050508", bbox_inches="tight")
print("wrote", OUT / "height_fields_3d.png")
