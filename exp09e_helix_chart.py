#!/usr/bin/env python3
"""exp09e — number chart in HELICAL coordinates (Kantamneni & Tegmark 2025).

Numbers in LLMs are encoded on a generalized helix: a linear monotone
component plus a circular (Fourier) component of period p. We fit the helix
basis directly from digit-token unembedding rows:

  rows(d) ~ alpha*d (linear)  +  R*(cos(w*d), sin(w*d)) (circular)

fit w by scanning periods and maximizing explained variance of the circular
component on the residual. Then chart the activation plane in
(linear-direction, circular-direction) coordinates. If the helix hypothesis
holds on Qwen3-1.7B, the answer-value chart should show periodic band
structure with the fitted period — a direct visual confirmation, and a
discriminator between the helix / linear / digit-wise encoding hypotheses.
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
OUT = ROOT / "runs" / "exp09e"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
RES = 32
LAYER = int(os.environ.get("FB_LAYER", "20"))
MAX_NEW = 12
A, B_, TRUTH = 37, 28, 65
PROMPT = f"Q: What is {A} + {B_}?\nA:"

device = "mps"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(device).eval()

digit_ids = [tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"]
U = model.lm_head.weight.detach().float()
rows = U[digit_ids]                                   # (10, d)
dvals = np.arange(10, dtype=float)

# --- fit helix: linear component + circular residual ---
rows_np = rows.cpu().numpy()
lin_dir = np.polyfit(dvals, rows_np, 1)               # slope per feature
linear_pred = lin_dir[0][None, :] * dvals[:, None] + lin_dir[1][None, :]
resid = rows_np - linear_pred                         # (10, d) circular part
# scan period: project residual on (cos, sin) of digit*2pi/p, keep best var
best = (None, 0, None, None)
for p10 in np.linspace(0.8, 20, 400):
    theta = dvals / p10 * 2 * np.pi
    C, S = np.cos(theta), np.sin(theta)
    # fit residual ~ C*cvec + S*svec (least squares per feature)
    A_mat = np.stack([C, S], 1)
    coef, *_ = np.linalg.lstsq(A_mat, resid, rcond=None)
    pred = A_mat @ coef
    var = ((resid - pred) ** 2).sum()
    circ_var = (pred ** 2).sum() / ((resid - pred) ** 2).sum() + (pred ** 2).sum()
    score = pred.var() / max(resid.var(), 1e-12)
    if score > best[1]:
        best = (p10, score, coef, pred)
p10, score, coef, pred = best
print(f"best helix period (in digit units): {p10:.2f}, "
      f"circular-variance fraction {score:.2f}", flush=True)

# basis vectors in feature space:
#   u = linear direction (unit), v = circular direction (unit, from coef)
lin_d = lin_dir[0] / np.linalg.norm(lin_dir[0])
# circular direction in feature space: top PCA of the circular prediction
pred = A_mat @ coef                                    # (10, d) circular part
circ = pred - pred.mean(0)
# PCA the circular directions to one principal circular axis
circ_c = circ - circ.mean(0)
_, _, Vt = np.linalg.svd(circ_c, full_matrices=False)
circ_d = Vt[0] / np.linalg.norm(Vt[0])

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
prompt_len = ids.shape[1]
with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states
base = hs[LAYER][0, -1].float()
rms = base.square().mean().sqrt().item()
SCALE = float(os.environ.get("FB_SCALE", "3.0"))   # semantic dirs are efficient
u = (torch.tensor(lin_d, dtype=torch.float32) / np.linalg.norm(lin_d) * rms).to(device).to(model.dtype)
v = (torch.tensor(circ_d, dtype=torch.float32) / np.linalg.norm(circ_d) * rms).to(device).to(model.dtype)

state = {"delta": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if hidden.shape[1] == prompt_len and state["delta"] is not None:
        hidden[0, prompt_len - 1, :] += state["delta"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = model.model.layers[LAYER].register_forward_hook(hook)

lin = torch.linspace(-SCALE, SCALE, RES)
pts = [(a, b) for a in lin.tolist() for b in lin.tolist()]
results = []
t0 = time.time()
for i, (a, b) in enumerate(pts):
    state["delta"] = (a * u + b * v).float()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
    m = re.findall(r"-?\d+", text.split("=")[-1] if "=" in text else text)
    results.append(dict(a=a, b=b, answer=int(m[0]) if m else None))
    if i % 128 == 0:
        print(f"[{i+1}/{len(pts)}] ({a:+.2f},{b:+.2f}) -> {results[-1]['answer']} "
              f"({time.time()-t0:.0f}s)", flush=True)
handle.remove()
(OUT / f"helix_chart_L{LAYER}.json").write_text(json.dumps(
    dict(period_digits=float(p10), results=results), indent=1))

# ---- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
answers = [r["answer"] for r in results]
known = [a for a in answers if a is not None]
print(f"answers: {len(set(known))} distinct, range {min(known)}..{max(known)}")
amap = np.array([a if a is not None else np.nan for a in answers],
                dtype=float).reshape(RES, RES)
fig, ax = plt.subplots(figsize=(8.6, 8.6), dpi=130)
fig.patch.set_facecolor("#050508")
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#2a2a33")
im = ax.imshow(np.ma.masked_invalid(np.clip(amap, 0, 95)), cmap=cmap,
               interpolation="nearest", origin="lower")
ax.set_title(f"HELIX-coordinate chart (period {p10:.1f} digits), layer {LAYER}\n"
             f"x = linear magnitude dir, y = circular dir; "
             f"{len(set(known))} distinct answers", fontsize=9,
             color="#c9d4e0", loc="left")
fig.colorbar(im, ax=ax, fraction=0.04)
ax.set_facecolor("#050508"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#1c2430")
fig.savefig(OUT / f"helix_chart_L{LAYER}.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / f"helix_chart_L{LAYER}.png")
