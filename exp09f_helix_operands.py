#!/usr/bin/env python3
"""exp09f — helix test at the OPERAND token positions (the decisive variant).

exp09e patched the generation position and found the answer immune — the
answer is committed before "A:". Kantamneni & Tegmark's Clock algorithm acts
on the OPERAND digits inside the prompt. Here we patch the residual stream at
every digit-token position of the two operands (37 and 28), along the fitted
helix directions (linear + circular, period fitted from digit unembeddings).

Sweep scales. Expected if the helix hypothesis holds on this model: the
answer shifts by predictable amounts (neighboring numbers), with band
structure whose period matches the fitted helix period.
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
OUT = ROOT / "runs" / "exp09f"
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

def parse_num(text):
    m = re.findall(r"-?\d+", text.split("=")[-1] if "=" in text else text)
    return int(m[0]) if m else None

# ---- helix basis from digit unembedding rows (same fit as exp09e) ----
digit_ids = [tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"]
U = model.lm_head.weight.detach().float().cpu()
rows = U[digit_ids].numpy()
dvals = np.arange(10, dtype=float)
lin_dir = np.polyfit(dvals, rows, 1)[0]
linear_pred = lin_dir[None, :] * dvals[:, None] + np.polyfit(
    dvals, rows, 1)[1][None, :]
resid = rows - linear_pred
best = (None, -1, None)
for p10 in np.linspace(0.8, 20, 400):
    theta = dvals / p10 * 2 * np.pi
    A_mat = np.stack([np.cos(theta), np.sin(theta)], 1)
    coef, *_ = np.linalg.lstsq(A_mat, resid, rcond=None)
    pred = A_mat @ coef
    score = pred.var() / max(resid.var(), 1e-12)
    if score > best[1]:
        best = (p10, score, pred)
p10, hscore, pred = best
pred_c = pred - pred.mean(0)
_, _, Vt = np.linalg.svd(pred_c, full_matrices=False)
print(f"helix fit: period {p10:.2f} digits, circ-var {hscore:.2f}", flush=True)

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
prompt_len = ids.shape[1]
# locate digit-token positions (the operands 3,7 and 2,8)
digit_pos = [i for i in range(prompt_len)
             if tok.decode([ids[0, i].item()]) in "0123456789"]
print("digit positions:", digit_pos,
      [tok.decode([ids[0, i].item()]) for i in digit_pos], flush=True)

with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states
base = hs[LAYER][0, -1].float()
rms = base.square().mean().sqrt().item()

# directions: linear (u) and circular (v) from the helix fit, unit-scaled
lin_u = lin_dir / np.linalg.norm(lin_dir)
circ_d = Vt[0] / np.linalg.norm(Vt[0])
u_t = torch.tensor(lin_u, dtype=torch.float32) * rms
v_t = torch.tensor(circ_d, dtype=torch.float32) * rms

state = {"delta": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if hidden.shape[1] == prompt_len and state["delta"] is not None:
        for p in digit_pos:
            hidden[0, p, :] += state["delta"].to(hidden.dtype)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

handle = model.model.layers[LAYER].register_forward_hook(hook)
lin = torch.linspace(-3.0, 3.0, RES)   # semantic dirs: small scales suffice
pts = [(a, b) for a in lin.tolist() for b in lin.tolist()]
results = []
t0 = time.time()
for i, (a, b) in enumerate(pts):
    state["delta"] = (a * u_t + b * v_t).to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
    results.append(dict(a=a, b=b, answer=parse_num(text)))
    if i % 128 == 0:
        print(f"[{i+1}/{len(pts)}] ({a:+.2f},{b:+.2f}) -> {results[-1]['answer']} "
              f"({time.time()-t0:.0f}s)", flush=True)
handle.remove()
(OUT / f"helix_operand_chart_L{LAYER}.json").write_text(json.dumps(results, indent=1))

# ---- figure ----
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
ax.set_title(f"HELIX directions at OPERAND positions, layer {LAYER}\n"
             f"x = linear dir, y = circular dir (period {p10:.1f}); "
             f"{len(set(known))} distinct answers", fontsize=9,
             color="#c9d4e0", loc="left")
fig.colorbar(im, ax=ax, fraction=0.04)
ax.set_facecolor("#050508"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#1c2430")
fig.savefig(OUT / f"helix_operand_chart_L{LAYER}.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / f"helix_operand_chart_L{LAYER}.png")
