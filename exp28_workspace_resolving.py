#!/usr/bin/env python3
"""exp28 — the workspace resolving: J-lens slice animation over generation.

For a multi-hop prompt, generate greedily. After each new output token,
recompute the J-lens top-1 token at EVERY (position, layer). Each output
token = one frame: columns = input+output positions so far, rows = layers.
The movie shows the workspace resolving as the prompt is read (frames 0)
and as each output token is produced.

Prompt chosen for a visible multi-hop intermediate: maple leaf -> Canada ->
Ottawa.
"""
import json, os, subprocess
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transformers, jlens

MONO = ["DejaVu Sans Mono", "Hiragino Sans GB", "Arial Unicode MS"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = MONO
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp28"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-1.7b_jacobian_lens.pt")
model = jlens.from_hf(hf, tok)

PROMPT = ("Q: What is the capital of the country whose flag features a "
          "maple leaf?\nA:")
N_OUT = 14

ids = tok(PROMPT, return_tensors="pt").input_ids.to("mps")
prompt_len = ids.shape[1]
gen = []
frames_meta = []

layer_logits = {}
def grab(layer, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    layer_logits[layer] = hidden

fig = plt.figure(figsize=(15.5, 10.5), dpi=110)
VOID, INK = "#050508", "#c9d4e0"

def render_frame(step, new_tok, pinned_tracks):
    """slice grid: rows=layers, cols=positions; text=top-1 lens token."""
    ax = fig.add_subplot(111)
    ax.clear()
    ax.set_facecolor(VOID)
    ax.axis("off")
    n_layers = max(layer_logits) + 1
    grid = np.empty((n_layers, ids.shape[1]), dtype=object)
    grid[:] = ""
    with torch.no_grad():
        for L in layer_logits:
            if L not in lens.jacobians:
                continue
            tr = layer_logits[L][0].to(torch.bfloat16) @ lens.jacobians[L].to("mps").to(torch.bfloat16).T
            lg = hf.lm_head(hf.model.norm(tr))
            top1 = lg.argmax(-1).cpu()
            for p in range(ids.shape[1]):
                grid[L, p] = tok.decode([top1[p]]).strip() or "·"
    im = ax.imshow(np.zeros((n_layers, ids.shape[1])), cmap="Greys",
                   vmin=-1, vmax=0, aspect="auto")
    p0 = max(0, ids.shape[1] - 26)   # crop to the tail for readability
    for L in range(n_layers):
        for p in range(p0, ids.shape[1]):
            fresh = p >= prompt_len + max(0, step - 1)
            ax.text(p - p0, L, grid[L, p], ha="center", va="center",
                    fontsize=7,
                    color="#eafcf7" if fresh else "#93a1b3",
                    family=MONO)
    ax.axvline(prompt_len - 0.5 - p0, color="#e07a5f", lw=1.2, ls="--")
    tok_disp = new_tok.replace("\\n", "\\n").strip()[:24] or "·"
    ax.set_title(f"the workspace resolving - step {step}: new token "
                 f"[{tok_disp}]   (prompt | output split = dashed)",
                 fontsize=12, color=INK, family=MONO, loc="left")
    ax.set_ylabel("layer", color=INK, fontsize=8)
    ax.set_xlabel("token position (input then output)", color=INK, fontsize=8)
    ax.tick_params(colors="#5a6a7a", labelsize=6)
    fig.savefig(OUT / f"frame_{step:03d}.png", facecolor=VOID,
                bbox_inches="tight")

# frame 0: prompt only
handles = [hf.model.layers[L].register_forward_hook(
    (lambda L: lambda layer, inp, out: grab(L, inp, out))(L))
    for L in range(len(hf.model.layers))]
with torch.no_grad():
    hf(ids, output_hidden_states=False)
render_frame(0, "<prompt read>", {})

cur = ids
for step in range(1, N_OUT + 1):
    with torch.no_grad():
        out = hf(cur)
    nxt = out.logits[0, -1].argmax().item()
    gen.append(nxt)
    cur = torch.cat([cur, torch.tensor([[nxt]], device="mps")], dim=1)
    with torch.no_grad():
        hf(cur)
    render_frame(step, tok.decode([nxt]), {})
    frames_meta.append(dict(step=step, token=tok.decode([nxt])))
    if step == 1:
        pass

for h in handles:
    h.remove()
json.dump(dict(prompt=PROMPT, frames=frames_meta,
               full_output=tok.decode(gen)),
          open(OUT / "generation.json", "w"), indent=1)
print("output:", repr(tok.decode(gen)), flush=True)

# trim fig for encode: close and re-encode frames
plt.close(fig)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "3",
                "-i", str(OUT / "frame_%03d.png"), "-c:v", "libx264",
                "-crf", "20", "-pix_fmt", "yuv420p",
                str(OUT / "workspace_resolving.mp4")], check=True)
print("wrote", OUT / "workspace_resolving.mp4")
