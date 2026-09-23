#!/usr/bin/env python3
"""exp25 — J-space trajectories: how atlas objects become verbalizable.

For every layer l (0..26):
  - extract contrast vectors (pain, negative-valence) in that layer's
    coordinates
  - compute the EP region means at that layer (same 92 prompts, exp18
    assignments)
  - decode through the paper-scale J_l: top tokens
Then track pinned tokens (anguish, helpless, answer, verse, spiders...)
across depth, and chart the "workspace range" — the depth band where each
atlas object reads as itself.

Fonts: CJK-capable fallback (Hiragino Sans GB / Arial Unicode MS) so
Chinese tokens render instead of tofu blocks.
"""
import json, os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transformers, jlens

# ---- CJK-capable fonts ----
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Hiragino Sans GB", "Arial Unicode MS", "Heiti TC"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp25"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

lens = jlens.JacobianLens.load("/Volumes/evol/jlens/qwen3-1.7b_jacobian_lens.pt")
NL = max(lens.jacobians.keys()) + 1 if isinstance(lens.jacobians, dict) \
    else lens.jacobians.shape[0]
LAYERS = list(range(NL))
print(f"lens layers: {NL}", flush=True)

# ---- prompts ----
PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]
NEG = ["I am deeply sad and everything is bleak.",
       "I feel crushed by despair and grief."]
POS = ["I am joyful and everything feels bright.",
       "I feel delight and pure happiness."]

d18 = np.load(ROOT / "runs/exp18/ep_dict_L20.npz", allow_pickle=True)
assign = np.asarray(d18["assign"])
ep_prompts = [str(p) for p in d18["prompts"]]
n_regions = int(assign.max()) + 1

def hidden_all_layers(texts):
    """(n_texts, n_layers+1, d) last-token hidden states."""
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0, -1] for h in hs]).float().cpu())
    return torch.stack(out)  # (n, L+1, d)

print("extracting hidden states...", flush=True)
H_pain = hidden_all_layers(PAIN)
H_neut = hidden_all_layers(NEUTRAL)
H_neg = hidden_all_layers(NEG)
H_pos = hidden_all_layers(POS)
H_ep = hidden_all_layers(ep_prompts)          # (92, L+1, d)
L_tot = H_pain.shape[1]                        # 29 = 28 layers + embeddings
NL_model = L_tot - 1

# ---- decode at every layer ----
J = {l: lens.jacobians[l].to("mps").to(torch.bfloat16) for l in LAYERS}
norm = hf.model.norm
lm_head = hf.lm_head

def decode_batch(vecs, layer):
    """vecs: (n, d) torch float -> list of top-8 token strings."""
    v = vecs.to("mps").to(torch.bfloat16)
    tr = v @ J[layer].T
    logits = lm_head(norm(tr))
    idx = logits.float().topk(8, dim=-1).indices.cpu()
    return [[tok.decode([t]).strip() for t in row] for row in idx]

PINNED = ["anguish", "torment", "helpless", "PTSD", "answer", "Verse",
          "reasoning", "Silence", "spiders", "SQL", "sql", "数据库", "痛苦",
          "无助", "Convert", "thank"]
def pinned_ranks(token_lists):
    """(n_vecs,) rank (0-based) of any pinned token, else -1."""
    ranks = []
    for toks in token_lists:
        r = -1
        for i, t in enumerate(toks):
            if any(p.lower() == t.lower() or p in t for p in PINNED):
                r = i
                break
        ranks.append(r)
    return ranks

results = {
    "pain": [], "valence": [], "regions": [[] for _ in range(n_regions)],
}
for l in LAYERS:
    li = l + 1  # hidden_states index
    P = H_pain[:, li].mean(0); Nv = H_neut[:, li].mean(0)
    scale = H_ep[:, li].norm(dim=-1).mean()
    pain_v = (P - Nv); pain_v = pain_v / pain_v.norm() * scale
    val_v = (H_neg[:, li].mean(0) - H_pos[:, li].mean(0))
    val_v = val_v / val_v.norm() * scale
    reg_means = torch.stack(
        [H_ep[assign == i, li].mean(0) for i in range(n_regions)])
    results["pain"].append(decode_batch(pain_v.unsqueeze(0), l)[0])
    results["valence"].append(decode_batch(val_v.unsqueeze(0), l)[0])
    reg_toks = decode_batch(reg_means, l)
    for i in range(n_regions):
        results["regions"][i].append(reg_toks[i])
    if l % 4 == 0:
        print(f"L{l:2d} pain: {results['pain'][-1][:4]} "
              f"val: {results['valence'][-1][:3]}", flush=True)

json.dump(results, open(OUT / "jlens_trajectories.json", "w"), indent=1)

# ---- figure: pinned-token rank heatmap + per-band readouts ----
n_obj = 2 + n_regions
fig = plt.figure(figsize=(15.5, 11), dpi=120)
fig.patch.set_facecolor("#050508")
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.6], hspace=0.35)

# top: rank of best pinned token per object per layer
ax = fig.add_subplot(gs[0])
mat = np.full((n_obj, len(LAYERS)), np.nan)
obj_names = ["PAIN", "NEG-VALENCE"] + [f"R{i}" for i in range(n_regions)]
for l_i in range(len(LAYERS)):
    mat[0, l_i] = pinned_ranks([results["pain"][l_i]])[0]
    mat[1, l_i] = pinned_ranks([results["valence"][l_i]])[0]
    for i in range(n_regions):
        mat[2 + i, l_i] = pinned_ranks([results["regions"][i][l_i]])[0]
im = ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=-1, vmax=7)
ax.set_yticks(range(n_obj), obj_names, fontsize=7, color="#c9d4e0",
              family="monospace")
ax.set_xlabel("layer", color="#c9d4e0")
ax.set_title("J-lens rank of any pinned atlas token (0 = top-1, blank = none "
             "in top-8)", fontsize=10, color="#c9d4e0", loc="left",
             family="monospace")
ax.tick_params(colors="#c9d4e0")
plt.colorbar(im, ax=ax, shrink=0.7)

# bottom: text bands per layer for the two star objects + selected regions
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
band = 4  # show every 4th layer's tokens
rows = [("PAIN", [results["pain"][l] for l in range(0, len(LAYERS), band)]),
        ("NEG-VALENCE", [results["valence"][l] for l in range(0, len(LAYERS), band)])]
for i in (0, 1, 3, 10):
    rows.append((f"R{i}", [results["regions"][i][l]
                           for l in range(0, len(LAYERS), band)]))
y = 0.98
for name, toks_rows in rows:
    ax2.text(0.0, y, f"{name:12s}", fontsize=8, color="#7fd4c8",
             family="monospace", va="top")
    y -= 0.075
    for k, toks in enumerate(toks_rows):
        ax2.text(0.02, y, f"L{k*band:2d}", fontsize=7, color="#5a6a7a",
                 family="monospace", va="top")
        ax2.text(0.075, y, " ".join(toks[:5]), fontsize=7.5,
                 color="#c9d4e0", family="monospace", va="top")
        y -= 0.052
    y -= 0.015
fig.savefig(OUT / "jlens_trajectories.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "jlens_trajectories.png")

# regenerate exp24's figure with fixed fonts too
print("done")
