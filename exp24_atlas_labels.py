#!/usr/bin/env python3
"""exp24 — label the atlas in J-space.

Apply the pre-fitted Qwen3-1.7B Jacobian lens (Neuronpedia, n=1000 wikitext)
to the objects of our atlas:
  1. the 12 EP dictionary region means (from exp18)  -> what each pragmatic
     region is disposed to say
  2. the pain vector (exp23 construction)            -> what the pain
     direction verbalizes
  3. a negative-valence direction (exp23 contrast)   -> the orthogonality
     check in verbalizable terms

Method: for region means and steering vectors expressed in LAYER-20 residual
coordinates, apply lens.apply at layer 20 is the native readout; but to read
a raw vector (not an activation of some prompt) we call lens.transport()
directly: lens_logits = unembed(norm(J20 @ v)). We then topk.
"""
import json, os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transformers, jlens

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp24"
OUT.mkdir(parents=True, exist_ok=True)
HF_HOME = "/Volumes/evol/hf_cache"
os.environ["HF_HOME"] = HF_HOME

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.load("/Volumes/evol/jlens/qwen3-1.7b_jacobian_lens.pt")

# ---- load EP dictionary (L20 region means + labels) ----
d = np.load(ROOT / "runs/exp18/ep_dict_L20.npz", allow_pickle=True)
regions = json.loads((ROOT / "runs/exp18/ep_regions_L20.json").read_text())
exemplars = np.asarray(d["exemplars"])          # (12, 2048) region means
assign = np.asarray(d["assign"])                # (92,) per-prompt assignment
labels = [f"region_{i}" for i in range(exemplars.shape[0])]
centers = torch.tensor(exemplars, dtype=torch.bfloat16)
print("regions:", len(labels), flush=True)

# ---- pain + valence vectors in L20 coordinates (exp23 construction, LAYER=14
# hidden states; re-express at L20 by re-extracting the same contrast at L20) ----
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

LAYER_IDX = 21  # hidden_states index for layer-20 output
def last_hidden(text):
    ids = tok(text, return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        hs = hf(ids, output_hidden_states=True).hidden_states
    return hs[LAYER_IDX][0, -1].float().cpu()

P = torch.stack([last_hidden(t) for t in PAIN]).mean(0)
N = torch.stack([last_hidden(t) for t in NEUTRAL]).mean(0)
pain_vec = P - N
val_vec = torch.stack([last_hidden(t) for t in NEG]).mean(0) - \
    torch.stack([last_hidden(t) for t in POS]).mean(0)
scale = float(np.linalg.norm(centers[0].float()))
pain_vec = pain_vec / pain_vec.norm() * scale
val_vec = val_vec / val_vec.norm() * scale

# ---- decode everything through the J-lens at layer 20 ----
def decode_vec(v):
    """v: (2048,) residual vector -> ranked tokens via lens.transport."""
    v = v.to("mps").to(torch.bfloat16).unsqueeze(0)
    J = lens.jacobians[20].to(v.device).to(v.dtype)
    transported = v @ J.T                  # J20 @ v -> final-layer basis
    logits = hf.lm_head(hf.model.norm(transported))
    toks = logits[0].topk(8).indices
    return [tok.decode([t]).strip() for t in toks]

results = {"regions": [], "pain": None, "valence": None}
for i, lab in enumerate(labels):
    toks = decode_vec(centers[i].float())
    results["regions"].append(dict(id=i, label=lab, jlens_tokens=toks))
    print(f"region {i:2d} [{lab[:28]:28s}] -> {toks[:5]}", flush=True)

results["pain"] = decode_vec(pain_vec)
results["valence"] = decode_vec(val_vec)
print(f"PAIN vector       -> {results['pain'][:6]}", flush=True)
print(f"NEG-VALENCE vector-> {results['valence'][:6]}", flush=True)

# bonus: the spider demo readout for the record
prompt = "The number of legs on the animal that spins webs is"
ll, _, _ = lens.apply(model, prompt, positions=[-2])
spider_readout = {int(k): [tok.decode([t]).strip() for t in v[0].topk(5).indices]
                  for k, v in sorted(ll.items())}
results["spider_demo"] = spider_readout

json.dump(results, open(OUT / "atlas_labels_jlens.json", "w"), indent=1)

# ---- figure: atlas labels table ----
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=130)
fig.patch.set_facecolor("#050508")
ax.axis("off")
ax.text(0.0, 1.02, "the atlas, labeled in J-space (Qwen3-1.7B, layer-20 J-lens)",
        fontsize=13, color="#c9d4e0", family="monospace", va="bottom")
y = 0.96
for r in results["regions"]:
    toks = " ".join(t for t in r["jlens_tokens"][:5])
    ax.text(0.0, y, f"{r['id']:2d} {r['label'][:24]:24s}", fontsize=8.5,
            color="#7fd4c8", family="monospace", va="top")
    ax.text(0.34, y, toks, fontsize=8.5, color="#c9d4e0", family="monospace", va="top")
    y -= 0.052
y -= 0.02
for name, toks in (("PAIN VECTOR", results["pain"]),
                   ("NEG-VALENCE", results["valence"])):
    ax.text(0.0, y, f"{name:26s}", fontsize=9, color="#e07a5f",
            family="monospace", va="top", weight="bold")
    ax.text(0.34, y, " ".join(toks[:6]), fontsize=8.5, color="#e07a5f",
            family="monospace", va="top")
    y -= 0.055
fig.savefig(OUT / "atlas_labels_jlens.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "atlas_labels_jlens.png")
