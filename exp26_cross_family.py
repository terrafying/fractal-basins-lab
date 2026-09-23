#!/usr/bin/env python3
"""exp26 — cross-family J-space trajectories: does the L8-affect / L20-pragmatics
stratification replicate in Gemma-3-1b (different family, tokenizer, lens)?

Same contrast protocol as exp25, minus EP regions (L20-specific assignments
do not transfer): pain, negative-valence, plus two pragmatic contrasts
(question vs statement, instruction vs statement) so we have a pragmatics
probe that is layer-relative like the affect probes.
"""
import json, os
from pathlib import Path

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
OUT = ROOT / "runs" / "exp26"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "allenai/OLMo-3-1025-7B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/olmo3-7b_jacobian_lens.pt")
NL = max(lens.jacobians.keys()) + 1
LAYERS = list(range(NL))
print(f"OLMo-3-7B lens layers: {NL}", flush=True)

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
QUEST = ["What causes the seasons on Earth?",
         "Who wrote Pride and Prejudice?",
         "How do vaccines work?",
         "Why is the sky blue?",
         "What is the boiling point of water?"]
INSTR = ["Write a haiku about rain.",
         "Summarize this article in one sentence.",
         "Fix the bug in this Python function.",
         "Translate this paragraph into French.",
         "List three uses for vinegar."]

def hidden_all_layers(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0, -1] for h in hs]).float().cpu())
    return torch.stack(out)

print("extracting...", flush=True)
H = {k: hidden_all_layers(v) for k, v in
     dict(pain=PAIN, neut=NEUTRAL, neg=NEG, pos=POS, quest=QUEST,
          instr=INSTR).items()}
L_tot = H["pain"].shape[1]

J = {l: lens.jacobians[l].to("mps").to(torch.bfloat16) for l in LAYERS}
norm = hf.model.norm
# gemma-3 ties embeddings; lm_head:
lm_head = hf.get_output_embeddings()

def decode_batch(vecs, layer):
    v = vecs.to("mps").to(torch.bfloat16)
    tr = v @ J[layer].T
    logits = lm_head(norm(tr))
    idx = logits.float().topk(8, dim=-1).indices.cpu()
    return [[tok.decode([t]).strip() for t in row] for row in idx]

traj = {k: [] for k in ("pain", "valence", "question", "instruction")}
for l in LAYERS:
    li = l + 1 if l + 1 < L_tot else l
    scale = H["neut"][:, li].norm(dim=-1).mean()
    def contrast(a, b):
        v = H[a][:, li].mean(0) - H[b][:, li].mean(0)
        return v / v.norm() * scale
    for name, (a, b) in dict(pain=("pain", "neut"), valence=("neg", "pos"),
                             question=("quest", "neut"),
                             instruction=("instr", "neut")).items():
        traj[name].append(decode_batch(contrast(a, b).unsqueeze(0), l)[0])
    if l % 4 == 0:
        print(f"L{l:2d} pain: {traj['pain'][-1][:4]}", flush=True)

json.dump(dict(n_layers=NL, traj=traj),
          open(OUT / "olmo_trajectories.json", "w"), indent=1)

# ---- figure ----
MONO_F = dict(fontsize=8, family=MONO, va="top", color="#c9d4e0")
fig, ax2 = plt.subplots(figsize=(15, 9), dpi=120)
fig.patch.set_facecolor("#050508")
ax2.axis("off")
ax2.text(0.0, 1.01, "gemma-3-1b-pt: J-space trajectories (cross-family "
         "check)", fontsize=12, color="#c9d4e0", family=MONO, va="bottom")
y = 0.97
for name in ("pain", "valence", "question", "instruction"):
    ax2.text(0.0, y, name.upper(), fontsize=9, color="#7fd4c8", family=MONO,
             va="top")
    y -= 0.058
    for l in range(0, NL, max(1, NL // 8)):
        ax2.text(0.005, y, f"L{l:2d}", fontsize=7, color="#5a6a7a",
                 family=MONO, va="top")
        ax2.text(0.05, y, "  ".join(traj[name][l][:5]), **MONO_F)
        y -= 0.043
    y -= 0.015
fig.savefig(OUT / "olmo_trajectories.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "olmo_trajectories.png")
