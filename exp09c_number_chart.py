#!/usr/bin/env python3
"""exp09c — number-manifold chart: activation basins along SEMANTIC directions.

Instead of random perturbation directions, build (u, v) from the unembedding
rows of the digit tokens ("0".."9"), via PCA. The (a, b) plane then spans the
model's number-representation subspace. Scanning it and recording the model's
answer turns the map into a CHART of the number manifold: if numbers live on
a ring/helix (as the literature suggests), the answer-value field should show
bands/periodic structure — interpretable geometry, not abstract fractality.

Task: 37 + 28 (truth 65), Qwen3-1.7B, patch at last prompt token, layer L.
"""
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp09c"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("FB_HF_MODEL", "Qwen/Qwen3-1.7B")
RES = int(os.environ.get("FB_RES", "32"))
LAYER = int(os.environ.get("FB_LAYER", "20"))
SCALE = float(os.environ.get("FB_SCALE", "24.0"))   # rms units; semantic dirs are strong
MAX_NEW = 12
A, B_, TRUTH = 37, 28, 65
PROMPT = f"Q: What is {A} + {B_}?\nA:"


def parse_num(text):
    m = re.findall(r"-?\d+", text.split("=")[-1] if "=" in text else text)
    return int(m[0]) if m else None


def main():
    device = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    n_layers = model.config.num_hidden_layers
    LAYER = min(int(os.environ.get("FB_LAYER", "20")), n_layers - 1)

    # digit token ids (leading-space and bare forms)
    digit_ids = []
    for d in "0123456789":
        tid = tok.encode(d, add_special_tokens=False)
        digit_ids.append(tid[0])
    print("digit token ids:", digit_ids, [tok.decode([t]) for t in digit_ids])

    U = model.lm_head.weight.detach().float()          # (vocab, d)
    rows = U[digit_ids]                                 # (10, d)
    # PCA top-2 of digit rows
    rows_c = rows - rows.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(rows_c, full_matrices=False)
    p1, p2 = Vt[0], Vt[1]
    print(f"digit-row PCA: var explained p1={float((rows_c @ p1).square().sum()/rows_c.square().sum()):.2f}, "
          f"p2={float((rows_c @ p2).square().sum()/rows_c.square().sum()):.2f}")

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    base = hs[LAYER][0, -1].float()
    rms = base.square().mean().sqrt().item()
    # scale semantic dirs to comparable magnitude: digit rows are lm-head scale,
    # hidden residual scale differs; normalize each direction to ~ rms of base
    u = (p1 / p1.square().mean().sqrt() * rms).to(device).to(model.dtype)
    v = (p2 / p2.square().mean().sqrt() * rms).to(device).to(model.dtype)

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
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        results.append(dict(a=a, b=b, answer=parse_num(text), raw=text[:40]))
        if i % 64 == 0:
            print(f"[{i+1}/{len(pts)}] ({a:+.1f},{b:+.1f}) -> "
                  f"{results[-1]['answer']} ({time.time()-t0:.0f}s)", flush=True)
    handle.remove()
    (OUT / "number_chart.json").write_text(json.dumps(results, indent=1))

    # ---- analysis + figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    answers = [r["answer"] for r in results]
    known = [a for a in answers if a is not None]
    print(f"\nanswers: {len(set(known))} distinct; range "
          f"{min(known) if known else '-'}..{max(known) if known else '-'}; "
          f"unparsed {answers.count(None)}")
    print("top:", __import__("collections").Counter(answers).most_common(8))

    amap = np.array([a if a is not None else np.nan for a in answers],
                    dtype=float).reshape(RES, RES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), dpi=130)
    fig.patch.set_facecolor("#050508")
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#2a2a33")
    im = axes[0].imshow(np.ma.masked_invalid(amap), cmap=cmap,
                        interpolation="nearest", origin="lower")
    axes[0].set_title(f"answer value chart ({len(set(known))} distinct values)\n"
                      f"digit-PCA directions, layer {LAYER}", fontsize=9,
                      color="#c9d4e0", loc="left")
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    # zoom on the truth-adjacent band structure: answers within +-20 of truth
    band = np.where(np.abs(amap - TRUTH) <= 20, amap, np.nan)
    im2 = axes[1].imshow(np.ma.masked_invalid(band), cmap=cmap,
                         interpolation="nearest", origin="lower")
    axes[1].set_title(f"band structure near truth ({TRUTH} +- 20)", fontsize=9,
                      color="#c9d4e0", loc="left")
    fig.colorbar(im2, ax=axes[1], fraction=0.04)
    for ax in axes:
        ax.set_facecolor("#050508"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#1c2430")
    fig.savefig(OUT / "number_manifold_chart.png", facecolor="#050508",
                bbox_inches="tight")
    print("wrote", OUT / "number_manifold_chart.png")


if __name__ == "__main__":
    main()
