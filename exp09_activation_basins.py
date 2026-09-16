#!/usr/bin/env python3
"""Exp 09 — fractal basins in the ACTIVATION SPACE of an ordinary autoregressive LLM.

Model: Qwen3-1.7B (HF, bf16, MPS). Task: 4x4 sudoku, short completion.
Protocol (mirrors the latent-slice experiments):
- Embed the prompt, run a forward pass to the LAST prompt token, take the
  residual stream at layer L.
- Sample a random orthonormal 2-plane (u, v) in that layer's residual space
  (QR of a Gaussian -- identical machinery to exp01).
- For each point (a, b) on a grid: patch the residual stream with
  h + a*u + b*v at that position/layer for ALL generation steps (hook), and
  generate to completion at temperature 0.
- Record: final answer identity (parsed 4x4 grid), CoT length (tokens).
- Render: identity-basin map + token-length map over the (a, b) plane;
  compute basin entropy / uncertainty exponent on the token-length field,
  and the approximate Wada statistic on the identity field.

This is the ordinary-LLM analog of the latent-slice experiments: if answer
basins in activation space are fractal/interlocked, the dynamical-systems
view extends to GPT-class models.
"""
import json, os, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp09"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("FB_HF_MODEL", "Qwen/Qwen3-1.7B")
RES = int(os.environ.get("FB_RES", "32"))          # 32^2 = 1024 generations
LAYER = int(os.environ.get("FB_LAYER", "20"))       # residual-stream layer
SCALE = float(os.environ.get("FB_SCALE", "96.0"))   # perturbation half-width (x RMS)
MAX_NEW = int(os.environ.get("FB_MAXNEW", "12"))

A, B_, TRUTH = 37, 28, 65                          # carry sum: model's decision edge
PROMPT = f"Q: What is {A} + {B_}?\nA:"


def parse_num(text):
    m = re.findall(r"-?\d+", text.split("=")[-1] if "=" in text else text)
    return int(m[0]) if m else None

PROMPT_TEMPLATE = ("This 4x4 sudoku uses digits 1-4 (each once per row, column, 2x2 box). "
                   "Grid (0 = empty):\n{grid}\nWhat digit goes in row {r}, column {c}? "
                   "Answer with a single digit and nothing else.")


def parse_grid(text):
    digits = []
    for line in text.strip().splitlines():
        ds = "".join(ch for ch in line if ch in "1234")
        if len(ds) == 4:
            digits.append(ds)
        if len(digits) == 4:
            break
    return tuple(digits) if len(digits) == 4 else ("__UNPARSED__",)


def main():
    device = ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    n_layers = model.config.num_hidden_layers
    LAYER = min(int(os.environ.get("FB_LAYER", "20")), n_layers - 1)
    print(f"task: {A} + {B_} = {TRUTH} (carry sum)", flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    inputs = {"input_ids": ids}
    prompt_len = ids.shape[1]

    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True).hidden_states
    base = hs[LAYER][0, -1].float()                    # residual at last prompt token
    rms = base.square().mean().sqrt().item()
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(base.numel(), 2, generator=g))
    u = (Q[:, 0] * rms).to(device).to(model.dtype)
    v = (Q[:, 1] * rms).to(device).to(model.dtype)
    print(f"layer {LAYER}/{n_layers}, base rms {rms:.2f}, "
          f"perturb half-width {SCALE}x rms", flush=True)

    # generation-time hook: add (a*u + b*v) to the patched position every step.
    # We patch position prompt_len-1 of the CURRENT sequence (kv-cached decode
    # recomputes only the new token, so we instead patch ALL positions lightly:
    # simplest correct approach = re-add at the last prompt position each step
    # via a forward hook on the layer output.
    state = {"delta": torch.zeros_like(base)}

    def hook(module, inp, out):
        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        # patch only during prefill (full prompt in flight); the KV cache
        # carries the perturbation through all decode steps
        if hidden.shape[1] == prompt_len:
            hidden[0, prompt_len - 1, :] += state["delta"].to(hidden.dtype)
        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

    layer_module = model.model.layers[LAYER]
    handle = layer_module.register_forward_hook(hook)

    lin = torch.linspace(-SCALE, SCALE, RES)
    grid_pts = [(a, b) for a in lin.tolist() for b in lin.tolist()]
    results = []
    t0 = time.time()
    for i, (a, b) in enumerate(grid_pts):
        state["delta"] = (a * u + b * v).float()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        ans = parse_num(text)
        ntok = int(out.shape[1] - prompt_len)
        results.append(dict(a=a, b=b, answer=ans, correct=ans == TRUTH,
                            tokens=ntok, raw=text[:60]))
        if i % 16 == 0:
            print(f"[{i+1}/{len(grid_pts)}] ({a:+.2f},{b:+.2f}) -> {ans} "
                  f"(truth {TRUTH}) tok={ntok}  ({time.time()-t0:.0f}s)", flush=True)
    handle.remove()

    (OUT / "activation_basins.json").write_text(json.dumps(results, indent=1))

    # ---- analysis ----
    answers = [r["answer"] for r in results]
    ids = Counter(answers)
    n_correct = sum(r["correct"] for r in results)
    print(f"\nanswer distribution: {dict(ids.most_common(8))}  (truth {TRUTH})")
    print(f"correct: {n_correct}/{len(results)}")
    toks = np.array([r["tokens"] for r in results])
    print(f"tokens: min {toks.min()} med {int(np.median(toks))} max {toks.max()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from loopscape.metrics import basin_entropy, uncertainty_exponent
    # numeric answer field: color by |answer - TRUTH| (semantic distance);
    # unparsed -> masked out (gray)
    UNKNOWN = abs(TRUTH) + 100
    vals = [r["answer"] if r["answer"] is not None else None for r in results]
    amap = np.array([abs(v - TRUTH) if v is not None else np.nan
                     for v in vals], dtype=float).reshape(RES, RES)
    tmap = toks.reshape(RES, RES)
    be = basin_entropy(tmap, box_size=5)
    ue = uncertainty_exponent(tmap, box_size=5)
    n_ans = len(set(v for v in vals if v is not None))
    print(f"token-length field: Sb {be['basin_entropy']:.3f} "
          f"alpha {ue['uncertainty_exponent']:.3f}; distinct answers {n_ans}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), dpi=130)
    fig.patch.set_facecolor("#050508")
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#2a2a33")
    im = axes[0].imshow(np.ma.masked_invalid(amap), cmap=cmap,
                        interpolation="nearest", origin="lower")
    axes[0].set_title(f"Qwen3-1.7B layer {LAYER}: |answer - {TRUTH}| "
                      f"({n_ans} distinct answers, gray=unparsed)", fontsize=9,
                      color="#c9d4e0", loc="left")
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    axes[1].imshow(tmap, cmap="magma", interpolation="spline36", origin="lower")
    axes[1].set_title(f"CoT length (tokens)  Sb={be['basin_entropy']:.2f} "
                      f"alpha={ue['uncertainty_exponent']:.2f}", fontsize=9,
                      color="#c9d4e0", loc="left")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#1c2430")
    fig.savefig(OUT / "activation_basins.png", facecolor="#050508",
                bbox_inches="tight")
    print("wrote", OUT / "activation_basins.png")


if __name__ == "__main__":
    main()
