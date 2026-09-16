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
import json, os, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp09"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("FB_HF_MODEL", "Qwen/Qwen3-1.7B")
RES = int(os.environ.get("FB_RES", "16"))          # 16^2 = 256 generations
LAYER = int(os.environ.get("FB_LAYER", "12"))       # residual-stream layer
SCALE = float(os.environ.get("FB_SCALE", "2.0"))    # perturbation half-width (x RMS)
MAX_NEW = int(os.environ.get("FB_MAXNEW", "8"))

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
    LAYER = min(int(os.environ.get("FB_LAYER", str(n_layers // 2))), n_layers - 1)

    # puzzle: solve p4_med by brute force, blank one interior cell, ask for it
    ROWS = ["0300", "4000", "0003", "0020"]
    g = [list(r) for r in ROWS]
    def ok(r, c, v):
        for i in range(4):
            if g[r][i] == v or g[i][c] == v:
                return False
        br, bc = 2 * (r // 2), 2 * (c // 2)
        return v not in [g[br + i][bc + j] for i in range(2) for j in range(2)]
    def rec(pos=0):
        while pos < 16 and g[pos // 4][pos % 4] != "0":
            pos += 1
        if pos == 16:
            return True
        r, c = pos // 4, pos % 4
        for v in "1234":
            if ok(r, c, v):
                g[r][c] = v
                if rec(pos + 1):
                    return True
                g[r][c] = "0"
        return False
    assert rec(), "puzzle unsolvable"
    solution = ["".join(row) for row in g]
    # choose a genuinely ambiguous-looking cell: multiple candidates survive
    # local row/col/box constraints, yet the true solution forces one value —
    # the model's decision boundary, not a trivial row-read.
    orig = [list(r) for r in ROWS]
    def cands(r, c):
        used = {orig[r][i] for i in range(4)} | {orig[i][c] for i in range(4)}
        br, bc = 2 * (r // 2), 2 * (c // 2)
        used |= {orig[br + i][bc + j] for i in range(2) for j in range(2)}
        return [v for v in "1234" if v not in used and orig[r][c] == "0"]
    best = None
    for r in range(4):
        for c in range(4):
            if orig[r][c] == "0":
                n = len(cands(r, c))
                if n >= 2 and (best is None or n > best[0]):
                    best = (n, r, c)
    assert best, "no ambiguous cell"
    _, QR, QC = best
    truth = solution[QR][QC]
    grid_str = "\n".join("".join(r) for r in orig)
    PROMPT = PROMPT_TEMPLATE.format(grid=grid_str, r=QR + 1, c=QC + 1)
    print(f"probing cell ({QR},{QC}), local candidates {best[0]}, "
          f"truth {truth}\nprompt:\n{PROMPT}", flush=True)

    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                  add_generation_prompt=True,
                                  enable_thinking=False,
                                  return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    inputs = {"input_ids": ids.to(device)}
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
        ds = [ch for ch in text if ch in "1234"]
        ans = ds[0] if ds else "?"
        ntok = int(out.shape[1] - prompt_len)
        results.append(dict(a=a, b=b, digit=ans, correct=ans == truth,
                            tokens=ntok, raw=text[-40:]))
        if i % 16 == 0:
            print(f"[{i+1}/{len(grid_pts)}] ({a:+.2f},{b:+.2f}) -> {ans} "
                  f"(truth {truth}) tok={ntok}  ({time.time()-t0:.0f}s)", flush=True)
    handle.remove()

    (OUT / "activation_basins.json").write_text(json.dumps(results, indent=1))

    # ---- analysis ----
    ids = Counter(r["digit"] for r in results)
    n_correct = sum(r["correct"] for r in results)
    print(f"\nanswer digits: {dict(ids)}  (truth {truth})")
    print(f"correct: {n_correct}/{len(results)}")
    toks = np.array([r["tokens"] for r in results])
    print(f"tokens: min {toks.min()} med {int(np.median(toks))} max {toks.max()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from loopscape.metrics import basin_entropy, uncertainty_exponent
    lut = {k: i for i, k in enumerate("1234?")}
    lab = np.array([lut.get(r["digit"], 4) for r in results]).reshape(RES, RES)
    tmap = toks.reshape(RES, RES)
    be = basin_entropy(tmap, box_size=5)
    ue = uncertainty_exponent(tmap, box_size=5)
    print(f"token-length field: Sb {be['basin_entropy']:.3f} "
          f"alpha {ue['uncertainty_exponent']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), dpi=130)
    fig.patch.set_facecolor("#050508")
    cmap = plt.get_cmap("turbo", 5)
    im = axes[0].imshow(lab, cmap=cmap, interpolation="nearest", origin="lower",
                        vmin=0, vmax=4)
    axes[0].set_title(f"Qwen3-1.7B layer {LAYER}: chosen digit "
                      f"(truth {truth})", fontsize=9, color="#c9d4e0", loc="left")
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
