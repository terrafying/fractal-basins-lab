#!/usr/bin/env python3
"""exp09b2 — find (layer, scale) where activation perturbation flips the
arithmetic answer. 1D scan along u with digit-distribution counts."""
import re, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device).eval()

A, B_, TRUTH = 37, 28, 65
PROMPT = f"Q: What is {A} + {B_}?\nA:"
enc = tok(PROMPT, return_tensors="pt")
ids = enc.input_ids.to(device)
prompt_len = ids.shape[1]

state = {"delta": None}
def make_hook():
    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if hidden.shape[1] == prompt_len and state["delta"] is not None:
            hidden[0, prompt_len - 1, :] += state["delta"].to(hidden.dtype)
        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
    return hook

with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states

n_layers = model.config.num_hidden_layers
for L in (2, 8, 14, 20, 26):
    base = hs[L][0, -1].float()
    rms = base.square().mean().sqrt().item()
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(base.numel(), 1, generator=g))
    u = (Q[:, 0] * rms).to(device).to(model.dtype)
    handle = model.model.layers[L].register_forward_hook(make_hook())
    out_line = []
    for s in (4, 8, 16, 32, 64, 128):
        counts = {}
        for a in (-s, -s / 2, s / 2, s):
            state["delta"] = (a * u).float()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=12, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
            m = re.findall(r"-?\d+", text.split("=")[-1] if "=" in text else text)
            ans = int(m[0]) if m else None
            counts[ans] = counts.get(ans, 0) + 1
        out_line.append(f"s{s}:{counts}")
        print(f"layer {L:2d} {out_line[-1]}", flush=True)
    handle.remove()
