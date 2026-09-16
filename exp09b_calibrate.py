#!/usr/bin/env python3
"""exp09b — calibrate the (layer, scale) regime where activation perturbations
flip Qwen3-1.7B's answer. 1D scan along u; counts digit distribution.
"""
import os, sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

device = "mps"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device).eval()

PROMPT = ("This 4x4 sudoku uses digits 1-4 (each once per row, column, 2x2 box). "
          "Answer with a single digit only.\n\n"
          "Grid:\n2030\n0100\n0002\n0400\nWhat digit goes in row 1, column 2? Answer: 3\n\n"
          "Grid:\n0102\n3004\n0000\n0201\nWhat digit goes in row 3, column 3? Answer: 4\n\n"
          "Grid:\n0040\n2010\n0300\n0002\nWhat digit goes in row 4, column 1? Answer: 2\n\n"
          "Grid:\n0300\n4000\n0003\n0020\n"
          "What digit goes in row 1, column 4? Answer:")
enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                              add_generation_prompt=True, enable_thinking=False,
                              return_dict=True, return_tensors="pt")
ids = enc["input_ids"].to(device)
prompt_len = ids.shape[1]

with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states

state = {"delta": None}

def make_hook():
    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if hidden.shape[1] == prompt_len and state["delta"] is not None:
            hidden[0, prompt_len - 1, :] += state["delta"].to(hidden.dtype)
        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
    return hook

n_layers = model.config.num_hidden_layers
scales = [0.5, 1, 2, 4, 8, 16, 32]
for L in (6, 14, 22, 27):
    base = hs[L][0, -1].float()
    rms = base.square().mean().sqrt().item()
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(base.numel(), 1, generator=g))
    u = (Q[:, 0] * rms).to(device).to(model.dtype)
    handle = model.model.layers[L].register_forward_hook(make_hook())
    row = []
    for s in scales:
        for a in (-s, s):
            state["delta"] = (a * u).float()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=12, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, prompt_len:], skip_special_tokens=False)
            if s == 0.5 and a < 0:
                print(f"  [dbg L{L} raw] {text!r}", flush=True)
            d = [c for c in text if c in "1234"]
            row.append(d[0] if d else "?")
        print(f"layer {L:2d} scale {s:5.1f}: {' '.join(row[-2:])}", flush=True)
    handle.remove()
    print(f"layer {L}: full scan {' '.join(row)}", flush=True)
