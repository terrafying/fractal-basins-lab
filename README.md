# Fractal Basins Lab

Probe the fractal convergence landscape of looped reasoning models — an
autonomous-computational-lab transplant of Buehler's agent-lab methodology onto
the dynamical system of "Fractal basins trap latent reasoning" (Lai, Bao,
Quinn, Gilpin, 2026, [arXiv:2609.04963](https://arxiv.org/abs/2609.04963)).

The probe target is a tiny open-weights recurrent reasoner (EqR, ~27M params),
so all measurement is free local GPU; frontier models are used only to design
experiments and interpret results. Protocol details and the metamaterial
analogy table: [AGENTS.md](AGENTS.md).

## Quickstart (any machine / Colab T4)

```bash
git clone https://github.com/terrafying/fractal-basins-lab
cd fractal-basins-lab
pip install "loopscape @ git+https://github.com/GilpinLab/loopscape"
FB_RES=64 FB_SEEDS=0 python exp01_sudoku_slices.py   # 16k trajectories
python render_basins.py && python render_basins.py --drift
```

Or open `colab/fractal_basins_colab.ipynb` in Colab with a T4 runtime.

## Contents

| file | role |
|---|---|
| `exp01_sudoku_slices.py` | basin maps (settling-time fields) across puzzle difficulty; env: `FB_RES`, `FB_SEEDS`, `FB_PUZZLES` |
| `exp02_state_capture.py` | per-loop decoded traces for animation |
| `render_basins.py` | stills + slow-drift animations (smoothed, outlier-robust) |
| `anim_crystallization.py` | solution crystallization over reasoning loops |
| `scripts/mini_submit.sh` / `mini_fetch.sh` | detached remote-GPU job protocol (works for any ssh host) |
| `runs/` | npz slices, traces, figures (gitignored) |

## Key numbers

- EqR sudoku: latent (97, 512), max_steps=24, noise_scale=0 (deterministic map).
- Slice: random orthonormal 2-plane (QR of Gaussian), grid [-1,1]^2.
- Throughput reference: M4 Pro MPS ~5 cond/s, M4 mini ~2.6 cond/s, T4 ~10-20 cond/s.
- Paper exclusions: drop slice if <90% solved or >1% hit the loop cap.
