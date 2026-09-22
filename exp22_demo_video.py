#!/usr/bin/env python3
"""exp22 — assemble the atlas demo video (Buehler-style walkthrough)."""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp22"
OUT.mkdir(parents=True, exist_ok=True)

SCENES = [
    (ROOT / "runs/exp04/hard_a/settling_200.png",
     "1. THE DECISION STAR - FPRM 200x200 settling field: hub-and-wedge geometry,"
     " interlocked settling basins (z=8.5)"),
    (ROOT / "runs/exp09c/number_manifold_chart_L8.png",
     "2. THE NUMBER MANIFOLD - Qwen3-1.7B layer 8: the number line, visible in"
     " activation space"),
    (ROOT / "runs/exp20/response_field_map.png",
     "3. THE RESPONSE FIELD - everything an ordinary LLM would say about Paris,"
     " mapped (note the French-language province)"),
    (ROOT / "runs/exp21/pure_geometry_model.png",
     "4. PURE GEOMETRY - 8 terms (angular harmonics + radial drift) reconstruct"
     " 71% of the map; 80% at layer 26"),
]

INK, VOID = "#c9d4e0", "#050508"
scene_pngs = []
for i, (img_path, caption) in enumerate(SCENES):
    fig, (cap_ax, img_ax) = plt.subplots(
        2, 1, figsize=(19.2, 10.8), dpi=100,
        gridspec_kw={"height_ratios": [1, 6]})
    fig.patch.set_facecolor(VOID)
    cap_ax.axis("off")
    cap_ax.text(0.01, 0.5, caption, fontsize=15, color=INK, va="center",
                family="monospace")
    img = mpimg.imread(img_path)
    img_ax.imshow(img)
    img_ax.axis("off")
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    p = OUT / f"scene_{i}.png"
    fig.savefig(p, facecolor=VOID)
    plt.close(fig)
    scene_pngs.append(p)
    print("rendered", p)

# concat with fades: encode each scene 4s, then xfade chain
inputs = []
for p in scene_pngs:
    inputs += ["-loop", "1", "-t", "5", "-i", str(p)]
n = len(scene_pngs)
fc = ""
prev = "[v0]"
for k in range(n):
    fc += f"[{k}:v]scale=1920:1080,setsar=1,format=yuv420p[v{k}];"
for k in range(1, n):
    outl = f"[x{k}]" if k < n - 1 else "[out]"
    fc += f"{prev}[v{k}]xfade=transition=fade:duration=1:offset={5*k-1}{outl};"
    prev = f"[x{k}]"
fc = fc.rstrip(";")
cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + [
    "-filter_complex", fc, "-map", "[out]", "-c:v", "libx264", "-crf", "20",
    "-pix_fmt", "yuv420p", str(OUT / "atlas_demo.mp4")]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("ffmpeg stderr:", r.stderr[-800:])
else:
    print("wrote", OUT / "atlas_demo.mp4")