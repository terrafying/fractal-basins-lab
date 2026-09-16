import numpy as np
d = np.load("runs/exp05/saddle_labels.npz")
fli, settling = d["fli"], d["settling"]
q = np.quantile(fli, [0.25, 0.5, 0.75])
masks = [("Q1 calmest", fli <= q[0]), ("Q2", (fli > q[0]) & (fli <= q[1])),
         ("Q3", (fli > q[1]) & (fli <= q[2])), ("Q4 wildest", fli > q[2])]
for name, m in masks:
    s = settling[m]
    print(f"{name}: n={m.sum()}  mean settling {s.mean():.2f}  frac>6 {(s>6).mean():.1%}  frac>10 {(s>10).mean():.1%}")
cost_all = int(settling.sum())
calm = fli <= np.median(fli)
cost_policy = 3 * len(fli) + int(settling[calm].sum())
print(f"run-all cost: {cost_all} loop-units; probe3+commit-calm-half: {cost_policy} ({cost_policy/cost_all:.0%})")
