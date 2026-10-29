"""Analyse one fr3duo_isaac_eval.py episode: arm motion vs the recorded demo, grippers, fingertip-to-box distance.

    .venv\\Scripts\\python.exe diagnostics\\analyse_isaac_episode.py eval_results\\isaac_<time>\\steps_ep00.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ref = json.loads((ROOT / "artifacts" / "eval_assets" / "handover_reference.json").read_text())
D = np.array(ref["demo_states"])
rows = [json.loads(l) for l in open(sys.argv[1])]
step = np.array([r["step"] for r in rows])
S = np.array([r["state"] for r in rows])
A = np.array([r["action"] for r in rows])
box = np.array([r["box"] for r in rows])
tips = np.array([r["left_tips"] for r in rows]) if "left_tips" in rows[0] else None
ready = np.array(ref["initial_state"])
print(f"{len(rows)} logged steps, step spacing {np.median(np.diff(step)):.0f} (1 = every 30 Hz step)")
print(" time | L-arm moved | demo L moved | L-grip cmd/state | R-arm moved | demo R moved | tips->box (mm) | box y (mm)")
for t in range(0, int(step[-1]) + 1, 150):
    i = int(np.searchsorted(step, t))
    if i >= len(rows):
        break
    d = D[min(t, len(D) - 1)]
    tb = f"{np.linalg.norm(tips[i].mean(0) - box[i]) * 1000:6.0f}" if tips is not None else "   n/a"
    print(f"{t / 30:5.1f}s | {np.abs(S[i, :7] - ready[:7]).sum():6.2f} | {np.abs(d[:7] - ready[:7]).sum():6.2f} | "
          f"{A[i, 7]:5.2f}/{S[i, 7]:4.2f} | {np.abs(S[i, 8:15] - ready[8:15]).sum():6.2f} | {np.abs(d[8:15] - ready[8:15]).sum():6.2f} | "
          f"{tb} | {box[i, 1] * 1000:7.1f}")
n = min(len(S), len(D))
idx = np.minimum(step[:n], len(D) - 1)
err = np.abs(S[:n][:, [*range(7), *range(8, 15)]] - D[idx][:, [*range(7), *range(8, 15)]])
print(f"mean joint distance to the demo at the same time: {err.mean():.3f} rad (max {err.max():.2f})")
if tips is not None:
    dist = np.linalg.norm(tips.mean(1) - box, axis=1) * 1000
    k = int(dist.argmin())
    print(f"closest fingertip-midpoint approach to the box: {dist[k]:.0f} mm at {step[k] / 30:.1f} s "
          f"(offset box - tips: {np.round((box[k] - tips[k].mean(0)) * 1000).tolist()} mm, x/y/z, y up)")
