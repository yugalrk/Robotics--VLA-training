"""When does the left gripper close, and how close is the arm to the demo's grasp pose then? Sim runs vs the demo.

    .venv\\Scripts\\python.exe diagnostics\\compare_grasp_runs.py [run_dir ...]   (default: the 3 latest policy_runs)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ref = json.loads((ROOT / "artifacts" / "eval_assets" / "handover_reference.json").read_text())
D, DA = np.array(ref["demo_states"]), np.array(ref["demo_actions"])
ARM = list(range(7))


def first_close(cmd):
    i = np.flatnonzero(cmd < 0.2)
    return int(i[0]) if len(i) else None


dg = first_close(DA[:, 7])
print(f"DEMO: left gripper closes at {dg / 30:.1f} s; arm moved {np.abs(D[dg, ARM] - D[0, ARM]).sum():.2f} rad from start")
runs = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "Environment" / "policy_runs").glob("2026*"))[-3:]
for r in runs:
    rows = [json.loads(l) for l in open(r / "steps.jsonl")]
    run = json.loads((r / "run.json").read_text())
    S, A = np.array([x["state"] for x in rows]), np.array([x["action"] for x in rows])
    info = run["policy"]
    g = first_close(A[:, 7])
    print(f"\n{r.name}  policy={info.get('policy_type')} n_action_steps={info.get('n_action_steps')}  steps={len(S)}")
    if g is None:
        print("  left gripper never closed")
        continue
    print(f"  left gripper closes at {g / 30:.1f} s ({(g - dg) / 30:+.1f} s vs demo); finger state 2 s later {S[min(g + 60, len(S) - 1), 7]:.2f} (0 = empty)")
    print(f"  arm distance to the demo GRASP pose when closing: {np.abs(S[g, ARM] - D[dg, ARM]).sum():.2f} rad (sum over 7 joints)")
    print(f"  arm distance to the demo pose at the same time:   {np.abs(S[g, ARM] - D[min(g, len(D) - 1), ARM]).sum():.2f} rad")
    best = int(np.argmin(np.abs(S[: g + 1, ARM] - D[dg, ARM]).sum(1)))
    print(f"  closest the arm got to the grasp pose before closing: {np.abs(S[best, ARM] - D[dg, ARM]).sum():.2f} rad at {best / 30:.1f} s")
