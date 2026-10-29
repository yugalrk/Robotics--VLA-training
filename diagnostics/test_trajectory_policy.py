"""Offline test of TrajectoryPolicy (the recorded handover as a closed-loop lookup table) with a simulated robot.

    .venv\\Scripts\\python.exe diagnostics\\test_trajectory_policy.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))
from fr3duo_policy_server import TrajectoryPolicy  # noqa: E402

DS = ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84"


def run(policy, gain, block=None, max_calls=1200):
    """Robot = first-order lag towards the commanded target each 1/6 s call; `block` = (start_s, end_s) frozen."""
    policy.reset()
    s = policy.S[0].copy()
    log = []
    for k in range(max_calls):
        a = policy.act({"observation.state": s}, "")
        t = k / 6
        if not (block and block[0] <= t < block[1]):
            s = s + gain * (a - s)
        log.append((t, policy.i))
        if policy.i >= len(policy.A) - 1:
            break
    return np.array(log)


p = TrajectoryPolicy(str(DS), episode=0, policy_hz=6)
print(f"trajectory: {len(p.A)} frames ({len(p.A) / 30:.1f} s), {p.step} frames per call at 6 Hz")
for name, gain, block in [("a) good tracking (gain 0.9)", 0.9, None),
                          ("b) blocked 2 s at t=8-10 s (approach)", 0.9, (8, 10)),
                          ("c) sluggish robot (gain 0.4)", 0.4, None)]:
    log = run(p, gain, block)
    t_end = log[-1, 0]
    msg = f"{name:40s} reached the end after {t_end:5.1f} s ({len(log)} calls)"
    if block:
        before = log[log[:, 0] < block[0]][-1, 1]
        during = log[(log[:, 0] >= block[0]) & (log[:, 0] < block[1])][:, 1]
        msg += f" | trajectory index while blocked: {int(before)} -> {int(during.max())} (waited: {during.max() - before <= 2 * p.step})"
    print(msg)
# perfect robot (always where the recording was): every served action must be the recorded action for
# 1/6 s after the robot's place in the recording, indices moving forward by at most one step per call
p.reset()
s, idx, errs = p.S[0].copy(), [], []
while p.i < len(p.A) - 1 and len(idx) < 2000:
    a = p.act({"observation.state": s}, "")
    k = min(p.i + p.step - 1, len(p.A) - 1)
    idx.append(p.i)
    errs.append(np.abs(a - p.A[k]).max())
    s = p.S[min(k + 1, len(p.S) - 1)]  # perfect tracking: the robot reaches the commanded pose
steps = np.diff(idx)
print(f"perfect tracking: {len(idx)} calls, served == recorded action at the robot's phase (max diff {max(errs):.6f}), "
      f"index steps {steps.min()}..{steps.max()} frames (limit {p.step}), monotonic: {bool((steps >= 0).all())}")
