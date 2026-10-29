"""How random are SmolVLA's outputs, and how much do the images matter? Measured on recorded frames.

    .venv\\Scripts\\python.exe diagnostics/ablation.py [--checkpoint outputs/curriculum_smolvla_expert10k/final]

On the 12 frames of eval_results/pi05_probe_frames.npz (held-out episodes 0 and 5), compares the first 10 predicted
actions (left+right arm joints) with what the demonstrator did:
- seed spread: same input, 5 different noise seeds -> how much the outputs differ from each other
- image ablations: all cameras black, images from a different moment, only the head / only the wrists blanked
- state ablation: correct images but the joint state of a different moment
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.policy_runner import CAMS, PolicyRunner

ROOT = Path(__file__).resolve().parents[1]
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
ARM = [*range(7), *range(8, 15)]
N = 10  # actions compared (the probe file stores 50 recorded frames = 10 actions at 6 Hz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/curriculum_smolvla_expert10k/final")
    args = ap.parse_args()
    F = np.load(ROOT / "eval_results" / "pi05_probe_frames.npz")
    imgs = np.stack([F["ext"], F["lw"], F["rw"]], 1)  # (12, 3, 224, 224, 3)
    S, DEMO = F["state"].astype(np.float32), F["actions"][:, ::5][:, :N]
    n = len(S)
    other = (np.arange(n) + 2) % n  # 2 entries later in the list = a different phase of the task
    runner = PolicyRunner(str(ROOT / args.checkpoint))

    def predict(img3, state, seed):
        torch.manual_seed(seed)
        b = {c: torch.from_numpy(img3[k]).permute(2, 0, 1).float() / 255.0 for k, c in enumerate(CAMS)}
        b["observation.state"] = torch.from_numpy(state)
        b["task"] = TASK
        with torch.no_grad():
            return runner.post(runner.policy.predict_action_chunk(runner.pre(b))).squeeze(0).float().cpu().numpy()[:N]

    def err(cond):
        return float(np.mean([np.abs(cond(i)[:, ARM] - DEMO[i][:, ARM]).mean() for i in range(n)]))

    black = np.zeros_like(imgs[0][0])  # one black 224x224x3 image
    hold = float(np.mean([np.abs(S[i][ARM] - DEMO[i][:, ARM]).mean() for i in range(n)]))
    normal = [[predict(imgs[i], S[i], s) for s in range(5)] for i in range(n)]
    e_normal = float(np.mean([np.abs(normal[i][s][:, ARM] - DEMO[i][:, ARM]).mean() for i in range(n) for s in range(5)]))
    spread = float(np.mean([np.std(np.stack(normal[i])[:, :, ARM], axis=0).mean() for i in range(n)]))
    first_spread = float(np.mean([np.std(np.stack(normal[i])[:, 0, ARM], axis=0).mean() for i in range(n)]))
    print(f"1) RANDOMNESS (same input, 5 noise seeds)")
    print(f"   spread between seeds, first action:     {first_spread:.4f} rad")
    print(f"   spread between seeds, 10 actions:       {spread:.4f} rad")
    print(f"   error vs the demonstrator (normal run): {e_normal:.4f} rad")
    print(f"   what pure noise would be: actions are drawn from N(0,1) in normalized space ~ the joint's std "
          f"(~0.2-1.2 rad), refined in 10 steps by the model")
    print(f"\n2) WHAT THE INPUTS CONTRIBUTE (error vs demonstrator, first 10 actions, rad; lower is better)")
    rows = [("normal (all inputs correct)", e_normal),
            ("all 3 cameras black", err(lambda i: predict(np.stack([black] * 3), S[i], 0))),
            ("images from a different moment", err(lambda i: predict(imgs[other[i]], S[i], 0))),
            ("head camera black, wrists correct", err(lambda i: predict(np.stack([black, imgs[i][1], imgs[i][2]]), S[i], 0))),
            ("wrist cameras black, head correct", err(lambda i: predict(np.stack([imgs[i][0], black, black]), S[i], 0))),
            ("correct images, state of a different moment", err(lambda i: predict(imgs[i], S[other[i]], 0))),
            ("baseline: just hold the current pose", hold)]
    for name, e in rows:
        print(f"   {name:45s} {e:.4f}  ({e / e_normal:4.1f}x normal)")


if __name__ == "__main__":
    main()
