"""Send real handover frames to the policy server on :8090 and compare the served actions with the recorded ones.

    .venv\\Scripts\\python.exe diagnostics\\check_pi05_offline.py [--episode 0] [--replan 25]

For a few start frames: /reset, then `replan` /act calls with that frame's observation (the server serves its
chunk one action per call), compared with the recorded actions[t : t+replan] at 30 fps. Low error = the
server's input mapping, normalization and prompt match the training setup.
"""
import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAMS = ["observation.images.external_camera", "observation.images.left_wrist_camera", "observation.images.right_wrist_camera"]
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
URL = "http://127.0.0.1:8090"


def post(path, data):
    return json.loads(urllib.request.urlopen(urllib.request.Request(URL + path, data=data, method="POST"), timeout=300).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--replan", type=int, default=25)
    ap.add_argument("--frames", default="0,200,330,600,900,1200")
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from fr3duo.data import DATASET_ROOT, REPO_ID

    ds = LeRobotDataset(REPO_ID, root=DATASET_ROOT, episodes=[args.episode], video_backend="pyav")
    info = json.loads(urllib.request.urlopen(URL + "/info").read())
    print("server:", info["policy_type"], info["checkpoint"])
    A = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(len(ds.hf_dataset))])
    arm = [*range(7), *range(8, 15)]
    rows = []
    for t in map(int, args.frames.split(",")):
        item = ds[t]
        obs = {"observation.state": item["observation.state"].numpy().astype(np.float32), "task": np.array(TASK)}
        for c in CAMS:
            obs[c] = (item[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        b = io.BytesIO()
        np.savez(b, **obs)
        body = b.getvalue()
        post("/reset", b"")
        served, lat = [], []
        for _ in range(args.replan):
            r = post("/act", body)
            served.append(r["action"])
            lat.append(r["latency_ms"])
        served = np.array(served)
        ref = A[t: t + args.replan]
        n = len(ref)
        err = np.abs(served[:n, arm] - ref[:, arm]).mean()
        hold = np.abs(obs["observation.state"][arm] - ref[:, arm]).mean()  # baseline: just hold the current pose
        grip = np.abs(served[:n][:, [7, 15]] - ref[:, [7, 15]]).mean()
        rows.append((t, err, hold, grip, lat[0]))
        print(f"frame {t:5d} ({t / 30:5.1f} s): arm error {err:.4f} rad (hold-pose baseline {hold:.4f}) | "
              f"gripper error {grip:.3f} | first-call latency {lat[0]:.0f} ms")
    e = np.array(rows)
    print(f"mean arm error {e[:, 1].mean():.4f} rad vs hold-pose baseline {e[:, 2].mean():.4f} rad")


if __name__ == "__main__":
    main()
