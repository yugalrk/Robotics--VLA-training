"""Export a few handover frames (state, 3 images, next 50 recorded actions) to eval_results/pi05_probe_frames.npz."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from fr3duo.data import DATASET_ROOT, REPO_ID  # noqa: E402

CAMS = ["observation.images.external_camera", "observation.images.left_wrist_camera", "observation.images.right_wrist_camera"]
out = {k: [] for k in ["state", "actions", "ext", "lw", "rw", "where"]}
for ep in [0, 5]:
    ds = LeRobotDataset(REPO_ID, root=DATASET_ROOT, episodes=[ep], video_backend="pyav")
    A = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(len(ds.hf_dataset))])
    for t in [0, 200, 330, 600, 900, 1200]:
        it = ds[t]
        out["state"].append(it["observation.state"].numpy())
        out["actions"].append(np.pad(A[t:t + 50], ((0, max(0, t + 50 - len(A))), (0, 0)), mode="edge"))
        for key, c in zip(["ext", "lw", "rw"], CAMS):
            out[key].append((it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        out["where"].append((ep, t))
        print("ep", ep, "frame", t, "task", it["task"])
np.savez(ROOT / "eval_results" / "pi05_probe_frames.npz", **{k: np.array(v) for k, v in out.items()})
print("saved", {k: np.array(v).shape for k, v in out.items()})
