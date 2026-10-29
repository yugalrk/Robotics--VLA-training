"""Loader + train/val split for the FR3 Duo handover dataset (84 episodes, clinical scene).

Usage:
    python data_prep/make_splits.py   # writes configs/splits.json and checks that LeRobot can load a sample
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84"
REPO_ID = "local/fr3_duo_handover_v9_night_84"  # LeRobot needs a name; data is read from DATASET_ROOT
SPLITS_FILE = ROOT / "configs" / "splits.json"

CAMERAS = [
    "observation.images.external_camera",
    "observation.images.left_wrist_camera",
    "observation.images.right_wrist_camera",
]


def light_table():
    """Per-episode lighting (the only thing randomized in this dataset)."""
    rows = []
    for line in open(DATASET_ROOT / "meta/fr3_episodes_extra.jsonl"):
        e = json.loads(line)
        light = e["randomization"]["applied"]["light"]
        rows.append((e["episode_index"], light["rect"]["intensity_to"] + light["dome"]["intensity_to"],
                     light["rect"]["color_temperature_K"]))
    return np.array(rows)


def make_splits(seed=0):
    """Hold out 10 episodes: the 3 darkest, 3 brightest, and 4 random ones.

    All 84 episodes share the same joint trajectory, so lighting is the only axis
    along which we can test generalization.
    """
    t = light_table()
    order = t[np.argsort(t[:, 1]), 0].astype(int)
    val = set(order[:3]) | set(order[-3:])
    rest = [i for i in order if i not in val]
    val |= set(np.random.default_rng(seed).choice(rest, 4, replace=False).tolist())
    val = sorted(int(i) for i in val)
    train = sorted(int(i) for i in t[:, 0] if int(i) not in val)
    return {"train": train, "val": val,
            "note": "val = 3 darkest + 3 brightest + 4 random episodes (by rect+dome light intensity)"}


def load_dataset(episodes=None, delta_timestamps=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(REPO_ID, root=DATASET_ROOT, episodes=episodes,
                          delta_timestamps=delta_timestamps, video_backend="pyav")


def main():
    splits = make_splits()
    SPLITS_FILE.write_text(json.dumps(splits, indent=1))
    print(f"train {len(splits['train'])} eps, val {len(splits['val'])} eps: {splits['val']}")

    ds = load_dataset(episodes=splits["val"][:1])
    print(f"fps={ds.fps}  frames={ds.num_frames}  episodes={ds.num_episodes}")
    s = ds[0]
    for k, v in s.items():
        print(f"  {k:40s} {tuple(v.shape) if hasattr(v, 'shape') else v}")


if __name__ == "__main__":
    main()
