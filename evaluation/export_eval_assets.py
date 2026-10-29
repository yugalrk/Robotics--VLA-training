"""Export what the Isaac Sim eval needs from the handover dataset into one JSON file.

Writes artifacts/eval_assets/handover_reference.json with:
- initial 16-D state (frame 0; identical in every episode)
- one demo episode's 30 fps actions + states (for the replay sanity check)
- lighting settings of the held-out (val) episodes, so the eval uses unseen lighting
- the demo's final box pose and the task instruction

Usage:
    python evaluation/export_eval_assets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.data import DATASET_ROOT, SPLITS_FILE

OUT = Path(__file__).resolve().parents[1] / "artifacts/eval_assets/handover_reference.json"


def main():
    val = json.loads(SPLITS_FILE.read_text())["val"]
    df = pd.read_parquet(DATASET_ROOT / "data/chunk-000/file-000.parquet")
    ep = df[df.episode_index == val[0]]
    S, A = np.stack(ep["observation.state"]), np.stack(ep["action"])
    extra = {e["episode_index"]: e for e in map(json.loads, open(DATASET_ROOT / "meta/fr3_episodes_extra.jsonl"))}
    conv = json.loads((DATASET_ROOT / "meta/fr3_conversion.json").read_text())

    ref = {
        "task": conv["dataset_cfg"]["task"],
        "fps": 30,
        "state_names": json.loads((DATASET_ROOT / "meta/info.json").read_text())["features"]["observation.state"]["names"],
        "initial_state": S[0].round(6).tolist(),
        "demo_episode": int(val[0]),
        "demo_actions": A.round(6).tolist(),
        "demo_states": S.round(6).tolist(),
        "hold_cut_frame": int(extra[val[0]]["hold_cut"]["dropped_frames"][0]),
        "demo_final_box_pose": extra[val[0]]["final_box_pose"],
        "gripper_contract": conv["dataset_cfg"]["gripper_contract"],
        "gripper_drive_recorded": extra[val[0]]["gripper_drive"],
        "camera_params": conv["recorder"]["camera_params"],
        "val_lighting": [{"episode_index": int(e), **extra[e]["randomization"]["applied"]["light"]} for e in val],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(ref))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): demo ep {val[0]}, {len(A)} frames, "
          f"{len(val)} held-out lighting configs")


if __name__ == "__main__":
    main()
