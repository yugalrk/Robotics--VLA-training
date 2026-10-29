"""Build a video-synced XYZ verification page for one dataset video file.

    .venv\\Scripts\\python.exe ui\\demo\\make_xyz_verify.py [--camera external_camera] [--chunk 0] [--file 0]

For every episode stored in videos/observation.images.<camera>/chunk-XXX/file-YYY.mp4, forward kinematics
(fr3duo/kinematics.py) turns each frame's joint angles (observation.state, the training matrix) into both grippers' centre
XYZ (room coordinates, metres, Y up). Writes eval_results/xyz_verify/: index.html (video next to the 3D path and
x/y/z-over-time plots, with a marker that follows the video), data.json, the video and plotly. Open it through the
eval-results server: http://127.0.0.1:8777/xyz_verify/
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fr3duo.kinematics import FK  # noqa: E402

DS = ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84"
SUB = ROOT / "Datasets/fr3_handover84_subtasks"  # per-frame sub-task labels (data_prep/make_subtasks.py), optional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="external_camera")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--file", type=int, default=0)
    args = ap.parse_args()
    key = f"videos/observation.images.{args.camera}/"
    eps = pd.concat(pd.read_parquet(f) for f in glob.glob(str(DS / "meta/episodes/*/*.parquet")))
    eps = eps[(eps[key + "chunk_index"] == args.chunk) & (eps[key + "file_index"] == args.file)].sort_values("episode_index")
    video = DS / f"videos/observation.images.{args.camera}/chunk-{args.chunk:03d}/file-{args.file:03d}.mp4"
    frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
                       for f in glob.glob(str(DS / "data/*/*.parquet")))
    sub_txt, sub_frames = {}, None
    if (SUB / "meta/subtasks.parquet").exists():
        s = pd.read_parquet(SUB / "meta/subtasks.parquet")
        sub_txt = {int(i): t for t, i in zip(s.index, s.subtask_index)}
        sub_frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "subtask_index"])
                               for f in glob.glob(str(SUB / "data/*/*.parquet")))
    fk = FK()
    out = {"camera": args.camera, "video": video.name, "fps": 30, "episodes": []}
    for e in eps.itertuples():
        ep = int(e.episode_index)
        d = frames[frames.episode_index == ep].sort_values("frame_index")
        S, A = np.stack(d["observation.state"]), np.stack(d["action"])
        L, R = fk.tcp_path("left", S[:, :7]), fk.tcp_path("right", S[:, 8:15])
        labels = []
        if sub_frames is not None:
            sf = sub_frames[sub_frames.episode_index == ep].sort_values("frame_index")
            labels = [sub_txt.get(int(i), "") for i in sf.subtask_index]
        out["episodes"].append({
            "episode": ep, "video_from": float(eps.loc[eps.episode_index == ep, key + "from_timestamp"].iloc[0]),
            "t": d["timestamp"].round(4).tolist(), "left": L.round(4).tolist(), "right": R.round(4).tolist(),
            "grip_left": S[:, 7].round(3).tolist(), "grip_right": S[:, 15].round(3).tolist(),
            "cmd_left": A[:, 7].round(3).tolist(), "cmd_right": A[:, 15].round(3).tolist(), "subtask": labels})
        print(f"episode {ep}: {len(d)} frames, video {out['episodes'][-1]['video_from']:.1f} s onwards | "
              f"left gripper range x {np.ptp(L[:, 0]) * 1000:.0f} mm, height {np.ptp(L[:, 1]) * 1000:.0f} mm, z {np.ptp(L[:, 2]) * 1000:.0f} mm")
    dst = ROOT / "eval_results" / "xyz_verify"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "data.json").write_text(json.dumps(out))
    shutil.copy(video, dst / "video.mp4")
    shutil.copy(ROOT / "ui/demo/vendor/plotly-2.35.2.min.js", dst / "plotly.min.js")
    shutil.copy(ROOT / "ui/demo/xyz_verify_template.html", dst / "index.html")
    print("wrote", dst, "-> open http://127.0.0.1:8777/xyz_verify/")


if __name__ == "__main__":
    main()
