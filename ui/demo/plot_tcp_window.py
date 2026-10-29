"""Map a 3.3 s window of a recorded episode to the gripper-centre XYZ trajectory and plot it (time = colour).

    .venv\\Scripts\\python.exe ui\\demo\\plot_tcp_window.py [--start 7.6] [--seconds 3.33] [--arm left] [--episode 0]

Forward kinematics (fr3duo/kinematics.py, the scene's FR3 chain = official Franka FR3v2 geometry) turns each frame's 7
joint angles into the gripper centre in room coordinates (metres, Y up). Writes eval_results/tcp_trajectory.html:
a 3D view plus top / front / side projections, light -> dark with time, start and end marked, the 20 policy steps
(6 Hz) as dots, and the hand-out scene's box for reference.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fr3duo.kinematics import FK  # noqa: E402

DS = ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84"
SCENE_BOX = {"min": [-4.265, 0.829, 22.721], "max": [-4.234, 0.901, 22.828]}  # hand-out scene box (bounding box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--arm", choices=["left", "right"], default="left")
    ap.add_argument("--start", type=float, default=7.6, help="window start (s); default ends at the left grasp, 10.9 s")
    ap.add_argument("--seconds", type=float, default=20 / 6, help="window length; 20 actions at 6 Hz = 3.33 s")
    ap.add_argument("--out", default="tcp_trajectory.html", help="file name in eval_results/")
    args = ap.parse_args()
    d = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                  for f in glob.glob(str(DS / "data/*/*.parquet")))
    d = d[d.episode_index == args.episode].sort_values("frame_index")
    S, A = np.stack(d["observation.state"]), np.stack(d["action"])
    a, b = int(round(args.start * 30)), int(round((args.start + args.seconds) * 30))
    sl = slice(0, 7) if args.arm == "left" else slice(8, 15)
    g = 7 if args.arm == "left" else 15
    fk = FK()
    P = fk.tcp_path(args.arm, S[a:b + 1, sl])
    t = np.arange(a, b + 1) / 30
    steps = np.arange(a, b + 1, 5)  # the 6 Hz policy steps inside the window
    data = {"x": P[:, 0].tolist(), "y": P[:, 1].tolist(), "z": P[:, 2].tolist(), "t": t.round(3).tolist(),
            "grip": S[a:b + 1, g].round(3).tolist(), "steps": ((steps - a)).tolist(),
            "arm": args.arm, "episode": args.episode, "box": SCENE_BOX,
            "path_len_mm": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * 1000),
            "displacement_mm": (np.round((P[-1] - P[0]) * 1000, 1)).tolist()}
    plotly = (ROOT / "ui" / "demo" / "vendor" / "plotly-2.35.2.min.js").read_text(encoding="utf-8")  # embedded: works offline
    html = (ROOT / "ui" / "demo" / "tcp_trajectory_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data)).replace("/*PLOTLY*/", plotly.replace("</script", "<\\/script"))
    out = ROOT / "eval_results" / args.out
    out.write_text(html, encoding="utf-8")
    print(f"{args.arm} gripper centre, episode {args.episode}, {t[0]:.2f}-{t[-1]:.2f} s ({len(P)} frames):")
    print(f"  start {np.round(P[0], 4).tolist()}  end {np.round(P[-1], 4).tolist()}")
    print(f"  path length {data['path_len_mm']:.1f} mm, start->end displacement (x, up, z) {data['displacement_mm']} mm")
    print("wrote", out)


if __name__ == "__main__":
    main()
