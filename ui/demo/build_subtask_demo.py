"""Build the sub-task demo page: how data_prep/make_subtasks.py cut one handover episode into 5 sub-tasks.

Re-runs the exact same detection code (imported from data_prep/make_subtasks.py) on one episode and writes
the signals, events, still pauses and boundaries, plus boundary snapshots from the three cameras.

Output: subtask_demo/ (index.html, data.json, video.mp4, snaps/*.png, plotly.min.js)
Usage:  python ui/demo/build_subtask_demo.py [--episode 0]
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fr3duo import subtasks as ms  # noqa: E402

OUT = ROOT / "build" / "subtask_demo"
CAMS = ["external_camera", "left_wrist_camera", "right_wrist_camera"]


def grab(src, E, ep, cam, t, fps):
    k = f"videos/observation.images.{cam}"
    e = E.loc[ep]
    f = src / f"videos/observation.images.{cam}/chunk-{e[k + '/chunk_index']:03d}/file-{e[k + '/file_index']:03d}.mp4"
    ts = e[k + "/from_timestamp"] + t / fps
    with av.open(str(f)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0, ts - 1) / s.time_base), stream=s)
        for fr in c.decode(s):
            if fr.time >= ts - 1e-3:
                return fr.to_ndarray(format="rgb24")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ep = ap.parse_args().episode
    src = ms.SRC["handover"]
    fps = json.loads((src / "meta/info.json").read_text())["fps"]
    df = pd.concat(pd.read_parquet(f) for f in glob.glob(str(src / "data/*/*.parquet")))
    g = df[df.episode_index == ep].sort_values("frame_index")
    S, A = np.stack(g["observation.state"]), np.stack(g["action"])

    # exactly what data_prep/make_subtasks.py does
    mv_l, mv_r = ms.moving(S, ms.L_ARM, fps), ms.moving(S, ms.R_ARM, fps)
    b, ev, still = ms.segment("handover", S, A, fps)
    problems = ms.check("handover", b, ev, still, S, A, fps)
    try:  # newer data_prep/make_subtasks.py defaults to 3 handover steps; the demo shows the 5 fine ones it always detects
        _, subs = ms.texts("handover", handover_subtasks=5)
    except TypeError:
        _, subs = ms.texts("handover")
    lc_done = ms.grip_settled(S, ms.L_G, ev["left_close"], fps) or ev["left_close"]
    rc_done = ms.grip_settled(S, ms.R_G, ev["right_close"], fps) or ev["right_close"]
    starts = [ms.first(mv_l, lc_done), ms.first(mv_r, lc_done), ev["left_open"], ms.first(mv_r, rc_done)]
    why = ["first left-arm motion after the left gripper finished closing",
           "first right-arm motion after that (the right arm starts its reach)",
           "the left gripper command starts opening",
           "first right-arm motion after the right gripper finished closing"]

    speed = lambda arm: np.r_[np.abs(np.diff(S[:, arm], axis=0)).sum(1) * fps, 0.0]  # noqa: E731
    OUT.mkdir(exist_ok=True)
    (OUT / "snaps").mkdir(exist_ok=True)
    E = pd.read_parquet(glob.glob(str(src / "meta/episodes/*/*.parquet"))[0]).set_index("episode_index")
    cuts = []
    for k, (e, n) in enumerate(zip(starts, still)):
        tiles = [grab(src, E, ep, c, b[k + 1], fps) for c in CAMS]
        Image.fromarray(np.concatenate(tiles, 1)).save(OUT / f"snaps/cut{k + 1}.png")
        cuts.append(dict(frame=b[k + 1], event_frame=int(e), pause_from=int(e - n), pause_to=int(e), why=why[k]))

    qa = {n: json.loads((ms.QA / f"{n}_qa.json").read_text()) for n in ["handover", "cubes"]}
    key = "videos/observation.images.external_camera/"
    out = dict(
        episode=ep, fps=fps, n=len(S), video_from=float(E.loc[ep, key + "from_timestamp"]),
        t=(np.arange(len(S)) / fps).round(3).tolist(),
        speed_l=speed(ms.L_ARM).round(4).tolist(), speed_r=speed(ms.R_ARM).round(4).tolist(),
        moving_l=mv_l.astype(int).tolist(), moving_r=mv_r.astype(int).tolist(),
        grip_cmd_l=A[:, ms.L_G].round(3).tolist(), grip_cmd_r=A[:, ms.R_G].round(3).tolist(),
        threshold=ms.STILL_RAD_S, bounds=[int(x) for x in b], subtasks=subs, cuts=cuts,
        events={k: (None if v is None else int(v)) for k, v in ev.items()},
        problems=problems,
        qa={n: dict(episodes=q["episodes"], flagged=q["flagged"], durations=q["duration_s_by_subtask"],
                    problems=q["problems"]) for n, q in qa.items()},
    )
    (OUT / "data.json").write_text(json.dumps(out))
    vid = src / f"videos/observation.images.external_camera/chunk-{E.loc[ep, key + 'chunk_index']:03d}/file-{E.loc[ep, key + 'file_index']:03d}.mp4"
    shutil.copy(vid, OUT / "video.mp4")
    shutil.copy(ROOT / "ui/demo/vendor/plotly-2.35.2.min.js", OUT / "plotly.min.js")
    shutil.copy(ROOT / "ui/demo/subtask_demo_template.html", OUT / "index.html")
    print(f"episode {ep}: bounds {b}, checks {'passed' if not problems else problems} -> {OUT}")


if __name__ == "__main__":
    main()
