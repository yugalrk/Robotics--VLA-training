"""Build the three-page demo site: 1 data (matrix + video), 2 robot model (FK + unique IK), 3 camera -> grid -> LUT.

    .venv\\Scripts\\python.exe ui\\demo\\build_site.py        -> build/site/  (serve it: ui/demo/serve_static.py --dir build/site --port 8779)

Page 1 uses the raw dataset only. Page 2 computes, for episode 0:
  - both arms' link positions from forward kinematics (the 3D robot model, stick figure)
  - the left arm's original joint trajectory recalculated from its gripper pose by inverse kinematics, twice:
      A) no extra rule: start from the recorded first pose, then the smallest joint change per frame
         -> the gripper pose is matched, but the 7th joint's freedom lets the joints drift (many solutions)
      B) one extra rule: joint 7 follows its recorded angle, the other 6 are solved from the pose
         -> 6 equations, 6 unknowns: one solution per frame (chosen continuous with the previous frame)
Page 3 is the box-grid LUT player (Datasets/.../Augmented/ui), copied in.
"""
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fr3duo.kinematics import FK, frame, rot  # noqa: E402
from fr3duo.retarget import jacobian, pose, rot_err  # noqa: E402

DS = ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84"
SUB = ROOT / "Datasets/fr3_handover84_subtasks"
AUG_UI = ROOT / "Datasets/fr3_duo_handover_84_ep/Augmented/ui"
SITE = ROOT / "build" / "site"
TPL = ROOT / "ui/demo/site"
EVERY = 2  # 15 Hz for the pages
NAMES = [f"left joint {i}" for i in range(1, 8)] + ["left gripper"] + [f"right joint {i}" for i in range(1, 8)] + ["right gripper"]


def load_frames():
    d = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
                  for f in glob.glob(str(DS / "data/*/*.parquet")))
    return d


def page_data(d):
    key = "videos/observation.images.external_camera/"
    eps = pd.concat(pd.read_parquet(f) for f in glob.glob(str(DS / "meta/episodes/*/*.parquet")))
    eps = eps[(eps[key + "chunk_index"] == 0) & (eps[key + "file_index"] == 0)].sort_values("episode_index")
    sub_txt, sub_frames = {}, None
    if (SUB / "meta/subtasks.parquet").exists():
        s = pd.read_parquet(SUB / "meta/subtasks.parquet")
        sub_txt = {int(i): t for t, i in zip(s.index, s.subtask_index)}
        sub_frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "subtask_index"]) for f in glob.glob(str(SUB / "data/*/*.parquet")))
    out = {"names": NAMES, "fps": 30 / EVERY, "episodes": []}
    for e in eps.itertuples():
        ep = int(e.episode_index)
        x = d[d.episode_index == ep].sort_values("frame_index").iloc[::EVERY]
        S, A = np.stack(x["observation.state"]), np.stack(x["action"])
        lab = []
        if sub_frames is not None:
            sf = sub_frames[sub_frames.episode_index == ep].sort_values("frame_index").iloc[::EVERY]
            lab = [sub_txt.get(int(i), "") for i in sf.subtask_index]
        out["episodes"].append({"episode": ep, "video_from": float(eps.loc[eps.episode_index == ep, key + "from_timestamp"].iloc[0]),
                                "t": x["timestamp"].round(3).tolist(), "state": S.round(4).tolist(), "action": A.round(4).tolist(), "subtask": lab})
    dst = SITE / "1-data"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "data.json").write_text(json.dumps(out))
    shutil.copy(DS / "videos/observation.images.external_camera/chunk-000/file-000.mp4", dst / "video.mp4")
    print(f"page 1: {len(out['episodes'])} episodes, {len(out['episodes'][0]['t'])} frames each at {30 / EVERY:.0f} Hz")


def link_points(fk, side, q):
    arm = fk.chain[side]
    T = arm["_T0"].copy()
    pts = [T[:3, 3].copy()]
    for (F0, axis, F1inv), qi in zip(arm["_J"], q):
        T = T @ F0 @ rot(axis, qi) @ F1inv
        pts.append(T[:3, 3].copy())
    pts.append((T @ np.r_[arm["tcp_in_link7"], 1])[:3])  # gripper centre
    return np.array(pts)


def ik_recover(fk, P, R, q0, fix7=None, iters=50, lam=1e-3):
    """Recover joints from gripper poses frame by frame. fix7: recorded joint-7 angles (rule B) or None (rule A:
    smallest change from the previous frame in the spare direction)."""
    out, q = [], q0.copy()
    lim = np.radians(np.array([j["limits_deg"] for j in fk.chain["left"]["joints"]]))
    for k in range(len(P)):
        if fix7 is not None:
            q[6] = fix7[k]
        q_prev = q.copy()
        for _ in range(iters):
            p, Rn = pose(fk, q)
            e = np.r_[P[k] - p, rot_err(R[k], Rn)]
            if np.linalg.norm(e) < 1e-8:
                break
            J = jacobian(fk, q)
            if fix7 is not None:
                J = J[:, :6]  # 6 x 6: joint 7 is given
                q[:6] += J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            else:
                Jp = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), np.eye(6))
                q += Jp @ e + (np.eye(7) - Jp @ J) @ (0.5 * (q_prev - q))  # spare direction: stay near the previous frame
            q = np.clip(q, lim[:, 0], lim[:, 1])
        out.append(q.copy())
    return np.array(out)


def page_model(d, fk):
    x = d[d.episode_index == 0].sort_values("frame_index").iloc[::EVERY]
    S, t = np.stack(x["observation.state"]).astype(float), x["timestamp"].to_numpy()
    QL, QR = S[:, :7], S[:, 8:15]
    poses = [pose(fk, q) for q in QL]
    P, R = np.array([p for p, _ in poses]), np.array([r for _, r in poses])
    QA = ik_recover(fk, P, R, QL[0])
    QB = ik_recover(fk, P, R, QL[0], fix7=QL[:, 6])
    err = lambda Q: max(np.linalg.norm(P[k] - pose(fk, Q[k])[0]) for k in range(len(Q))) * 1000
    stats = {"A_pose_err_mm": round(err(QA), 4), "B_pose_err_mm": round(err(QB), 4),
             "A_joint_diff_deg": round(float(np.degrees(np.abs(QA - QL).max())), 2), "B_joint_diff_deg": round(float(np.degrees(np.abs(QB - QL).max())), 4)}
    L = np.array([link_points(fk, "left", q) for q in QL]).round(4)
    Rr = np.array([link_points(fk, "right", q) for q in QR]).round(4)
    LA = np.array([link_points(fk, "left", q) for q in QA]).round(4)
    out = {"t": t.round(3).tolist(), "left": L.tolist(), "right": Rr.tolist(), "leftA": LA.tolist(),
           "q_rec": np.degrees(QL).round(3).tolist(), "q_A": np.degrees(QA).round(3).tolist(), "q_B": np.degrees(QB).round(3).tolist(),
           "grip_l": S[:, 7].round(3).tolist(), "grip_r": S[:, 15].round(3).tolist(), "stats": stats}
    dst = SITE / "2-model"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "data.json").write_text(json.dumps(out))
    print(f"page 2: {len(t)} frames; recalculation A (no extra rule): pose error {stats['A_pose_err_mm']} mm, joints differ up to "
          f"{stats['A_joint_diff_deg']} deg; B (joint 7 given): pose error {stats['B_pose_err_mm']} mm, joints differ up to {stats['B_joint_diff_deg']} deg")


def main():
    SITE.mkdir(exist_ok=True)
    d = load_frames()
    fk = FK()
    page_data(d)
    page_model(d, fk)
    if AUG_UI.exists():  # page 3: the LUT player (with its paths and camera assets)
        shutil.copytree(AUG_UI, SITE / "3-lut", dirs_exist_ok=True)
        shutil.copy(TPL.parent / "augment_ui_template.html", SITE / "3-lut/index.html")
    plotly = ROOT / "ui/demo/vendor/plotly-2.35.2.min.js"
    for page in ["1-data", "2-model"]:
        shutil.copy(plotly, SITE / page / "plotly.min.js")
    shutil.copy(TPL / "index.html", SITE / "index.html")
    shutil.copy(TPL / "data.html", SITE / "1-data/index.html")
    shutil.copy(TPL / "model.html", SITE / "2-model/index.html")
    print("wrote", SITE, "-> serve with: .venv\\Scripts\\python.exe ui\\demo\\serve_static.py --dir build\\site --port 8779")


if __name__ == "__main__":
    main()
