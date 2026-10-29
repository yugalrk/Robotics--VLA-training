"""Augment the handover dataset: retarget the recorded left-arm motion to a grid of box positions on the cart.

    .venv\\Scripts\\python.exe augmentation\\augment_box_grid.py                  # whole cart top, 5 cm grid, all CPU cores
    .venv\\Scripts\\python.exe augmentation\\augment_box_grid.py --dx -5 5 --dz -5 5 --step 1

All 84 recorded episodes are the same motion, so one recording (episode 0) is the template. For every box shift
(dx, dz) on the table plane (room X and Z; Y is up in this scene) the left arm's gripper path is shifted with the
method of augmentation/retarget_trajectory.py: shift blended in over the reach, held through grasp and lift, blended out
while carrying; recorded gripper orientation; per-frame damped-least-squares IK pulled towards the recorded joints;
FR3 joint limits. Right arm, gripper commands and timing stay as recorded.

Every cell gets a status:
  ok          IK reaches every shifted pose (< 0.5 mm, < 0.1 deg), inside the joint limits, joint steps <= 1.25x the recording
  out_of_reach  the arm cannot do it (the kinematic boundary)
  off_cart    the box would not stand on the cart top (inner area, minus half the box and a rim margin)
Only ok cells are written as episodes.

Output: Datasets/fr3_duo_handover_84_ep/Augmented/
  data/chunk-000/file-XXX.parquet  rows: episode_index, frame_index, timestamp, observation.state (16), action (16),
                                    box_dx_m, box_dz_m        (no camera images: see meta/info.json)
  meta/info.json, meta/episodes.parquet (ok cells), meta/cells.parquet (every cell + status), meta/augmentation.json
  ui/index.json, ui/paths/<episode>.json   for the player
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DS = ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84"
OUT = ROOT / "Datasets/fr3_duo_handover_84_ep/Augmented"
BLEND_IN, BLEND_OUT = (2.0, 7.6), (12.5, 20.0)
CART_TOP = {"x": (-5.154, -4.086), "z": (22.475, 23.089)}  # cart top surface (scene mesh, y 0.82-0.83)
CART_MARGIN = 0.03 + 0.055  # rim + half the box footprint
UI_EVERY = 4  # player paths at 7.5 Hz
CHUNK = 250  # episodes per parquet file
_G = {}


def _init(episode):
    from fr3duo.kinematics import FK
    from fr3duo.retarget import pose, smoothstep
    d = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
                  for f in glob.glob(str(DS / "data/*/*.parquet")))
    d = d[d.episode_index == episode].sort_values("frame_index")
    S, A, t = np.stack(d["observation.state"]).astype(float), np.stack(d["action"]).astype(float), d["timestamp"].to_numpy()
    fk = FK()
    lim = np.radians(np.array([j["limits_deg"] for j in fk.chain["left"]["joints"]]))
    w = smoothstep(t, *BLEND_IN) * (1 - smoothstep(t, *BLEND_OUT))
    _G.update(S=S, A=A, t=t, fk=fk, lim=lim, poses=[pose(fk, q) for q in S[:, :7]], w=w, active=np.flatnonzero(w > 0),
              rec_step=np.abs(np.diff(S[:, :7], axis=0)).max(), g=int(np.flatnonzero(A[:, 7] < 0.2)[0]))


def solve(cell):
    from fr3duo.retarget import ik, pose
    dx, dz = cell
    G = _G
    Q, lim = G["S"][:, :7], G["lim"]
    lo, hi = lim[:, 0] + 1e-4, lim[:, 1] - 1e-4
    shift = np.array([dx, 0.0, dz]) / 100
    Qn, err_p, err_r = Q.copy(), 0.0, 0.0
    q_prev = Q[G["active"][0] - 1].copy()
    for k in G["active"]:
        p, R = G["poses"][k]
        pt = p + G["w"][k] * shift
        qn, res = ik(G["fk"], pt, R, q_prev, Q[k], lo, hi)
        pn, _ = pose(G["fk"], qn)
        err_p, err_r = max(err_p, np.linalg.norm(pt - pn)), max(err_r, np.linalg.norm(res[3:]))
        Qn[k] = q_prev = qn
        if err_p > 0.01:  # clearly unreachable: stop early
            break
    step = np.abs(np.diff(Qn, axis=0)).max()
    margin = float(np.degrees(np.minimum(Qn - lim[:, 0], lim[:, 1] - Qn).min()))
    ok = bool(err_p < 5e-4 and np.degrees(err_r) < 0.1 and margin > 0 and step <= G["rec_step"] * 1.25)
    gp = pose(G["fk"], Qn[G["g"]])[0]
    stats = {"box_dx_mm": round(dx * 10, 1), "box_dz_mm": round(dz * 10, 1), "status": "ok" if ok else "out_of_reach",
             "max_pos_err_mm": round(err_p * 1000, 4), "max_rot_err_deg": round(float(np.degrees(err_r)), 4),
             "max_joint_change_deg": round(float(np.degrees(np.abs(Qn - Q).max())), 2), "max_step_deg": round(float(np.degrees(step)), 3),
             "min_limit_margin_deg": round(margin, 1), "grasp_x": round(float(gp[0]), 5), "grasp_y": round(float(gp[1]), 5), "grasp_z": round(float(gp[2]), 5)}
    return cell, stats, (Qn.astype(np.float32) if ok else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, nargs=2, default=None, help="dx range in cm (default: the cart top)")
    ap.add_argument("--dz", type=float, nargs=2, default=None, help="dz range in cm (default: the cart top)")
    ap.add_argument("--step", type=float, default=5.0, help="grid step in cm")
    ap.add_argument("--episode", type=int, default=0, help="template episode (all 84 are identical)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    _init(args.episode)
    S, A, t, fk, g = _G["S"], _G["A"], _G["t"], _G["fk"], _G["g"]
    from fr3duo.retarget import pose
    grasp0 = pose(fk, S[g, :7])[0]
    # cart limits for the box centre, as shifts of the recorded grasp point (cm)
    cx = [(CART_TOP["x"][0] + CART_MARGIN - grasp0[0]) * 100, (CART_TOP["x"][1] - CART_MARGIN - grasp0[0]) * 100]
    cz = [(CART_TOP["z"][0] + CART_MARGIN - grasp0[2]) * 100, (CART_TOP["z"][1] - CART_MARGIN - grasp0[2]) * 100]
    st = args.step  # grid aligned to multiples of the step, so the recorded spot (0, 0) is always a cell
    rx = args.dx or [np.floor(cx[0] / st) * st, np.ceil(cx[1] / st) * st]
    rz = args.dz or [np.floor(cz[0] / st) * st, np.ceil(cz[1] / st) * st]
    xs = np.round(np.arange(rx[0], rx[1] + 1e-9, args.step), 3)
    zs = np.round(np.arange(rz[0], rz[1] + 1e-9, args.step), 3)
    cells = [(dx, dz) for dz in zs for dx in xs]
    on_cart = [(cx[0] <= dx <= cx[1]) and (cz[0] <= dz <= cz[1]) for dx, dz in cells]
    todo = [c for c, oc in zip(cells, on_cart) if oc]
    print(f"grid dx {xs[0]}..{xs[-1]} cm, dz {zs[0]}..{zs[-1]} cm, step {args.step} cm: {len(cells)} cells, "
          f"{len(todo)} on the cart (box centre limits dx {cx[0]:.1f}..{cx[1]:.1f}, dz {cz[0]:.1f}..{cz[1]:.1f} cm); "
          f"{args.workers} workers", flush=True)

    results = {}
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.episode,)) as pool:
        for k, (cell, stats, Qn) in enumerate(pool.imap_unordered(solve, todo, chunksize=4)):
            results[cell] = (stats, Qn)
            if (k + 1) % 200 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                print(f"  {k + 1}/{len(todo)} cells, {sum(r[0]['status'] == 'ok' for r in results.values())} ok, "
                      f"{el:.0f} s, ~{el / (k + 1) * (len(todo) - k - 1):.0f} s left", flush=True)

    # write: episodes = ok cells, in grid order
    if OUT.exists():
        for sub in ["data", "meta", "ui/paths"]:
            shutil.rmtree(OUT / sub, ignore_errors=True)
    (OUT / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (OUT / "meta").mkdir(parents=True, exist_ok=True)
    (OUT / "ui/paths").mkdir(parents=True, exist_ok=True)
    allcells, eps, rows, file_i, n_rows = [], [], [], 0, 0
    Q = S[:, :7]
    poses_every = lambda Qx: np.array([pose(fk, q)[0] for q in Qx[::UI_EVERY]])

    def flush():
        nonlocal rows, file_i, n_rows
        if not rows:
            return
        df = pd.DataFrame(rows, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action", "box_dx_m", "box_dz_m"])
        df.to_parquet(OUT / f"data/chunk-000/file-{file_i:03d}.parquet", index=False, compression="zstd")
        n_rows += len(df)
        rows, file_i = [], file_i + 1

    for cell, oc in zip(cells, on_cart):
        if not oc:
            allcells.append({"box_dx_mm": round(cell[0] * 10, 1), "box_dz_mm": round(cell[1] * 10, 1), "status": "off_cart", "episode_index": -1})
            continue
        stats, Qn = results[cell]
        e = len(eps) if stats["status"] == "ok" else -1
        allcells.append({**stats, "episode_index": e})
        if e < 0:
            continue
        eps.append({**stats, "episode_index": e})
        Sn, An = S.copy(), A.copy()
        Sn[:, :7] = Qn
        An[:, :7] = A[:, :7] + np.vstack([Qn[1:] - Q[1:], np.zeros((1, 7))])  # action k targets frame k+1
        for k in range(len(t)):
            rows.append((e, k, float(t[k]), Sn[k].astype(np.float32).tolist(), An[k].astype(np.float32).tolist(), cell[0] / 100, cell[1] / 100))
        (OUT / "ui/paths" / f"{e}.json").write_text(json.dumps({"path": poses_every(Qn.astype(float)).round(5).tolist(),
                                                                "dq_deg": np.degrees(Qn - Q)[::UI_EVERY].round(2).tolist()}))
        if (e + 1) % CHUNK == 0:
            flush()
    flush()
    C, E = pd.DataFrame(allcells), pd.DataFrame(eps)
    C.to_parquet(OUT / "meta/cells.parquet", index=False)
    E.to_parquet(OUT / "meta/episodes.parquet", index=False)
    counts = C.status.value_counts().to_dict()
    info = {"name": "fr3_duo_handover_box_grid", "source": str(DS.relative_to(ROOT)), "template_episode": args.episode,
            "total_episodes": len(E), "frames_per_episode": len(t), "total_frames": n_rows, "fps": 30, "cells": counts,
            "features": {"observation.state": "16 = left q1..q7 (rad), left gripper, right q1..q7, right gripper",
                         "action": "16, same layout: joint targets + gripper commands (1 open, 0 closed)",
                         "box_dx_m / box_dz_m": "box shift from the recorded position, room X / Z (Y is up)"},
            "cameras": "NONE. The recorded videos show the box at its original position; render each episode in "
                       "Isaac Sim with the box moved before training a vision-based policy on this data."}
    (OUT / "meta/info.json").write_text(json.dumps(info, indent=1))
    (OUT / "meta/augmentation.json").write_text(json.dumps({
        "method": "Cartesian shift of the left gripper path + per-frame IK (damped least squares, 6-D pose, null space "
                  "pulled to the recorded joints, FR3 joint limits); see tools/retarget_trajectory.py",
        "grid_cm": {"dx": xs.tolist(), "dz": zs.tolist(), "step": args.step}, "cart_limits_cm": {"dx": cx, "dz": cz},
        "blend_in_s": BLEND_IN, "blend_out_s": BLEND_OUT, "grasp_time_s": float(t[g]),
        "status_rules": {"ok": "position error < 0.5 mm, orientation error < 0.1 deg, inside joint limits, joint step <= 1.25x the recording's",
                         "out_of_reach": "any ok condition fails (kinematic boundary)", "off_cart": "box centre outside the cart top minus rim and half the box"},
        "not_checked": "collisions (fingers / arm vs cart rim, tray rails, bottle) and the right arm at the handover"}, indent=1))
    (OUT / "ui/index.json").write_text(json.dumps({
        "dx_mm": (xs * 10).tolist(), "dz_mm": (zs * 10).tolist(), "step_mm": args.step * 10, "t": t[::UI_EVERY].round(3).tolist(), "grasp_t": float(t[g]),
        "orig": np.array([pose(fk, q)[0] for q in Q[::UI_EVERY]]).round(5).tolist(), "grip": S[::UI_EVERY, 7].round(3).tolist(),
        "blend": [BLEND_IN, BLEND_OUT], "cart_limits_cm": {"dx": cx, "dz": cz}, "cells": allcells}))
    shutil.copy(ROOT / "ui/demo/vendor/plotly-2.35.2.min.js", OUT / "ui/plotly.min.js")
    shutil.copy(ROOT / "ui/demo/augment_ui_template.html", OUT / "ui/index.html")
    ok = E if len(E) else None
    print(f"\ncells: {counts}")
    if ok is not None:
        print(f"ok episodes: {len(E)}; dx {E.box_dx_mm.min():.0f}..{E.box_dx_mm.max():.0f} mm, dz {E.box_dz_mm.min():.0f}..{E.box_dz_mm.max():.0f} mm; "
              f"largest joint change {E.max_joint_change_deg.max():.1f} deg")
    print(f"wrote {OUT}: {n_rows} rows in {file_i} files, {sum(f.stat().st_size for f in (OUT / 'data').rglob('*.parquet')) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
