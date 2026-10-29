"""Retarget the recorded left-arm handover motion to a moved box: Cartesian shift + inverse kinematics.

    .venv\\Scripts\\python.exe augmentation\\retarget_trajectory.py --dx 0.03            # box +3 cm along room X
    .venv\\Scripts\\python.exe augmentation\\retarget_trajectory.py --dx 0 --dy -0.064 --dz 0.031 --name scene_box

1. FK of the recorded joint angles -> gripper-centre position p(t) and orientation R(t) (fr3duo/kinematics.py).
2. Target p'(t) = p(t) + w(t) * shift, orientation unchanged. w(t) blends 0 -> 1 over the reach, stays 1 through
   descent, grasp and lift, and goes back to 0 while carrying (the box is in the gripper then, so the handover and
   placing stay as recorded).
3. IK per frame (damped least squares on position + orientation, 6 x 7 Jacobian), started from the previous
   frame's solution; the arm's spare 7th degree of freedom is pulled towards the recorded joint angles, so the
   unique answer is "reach the shifted pose while changing the joints as little as possible"; joint limits clamp.
Writes eval_results/retarget_<name>/ (index.html, data.json, plotly) and the joint trajectory (npz).
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fr3duo.kinematics import FK  # noqa: E402

DS = ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84"


def smoothstep(t, a, b):
    x = np.clip((t - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def pose(fk, q):
    T = fk.link7("left", q)
    return (T @ np.r_[fk.chain["left"]["tcp_in_link7"], 1])[:3], T[:3, :3]


def rot_err(Rt, R):  # axis-angle vector rotating R onto Rt (world frame)
    E = Rt @ R.T
    v = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    s, c = np.linalg.norm(v), (np.trace(E) - 1) / 2
    return v if s < 1e-12 else v * np.arctan2(s, c) / s


def jacobian(fk, q, eps=1e-6):
    p0, R0 = pose(fk, q)
    J = np.zeros((6, 7))
    for i in range(7):
        dq = q.copy()
        dq[i] += eps
        p, R = pose(fk, dq)
        J[:3, i] = (p - p0) / eps
        J[3:, i] = rot_err(R, R0) / eps
    return J


def ik(fk, p_t, R_t, q_seed, q_ref, lo, hi, iters=60, lam=1e-3, k_null=0.5, tol=1e-6):
    q = q_seed.copy()
    for _ in range(iters):
        p, R = pose(fk, q)
        e = np.r_[p_t - p, rot_err(R_t, R)]
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < tol:
            break
        J = jacobian(fk, q)
        JJt = J @ J.T + lam ** 2 * np.eye(6)
        Jp = J.T @ np.linalg.solve(JJt, np.eye(6))  # damped pseudo-inverse
        dq = Jp @ e + (np.eye(7) - Jp @ J) @ (k_null * (q_ref - q))  # task + pull towards the recording
        q = np.clip(q + dq, lo, hi)
    return q, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.03, help="box shift along room X (m)")
    ap.add_argument("--dy", type=float, default=0.0, help="box shift up (m)")
    ap.add_argument("--dz", type=float, default=0.0, help="box shift along room Z (m)")
    ap.add_argument("--blend_in", type=float, nargs=2, default=[2.0, 7.6], help="seconds over which the shift ramps in")
    ap.add_argument("--blend_out", type=float, nargs=2, default=[12.5, 20.0], help="seconds over which it ramps out")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--name", default="x_plus_3cm")
    args = ap.parse_args()
    shift = np.array([args.dx, args.dy, args.dz])
    d = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
                  for f in glob.glob(str(DS / "data/*/*.parquet")))
    d = d[d.episode_index == args.episode].sort_values("frame_index")
    S, A, t = np.stack(d["observation.state"]), np.stack(d["action"]), d["timestamp"].to_numpy()
    Q = S[:, :7].astype(float)
    fk = FK()
    lim = np.radians(np.array([j["limits_deg"] for j in fk.chain["left"]["joints"]]))
    lo, hi = lim[:, 0] + 1e-4, lim[:, 1] - 1e-4
    w = smoothstep(t, *args.blend_in) * (1 - smoothstep(t, *args.blend_out))
    P, Pn, Qn, err_p, err_r = [], [], [], [], []
    q_prev = Q[0].copy()
    for k in range(len(t)):
        p, R = pose(fk, Q[k])
        pt = p + w[k] * shift
        if w[k] == 0:  # outside the blend: the recording itself
            qn, e = Q[k].copy(), np.zeros(6)
        else:
            qn, e = ik(fk, pt, R, q_prev, Q[k], lo, hi)
        q_prev = qn
        pn, _ = pose(fk, qn)
        P.append(p), Pn.append(pn), Qn.append(qn)
        err_p.append(np.linalg.norm(pt - pn)), err_r.append(np.linalg.norm(e[3:]))
    P, Pn, Qn = np.array(P), np.array(Pn), np.array(Qn)
    dQ = Qn - Q
    step_new, step_rec = np.abs(np.diff(Qn, axis=0)).max(), np.abs(np.diff(Q, axis=0)).max()
    g = int(np.flatnonzero(A[:, 7] < 0.2)[0])
    print(f"shift {np.round(shift * 1000, 1).tolist()} mm, blend in {args.blend_in} s, out {args.blend_out} s, {len(t)} frames")
    print(f"reaching accuracy: max position error {max(err_p) * 1000:.4f} mm, max orientation error {np.degrees(max(err_r)):.4f} deg")
    print(f"at the grasp ({t[g]:.1f} s): original {np.round(P[g], 4).tolist()} -> retargeted {np.round(Pn[g], 4).tolist()} "
          f"(moved {np.round((Pn[g] - P[g]) * 1000, 1).tolist()} mm)")
    print("joint change, max |new - recorded| per joint (deg):", np.round(np.degrees(np.abs(dQ).max(0)), 2).tolist())
    print(f"smoothness: largest joint step per frame {np.degrees(step_new):.3f} deg (recording: {np.degrees(step_rec):.3f} deg)")
    margin = np.degrees(np.minimum(Qn - lim[:, 0], lim[:, 1] - Qn).min(0))
    print("smallest distance to a joint limit (deg):", np.round(margin, 1).tolist())
    out = ROOT / "eval_results" / f"retarget_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "left_arm_joints.npz", t=t, q_recorded=Q, q_retargeted=Qn, weight=w, shift=shift)
    data = {"name": args.name, "shift_mm": (shift * 1000).round(1).tolist(), "blend_in": args.blend_in, "blend_out": args.blend_out,
            "t": t.round(4).tolist(), "w": w.round(4).tolist(), "orig": P.round(5).tolist(), "new": Pn.round(5).tolist(),
            "q_orig": np.degrees(Q).round(3).tolist(), "q_new": np.degrees(Qn).round(3).tolist(),
            "grip": S[:, 7].round(3).tolist(), "grasp_t": float(t[g]),
            "stats": {"max_pos_err_mm": round(max(err_p) * 1000, 4), "max_rot_err_deg": round(float(np.degrees(max(err_r))), 4),
                      "max_joint_change_deg": round(float(np.degrees(np.abs(dQ).max())), 2),
                      "max_step_deg": round(float(np.degrees(step_new)), 3), "rec_step_deg": round(float(np.degrees(step_rec)), 3),
                      "min_limit_margin_deg": round(float(margin.min()), 1)}}
    (out / "data.json").write_text(json.dumps(data))
    shutil.copy(ROOT / "ui/demo/vendor/plotly-2.35.2.min.js", out / "plotly.min.js")
    shutil.copy(ROOT / "ui/demo/retarget_template.html", out / "index.html")
    print(f"wrote {out} -> http://127.0.0.1:8777/retarget_{args.name}/")


if __name__ == "__main__":
    main()
