"""Extract both FR3 arms' kinematic chain from the scene USD into geometry/fr3duo_scene_chain.json (needs pxr).

Run with Isaac Sim's Python and its USD libraries on the path (see the README note in geometry/). Per arm:
- link0 world pose (the arm base, fixed on the FR3 Duo mount)
- joints 1-7: frames in parent / child body (localPos/Rot 0/1) and rotation axis
- tcp_in_link7: the point between the two fingertip links, in link7's frame (gripper centre)
- q_authored: the joint angles of the pose stored in the file, and that pose's link7 world matrix (self-check)
"""
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
USD = ROOT / "Environment/Environment_Hackathon_2026/Environment_Hackathon_2026/Hackathon_FR3Duo_flattened.usd"
R = "/World/FR3Duo_withV_Mkit__1_"
stage = Usd.Stage.Open(str(USD))
xc = UsdGeom.XformCache()


def world(path):  # USD matrices are row-vector convention: transpose to column-vector 4x4
    return np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T


def quat(q):
    return [q.GetReal(), *q.GetImaginary()]


out = {"source": str(USD.name), "units": "metres, radians; world is Y-up", "arms": {}}
for side in ["left", "right"]:
    joints = []
    for i in range(1, 8):
        j = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{R}/{side}_fr3v2_link{i - 1}/{side}_fr3v2_joint{i}")) \
            if stage.GetPrimAtPath(f"{R}/{side}_fr3v2_link{i - 1}/{side}_fr3v2_joint{i}") else None
        if j is None:
            j = UsdPhysics.RevoluteJoint(next(p for p in Usd.PrimRange(stage.GetPrimAtPath(R)) if p.GetName() == f"{side}_fr3v2_joint{i}"))
        joints.append({"name": f"{side}_fr3v2_joint{i}", "axis": str(j.GetAxisAttr().Get()),
                       "pos0": list(j.GetLocalPos0Attr().Get()), "rot0": quat(j.GetLocalRot0Attr().Get()),
                       "pos1": list(j.GetLocalPos1Attr().Get()), "rot1": quat(j.GetLocalRot1Attr().Get()),
                       "body0": str(j.GetBody0Rel().GetTargets()[0]), "body1": str(j.GetBody1Rel().GetTargets()[0]),
                       "limits_deg": [j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()]})
    T0 = world(f"{R}/{side}_fr3v2_link0")
    T7 = world(f"{R}/{side}_fr3v2_link7")
    tips = [world(f"{R}/{side}_robotiq_85_{f}_finger_tip_link")[:3, 3] for f in ("left", "right")]
    tcp_w = (tips[0] + tips[1]) / 2
    tcp7 = np.linalg.inv(T7) @ np.r_[tcp_w, 1]
    # authored joint angles: rotation about the joint axis between the parent- and child-side joint frames
    q = []
    for jt in joints:
        Tp, Tc = world(jt["body0"]), world(jt["body1"])

        def frame(pos, rot):
            m = np.eye(4)
            m[:3, :3] = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(rot[0], *rot[1:])))).T
            m[:3, 3] = pos
            return m
        Fp, Fc = Tp @ frame(jt["pos0"], jt["rot0"]), Tc @ frame(jt["pos1"], jt["rot1"])
        Rrel = Fp[:3, :3].T @ Fc[:3, :3]
        a = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[jt["axis"]]
        q.append(float(np.arctan2(Rrel[a[1], a[0]], Rrel[a[0], a[0]])))
    out["arms"][side] = {"link0_world": T0.tolist(), "joints": joints, "tcp_in_link7": tcp7[:3].tolist(),
                         "q_authored": q, "link7_world_authored": T7.tolist(), "tcp_world_authored": tcp_w.tolist()}
    print(f"{side}: link0 at {np.round(T0[:3, 3], 4).tolist()}, tcp offset in link7 {np.round(tcp7[:3], 4).tolist()} "
          f"(|{np.linalg.norm(tcp7[:3]):.4f}| m), authored q {np.round(q, 3).tolist()}")
dst = ROOT / "geometry" / "fr3duo_scene_chain.json"
dst.write_text(json.dumps(out, indent=1))
print("wrote", dst)
