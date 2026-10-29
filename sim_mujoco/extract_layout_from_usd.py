"""Extract the FR3 Duo scene layout from the Isaac USD (run with any Python that has `usd-core`).

Writes sim_mujoco/fr3duo_layout.json: 4x4 world transforms (USD frame, Y-up, metres, column-vector convention)
of both arms' links, gripper bases and cameras, the joint positions the links were authored at, and the
bounding boxes of the box, tray and carts.
"""
import json
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

usd = sys.argv[1]
out = Path(__file__).resolve().parent / "fr3duo_layout.json"
st = Usd.Stage.Open(usd)
xc = UsdGeom.XformCache()
bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
R = "/World/FR3Duo_withV_Mkit__1_"


def T(path):
    m = np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(path)), dtype=float).T  # USD is row-vector
    # remove scale from the rotation part
    s = np.linalg.norm(m[:3, :3], axis=0)
    m[:3, :3] /= s
    return m.round(6).tolist()


def bbox(path):
    r = bc.ComputeWorldBound(st.GetPrimAtPath(path)).ComputeAlignedRange()
    return {"min": list(r.GetMin()), "max": list(r.GetMax())}


layout = {"source": usd, "up": "Y", "arms": {}, "cameras": {}, "objects": {}}
for side in ["left", "right"]:
    a = {"links": {}, "joint_pos_deg": {}}
    for i in range(8):
        a["links"][f"link{i}"] = T(f"{R}/{side}_fr3v2_link{i}")
    for i in range(1, 8):
        j = st.GetPrimAtPath(f"{R}/{side}_fr3v2_link{i - 1}/{side}_fr3v2_joint{i}")
        a["joint_pos_deg"][f"joint{i}"] = j.GetAttribute("physics:state:angular:physics:position").Get()
    a["gripper_base"] = T(f"{R}/{side}_robotiq_85_base_link")
    layout["arms"][side] = a
cams = {"head": f"{R}/mount_link/cover_link/head_link/zed_mini_camera_link/zed_mini_left_camera_frame/head_cam",
        "left_wrist": f"{R}/left_d405_color_frame/left_wrist_cam", "right_wrist": f"{R}/right_d405_color_frame/right_wrist_cam"}
for k, p in cams.items():
    c = UsdGeom.Camera(st.GetPrimAtPath(p))
    layout["cameras"][k] = {"T": T(p), "focal": c.GetFocalLengthAttr().Get(), "vaperture": c.GetVerticalApertureAttr().Get()}
for k, p in {"box": "/World/WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01", "tray": "/World/ServingTray",
             "cart_left": "/World/SM_SupplyCart_03a", "cart_right": "/World/SM_SupplyCart_01e",
             "robot": R, "mount": f"{R}/mount_link"}.items():
    layout["objects"][k] = {"bbox": bbox(p), "T": T(p)}
out.write_text(json.dumps(layout, indent=1))
print("wrote", out)
for side in ["left", "right"]:
    print(side, "joint pos deg:", {k: round(v, 1) for k, v in layout["arms"][side]["joint_pos_deg"].items()})
    print("   link0 pos", np.array(layout["arms"][side]["links"]["link0"])[:3, 3].round(3), " link7 pos", np.array(layout["arms"][side]["links"]["link7"])[:3, 3].round(3))
