"""MuJoCo replica of the Isaac Sim FR3 Duo handover scene.

Built from:
- MuJoCo Menagerie `franka_fr3` (x2) and `robotiq_2f85` (x2), attached the way the Isaac robot is assembled
  (gripper base 0.118 m along link7, rotated -90 deg about z: identical to Menagerie's attachment_site + 2f85 mount)
- fr3duo_layout.json, extracted from Hackathon_FR3Duo_flattened.usd by extract_layout_from_usd.py:
  arm base poses, wrist/head camera poses, box / tray / cart positions
- the recording metadata: box 31 x 72 x 107 mm, 0.10 kg, friction 0.9; square 224 x 224 cameras

Frames: USD is Y-up; MuJoCo is Z-up. p_mj = C @ (p_usd - origin), with C mapping (x, y, z) -> (x, -z, y) and the
robot's root as origin. Rotations: R_mj = C @ R_usd.

Usage:
    from fr3duo_scene import build_model
    model, info = build_model()
    python fr3duo_scene.py            # builds, checks kinematics against the USD, writes fr3duo_scene.xml + preview.png
"""
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MENAGERIE = HERE.parent / "third_party" / "mujoco_menagerie"
LAYOUT = json.loads((HERE / "fr3duo_layout.json").read_text())
C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
ORIGIN = np.array(LAYOUT["objects"]["robot"]["T"])[:3, 3]
TIMESTEP = 1 / 600  # 600 Hz physics: 100 substeps per 6 Hz policy step, 20 per 30 Hz demo frame
CAM_FOVY = {"head": 90.0, "left_wrist": 87.0, "right_wrist": 87.0}  # square 4.8 mm aperture, focal 2.4 / 2.529 mm
BOX_SIZE = np.array([0.031, 0.072, 0.107])  # USD x, y(up), z extents
GRIP_CLOSED_RAD = 0.8


def to_mj(T_usd):
    T = np.array(T_usd, float)
    out = np.eye(4)
    out[:3, :3] = C @ T[:3, :3]
    out[:3, 3] = C @ (T[:3, 3] - ORIGIN)
    return out


def quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).flatten())
    return q


def aabb_mj(name):
    b = LAYOUT["objects"][name]["bbox"]
    lo, hi = C @ (np.array(b["min"]) - ORIGIN), C @ (np.array(b["max"]) - ORIGIN)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def build_model(render_size=224):
    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = max(640, render_size)
    spec.visual.global_.offheight = max(480, render_size)
    spec.visual.quality.shadowsize = 2048
    spec.visual.map.znear = 0.0008  # near clip is relative to the scene size; keep fingers 5-10 cm from the wrist cameras visible
    w = spec.worldbody

    # ---------------------------------------------------------------- room (plain stand-ins for the Isaac props)
    floor_z = C @ (np.array([0, 0.0, 0]) - ORIGIN)
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.1], pos=[0, 0, floor_z[2]],
               rgba=[0.62, 0.62, 0.64, 1])
    for name, pos, size in [("wall_back", [0, 1.6, 0.3], [4, 0.02, 1.6]), ("wall_side", [-1.9, 0, 0.3], [0.02, 4, 1.6])]:
        w.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=[0.42, 0.55, 0.68, 1],
                   contype=0, conaffinity=0)
    w.add_light(name="key", pos=[0.3, -0.6, 2.2], dir=[0, 0.3, -1], diffuse=[0.75, 0.75, 0.75], castshadow=True)
    w.add_light(name="fill", pos=[-1.0, 1.0, 2.0], dir=[0.4, -0.4, -1], diffuse=[0.35, 0.35, 0.35], castshadow=False)

    # carts: a static slab whose top is the support surface (box rests on the left cart, tray on the right one)
    box_lo, box_hi = aabb_mj("box")
    tray_lo, tray_hi = aabb_mj("tray")
    for cart, top in [("cart_left", box_lo[2]), ("cart_right", tray_lo[2])]:
        lo, hi = aabb_mj(cart)
        c = (lo + hi) / 2
        half = (hi - lo) / 2
        w.add_geom(name=f"{cart}_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[c[0], c[1], top - 0.015],
                   size=[half[0], half[1], 0.015], rgba=[0.72, 0.74, 0.76, 1], friction=[0.8, 0.005, 0.0001])
        for sx in (-1, 1):
            for sy in (-1, 1):
                leg_h = (top - 0.03 - floor_z[2]) / 2
                w.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, leg_h, 0],
                           pos=[c[0] + sx * (half[0] - 0.03), c[1] + sy * (half[1] - 0.03), floor_z[2] + leg_h],
                           rgba=[0.6, 0.6, 0.62, 1], contype=0, conaffinity=0)

    # serving tray: floor + 4 rim walls (interior ~229 x 307 mm, as documented)
    tc = (tray_lo + tray_hi) / 2
    th = (tray_hi - tray_lo) / 2
    tray = w.add_body(name="tray", pos=[tc[0], tc[1], tray_lo[2]])
    floor_t, wall_t = 0.008, 0.01
    tray.add_geom(name="tray_floor", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, floor_t / 2],
                  size=[th[0], th[1], floor_t / 2], rgba=[0.86, 0.86, 0.84, 1])
    rim_h = (tray_hi[2] - tray_lo[2])
    for name, pos, size in [("tray_wx0", [-th[0] + wall_t / 2, 0, rim_h / 2], [wall_t / 2, th[1], rim_h / 2]),
                            ("tray_wx1", [th[0] - wall_t / 2, 0, rim_h / 2], [wall_t / 2, th[1], rim_h / 2]),
                            ("tray_wy0", [0, -th[1] + wall_t / 2, rim_h / 2], [th[0], wall_t / 2, rim_h / 2]),
                            ("tray_wy1", [0, th[1] - wall_t / 2, rim_h / 2], [th[0], wall_t / 2, rim_h / 2])]:
        tray.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=[0.8, 0.8, 0.78, 1])

    # the box (free body), placed exactly where the USD has it
    bc = (box_lo + box_hi) / 2
    box = w.add_body(name="box", pos=bc)
    box.add_freejoint(name="box_free")
    half = (box_hi - box_lo) / 2
    box.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=0.10, rgba=[0.93, 0.92, 0.88, 1],
                 friction=[0.9, 0.01, 0.001], condim=4, solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    box.add_site(name="box_center", size=[0.004, 0, 0])

    # ---------------------------------------------------------------- robot: two FR3 + Robotiq 2F-85
    for side in ["left", "right"]:
        T0 = to_mj(LAYOUT["arms"][side]["links"]["link0"])
        arm = mujoco.MjSpec.from_file(str(MENAGERIE / "franka_fr3/fr3.xml"))
        arm.meshdir = str(MENAGERIE / "franka_fr3/assets")
        for k in list(arm.keys):
            arm.delete(k)
        frame = w.add_frame(pos=T0[:3, 3], quat=quat(T0[:3, :3]))
        spec.attach(arm, prefix=f"{side}_", frame=frame)
        grip = mujoco.MjSpec.from_file(str(MENAGERIE / "robotiq_2f85/2f85.xml"))
        grip.meshdir = str(MENAGERIE / "robotiq_2f85/assets")
        for k in list(grip.keys):
            grip.delete(k)
        # The Isaac (ROS URDF) Robotiq base frame opens its fingers along x, Menagerie's along y: turn the mount
        # site by +90 deg about z so the fingers close along the same axis as in the recordings.
        site = spec.site(f"{side}_attachment_site")
        site.quat = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        spec.attach(grip, prefix=f"{side}_g_", site=site)
        # wrist camera, fixed to link7 at the pose it has in the USD
        T7 = np.array(LAYOUT["arms"][side]["links"]["link7"])
        Tc = np.array(LAYOUT["cameras"][f"{side}_wrist"]["T"])
        rel = np.linalg.inv(T7) @ Tc
        spec.body(f"{side}_fr3_link7").add_camera(name=f"{side}_wrist", pos=rel[:3, 3], fovy=CAM_FOVY[f"{side}_wrist"],
                                                  quat=quat(rel[:3, :3]))
    # gravity compensation on the robot (Isaac drives are very stiff, so the recorded arms do not sag)
    for b in spec.bodies:
        if b.name.startswith(("left_", "right_")):
            b.gravcomp = 1.0
    Th = to_mj(LAYOUT["cameras"]["head"]["T"])
    w.add_camera(name="head", pos=Th[:3, 3], quat=quat(Th[:3, :3]), fovy=CAM_FOVY["head"])
    # an outside view for videos (not used by the policy)
    w.add_camera(name="overview", pos=[0.2, -1.75, 0.95], xyaxes=[1, 0.05, 0, 0, 0.45, 0.9], fovy=55)

    model = spec.compile()
    info = {
        "arm_qpos": {s: [model.joint(f"{s}_fr3_joint{i}").qposadr[0] for i in range(1, 8)] for s in ["left", "right"]},
        "arm_act": {s: [model.actuator(f"{s}_fr3_joint{i}").id for i in range(1, 8)] for s in ["left", "right"]},
        "grip_act": {s: model.actuator(f"{s}_g_fingers_actuator").id for s in ["left", "right"]},
        "grip_qpos": {s: model.joint(f"{s}_g_right_driver_joint").qposadr[0] for s in ["left", "right"]},
        "box_qpos": model.joint("box_free").qposadr[0],
        "tray_lo": tray_lo, "tray_hi": tray_hi, "box_start": bc, "spec": spec,
    }
    return model, info


def check_kinematics(model, info):
    """Put both arms in the pose the USD links were authored at and compare link positions (mm)."""
    data = mujoco.MjData(model)
    errs = {}
    for side in ["left", "right"]:
        q = np.deg2rad([LAYOUT["arms"][side]["joint_pos_deg"][f"joint{i}"] for i in range(1, 8)])
        data.qpos[info["arm_qpos"][side]] = q
    mujoco.mj_kinematics(model, data)
    for side in ["left", "right"]:
        e = []
        for i in range(8):
            usd = to_mj(LAYOUT["arms"][side]["links"][f"link{i}"])
            b = model.body(f"{side}_fr3_link{i}").id
            e.append((np.linalg.norm(data.xpos[b] - usd[:3, 3]) * 1000,
                      np.degrees(np.arccos(np.clip((np.trace(data.xmat[b].reshape(3, 3).T @ usd[:3, :3]) - 1) / 2, -1, 1)))))
        errs[side] = e
    return errs


if __name__ == "__main__":
    model, info = build_model()
    print(f"model: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators, {model.ncam} cameras")
    for side, e in check_kinematics(model, info).items():
        print(f"{side} arm vs USD at the authored pose: max position error {max(p for p, _ in e):.2f} mm, "
              f"max rotation error {max(r for _, r in e):.2f} deg")
    (HERE / "fr3duo_scene.xml").write_text(info["spec"].to_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, 480, 640)
    imgs = []
    for cam in ["overview", "head", "left_wrist", "right_wrist"]:
        r.update_scene(data, camera=cam)
        imgs.append(r.render())
    from PIL import Image
    top = np.concatenate([imgs[0], imgs[1]], 1)
    bot = np.concatenate([imgs[2], imgs[3]], 1)
    Image.fromarray(np.concatenate([top, bot], 0)).save(HERE / "preview.png")
    print("wrote fr3duo_scene.xml and preview.png")
