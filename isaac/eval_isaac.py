"""Closed-loop evaluation of the FR3 Duo handover in Isaac Sim.

Runs INSIDE Isaac Sim's Python (needs only numpy + Pillow, both shipped with Isaac Sim).
The policy runs in a separate process (serving/serve_policy.py), reached over TCP.

Modes
-----
replay : play the recorded demo actions open-loop (30 fps, each held for 2 physics ticks, exactly as
         recorded). Checks that this sim setup reproduces the demonstration. If the box does not
         end up on the tray here, fix the setup before trusting any policy result.
policy : closed loop at 6 Hz. Every 10 physics ticks (60 Hz) the 3 cameras + joint state are sent
         to the policy server; the returned joint targets are interpolated over the next 10 ticks.

Usage (from the Isaac Sim install folder or an `isaacsim` pip env):
    python isaac/eval_isaac.py --mode replay --usd <path>/Hackathon_FR3Duo_flattened.usd
    (use the flattened file: it is the only one with box geometry + a rigged robot + cameras in one file)
    python isaac/eval_isaac.py --mode policy --usd <...> --server 127.0.0.1:5555 --episodes 10
    (Windows Isaac Sim install: use python.bat; add --headless false to watch)

Outputs (in --out): results.json (per-episode success, stage reached, box error, latency),
a GIF of the head camera per episode, and for replay a joint-tracking report.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["replay", "policy"], required=True)
ap.add_argument("--usd", required=True, help="Hackathon_FR3Duo_flattened.usd")
ap.add_argument("--ref", default=str(Path(__file__).resolve().parents[1] / "artifacts/eval_assets/handover_reference.json"))
ap.add_argument("--server", default="127.0.0.1:5555")
ap.add_argument("--episodes", type=int, default=10, help="policy mode: one per held-out lighting config")
ap.add_argument("--max_seconds", type=float, default=90.0)
ap.add_argument("--policy_hz", type=float, default=6.0)
ap.add_argument("--gripper_drive", choices=["recorded", "usd"], default="recorded",
                help="'recorded' = soft, force-limited drive from the dataset metadata; 'usd' = as in the file")
ap.add_argument("--no_tray_collider", dest="tray_collider", action="store_false",
                help="skip adding the tray's floor+walls collider (the hand-out tray mesh has none)")
ap.add_argument("--headless", default="true")
ap.add_argument("--out", default="eval_results")
args = ap.parse_args()

# --- Isaac Sim must start before any omni/isaacsim import -------------------------------------
try:
    from isaacsim import SimulationApp
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.kit import SimulationApp
app = SimulationApp({"headless": args.headless.lower() == "true", "width": 1280, "height": 720})

import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

try:  # Isaac Sim 5.x / 6.x
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation as SingleArticulation
    from omni.isaac.core.prims import RigidPrim as SingleRigidPrim
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.sensor import Camera

ROBOT = "/World/FR3Duo_withV_Mkit__1_"
BOX = "/World/WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01"
TRAY = "/World/ServingTray"
CAMS = {  # dataset key -> camera prim (from meta/fr3_conversion.json)
    "observation.images.external_camera": f"{ROBOT}/mount_link/cover_link/head_link/zed_mini_camera_link/zed_mini_left_camera_frame/head_cam",
    "observation.images.left_wrist_camera": f"{ROBOT}/left_d405_color_frame/left_wrist_cam",
    "observation.images.right_wrist_camera": f"{ROBOT}/right_d405_color_frame/right_wrist_cam",
}
PHYSICS_HZ = 60
REF = json.loads(Path(args.ref).read_text())
GC = REF["gripper_contract"]
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


# --- gripper conventions (dataset: 1 = open, 0 = closed) -----------------------------------------
def gripper_action_to_joint(a):
    """Dataset action = 1 - cmd/0.8  ->  cmd = 0.8 * (1 - a) rad on the *_gripper_ctrl joint."""
    return GC["closed_joint_rad"] * (1.0 - float(np.clip(a, 0, 1)))


def gripper_joint_to_state(q):
    """Measured ctrl-joint angle -> normalized state. Piecewise-linear through the contract's points
    (0 rad -> 1.0, hold 0.459 rad -> 0.51, closed 0.8 rad -> 0.0). Replay mode prints how well this
    matches the recorded state so it can be re-fitted if needed."""
    return float(np.interp(q, [0.0, GC["hold_joint_rad"], GC["closed_joint_rad"]], [1.0, GC["hold_norm"], 0.0]))


# --- scene setup ---------------------------------------------------------------------------------
omni.usd.get_context().open_stage(args.usd)
for _ in range(20):
    app.update()
stage = omni.usd.get_context().get_stage()


def add_tray_collider(interior=(0.229, 0.307), floor_thickness=0.02, wall_height=0.03, wall_thickness=0.01):
    """The hand-out's ServingTray mesh has no collision. The recordings used a static five-part collider
    (floor + 4 walls, interior ~229 x 307 mm). Recreate it, invisible to the cameras (purpose = guide).
    Floor top = top of the tray's bounding box (the demo's box ended standing with its bottom at ~that height)."""
    r = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath(TRAY)).ComputeAlignedRange()
    mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
    c = (mn + mx) / 2
    top = mx[1]
    ix, iz = interior
    parts = {  # name: (center, size)
        "floor": ((c[0], top - floor_thickness / 2, c[2]), (ix + 2 * wall_thickness, floor_thickness, iz + 2 * wall_thickness)),
        "wall_x0": ((c[0] - ix / 2 - wall_thickness / 2, top + wall_height / 2, c[2]), (wall_thickness, wall_height, iz)),
        "wall_x1": ((c[0] + ix / 2 + wall_thickness / 2, top + wall_height / 2, c[2]), (wall_thickness, wall_height, iz)),
        "wall_z0": ((c[0], top + wall_height / 2, c[2] - iz / 2 - wall_thickness / 2), (ix, wall_height, wall_thickness)),
        "wall_z1": ((c[0], top + wall_height / 2, c[2] + iz / 2 + wall_thickness / 2), (ix, wall_height, wall_thickness)),
    }
    for name, (ctr, size) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"/World/TrayCollider/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, ctr)))
        cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    print(f"tray collider: floor top y={top:.3f}, interior {ix}x{iz} m around ({c[0]:.3f}, {c[2]:.3f})", flush=True)


def prepare_stage():
    # Box: the flattened file has mass 0; the recordings used 0.10 kg and friction 0.9 / 0.8.
    box = stage.GetPrimAtPath(BOX)
    UsdPhysics.MassAPI.Apply(box).CreateMassAttr(0.10)
    mat_path = Sdf.Path("/World/PhysicsMaterials/box_material")
    mat = UsdShade.Material.Define(stage, mat_path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(0.9)
    pm.CreateDynamicFrictionAttr(0.8)
    pm.CreateRestitutionAttr(0.0)
    for p in [box, *box.GetAllChildren()]:
        if p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    # Gripper drive: the recordings used a soft, force-limited drive (values from episode metadata).
    if args.gripper_drive == "recorded":
        d = REF["gripper_drive_recorded"]
        for side in ["left", "right"]:
            j = stage.GetPrimAtPath(f"{ROBOT}/{side}_robotiq_85_base_link/{side}_gripper_ctrl")
            drv = UsdPhysics.DriveAPI.Get(j, "angular")
            # metadata is per radian; USD angular drives are per degree
            drv.GetStiffnessAttr().Set(d["stiffness"] * np.pi / 180)
            drv.GetDampingAttr().Set(d["damping"] * np.pi / 180)
            drv.GetMaxForceAttr().Set(d["maxForce"])
    if args.tray_collider:
        add_tray_collider()
    # Cameras: square 224x224 apertures as recorded.
    for key, path in CAMS.items():
        cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
        p = REF["camera_params"][{"observation.images.external_camera": "head",
                                  "observation.images.left_wrist_camera": "left_wrist",
                                  "observation.images.right_wrist_camera": "right_wrist"}[key]]
        cam.GetFocalLengthAttr().Set(p["focal_length"])
        cam.GetHorizontalApertureAttr().Set(p["horizontal_aperture"])
        cam.GetVerticalApertureAttr().Set(p["vertical_aperture"])


def set_lighting(cfg):
    rect, dome = UsdLux.RectLight(stage.GetPrimAtPath("/World/RectLight")), UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
    for light, c in [(rect, cfg["rect"]), (dome, cfg["dome"])]:
        light.GetIntensityAttr().Set(float(c["intensity_to"]))
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(float(c["color_temperature_K"]))


prepare_stage()
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / PHYSICS_HZ)
arms = {s: world.scene.add(SingleArticulation(prim_path=f"{ROBOT}/{s}_arm_root_joint", name=f"{s}_arm")) for s in ["left", "right"]}
box = world.scene.add(SingleRigidPrim(prim_path=BOX, name="box"))
cams = {}
for key, path in CAMS.items():
    cams[key] = Camera(prim_path=path, resolution=(224, 224), frequency=PHYSICS_HZ)
world.reset()
for c in cams.values():
    c.initialize()

idx = {s: [arms[s].get_dof_index(f"{s}_fr3v2_joint{i}") for i in range(1, 8)] + [arms[s].get_dof_index(f"{s}_gripper_ctrl")]
       for s in ["left", "right"]}
print("DOFs:", {s: arms[s].dof_names for s in arms}, flush=True)

tray_box = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath(TRAY)).ComputeAlignedRange()
TRAY_MIN, TRAY_MAX = np.array(tray_box.GetMin()), np.array(tray_box.GetMax())


# --- robot I/O ----------------------------------------------------------------------------------
def read_state():
    out = []
    for s in ["left", "right"]:
        q = arms[s].get_joint_positions()[idx[s]]
        out += list(q[:7]) + [gripper_joint_to_state(q[7])]
    return np.array(out, np.float32)


def raw_gripper_joints():
    return [float(arms[s].get_joint_positions()[idx[s][7]]) for s in ["left", "right"]]


def apply_target(a16):
    for k, s in enumerate(["left", "right"]):
        a = a16[8 * k: 8 * k + 8]
        q = np.r_[a[:7], gripper_action_to_joint(a[7])]
        arms[s].apply_action(ArticulationAction(joint_positions=q, joint_indices=np.array(idx[s])))


def reset_robot(state16):
    for k, s in enumerate(["left", "right"]):
        q = arms[s].get_joint_positions().copy()
        st = state16[8 * k: 8 * k + 8]
        q[idx[s][:7]] = st[:7]
        q[idx[s][7]] = 0.0  # open
        arms[s].set_joint_positions(q)
        arms[s].set_joint_velocities(np.zeros_like(q))
    apply_target(np.array(state16, np.float32))


def images():
    out = {}
    for key, c in cams.items():
        rgba = c.get_rgba()
        if rgba is None or rgba.size == 0:
            rgba = np.zeros((224, 224, 4), np.uint8)
        rgb = rgba[..., :3]
        out[key] = (rgb * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    return out


def box_pose():
    p, q = box.get_world_pose()
    return np.asarray(p, float), np.asarray(q, float)  # q = (w, x, y, z)


def box_tilt_deg(q):
    """Angle between the box's long axis in the demo's final (upright) pose and now; Y is up."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    up = np.abs(R.T @ np.array([0, 1, 0]))
    return float(np.degrees(np.arccos(np.clip(up.max(), -1, 1))))


# --- scoring --------------------------------------------------------------------------------------
class Tracker:
    """Stage progress: 1 lifted -> 2 carried at least halfway to the tray -> 3 over the tray -> 4 resting on the tray."""

    def __init__(self):
        self.p0, _ = box_pose()
        self.tray_c = (TRAY_MIN + TRAY_MAX) / 2
        self.max_lift, self.min_tray_dist, self.stage = 0.0, 1e9, 0
        self.hist = []

    def on_tray_xz(self, p, margin=0.01):
        return bool(np.all(p[[0, 2]] > TRAY_MIN[[0, 2]] + margin) and np.all(p[[0, 2]] < TRAY_MAX[[0, 2]] - margin))

    def update(self, t):
        p, q = box_pose()
        self.hist.append((t, *p))
        self.max_lift = max(self.max_lift, p[1] - self.p0[1])
        d = np.linalg.norm((p - self.tray_c)[[0, 2]])
        self.min_tray_dist = min(self.min_tray_dist, d)
        start_d = np.linalg.norm((self.p0 - self.tray_c)[[0, 2]])
        if self.max_lift > 0.03:
            self.stage = max(self.stage, 1)
        if self.stage >= 1 and d < 0.5 * start_d:
            self.stage = max(self.stage, 2)
        if self.stage >= 2 and self.on_tray_xz(p):
            self.stage = max(self.stage, 3)
        return p, q

    def result(self):
        p, q = box_pose()
        h = np.array(self.hist[-PHYSICS_HZ // 2:]) if len(self.hist) > 3 else None
        speed = float(np.linalg.norm(np.diff(h[:, 1:], axis=0), axis=1).sum() / (h[-1, 0] - h[0, 0])) if h is not None and h[-1, 0] > h[0, 0] else None
        resting = self.on_tray_xz(p) and p[1] > TRAY_MIN[1] and p[1] < TRAY_MAX[1] + 0.15 and (speed is None or speed < 0.01)
        if self.stage >= 3 and resting:
            self.stage = 4
        demo = np.array(REF["demo_final_box_pose"][:3])
        return {"success": bool(self.stage == 4), "stage": self.stage,
                "stage_name": ["not lifted", "lifted", "carried halfway", "over tray", "resting on tray"][self.stage],
                "final_box_pos": p.round(4).tolist(), "box_tilt_deg": round(box_tilt_deg(q), 2),
                "dist_to_demo_final_mm": round(float(np.linalg.norm(p - demo)) * 1000, 1),
                "max_lift_mm": round(self.max_lift * 1000, 1), "final_speed_m_s": speed}


def save_gif(frames, path, fps=6):
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    except Exception as e:  # never fail an eval because of the video
        print("gif failed:", e)


def strip(obs):
    return np.concatenate([obs[k] for k in CAMS], axis=1)


# --- episodes -------------------------------------------------------------------------------------
def settle(n=30):
    for _ in range(n):
        world.step(render=False)


def run_replay():
    world.reset()
    set_lighting(REF["val_lighting"][0])
    reset_robot(REF["initial_state"])
    settle()
    tr = Tracker()
    A, S = np.array(REF["demo_actions"]), np.array(REF["demo_states"])
    err, grip_pairs, frames = [], [], []
    # recording rule: frame k = state at raw row 2k, action[k] = target for the next 2 physics ticks
    for k in range(len(A)):
        if k % 5 == 0:
            world.render()
            frames.append(strip(images()))
        s = read_state()
        if k < REF["hold_cut_frame"]:  # after the cut the recorded timeline skips 2.5 s
            err.append(np.abs(s - S[k]))
            grip_pairs.append((raw_gripper_joints(), S[k][[7, 15]].tolist()))
        apply_target(A[k])
        world.step(render=False)
        world.step(render=False)
        tr.update(k / 30)
    settle(60)
    tr.update(len(A) / 30 + 1)
    err = np.array(err)
    res = tr.result()
    res["joint_tracking_rad"] = {"mean": round(float(err[:, [*range(7), *range(8, 15)]].mean()), 4),
                                 "max": round(float(err[:, [*range(7), *range(8, 15)]].max()), 4)}
    res["gripper_state_abs_err"] = round(float(err[:, [7, 15]].mean()), 3)
    # fit state = a + b * q per gripper, to check / replace the default joint->state mapping
    g = np.array([(q[i], st[i]) for q, st in grip_pairs for i in range(2)])
    b, a = np.polyfit(g[:, 0], g[:, 1], 1)
    res["gripper_fit_state_eq"] = f"state ~= {a:.3f} + {b:.3f} * q_rad"
    save_gif(frames, OUT / "replay_head_left_right.gif")
    (OUT / "replay.json").write_text(json.dumps(res, indent=1))
    print("REPLAY", json.dumps(res, indent=1))
    print("-> if success is false or joint tracking is poor, the sim setup differs from the recording "
          "(box pose/mass, gripper drive, physics); fix that before evaluating policies.")


def run_policy():
    from multiprocessing.connection import Client
    host, port = args.server.split(":")
    conn = Client((host, int(port)), authkey=b"fr3duo")
    conn.send({"cmd": "info"})
    info = conn.recv()
    print("policy:", info, flush=True)
    ticks_per_step = int(round(PHYSICS_HZ / args.policy_hz))
    results = []
    for e in range(args.episodes):
        light = REF["val_lighting"][e % len(REF["val_lighting"])]
        world.reset()
        set_lighting(light)
        reset_robot(REF["initial_state"])
        settle()
        conn.send({"cmd": "reset"})
        conn.recv()
        tr = Tracker()
        prev = np.array(REF["initial_state"], np.float32)
        lat, frames, actions = [], [], []
        t0 = time.time()
        for step in range(int(args.max_seconds * args.policy_hz)):
            world.render()
            obs = images()
            frames.append(strip(obs))
            obs["observation.state"] = read_state()
            obs["task"] = REF["task"]
            conn.send({"cmd": "act", "obs": obs})
            out = conn.recv()
            a = np.asarray(out["action"], np.float32)
            lat.append(out["latency_ms"])
            actions.append(a.tolist())
            for i in range(1, ticks_per_step + 1):  # interpolate joint targets between policy steps
                apply_target(prev + (a - prev) * i / ticks_per_step)
                world.step(render=False)
            prev = a
            tr.update(step / args.policy_hz)
        settle(60)
        tr.update(args.max_seconds + 1)
        r = {"episode": e, "lighting_from_val_episode": light["episode_index"], **tr.result(),
             "latency_ms": {"median": round(float(np.median(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1)},
             "wall_s": round(time.time() - t0, 1)}
        results.append(r)
        save_gif(frames, OUT / f"policy_ep{e:02d}.gif")
        print("EPISODE", json.dumps(r), flush=True)
        (OUT / "results.json").write_text(json.dumps({"policy": info, "episodes": results}, indent=1))
    conn.send({"cmd": "close"})
    sr = np.mean([r["success"] for r in results])
    stages = np.bincount([r["stage"] for r in results], minlength=5).tolist()
    summary = {"success_rate": float(sr), "episodes": len(results), "stage_counts_0to4": stages,
               "median_latency_ms": float(np.median([r["latency_ms"]["median"] for r in results]))}
    (OUT / "results.json").write_text(json.dumps({"policy": info, "summary": summary, "episodes": results}, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)


try:
    run_replay() if args.mode == "replay" else run_policy()
finally:
    app.close()
