"""Scored closed-loop evaluation of a FR3 Duo policy, run from Isaac Sim's Script Editor (Isaac Sim 6.2).

1. Start the policy server (Windows .venv):
       .venv\\Scripts\\python.exe serving\\fr3duo_policy_server.py --checkpoint outputs\\<run>\\final
2. Isaac Sim: Window > Script Editor > File > Open this file > Ctrl+Enter.
   A window "FR3 Duo eval" shows progress; Stop ends the run after the current step.

Builds on fr3duo_isaac_client.py (scene loading, cameras, robot I/O, HTTP client) and adds what a scored
evaluation needs, taken from isaac/eval_isaac.py:
- the task physics the hand-out scene lacks: box mass 0.10 kg + friction 0.9/0.8, the soft force-limited
  gripper drive of the recordings, an invisible collider for the serving tray (floor + 4 walls)
- policy timing of the training data: one policy action every 5 control steps (6 Hz), interpolated
- per episode the lighting of one held-out episode (artifacts/eval_assets/handover_reference.json)
- box tracking and the isaac/eval_isaac.py stage score: 0 not lifted, 1 lifted, 2 carried halfway, 3 over the
  tray, 4 resting on the tray (= success)

Results: <project>/eval_results/isaac_<time>/results.json (+ steps_epXX.jsonl per episode).
"""
import asyncio
import importlib.util
import json
import os
import time

import numpy as np

# <project> = the folder above isaac/; falls back to the original machine's path when the Script Editor runs a
# temporary copy of this file.
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ""
PROJECT = (os.path.dirname(_here) if os.path.isfile(os.path.join(_here, "fr3duo_isaac_client.py"))
           else r"C:\Users\User\Desktop\Hack2026\C2_Teach-the-robot_VLA")
EPISODES = 3                 # each uses the lighting of a different held-out episode
MAX_STEPS = 2400             # 80 s at 30 Hz; recorded episodes are 75 s
ACTION_STRIDE = 5            # 30 Hz control / 6 Hz policy (the checkpoint's actions are 1/6 s apart)

BOX = "/World/WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01"
TRAY = "/World/ServingTray"

# Reuse the organisers' client without opening its window.
os.environ["FR3DUO_NO_AUTORUN"] = "1"
_spec = importlib.util.spec_from_file_location("fr3duo_client", os.path.join(PROJECT, "isaac", "fr3duo_isaac_client.py"))
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)
REF = json.load(open(os.path.join(PROJECT, "artifacts", "eval_assets", "handover_reference.json")))
GC = REF["gripper_contract"]


def gripper_joint_to_state(q):
    """Measured ctrl-joint angle -> dataset gripper state, through the contract's points (same as isaac/eval_isaac.py):
    0 rad -> 1.0, holding the box 0.459 rad -> 0.51, closed 0.8 rad -> 0.0."""
    return float(np.interp(q, [0.0, GC["hold_joint_rad"], GC["closed_joint_rad"]], [1.0, GC["hold_norm"], 0.0]))


client.gripper_to_norm = gripper_joint_to_state  # Arm.read() uses this


def task_physics(stage):
    """Box mass/friction, recorded gripper drive, tray collider (session layer, nothing saved to disk)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        box = stage.GetPrimAtPath(BOX)
        UsdPhysics.MassAPI.Apply(box).CreateMassAttr(0.10)
        mat = UsdShade.Material.Define(stage, Sdf.Path("/World/PhysicsMaterials/box_material"))
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(0.9)
        pm.CreateDynamicFrictionAttr(0.8)
        pm.CreateRestitutionAttr(0.0)
        for p in [box, *box.GetAllChildren()]:
            if p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        d = REF["gripper_drive_recorded"]  # per radian; USD angular drives are per degree
        for side in ["left", "right"]:
            drv = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{client.ROBOT}/{side}_robotiq_85_base_link/{side}_gripper_ctrl"), "angular")
            drv.GetStiffnessAttr().Set(d["stiffness"] * np.pi / 180)
            drv.GetDampingAttr().Set(d["damping"] * np.pi / 180)
            drv.GetMaxForceAttr().Set(d["maxForce"])
        r = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath(TRAY)).ComputeAlignedRange()
        mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
        c, top, (ix, iz), ft, wh, wt = (mn + mx) / 2, mx[1], (0.229, 0.307), 0.02, 0.03, 0.01
        parts = {"floor": ((c[0], top - ft / 2, c[2]), (ix + 2 * wt, ft, iz + 2 * wt)),
                 "wall_x0": ((c[0] - ix / 2 - wt / 2, top + wh / 2, c[2]), (wt, wh, iz)),
                 "wall_x1": ((c[0] + ix / 2 + wt / 2, top + wh / 2, c[2]), (wt, wh, iz)),
                 "wall_z0": ((c[0], top + wh / 2, c[2] - iz / 2 - wt / 2), (ix, wh, wt)),
                 "wall_z1": ((c[0], top + wh / 2, c[2] + iz / 2 + wt / 2), (ix, wh, wt))}
        for name, (ctr, size) in parts.items():
            cube = UsdGeom.Cube.Define(stage, f"/World/TrayCollider/{name}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, ctr)))
            cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return mn, mx


def set_lighting(stage, cfg):
    from pxr import Usd, UsdLux

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path, c in [("/World/RectLight", cfg["rect"]), ("/World/DomeLight", cfg["dome"])]:
            light = UsdLux.LightAPI(stage.GetPrimAtPath(path))
            light.GetIntensityAttr().Set(float(c["intensity_to"]))
            light.CreateEnableColorTemperatureAttr().Set(True)
            light.CreateColorTemperatureAttr().Set(float(c["color_temperature_K"]))


class BoxTracker:
    """isaac/eval_isaac.py's stage score. Y is up."""

    def __init__(self, tray_min, tray_max):
        try:
            from isaacsim.core.experimental.prims import RigidPrim
            self.prim = RigidPrim(BOX)
        except Exception:
            from isaacsim.core.experimental.prims import XformPrim
            self.prim = XformPrim(BOX)
        self.tmin, self.tmax = tray_min, tray_max
        self.tray_c = (tray_min + tray_max) / 2
        self.p0 = self.pos()
        self.max_lift, self.stage, self.hist = 0.0, 0, []

    def pos(self):
        p = self.prim.get_world_poses()[0]
        return np.asarray(p.numpy() if hasattr(p, "numpy") else p, float).reshape(-1, 3)[0]

    def on_tray(self, p, margin=0.01):
        return bool(np.all(p[[0, 2]] > self.tmin[[0, 2]] + margin) and np.all(p[[0, 2]] < self.tmax[[0, 2]] - margin))

    def update(self, t):
        p = self.pos()
        self.hist.append((t, *p))
        self.max_lift = max(self.max_lift, p[1] - self.p0[1])
        d, d0 = np.linalg.norm((p - self.tray_c)[[0, 2]]), np.linalg.norm((self.p0 - self.tray_c)[[0, 2]])
        if self.max_lift > 0.03:
            self.stage = max(self.stage, 1)
        if self.stage >= 1 and d < 0.5 * d0:
            self.stage = max(self.stage, 2)
        if self.stage >= 2 and self.on_tray(p):
            self.stage = max(self.stage, 3)
        return p

    def result(self):
        p = self.pos()
        h = np.array(self.hist[-15:])
        speed = float(np.linalg.norm(np.diff(h[:, 1:], axis=0), axis=1).sum() / max(h[-1, 0] - h[0, 0], 1e-6)) if len(h) > 2 else None
        resting = self.on_tray(p) and self.tmin[1] < p[1] < self.tmax[1] + 0.15 and (speed is None or speed < 0.01)
        if self.stage >= 3 and resting:
            self.stage = 4
        demo = np.array(REF["demo_final_box_pose"][:3])
        return {"success": self.stage == 4, "stage": self.stage,
                "stage_name": ["not lifted", "lifted", "carried halfway", "over tray", "resting on tray"][self.stage],
                "max_lift_mm": round(self.max_lift * 1000, 1), "final_box_pos": p.round(4).tolist(),
                "dist_to_demo_final_mm": round(float(np.linalg.norm(p - demo)) * 1000, 1), "final_speed_m_s": speed}


class Evaluator:
    def __init__(self):
        self.stop_requested, self.running, self.status = False, False, print

    def stop(self):
        self.stop_requested = True

    async def episode(self, e, policy, info, out_dir):
        import carb
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager

        app, timeline, ctx = omni.kit.app.get_app(), omni.timeline.get_timeline_interface(), omni.usd.get_context()
        light = REF["val_lighting"][e % len(REF["val_lighting"])]
        timeline.stop()
        await app.next_update_async()
        opened, err = await ctx.open_stage_async(client.SCENE)
        if not opened:
            raise RuntimeError(f"could not open {client.SCENE}: {err}")
        for _ in range(600):
            stage = ctx.get_stage()
            if stage is not None and ctx.get_stage_state() == omni.usd.StageState.OPENED:
                break
            await app.next_update_async()
        client.prepare_stage(stage)
        tray_min, tray_max = task_physics(stage)
        set_lighting(stage, light)
        SimulationManager.set_physics_dt(1.0 / client.PHYSICS_HZ)
        RenderingManager.set_dt(1.0 / client.CONTROL_HZ)
        carb.settings.get_settings().set_string("/rtx/rendermode", client.RENDER_MODE)
        carb.settings.get_settings().set_bool("/omni/replicator/captureOnPlay", True)
        products, annotators = [], {}
        try:
            for feature, (cam, _) in client.CAMERAS.items():
                rp = rep.create.render_product(cam, (client.IMAGE_SIZE, client.IMAGE_SIZE))
                an = rep.AnnotatorRegistry.get_annotator("rgb")
                an.attach([rp])
                products.append(rp)
                annotators[feature] = an
            timeline.play()
            for _ in range(3):
                await app.next_update_async()
            arms = [client.Arm("left"), client.Arm("right")]
            for arm in arms:
                arm.teleport_ready()
            await client.sim_steps(15)  # settle
            await client.grab_images(annotators)
            tracker = BoxTracker(tray_min, tray_max)
            from isaacsim.core.experimental.prims import XformPrim
            tips = XformPrim([f"{client.ROBOT}/left_robotiq_85_{f}_finger_tip_link" for f in ("left", "right")])
            policy.reset()
            stride = 1 if str(info["policy_type"]).startswith(("replay", "pi05")) else ACTION_STRIDE  # replay = recorded 30 fps actions
            lat, t0 = [], time.perf_counter()
            with open(os.path.join(out_dir, f"steps_ep{e:02d}.jsonl"), "w") as log:
                for step in range(MAX_STEPS):
                    if self.stop_requested or not timeline.is_playing():
                        break
                    if step % stride == 0:
                        obs = {"observation.state": np.concatenate([a.read() for a in arms])}
                        obs.update(await client.grab_images(annotators))
                        action, latency_ms, _ = policy.act(obs, REF["task"])
                        lat.append(latency_ms)
                        start = obs["observation.state"] if step == 0 else target
                    target = start + (action - start) * ((step % stride + 1) / stride)
                    arms[0].command(target[:8])
                    arms[1].command(target[8:])
                    await client.sim_steps(1)
                    p = tracker.update(step / client.CONTROL_HZ)
                    if step % stride == 0:
                        tp = tips.get_world_poses()[0]
                        tp = np.asarray(tp.numpy() if hasattr(tp, "numpy") else tp, float).reshape(-1, 3)
                        log.write(json.dumps({"step": step, "state": obs["observation.state"].tolist(),
                                              "action": action.tolist(), "box": p.round(4).tolist(),
                                              "left_tips": tp.round(4).tolist()}) + "\n")
                    if step % 150 == 0:
                        self.status(f"episode {e + 1}/{EPISODES} | {step / client.CONTROL_HZ:.0f}/{MAX_STEPS / client.CONTROL_HZ:.0f} s sim | "
                                    f"stage {tracker.stage} | lift {tracker.max_lift * 1000:.0f} mm")
            await client.sim_steps(30)  # let the box come to rest
            tracker.update(MAX_STEPS / client.CONTROL_HZ + 1)
            r = {"episode": e, "lighting_from_val_episode": light["episode_index"], "sim_steps": step + 1,
                 **tracker.result(), "latency_ms_median": round(float(np.median(lat)), 1) if lat else None,
                 "wall_s": round(time.perf_counter() - t0, 1)}
            return r
        finally:
            for rp in products:
                rp.destroy()
            timeline.stop()

    async def run(self):
        if self.running:
            return
        self.running, self.stop_requested = True, False
        try:
            policy = client.PolicyClient(client.SERVER, timeout=120)
            info = policy.info()
            self.status(f"policy {info['policy_type']} | {info['checkpoint']}")
            out_dir = os.path.join(PROJECT, "eval_results", "isaac_" + time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(out_dir, exist_ok=True)
            results = []
            for e in range(EPISODES):
                if self.stop_requested:
                    break
                r = await self.episode(e, policy, info, out_dir)
                results.append(r)
                print("[fr3duo eval] EPISODE", json.dumps(r))
                summary = {"episodes": len(results), "success_rate": float(np.mean([x["success"] for x in results])),
                           "stage_counts_0to4": np.bincount([x["stage"] for x in results], minlength=5).tolist(),
                           "action_stride": ACTION_STRIDE, "max_steps": MAX_STEPS}
                with open(os.path.join(out_dir, "results.json"), "w") as f:
                    json.dump({"policy": info, "summary": summary, "episodes": results}, f, indent=1)
            self.status(f"done -> {out_dir} | " + json.dumps(summary))
            print("[fr3duo eval] SUMMARY", json.dumps(summary), out_dir)
        except Exception as exc:
            self.status(f"ERROR: {type(exc).__name__}: {exc}")
            raise
        finally:
            self.running = False


def show_window(ev):
    import omni.ui as ui

    window = ui.Window("FR3 Duo eval", width=520, height=150)
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label(f"{EPISODES} episodes x {MAX_STEPS / client.CONTROL_HZ:.0f} s, policy every {ACTION_STRIDE} steps (6 Hz)", height=18)
            with ui.HStack(height=30, spacing=6):
                ui.Button("Start", clicked_fn=lambda: asyncio.ensure_future(ev.run()))
                ui.Button("Stop", clicked_fn=ev.stop)
            status = ui.Label("idle", word_wrap=True)

    def show(msg):
        print("[fr3duo eval]", msg)
        status.text = str(msg)

    ev.status = show
    return window


import omni.kit.app  # noqa: E402

_mgr = omni.kit.app.get_app().get_extension_manager()
for _ext in ("isaacsim.core.experimental.prims", "isaacsim.core.simulation_manager",
             "isaacsim.core.rendering_manager", "omni.replicator.core"):
    _mgr.set_extension_enabled_immediate(_ext, True)
fr3duo_evaluator = Evaluator()
fr3duo_eval_window = show_window(fr3duo_evaluator)
asyncio.ensure_future(fr3duo_evaluator.run())  # starts right away; Start re-runs after it finishes
