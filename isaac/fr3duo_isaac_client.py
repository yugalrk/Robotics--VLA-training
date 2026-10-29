"""Run a trained FR3 Duo policy in Isaac Sim 6.2, closed loop.

1. In WSL, start the policy server (fr3duo_policy_server.py) with your checkpoint, or --dummy.
2. In Isaac Sim: Window > Script Editor > File > Open this file > Ctrl+Enter.
   A window "FR3 Duo policy" opens: type the instruction, press Start. Stop (or the timeline's
   Stop button) ends the run.

Every control step (30 Hz) the client renders the head and both wrist cameras at 224x224, reads
the 16 joint values, sends them with the instruction to the server, and applies the returned 16
joint targets for two 60 Hz physics steps. The simulation waits while the policy thinks, so the
timing matches the recorded dataset even when inference is slower than real time.

Dataset contract (FR3Duo_Handover_Dataset_lerobot3/meta/info.json and fr3_conversion.json):
    state / action = [left q1..q7, left gripper, right q1..q7, right gripper], arm joints in rad
    action gripper: 1 = open, 0 = closed, gripper joint = 0.8 rad * (1 - value)
    state gripper:  ASSUMED 1 - joint / 0.8 rad, see gripper_to_norm()

Per-step logs go to policy_runs/<time>/steps.jsonl next to this file. With the --dummy server
the robot just holds its pose and the run is marked FAKE.
"""
import asyncio
import io
import json
import os
import time
import urllib.request

import numpy as np

SERVER = "http://127.0.0.1:8090"     # not "localhost": Windows tries IPv6 first, which costs ~2 s per request
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
MAX_STEPS = 2400                    # recorded episodes are 2247 steps (75 s) long
CONTROL_HZ = 30
PHYSICS_HZ = 60

DEFAULT_HERE = r"C:\Users\User\Desktop\Hack2026\C2_Teach-the-robot_VLA\Environment"
ENV_SUBDIR = os.path.join("Environment_Hackathon_2026", "Environment_Hackathon_2026")
# The Script Editor runs a temporary copy of this file, so __file__ can point into %TEMP%:
# only use it when the environment folder is actually next to it.
# This file lives in <project>/isaac/; the scene is in <project>/Environment/.
_file_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
_env_home = os.path.join(os.path.dirname(_file_dir), "Environment") if _file_dir else None
HERE = _env_home if _env_home and os.path.isdir(os.path.join(_env_home, ENV_SUBDIR)) else DEFAULT_HERE
ENV_DIR = os.path.join(HERE, ENV_SUBDIR)
SCENE = os.path.join(ENV_DIR, "Hackathon_FR3Duo_flattened.usd")
AUTHOR_PREFIX = "C:/Users/z0045ytr/"
TEXTURE_ROOT = os.path.join(ENV_DIR, "Collected_Hackathon_FR3Duo_v2", "C", "Users", "z0045ytr")
LOG_ROOT = os.path.join(HERE, "policy_runs")

ROBOT = "/World/FR3Duo_withV_Mkit__1_"
ARTICULATIONS = {"left": ROBOT + "/left_arm_root_joint", "right": ROBOT + "/right_arm_root_joint"}
# feature name -> (camera prim, focal length); recorder v9 renders 224x224 with 4.8 x 4.8 apertures
CAMERAS = {
    "observation.images.external_camera": (
        ROBOT + "/mount_link/cover_link/head_link/zed_mini_camera_link/zed_mini_left_camera_frame/head_cam", 2.4),
    "observation.images.left_wrist_camera": (ROBOT + "/left_d405_color_frame/left_wrist_cam", 2.5290722846984863),
    "observation.images.right_wrist_camera": (ROBOT + "/right_d405_color_frame/right_wrist_cam", 2.5290722846984863),
}
CAMERA_APERTURE = 4.8
IMAGE_SIZE = 224
REPLICATOR_CAPTURE_ON_PLAY = True
RENDER_MODE = "RealTimePathTracing"   # "RTX - Real-Time 2.0" in the viewport menu
# Control steps per model action, with linear interpolation in between. The step_003000 checkpoint's
# chunks match the recorded actions best at a 5-frame spacing (offline check), so 5 should replay the
# demonstration speed. 1 = one action per 1/30 s step (the tested behaviour). 5 is NOT yet tested in the sim.
ACTION_STRIDE = 5                   # the curriculum checkpoints (training/train_curriculum.py) run at 6 Hz = every 5th 30 Hz step
TIMELINE_END_S = 3600.0             # authored timeline length; any value well above a run (MAX_STEPS / CONTROL_HZ = 80 s)
ARM_READY_RAD = [0.0, -0.785, 0.0, -2.3566, 0.0, 1.5708, 0.7854]   # first frame of every recorded episode
GRIPPER_CLOSED_RAD = 0.8


def gripper_to_norm(joint_rad):
    # ASSUMPTION: the dataset's gripper_joint_to_norm() is not in the folder. This linear form mirrors
    # the documented action rule and gives 0.995 when open, as in the data; check it against the
    # converter script (scripts/convert_raw_to_lerobot_v3.py) when you get it.
    return 1.0 - joint_rad / GRIPPER_CLOSED_RAD


def norm_to_gripper(value):
    return GRIPPER_CLOSED_RAD * (1.0 - float(np.clip(value, 0.0, 1.0)))


class PolicyClient:
    def __init__(self, url, timeout=60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, path, body=None):
        request = urllib.request.Request(self.url + path, data=body, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"policy server {path}: {err.read().decode(errors='replace')}") from None

    def info(self):
        return self._call("/info")

    def reset(self):
        return self._call("/reset", b"")

    def act(self, observation, task):
        buffer = io.BytesIO()
        np.savez(buffer, task=np.array(task), **observation)
        reply = self._call("/act", buffer.getvalue())
        return np.asarray(reply["action"], dtype=np.float32), reply["latency_ms"], reply["fake"]


def prepare_stage(stage):
    """Session-layer fixes: local textures, 60 Hz physics, recorder camera intrinsics.

    Also gives the stage a time range. The flattened scene has none, so Play loops over a single
    frame: sim time stays at 0, physics never steps and the cameras never render. Isaac Sim reads
    the range from the root layer only, so that one edit is made there, in memory (the file on disk
    changes only if you save the stage)."""
    from pxr import Sdf, Usd, UsdGeom

    with Usd.EditContext(stage, stage.GetRootLayer()):
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(TIMELINE_END_S * stage.GetTimeCodesPerSecond())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                value = attr.Get()
                if value and value.path.startswith(AUTHOR_PREFIX):
                    local = os.path.join(TEXTURE_ROOT, *value.path[len(AUTHOR_PREFIX):].split("/"))
                    if os.path.exists(local):
                        attr.Set(Sdf.AssetPath(local.replace("\\", "/")))
        scene = stage.GetPrimAtPath("/World/PhysicsScene")
        scene.AddAppliedSchema("PhysxSceneAPI")
        scene.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(PHYSICS_HZ)
        for camera_path, focal_length in CAMERAS.values():
            camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
            camera.CreateFocalLengthAttr().Set(focal_length)
            camera.CreateHorizontalApertureAttr().Set(CAMERA_APERTURE)
            camera.CreateVerticalApertureAttr().Set(CAMERA_APERTURE)


class Arm:
    def __init__(self, side):
        from isaacsim.core.experimental.prims import Articulation

        self.side = side
        self.articulation = Articulation(ARTICULATIONS[side])
        names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)] + [f"{side}_gripper_ctrl"]
        self.dofs = self.articulation.get_dof_indices(names).numpy()

    def read(self):
        """[q1..q7 (rad), gripper (normalized)] as in observation.state."""
        q = self.articulation.get_dof_positions().numpy()[0, self.dofs]
        return np.append(q[:7], gripper_to_norm(q[7])).astype(np.float32)

    def command(self, action8):
        targets = np.append(action8[:7], norm_to_gripper(action8[7])).astype(np.float32)
        self.articulation.set_dof_position_targets(targets[None], dof_indices=self.dofs)

    def teleport_ready(self):
        ready = np.array([ARM_READY_RAD + [0.0]], dtype=np.float32)
        self.articulation.set_dof_positions(ready, dof_indices=self.dofs)
        self.articulation.set_dof_position_targets(ready, dof_indices=self.dofs)


async def sim_steps(n):
    """Advance the simulation by n control steps (one app update = 1/CONTROL_HZ s, see RenderingManager.set_dt)."""
    import omni.kit.app

    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def grab_images(annotators, max_updates=120):
    """Latest RGB frames from all cameras. The cameras render at 30 Hz of sim time, so a new frame
    arrives every control step; before the first one the annotators return an empty array."""
    import omni.kit.app
    import omni.timeline

    for _ in range(max_updates):
        frames = {feature: annotator.get_data() for feature, annotator in annotators.items()}
        if all(np.ndim(f) == 3 and f.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE) for f in frames.values()):
            return {feature: np.ascontiguousarray(f[:, :, :3]) for feature, f in frames.items()}
        await omni.kit.app.get_app().next_update_async()
    timeline = omni.timeline.get_timeline_interface()
    shapes = {feature: np.shape(f) for feature, f in frames.items()}
    raise RuntimeError(f"cameras produced no {IMAGE_SIZE}x{IMAGE_SIZE} images: {shapes} "
                       f"(sim time {timeline.get_current_time():.3f} s; if it stays at 0 the timeline is not advancing)")


class Runner:
    def __init__(self):
        self.stop_requested = False
        self.running = False
        self.status = print

    def stop(self):
        self.stop_requested = True

    async def run(self, task=TASK, max_steps=MAX_STEPS, load_scene=True):
        import carb
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager

        if self.running:
            self.status("already running")
            return None
        self.running, self.stop_requested = True, False
        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()
        render_products = []
        try:
            policy = PolicyClient(SERVER)
            info = policy.info()          # fails early if the server is not running
            self.status(f"policy {info['policy_type']} | {'FAKE (dummy)' if info['fake'] else info['checkpoint']}")

            timeline.stop()
            await app.next_update_async()
            context = omni.usd.get_context()
            if load_scene or context.get_stage_url().replace("\\", "/") != SCENE.replace("\\", "/"):
                self.status("loading scene ...")
                opened, error = await context.open_stage_async(SCENE)
                if not opened:
                    raise RuntimeError(f"could not open {SCENE}: {error}")
            # In the GUI the stage can still be attaching after open_stage_async returns.
            for _ in range(600):
                stage = context.get_stage()
                if stage is not None and context.get_stage_state() == omni.usd.StageState.OPENED:
                    break
                await app.next_update_async()
            else:
                raise RuntimeError(f"stage did not finish opening: {SCENE}")
            prepare_stage(stage)
            # One app update = one 30 Hz control step = two 60 Hz physics substeps. The cameras render
            # at 30 Hz of sim time, so every timeline frame has a camera frame; with a 60 Hz timeline
            # Replicator waits for camera frames that never come and the timeline stalls.
            SimulationManager.set_physics_dt(1.0 / PHYSICS_HZ)
            RenderingManager.set_dt(1.0 / CONTROL_HZ)

            # The scene file stores PathTracing as its render mode. With path tracing, Replicator renders many
            # subframes per captured camera frame and keeps the timeline paused meanwhile, so the run stalls.
            carb.settings.get_settings().set_string("/rtx/rendermode", RENDER_MODE)
            carb.settings.get_settings().set_bool("/omni/replicator/captureOnPlay", REPLICATOR_CAPTURE_ON_PLAY)
            annotators = {}
            for feature, (camera_path, _) in CAMERAS.items():
                render_product = rep.create.render_product(camera_path, (IMAGE_SIZE, IMAGE_SIZE))
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([render_product])
                render_products.append(render_product)
                annotators[feature] = annotator

            timeline.play()
            for _ in range(3):
                await app.next_update_async()
            arms = [Arm("left"), Arm("right")]
            for arm in arms:
                arm.teleport_ready()
            await sim_steps(5)   # settle
            await grab_images(annotators)
            policy.reset()

            run_dir = os.path.join(LOG_ROOT, time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                json.dump({"task": task, "server": SERVER, "policy": info, "scene": SCENE,
                           "max_steps": max_steps, "control_hz": CONTROL_HZ, "physics_hz": PHYSICS_HZ}, f, indent=1)
            # a --replay-episode server returns the recorded 30 fps actions one by one: play them every step
            stride = 1 if str(info["policy_type"]).startswith(("replay", "pi05")) else ACTION_STRIDE
            latencies, fake, wall_start, step = [], info["fake"], time.perf_counter(), 0
            with open(os.path.join(run_dir, "steps.jsonl"), "w") as log:
                for step in range(max_steps):
                    if self.stop_requested or not timeline.is_playing():
                        self.status(f"stopped at step {step}")
                        break
                    observation = {"observation.state": np.concatenate([arm.read() for arm in arms])}
                    if step % stride == 0:
                        observation.update(await grab_images(annotators))
                        action, latency_ms, fake = policy.act(observation, task)
                        start = action if step == 0 else target
                    target = start + (action - start) * ((step % stride + 1) / stride)
                    arms[0].command(target[:8])
                    arms[1].command(target[8:])
                    await sim_steps(1)
                    latencies.append(latency_ms)
                    log.write(json.dumps({"step": step, "wall_s": time.perf_counter() - wall_start,
                                          "sim_s": timeline.get_current_time(),
                                          "state": observation["observation.state"].tolist(),
                                          "action": action.tolist(), "latency_ms": latency_ms}) + "\n")
                    if step % CONTROL_HZ == 0:
                        self.status(f"step {step}/{max_steps} | policy {latency_ms:.0f} ms" + (" | FAKE" if fake else ""))
                else:
                    step = max_steps
            summary = {"steps": step, "sim_seconds": step / CONTROL_HZ,
                       "wall_seconds": time.perf_counter() - wall_start,
                       "policy_latency_ms_mean": float(np.mean(latencies)) if latencies else None,
                       "fake": fake, "log_dir": run_dir}
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1)
            self.status(f"done: {summary}")
            return summary
        except Exception as exc:
            self.status(f"ERROR: {type(exc).__name__}: {exc}")
            raise
        finally:
            for render_product in render_products:
                render_product.destroy()
            self.running = False


def show_window(runner):
    import omni.ui as ui

    window = ui.Window("FR3 Duo policy", width=460, height=210)
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label("Instruction (use a training sentence exactly; this scene has the handover only, no cubes)", height=18)
            task_field = ui.StringField(multiline=True, height=60)
            task_field.model.set_value(TASK)
            with ui.HStack(height=30, spacing=6):
                ui.Button("Start", clicked_fn=lambda: asyncio.ensure_future(
                    runner.run(task_field.model.get_value_as_string())))
                ui.Button("Stop", clicked_fn=runner.stop)
            status = ui.Label("idle", word_wrap=True)

    def show(message):
        print("[fr3duo]", message)
        status.text = str(message)

    runner.status = show
    return window


if not os.environ.get("FR3DUO_NO_AUTORUN"):
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in ("isaacsim.core.experimental.prims", "isaacsim.core.simulation_manager",
                      "isaacsim.core.rendering_manager", "omni.replicator.core"):
        manager.set_extension_enabled_immediate(extension, True)
    fr3duo_runner = Runner()
    fr3duo_window = show_window(fr3duo_runner)
