"""Closed-loop evaluation of FR3 Duo policies in the MuJoCo replica (fr3duo_scene.py). Runs on a laptop.

Modes
  replay : play a recorded demo's actions open-loop (30 fps, each held 2/60 s like the recording) -> checks that
           the replica reproduces the demonstration (box should end on the tray)
  policy : run a trained checkpoint closed-loop at 6 Hz: render the 3 cameras (224x224), read the 16-D state, ask
           the policy, interpolate its joint targets over the next 1/6 s of physics

Every episode is scored with the same stages as isaac/eval_isaac.py
  0 not lifted -> 1 lifted (>3 cm) -> 2 carried halfway to the tray -> 3 over the tray -> 4 resting on the tray
and saved as an MP4: outside view | head | left wrist | right wrist (the three right panels are what the policy sees),
with a live status overlay (stage, gripper command / opening / fingers touching the box, box height), plus a
<name>_trace.csv with the same values per recorded frame.

Usage (from the project root):
  python sim_mujoco/eval_mujoco.py --mode replay
  python sim_mujoco/eval_mujoco.py --mode policy --checkpoint outputs/smolvla_handover/step_003000 --episodes 3
  options: --n_action_steps 5 (re-plan more often) --instruction "..." --max_seconds 90 --box_offset 0.02 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from fr3duo_scene import TIMESTEP, build_model  # noqa: E402

REF = json.loads((ROOT / "artifacts/eval_assets/handover_reference.json").read_text())
GC = REF["gripper_contract"]
CAMS = {"observation.images.external_camera": "head", "observation.images.left_wrist_camera": "left_wrist",
        "observation.images.right_wrist_camera": "right_wrist"}
SIDES = ["left", "right"]


def grip_action_to_ctrl(a):
    """Dataset gripper action (1 open, 0 closed) -> joint target 0.8*(1-a) rad -> Robotiq ctrl 0..255."""
    q = GC["closed_joint_rad"] * (1.0 - float(np.clip(a, 0, 1)))
    return 255.0 * q / GC["closed_joint_rad"]


def grip_joint_to_state(q):
    return float(np.interp(q, [0.0, GC["hold_joint_rad"], GC["closed_joint_rad"]], [1.0, GC["hold_norm"], 0.0]))


class Sim:
    def __init__(self, box_offset=(0.0, 0.0), video_size=(320, 240)):
        self.model, self.info = build_model()
        self.data = mujoco.MjData(self.model)
        self.r224 = mujoco.Renderer(self.model, 224, 224)
        self.rvid = mujoco.Renderer(self.model, video_size[1], video_size[0])
        self.box_offset = np.array([*box_offset, 0.0])
        self.frames = []
        self.trace = []

        self.box_pos0, self.box_quat0 = self.demo_box_pose()
        self.sag = self.build_sag_table()

    def build_sag_table(self):
        """Isaac's recorded arm does not end up exactly at its command: joint 1 rests up to 0.08 rad away (the arms
        hang on a tilted V-mount and that drive is soft; friction adds hysteresis), the other joints within 5 mrad.
        MuJoCo's arm follows commands closely, so without correction it would be 4-5 cm away from where the
        recorded robot was. Table of (command -> resting offset) from the demo's still frames; at run time the
        offset of the nearest commands (5-NN, inverse-distance weights) is added to the target."""
        A, S = np.array(REF["demo_actions"]), np.array(REF["demo_states"])
        tab = {}
        for j, s in enumerate(SIDES):
            sl = slice(8 * j, 8 * j + 7)
            still = [k for k in range(1, len(S) - 1) if np.abs(S[k + 1, sl] - S[k - 1, sl]).max() * 15 < 0.02]
            tab[s] = (A[still, sl], S[still, sl] - A[still, sl])
        return tab

    def sag_offset(self, side, cmd7):
        keys, offs = self.sag[side]
        dist = np.linalg.norm(keys - cmd7, axis=1)
        nn = np.argpartition(dist, 5)[:5]
        w = 1.0 / (dist[nn] + 1e-3)
        return (offs[nn] * w[:, None]).sum(0) / w.sum()

    def fk(self, state16):
        d = mujoco.MjData(self.model)
        for j, s in enumerate(SIDES):
            d.qpos[self.info["arm_qpos"][s]] = state16[8 * j: 8 * j + 7]
        mujoco.mj_kinematics(self.model, d)
        return d

    def demo_box_pose(self, grip_depth=0.025):
        """Where the box must have been in the recordings (the hand-out USD pose is overridden in the recordings).
        Derived from the two demo grasps, using the recorded joint states:
        - at the handover both grippers hold the box: the thin side (31 mm) is between the fingers of both, the long
          side (107 mm) runs from the left grasp to the right grasp (68 mm apart, so each grips ~2 cm from an end),
          the 72 mm side points along the left gripper's approach, the left fingertips `grip_depth` inside the top.
        - that pose, fixed in the left gripper's frame, is carried back to the moment the left gripper closes:
          that is the start pose on the cart. The cart's support surface is moved to the box's underside.
        Also resizes the box geom."""
        A, S = np.array(REF["demo_actions"]), np.array(REF["demo_states"])
        kl, kr = int(np.argmax(A[:, 7] < 0.25)), int(np.argmax(A[:, 15] < 0.25))
        m = self.model
        half = np.array([0.031, 0.107, 0.072]) / 2  # box axes: x thin (between fingers), y long, z = 72 mm side
        d = self.fk(S[kr])
        gb = m.body("left_g_base").id
        RG, pG = d.xmat[gb].reshape(3, 3).copy(), d.xpos[gb].copy()
        pL = d.site_xpos[m.site("left_g_pinch").id].copy()
        pR = d.site_xpos[m.site("right_g_pinch").id].copy()
        spread = d.xpos[m.body("left_g_left_pad").id] - d.xpos[m.body("left_g_right_pad").id]
        x = spread / np.linalg.norm(spread)
        v = pR - pL
        z = RG[:, 2] - (RG[:, 2] @ x) * x  # 72 mm side along the left gripper's approach (box sits flat when picked)
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        if y @ v < 0:
            x, y = -x, -y
        c = pL + 0.5 * (v @ y) * y + (half[2] - grip_depth) * z
        R_box_G = RG.T @ np.column_stack([x, y, z])
        c_G = RG.T @ (c - pG)
        # back to the left grasp moment
        d0 = self.fk(S[kl])
        R0, p0 = d0.xmat[gb].reshape(3, 3), d0.xpos[gb]
        Rb = R0 @ R_box_G
        cb = R0 @ c_G + p0
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(Rb).flatten())
        g = m.geom("box_geom").id
        m.geom_size[g] = half
        m.geom_rbound[g] = np.linalg.norm(half)
        bottom = cb[2] - np.abs(Rb[2]) @ half  # lowest point of the box
        top_geom = m.geom("cart_left_top").id
        self.cart_shift = bottom - (m.geom_pos[top_geom][2] + m.geom_size[top_geom][2])
        m.geom_pos[top_geom][2] += self.cart_shift
        self.box_pose_check = {"right_pinch_in_box_mm": (Rb.T @ (R0 @ (RG.T @ (pR - pG)) + p0 - cb) * 1000).round(1).tolist(),
                               "box_up_axis_tilt_deg": round(float(np.degrees(np.arccos(abs(Rb[2, 2])))), 1),
                               "support_raised_mm": round(float(self.cart_shift) * 1000, 1)}
        return cb + np.array([0, 0, 0.0005]), q

    # ---------------------------------------------------------------- robot I/O
    def reset(self, state16):
        mujoco.mj_resetData(self.model, self.data)
        i = self.info
        self.data.qpos[i["box_qpos"]:i["box_qpos"] + 3] = self.box_pos0 + self.box_offset
        self.data.qpos[i["box_qpos"] + 3:i["box_qpos"] + 7] = self.box_quat0
        for k, s in enumerate(SIDES):
            self.data.qpos[i["arm_qpos"][s]] = state16[8 * k: 8 * k + 7]
        self.apply(np.array(state16, np.float32))
        mujoco.mj_forward(self.model, self.data)
        self.step(int(0.5 / TIMESTEP))  # let the box settle on the cart
        self.box0 = self.box_pos().copy()

    def apply(self, a16, emulate_sag=True):
        for k, s in enumerate(SIDES):
            tgt = np.array(a16[8 * k: 8 * k + 7], float)
            if emulate_sag:
                tgt += self.sag_offset(s, tgt)
            self.data.ctrl[self.info["arm_act"][s]] = tgt
            self.data.ctrl[self.info["grip_act"][s]] = grip_action_to_ctrl(a16[8 * k + 7])

    def step(self, n):
        mujoco.mj_step(self.model, self.data, nstep=n)

    def state(self):
        out = []
        for s in SIDES:
            out += list(self.data.qpos[self.info["arm_qpos"][s]]) + [grip_joint_to_state(self.data.qpos[self.info["grip_qpos"][s]])]
        return np.array(out, np.float32)

    def images(self):
        out = {}
        for key, cam in CAMS.items():
            self.r224.update_scene(self.data, camera=cam)
            out[key] = self.r224.render().copy()
        return out

    def box_pos(self):
        a = self.info["box_qpos"]
        return self.data.qpos[a:a + 3]

    def touching(self, side):
        """Which of this gripper's two finger pads currently touch the box."""
        box_g = self.model.geom("box_geom").id
        hit = {"left_pad": False, "right_pad": False}
        for c in self.data.contact[:self.data.ncon]:
            if box_g not in (c.geom1, c.geom2):
                continue
            body = self.model.body(self.model.geom_bodyid[c.geom2 if c.geom1 == box_g else c.geom1]).name
            for f in hit:
                if body.startswith(f"{side}_g_{f.split('_')[0]}") and ("pad" in body or "follower" in body):
                    hit[f] = True
        return hit

    def snapshot(self, t, cmd16, tracker):
        """One row of trace.csv: what the robot was told, what the grippers did, where the box is."""
        p = self.box_pos()
        row = {"t_s": round(t, 2), "stage": tracker.stage,
               "box_x": round(p[0], 4), "box_y": round(p[1], 4), "box_z": round(p[2], 4),
               "box_lift_mm": round((p[2] - tracker.p0[2]) * 1000, 1),
               "box_to_tray_mm": round(float(np.linalg.norm((p - tracker.tray_c)[:2])) * 1000, 1)}
        for k, s in enumerate(SIDES):
            q = self.data.qpos[self.info["grip_qpos"][s]]
            touch = self.touching(s)
            pinch = self.data.site_xpos[self.model.site(f"{s}_g_pinch").id]
            row.update({f"{s}_grip_cmd": round(float(cmd16[8 * k + 7]), 2), f"{s}_grip_angle_rad": round(float(q), 3),
                        f"{s}_grip_state": round(grip_joint_to_state(q), 2),
                        f"{s}_touch_pads": int(touch["left_pad"]) + int(touch["right_pad"]),
                        f"{s}_pinch_to_box_mm": round(float(np.linalg.norm(pinch - p)) * 1000, 1)})
        self.trace.append(row)
        return row

    def record(self, imgs=None, row=None):
        self.rvid.update_scene(self.data, camera="overview")
        over = self.rvid.render()
        imgs = imgs or self.images()
        h = over.shape[0]
        small = [np.asarray(_resize(imgs[k], h)) for k in CAMS]
        frame = np.concatenate([over, *small], 1)
        if row is not None:
            frame = overlay(frame, row)
        self.frames.append(frame)


STAGES = ["not lifted", "lifted", "carried halfway", "over tray", "resting on tray"]


def overlay(frame, row):
    """Burn the live status into the video: time, stage, per gripper command / opening / finger contact, box."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    def grip(s):
        cmd = row[f"{s}_grip_cmd"]
        word = "open" if cmd > 0.75 else "pre-grasp" if cmd > 0.25 else "CLOSE"
        touch = row[f"{s}_touch_pads"]
        return (f"{s[0].upper()} gripper: cmd {word} ({cmd:.2f}) | angle {row[f'{s}_grip_angle_rad']:.2f} rad | "
                f"fingers on box {touch}/2 | {row[f'{s}_pinch_to_box_mm']:.0f} mm from box")
    lines = [f"t={row['t_s']:5.1f}s  stage {row['stage']}: {STAGES[row['stage']]}  |  box lift {row['box_lift_mm']:+.0f} mm, "
             f"{row['box_to_tray_mm']:.0f} mm from tray centre", grip("left"), grip("right")]
    d.rectangle([0, 0, 470, 14 * len(lines) + 4], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        colour = (255, 255, 0) if i == 0 else (120, 255, 120) if f"on box 2/2" in ln else (255, 255, 255)
        d.text((4, 2 + 14 * i), ln, fill=colour)
    return np.asarray(img)


def save_trace(rows, path):
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _resize(img, h):
    from PIL import Image
    return Image.fromarray(img).resize((h, h))


# ---------------------------------------------------------------- scoring
class Tracker:
    def __init__(self, sim):
        self.sim = sim
        self.lo, self.hi = sim.info["tray_lo"], sim.info["tray_hi"]
        self.tray_c = (self.lo + self.hi) / 2
        self.p0 = sim.box0.copy()
        self.start_d = np.linalg.norm((self.p0 - self.tray_c)[:2])
        self.stage, self.max_lift, self.hist = 0, 0.0, []

    def on_tray_xy(self, p, margin=0.01):
        return bool(np.all(p[:2] > self.lo[:2] + margin) and np.all(p[:2] < self.hi[:2] - margin))

    def update(self, t):
        p = self.sim.box_pos().copy()
        self.hist.append((t, *p))
        self.max_lift = max(self.max_lift, p[2] - self.p0[2])
        d = np.linalg.norm((p - self.tray_c)[:2])
        if self.max_lift > 0.03:
            self.stage = max(self.stage, 1)
        if self.stage >= 1 and d < 0.5 * self.start_d:
            self.stage = max(self.stage, 2)
        if self.stage >= 2 and self.on_tray_xy(p):
            self.stage = max(self.stage, 3)

    def result(self):
        p = self.sim.box_pos().copy()
        h = np.array(self.hist[-10:])
        speed = float(np.linalg.norm(np.diff(h[:, 1:], axis=0), axis=1).sum() / max(h[-1, 0] - h[0, 0], 1e-6)) if len(h) > 2 else 0.0
        resting = self.on_tray_xy(p) and self.lo[2] - 0.01 < p[2] < self.hi[2] + 0.12 and speed < 0.01
        if self.stage >= 3 and resting:
            self.stage = 4
        return {"success": self.stage == 4, "stage": self.stage,
                "stage_name": ["not lifted", "lifted", "carried halfway", "over tray", "resting on tray"][self.stage],
                "max_lift_mm": round(self.max_lift * 1000, 1), "final_box_pos": p.round(4).tolist(),
                "final_dist_to_tray_center_mm": round(float(np.linalg.norm((p - self.tray_c)[:2])) * 1000, 1),
                "final_speed_m_s": round(speed, 4)}


def save_video(frames, path, fps):
    import av
    with av.open(str(path), "w") as c:
        try:
            s = c.add_stream("libx264", rate=fps)
        except Exception:
            s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = frames[0].shape[1] // 2 * 2, frames[0].shape[0] // 2 * 2, "yuv420p"
        for f in frames:
            f = f[: s.height, : s.width]
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


# ---------------------------------------------------------------- modes
def run_replay(args, out):
    sim = Sim(args.box_offset)
    A, S = np.array(REF["demo_actions"]), np.array(REF["demo_states"])
    sim.reset(A[0] * 0 + np.array(REF["initial_state"]))
    tr = Tracker(sim)
    sub = int(round((1 / 30) / TIMESTEP))
    err = []
    for k in range(len(A)):
        if k < REF["hold_cut_frame"]:
            err.append(np.abs(sim.state() - S[k]))
        if k % 5 == 0:
            sim.record(row=sim.snapshot(k / 30, A[k], tr))
        sim.apply(A[k])
        sim.step(sub)
        tr.update(k / 30)
    sim.step(int(1.0 / TIMESTEP))
    tr.update(len(A) / 30 + 1)
    sim.record(row=sim.snapshot(len(A) / 30 + 1, A[-1], tr))
    e = np.array(err)
    res = {**tr.result(), "joint_tracking_rad_mean": round(float(e[:, [*range(7), *range(8, 15)]].mean()), 4),
           "joint_tracking_rad_max": round(float(e[:, [*range(7), *range(8, 15)]].max()), 4),
           "gripper_state_err_mean": round(float(e[:, [7, 15]].mean()), 3)}
    save_video(sim.frames, out / "replay.mp4", 6)
    save_trace(sim.trace, out / "replay_trace.csv")
    (out / "replay.json").write_text(json.dumps(res, indent=1))
    print("REPLAY", json.dumps(res))


def run_policy(args, out):
    from fr3duo.policy_runner import PolicyRunner
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    runner = PolicyRunner(ROOT / args.checkpoint, n_action_steps=args.n_action_steps)
    sim = Sim(args.box_offset)
    sub = int(round((1 / 6) / TIMESTEP))
    task = args.instruction or REF["task"]
    results = []
    for ep in range(args.episodes):
        sim.frames, sim.trace = [], []
        sim.reset(np.array(REF["initial_state"]))
        runner.reset()
        tr = Tracker(sim)
        prev = np.array(REF["initial_state"], np.float32)
        lat = []
        t0 = time.time()
        for step in range(int(args.max_seconds * 6)):
            imgs = sim.images()
            obs = {**imgs, "observation.state": sim.state(), "task": task}
            o = runner.act(obs)
            a = np.asarray(o["action"], np.float32)
            sim.record(imgs, row=sim.snapshot(step / 6, a, tr))
            lat.append(o["latency_ms"])
            n = 10
            for i in range(1, n + 1):  # interpolate the joint targets over the 1/6 s step
                sim.apply(prev + (a - prev) * i / n)
                sim.step(sub // n)
            prev = a
            tr.update(step / 6)
            if tr.stage == 4 or (step > 30 and tr.result()["stage"] == 4):
                break
        sim.step(int(1.0 / TIMESTEP))
        tr.update(args.max_seconds)
        sim.record(row=sim.snapshot(step / 6 + 1, prev, tr))
        r = {"episode": ep, "instruction": task, **tr.result(), "policy_steps": step + 1,
             "latency_ms_median": round(float(np.median(lat)), 1), "wall_s": round(time.time() - t0, 1)}
        save_video(sim.frames, out / f"policy_ep{ep:02d}.mp4", 6)
        save_trace(sim.trace, out / f"policy_ep{ep:02d}_trace.csv")
        results.append(r)
        print("EPISODE", json.dumps(r), flush=True)
        (out / "results.json").write_text(json.dumps({"checkpoint": args.checkpoint, "episodes": results}, indent=1))
    sr = float(np.mean([r["success"] for r in results]))
    summary = {"checkpoint": args.checkpoint, "success_rate": sr, "stages": [r["stage"] for r in results],
               "box_offset_m": list(args.box_offset), "instruction": task}
    (out / "results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "policy"], required=True)
    ap.add_argument("--checkpoint", default="outputs/smolvla_handover/step_003000")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_seconds", type=float, default=90)
    ap.add_argument("--n_action_steps", type=int, default=None)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--box_offset", type=float, nargs=2, default=[0.0, 0.0], help="move the box start (m, x y)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = ROOT / (args.out or f"eval_results/mujoco/{args.mode}" + ("" if args.mode == "replay" else "_" + Path(args.checkpoint).parent.name))
    out.mkdir(parents=True, exist_ok=True)
    (run_replay if args.mode == "replay" else run_policy)(args, out)
