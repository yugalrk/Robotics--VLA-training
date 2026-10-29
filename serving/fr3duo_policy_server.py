"""Policy server for the FR3 Duo Isaac Sim bridge. Runs in WSL, next to LeRobot.

    source ~/shs-hack-2026-vla-pipeline/.venv/bin/activate
    python /mnt/c/Users/User/Desktop/Hack2026/C2_Teach-the-robot_VLA/serving/fr3duo_policy_server.py \
        --checkpoint outputs/.../checkpoints/last/pretrained_model

    # bridge test without a trained model: holds the starting pose, flagged FAKE
    python .../fr3duo_policy_server.py --dummy

The Isaac Sim client (fr3duo_isaac_client.py) sends one observation per 30 Hz control step and
gets back one 16-value action. Protocol (HTTP on 127.0.0.1; from Windows use http://127.0.0.1:8090,
not localhost, which first tries IPv6 and adds ~2 s per request):

    GET  /info   -> {"checkpoint", "policy_type", "input_features", "output_features", "fake"}
    POST /reset  -> clears the policy's action queue; call at the start of every episode
    POST /act    body: .npz with one array per policy input feature, named exactly like the
                       feature ("observation.state", "observation.images.external_camera", ...),
                       images as uint8 (H, W, 3), plus "task" (a 0-d string array)
                 -> {"action": [floats], "latency_ms": float, "fake": bool}

Dataset contract this server checks the checkpoint against (FR3Duo_Handover_Dataset_lerobot3):
    observation.state (16,), action (16,), three 224x224 RGB cameras:
    observation.images.external_camera / left_wrist_camera / right_wrist_camera.
"""
import argparse
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

FR3DUO_STATE_DIM = 16
FR3DUO_ACTION_DIM = 16
FR3DUO_CAMERAS = [
    "observation.images.external_camera",
    "observation.images.left_wrist_camera",
    "observation.images.right_wrist_camera",
]
FR3DUO_TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."


class DummyPolicy:
    """Holds the pose of the first observation after /reset. For testing the bridge only; FAKE results."""

    fake = True
    checkpoint = None
    policy_type = "dummy-hold-start-pose"
    input_features = {"observation.state": [FR3DUO_STATE_DIM]} | {k: [3, 224, 224] for k in FR3DUO_CAMERAS}
    output_features = {"action": [FR3DUO_ACTION_DIM]}

    def __init__(self):
        self.hold = None

    def reset(self):
        self.hold = None

    def act(self, obs, task):
        if self.hold is None:
            self.hold = np.asarray(obs["observation.state"], dtype=np.float32).copy()
        return self.hold


class ReplayPolicy:
    """Plays back the recorded actions of one dataset episode (open loop). For checking that the
    simulation reproduces the recordings; not a policy result, so it is flagged FAKE."""

    fake = True
    checkpoint = None
    input_features = DummyPolicy.input_features
    output_features = DummyPolicy.output_features

    def __init__(self, dataset_root, episode):
        import glob
        import os

        import pandas as pd

        files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "*.parquet")))
        df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "action"]) for f in files)
        df = df[df.episode_index == episode].sort_values("frame_index")
        if df.empty:
            raise SystemExit(f"episode {episode} not found in {dataset_root}")
        self.actions = np.stack(df["action"].to_numpy()).astype(np.float32)
        self.policy_type = f"replay-episode-{episode}"
        self.step = 0

    def reset(self):
        self.step = 0

    def act(self, obs, task):
        action = self.actions[min(self.step, len(self.actions) - 1)]
        self.step += 1
        return action


class LeRobotPolicy:
    fake = False

    def __init__(self, checkpoint, device, n_action_steps=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        try:
            from lerobot.policies import make_pre_post_processors
        except ImportError:  # lerobot 0.4.4 (the Windows .venv)
            from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies.utils import prepare_observation_for_inference

        self._torch = torch
        self._prepare = prepare_observation_for_inference
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg.device = str(self.device)
        if n_action_steps:  # actions executed from each predicted chunk before predicting a new one
            if not 1 <= n_action_steps <= cfg.chunk_size:
                raise SystemExit(f"--n_action_steps must be between 1 and the chunk size {cfg.chunk_size}")
            cfg.n_action_steps = n_action_steps
        self.n_action_steps, self.chunk_size = cfg.n_action_steps, getattr(cfg, "chunk_size", None)
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
        self.policy.to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.checkpoint = checkpoint
        self.policy_type = cfg.type
        self.input_features = {k: list(v.shape) for k, v in cfg.input_features.items()}
        self.output_features = {k: list(v.shape) for k, v in cfg.output_features.items()}

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def act(self, obs, task):
        frame = self._prepare(dict(obs), self.device, task=task)
        with self._torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(self.preprocessor(frame)))
        return action.squeeze(0).float().cpu().numpy()


class TrajectoryPolicy:
    """No learned model: the recorded handover as a closed-loop lookup table. All 84 handover episodes have the
    same start pose and identical actions (only the lighting differs), so one recorded trajectory is the task.

    act(): find where along the recording the robot is now (nearest recorded state among the next `step` frames:
    never more than 1/policy_hz s ahead per call), then return the recorded action for 1/policy_hz s later. A robot
    that lags or is blocked is waited for. Where the recording itself stands still (e.g. a gripper closing), nearest-
    state matching cannot see progress, so there the index advances on time once the robot is at that pose."""

    fake = False
    policy_type = "trajectory-handover"
    input_features = DummyPolicy.input_features
    output_features = DummyPolicy.output_features

    def __init__(self, dataset_root, episode=0, policy_hz=6.0, fps=30, still_rad=0.002, near_rad=0.02):
        import glob
        import os

        import pandas as pd

        files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "*.parquet")))
        df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"]) for f in files)
        d = df[df.episode_index == episode].sort_values("frame_index")
        if d.empty:
            raise SystemExit(f"episode {episode} not found in {dataset_root}")
        self.S = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
        self.A = np.stack(d["action"].to_numpy()).astype(np.float32)
        self.step = max(1, int(round(fps / policy_hz)))
        self.still, self.near = still_rad, near_rad
        self.w = np.ones(16, np.float32)
        self.w[[7, 15]] = 0.2  # gripper readings depend on the sim's joint->state mapping: trust them less
        self.checkpoint = f"trajectory: {dataset_root} episode {episode} ({len(self.A)} frames)"
        self.n_action_steps, self.chunk_size = 1, 1
        self.reset()

    def reset(self):
        self.i = 0

    def dist(self, a, b):
        return float((np.abs(np.asarray(a, np.float32) - b) * self.w).mean())

    def phase(self, state):
        lo, nxt = self.i, min(self.i + self.step, len(self.S) - 1)
        d = (np.abs(self.S[lo:nxt + 1] - np.asarray(state, np.float32)) * self.w).mean(1)
        j = lo + int(np.argmax(d <= d.min() + 1e-6))  # earliest best match: never skip ahead within ties
        if self.dist(self.S[nxt], self.S[lo]) < self.still and d[0] < self.near:  # recording stands still here
            j = nxt
        return j

    def act(self, obs, task):
        self.i = self.phase(obs["observation.state"])
        return self.A[min(self.i + self.step - 1, len(self.A) - 1)].copy()


class GraspDeepener:
    """Inference-time grasp shaping, no retraining. When the model first commands a gripper to close, keep that
    gripper at its pre-grasp opening and continue the arm's approach (mean joint velocity of the last second of
    executed actions, slowly decaying, capped at max_extra_rad per joint) for `extend_s`, then close and hold the arm
    still for `hold_s`, then hand
    back to the model (re-planning from the new pose). Each arm is shaped once, at its first grasp."""

    ARMS = {"left": (list(range(7)), 7), "right": (list(range(8, 15)), 15)}

    def __init__(self, policy, extend_s, hold_s=0.7, hz=6.0, close_below=0.2, decay=0.95, max_extra_rad=0.15):
        self.p, self.extend, self.hold = policy, int(round(extend_s * hz)), int(round(hold_s * hz))
        self.window = max(2, int(round(hz)))  # approach velocity = mean over the last second of executed actions
        self.close_below, self.decay, self.max_extra = close_below, decay, max_extra_rad
        for k in ("fake", "checkpoint", "policy_type", "input_features", "output_features", "n_action_steps", "chunk_size"):
            setattr(self, k, getattr(policy, k, None))
        self.grasp_extend_s, self.grasp_hold_s = extend_s, hold_s
        self.reset()

    def reset(self):
        self.p.reset()
        self.hist, self.plan, self.done = [], [], set()

    def act(self, obs, task):
        if self.plan:  # shaping in progress: replay the prepared actions, then let the model re-plan
            a = self.plan.pop(0)
            if not self.plan:
                self.p.reset()
        else:
            a = np.asarray(self.p.act(obs, task), np.float32)
            for side, (arm, g) in self.ARMS.items():
                prev = self.hist[-1] if self.hist else None
                if side in self.done or prev is None or len(self.hist) < 3:
                    continue
                if prev[g] >= self.close_below and a[g] < self.close_below:  # the model starts closing this gripper
                    self.done.add(side)
                    self.plan = self._shape(arm, g, a[g])
                    a = self.plan.pop(0)
                    break
        self.hist = (self.hist + [np.asarray(a, np.float32)])[-(self.window + 2):]
        return a

    def _shape(self, arm, g, close_cmd):
        base = self.hist[-1].copy()
        w = min(self.window, len(self.hist) - 1)
        v = (self.hist[-1][arm] - self.hist[-1 - w][arm]) / w  # mean joint velocity per policy step over the last second
        plan, extra, cur = [], np.zeros(len(arm), np.float32), base.copy()
        for i in range(self.extend):
            step = v * self.decay ** (i + 1)
            extra = np.clip(extra + step, -self.max_extra, self.max_extra)
            cur = base.copy()
            cur[arm] = base[arm] + extra  # keep the gripper at its pre-grasp opening while approaching
            plan.append(cur.copy())
        closing = cur.copy()
        closing[g] = close_cmd
        plan += [closing.copy() for _ in range(max(1, self.hold))]
        return plan


def contract_problems(policy):
    """Differences between the checkpoint's features and the FR3 Duo dataset contract."""
    problems = []
    expected = {"observation.state": [FR3DUO_STATE_DIM]} | {k: [3, 224, 224] for k in FR3DUO_CAMERAS}
    for key, shape in expected.items():
        got = policy.input_features.get(key)
        if got is None:
            problems.append(f"missing input {key} {shape}")
        elif got != shape:
            problems.append(f"input {key} has shape {got}, expected {shape}")
    for key in sorted(set(policy.input_features) - set(expected)):
        problems.append(f"unexpected input {key} {policy.input_features[key]}")
    if policy.output_features.get("action") != [FR3DUO_ACTION_DIM]:
        problems.append(f"action has shape {policy.output_features.get('action')}, expected [{FR3DUO_ACTION_DIM}]")
    return problems


def make_handler(policy):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/info":
                return self._reply(404, {"error": f"unknown path {self.path}"})
            self._reply(200, {
                "checkpoint": policy.checkpoint,
                "policy_type": policy.policy_type,
                "input_features": policy.input_features,
                "output_features": policy.output_features,
                "fake": policy.fake,
                "n_action_steps": getattr(policy, "n_action_steps", None),
                "grasp_extend_s": getattr(policy, "grasp_extend_s", 0),
                "grasp_hold_s": getattr(policy, "grasp_hold_s", 0),
                "chunk_size": getattr(policy, "chunk_size", None),
            })

        def do_POST(self):
            try:
                if self.path == "/reset":
                    policy.reset()
                    return self._reply(200, {"ok": True})
                if self.path != "/act":
                    return self._reply(404, {"error": f"unknown path {self.path}"})
                body = self.rfile.read(int(self.headers["Content-Length"]))
                with np.load(io.BytesIO(body), allow_pickle=False) as npz:
                    arrays = {k: npz[k] for k in npz.files}
                task = str(arrays.pop("task", FR3DUO_TASK))
                missing = [k for k in policy.input_features if k not in arrays]
                if missing:
                    return self._reply(400, {"error": f"observation is missing {missing}"})
                obs = {k: arrays[k] for k in policy.input_features}
                start = time.perf_counter()
                action = policy.act(obs, task)
                self._reply(200, {
                    "action": action.tolist(),
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "fake": policy.fake,
                })
            except Exception as exc:  # report to the client instead of dropping the connection
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="pretrained_model folder or Hugging Face repo id")
    source.add_argument("--dummy", action="store_true", help="hold-pose policy for bridge tests (FAKE results)")
    source.add_argument("--trajectory", action="store_true",
                        help="no model: the recorded handover as a closed-loop lookup table (see TrajectoryPolicy)")
    source.add_argument("--replay-episode", type=int, metavar="N",
                        help="play back the recorded actions of dataset episode N (sim check, FAKE results)")
    parser.add_argument("--dataset", default=str(Path(__file__).resolve().parents[1] / "Environment" / "Environment_Hackathon_2026"
                                                 / "Environment_Hackathon_2026" / "FR3Duo_Handover_Dataset_lerobot3"),
                        help="dataset root for --trajectory / --replay-episode (default: next to this script)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="auto", help="cuda, cpu or auto")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="actions executed from each predicted chunk before re-planning (default: the checkpoint's, "
                             "20 = the whole 3.3 s chunk; 5 re-plans every 0.8 s at 6 Hz)")
    parser.add_argument("--grasp_extend", type=float, default=0.0,
                        help="seconds to keep approaching (gripper held open) when the model first closes a gripper; 0 = off")
    parser.add_argument("--grasp_hold", type=float, default=0.7, help="seconds to hold the arm still while the gripper closes")
    parser.add_argument("--policy_hz", type=float, default=6.0, help="rate the client calls /act at (the sim scripts: 6 Hz)")
    parser.add_argument("--any-robot", action="store_true",
                        help="serve a checkpoint that does not match the FR3 Duo contract (for testing only)")
    args = parser.parse_args()

    if args.dummy:
        policy = DummyPolicy()
    elif args.trajectory:
        policy = TrajectoryPolicy(args.dataset, policy_hz=args.policy_hz)
    elif args.replay_episode is not None:
        policy = ReplayPolicy(args.dataset, args.replay_episode)
    else:
        policy = LeRobotPolicy(args.checkpoint, args.device, args.n_action_steps)
        if args.grasp_extend > 0:
            policy = GraspDeepener(policy, args.grasp_extend, args.grasp_hold, args.policy_hz)
            print(f"grasp deepening: approach {args.grasp_extend} s further with the gripper open, then close and hold {args.grasp_hold} s")
    problems = contract_problems(policy)
    if problems:
        print("Checkpoint does not match the FR3 Duo dataset contract:")
        for problem in problems:
            print("  -", problem)
        if not args.any_robot:
            raise SystemExit("Refusing to serve it to the FR3 Duo client (use --any-robot only for tests).")
    if policy.fake:
        print(f"{policy.policy_type}: not a trained policy. Results are FAKE.")
    print(f"policy {policy.policy_type} from {policy.checkpoint} | inputs {policy.input_features}"
          + (f" | executes {policy.n_action_steps} of {policy.chunk_size} actions per prediction" if getattr(policy, "n_action_steps", None) else ""))
    print(f"serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    HTTPServer((args.host, args.port), make_handler(policy)).serve_forever()


if __name__ == "__main__":
    main()
