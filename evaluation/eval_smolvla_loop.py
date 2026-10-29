"""Closed-loop evaluation of a SmolVLA checkpoint on the dual-arm FR3 (mock env, swap in the real robot API).

The checkpoint decides the observation layout, not this script. The FR3 Duo SmolVLA checkpoints here expect:
    3 cameras  observation.images.{external,left_wrist,right_wrist}_camera   float [3, 224, 224] in [0, 1]
    state      observation.state  [16] = left (7 joints rad + gripper 0..1 open) + right (same)
    action     [16], same layout as the state: absolute joint targets + gripper command (1 = open, 0 = closed)
    task       the training instruction (SmolVLA is language-conditioned)
Grippers count as DoF here, so 14 arm joints + 2 grippers = 16. Mismatched shapes fail loudly at startup.

Usage:
    python evaluation/eval_smolvla_loop.py --checkpoint outputs/smolvla_handover/step_003000 --episodes 5 --max_steps 300
Writes <out>/steps.jsonl (one line per step, streamed) and <out>/steps.csv + <out>/episodes.csv at the end.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
# env observation key -> policy input key
CAMERA_KEYS = {
    "pixels/cam_head": "observation.images.external_camera",
    "pixels/cam_left": "observation.images.left_wrist_camera",
    "pixels/cam_right": "observation.images.right_wrist_camera",
}
STATE_DIM = 16
IMG_SHAPE = (3, 224, 224)
CONTROL_HZ = 6  # the checkpoints were trained on 30 fps data sampled every 5th frame


class DualFr3Env:
    """Mock of the dual-FR3 robot. Replace the bodies of reset() and step() with your hardware calls.

    reset() -> obs dict, step(action) -> (next_obs, reward, done, info); obs is as in CAMERA_KEYS + "state".
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros(STATE_DIM, np.float32)

    def _observe(self):
        # REAL ROBOT: read the 3 cameras (RGB, resized to 224x224) and both arms' joints + gripper opening.
        obs = {k: torch.rand(IMG_SHAPE) for k in CAMERA_KEYS}  # float CHW in [0, 1]
        obs["state"] = torch.from_numpy(self.q.copy())
        return obs

    def reset(self):
        # REAL ROBOT: move both arms to the start pose and open the grippers, then wait until settled.
        self.q = np.array([-0.19, -0.34, 0.07, -2.24, 0.76, 1.73, -1.03, 1.0,
                           -0.17, -0.44, -0.15, -2.27, 0.09, 1.86, 1.07, 1.0], np.float32)
        return self._observe()

    def step(self, action):
        # REAL ROBOT: send the 16 joint/gripper targets, wait one control period (1 / CONTROL_HZ s), read back.
        action = np.asarray(action, np.float32)
        self.q = self.q + 0.5 * (action - self.q)  # crude "the robot moves halfway to the target"
        reward, done = 0.0, False  # REAL ROBOT: task-success check (e.g. box on tray) goes here
        return self._observe(), reward, done, {}


def to_policy_batch(obs, task):
    """Env observation -> the un-batched dict the LeRobot preprocessor expects."""
    batch = {policy_key: obs[env_key].float() for env_key, policy_key in CAMERA_KEYS.items()}
    batch["observation.state"] = obs["state"].float()
    batch["task"] = task
    return batch


def check_layout(policy):
    """Fail early if the env's observations do not match what the checkpoint was trained on."""
    feats = policy.config.input_features
    for key in CAMERA_KEYS.values():
        assert key in feats, f"checkpoint has no camera {key}; it expects {list(feats)}"
        assert tuple(feats[key].shape) == IMG_SHAPE, f"{key}: checkpoint {feats[key].shape}, env {IMG_SHAPE}"
    assert tuple(feats["observation.state"].shape) == (STATE_DIM,), \
        f"state: checkpoint {feats['observation.state'].shape}, env {STATE_DIM}"
    assert tuple(policy.config.output_features["action"].shape) == (STATE_DIM,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--out", default="eval_results/smolvla_loop")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.config.device = args.device
    policy.to(args.device).eval()
    check_layout(policy)
    # The checkpoint's own normalization stats + tokenizer; without them the actions are meaningless.
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.checkpoint, preprocessor_overrides={"device_processor": {"device": args.device}})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = DualFr3Env()
    rows, episodes = [], []
    with open(out / "steps.jsonl", "w") as log:
        for ep in range(args.episodes):
            obs = env.reset()
            policy.reset()
            preprocess.reset()
            postprocess.reset()
            ep_reward, done, t_ep = 0.0, False, time.time()
            for step in range(args.max_steps):
                t0 = time.perf_counter()
                with torch.no_grad():
                    action = postprocess(policy.select_action(preprocess(to_policy_batch(obs, args.task))))
                action = action.squeeze(0).float().cpu().numpy()
                latency_ms = (time.perf_counter() - t0) * 1000
                row = {"episode": ep, "step": step, "timestamp": time.time(), "latency_ms": round(latency_ms, 2),
                       "state": obs["state"].tolist(), "action": action.tolist()}
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                row.update(reward=reward, done=done)
                log.write(json.dumps(row) + "\n")
                log.flush()  # a crash keeps everything logged so far
                rows.append(row)
                if done:
                    break
            episodes.append({"episode": ep, "steps": step + 1, "done": done, "return": ep_reward,
                             "wall_s": round(time.time() - t_ep, 1)})
            print(f"episode {ep}: {step + 1} steps, done={done}, return={ep_reward:.2f}", flush=True)

    # flat CSV: one column per state / action dimension
    df = pd.DataFrame(rows)
    df = pd.concat([df.drop(columns=["state", "action"]),
                    pd.DataFrame(df["state"].tolist()).add_prefix("state_"),
                    pd.DataFrame(df["action"].tolist()).add_prefix("action_")], axis=1)
    df.to_csv(out / "steps.csv", index=False)
    pd.DataFrame(episodes).to_csv(out / "episodes.csv", index=False)
    print(f"wrote {out / 'steps.jsonl'}, {out / 'steps.csv'}, {out / 'episodes.csv'}")


if __name__ == "__main__":
    main()
