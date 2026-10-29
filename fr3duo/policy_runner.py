"""Serve a trained LeRobot policy (ACT / SmolVLA checkpoint dir) over TCP.

The simulator (or a real robot) sends observations, the server returns the next 16-D action.
Uses only the Python standard library for transport (multiprocessing.connection: pickled dicts
over TCP), so the client side (e.g. Isaac Sim's Python) needs nothing but numpy.

Protocol (each message is a dict):
    -> {"cmd": "reset"}                                    <- {"ok": True}
    -> {"cmd": "act", "obs": {                             <- {"action": float32[16], "latency_ms": float,
           "observation.images.external_camera": uint8[224,224,3],      "replanned": bool}
           "observation.images.left_wrist_camera": uint8[224,224,3],
           "observation.images.right_wrist_camera": uint8[224,224,3],
           "observation.state": float32[16],
           "task": str (optional, used by language-conditioned policies)}}
    -> {"cmd": "info"}                                     <- {"checkpoint": ..., "fps": ..., ...}
    -> {"cmd": "close"}

Usage:
    python serving/serve_policy.py --checkpoint outputs/act_mixed/step_006000 --port 5555
    python serving/serve_policy.py --checkpoint ... --n_action_steps 5      # re-plan every 5 actions
    python serving/serve_policy.py --checkpoint ... --selftest                # check against dataset frames, no server
"""
import argparse
import json
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

AUTHKEY = b"fr3duo"
CAMS = ["observation.images.external_camera", "observation.images.left_wrist_camera",
        "observation.images.right_wrist_camera"]


class PolicyRunner:
    def __init__(self, checkpoint, device="cuda", n_action_steps=None, temporal_ensemble=None):
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = device
        if temporal_ensemble is not None:
            cfg.temporal_ensemble_coeff, cfg.n_action_steps = temporal_ensemble, 1
        elif n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, checkpoint, preprocessor_overrides={"device_processor": {"device": device}})
        self.cfg, self.device, self.checkpoint = cfg, device, str(checkpoint)
        self.reset()

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    @torch.no_grad()
    def act(self, obs):
        t0 = time.perf_counter()
        batch = {k: torch.from_numpy(np.ascontiguousarray(obs[k])).permute(2, 0, 1).float() / 255.0 for k in CAMS}
        batch["observation.state"] = torch.as_tensor(np.asarray(obs["observation.state"], np.float32))
        batch["task"] = obs.get("task", "")
        queue = getattr(self.policy, "_action_queue", None)  # ACT
        if queue is None:
            queue = getattr(self.policy, "_queues", {}).get("action")  # SmolVLA, pi0, ...
        replanned = queue is None or len(queue) == 0
        action = self.post(self.policy.select_action(self.pre(batch)))
        if self.device == "cuda":
            torch.cuda.synchronize()
        return {"action": action.squeeze(0).cpu().numpy().astype(np.float32),
                "latency_ms": (time.perf_counter() - t0) * 1000, "replanned": replanned}

    def info(self):
        return {"checkpoint": self.checkpoint, "type": self.cfg.type, "chunk_size": getattr(self.cfg, "chunk_size", None),
                "n_action_steps": self.cfg.n_action_steps, "fps": 6}


def selftest(runner, n=60):
    """Feed held-out handover frames (6 Hz) through the policy and compare with the demo actions."""
    from fr3duo.data import load_dataset, SPLITS_FILE
    ep = json.loads(Path(SPLITS_FILE).read_text())["val"][0]
    ds = load_dataset(episodes=[ep])
    runner.reset()
    errs, lat = [], []
    for k in range(n):
        s = ds[k * 5]  # 30 fps -> 6 fps
        obs = {c: (s[c].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in CAMS}
        obs["observation.state"] = s["observation.state"].numpy()
        obs["task"] = s["task"]  # language-conditioned policies (SmolVLA) need the instruction
        out = runner.act(obs)
        errs.append(np.abs(out["action"] - s["action"].numpy()))
        lat.append(out["latency_ms"])
    errs = np.array(errs)
    print(f"held-out handover ep {ep}, first {n} steps (~{n / 6:.0f} s):")
    print(f"  mean |action error| arm joints: {errs[:, [*range(7), *range(8, 15)]].mean():.4f} rad, "
          f"grippers: {errs[:, [7, 15]].mean():.3f}")
    print(f"  latency: median {np.median(lat):.1f} ms, max {np.max(lat):.1f} ms (replans are the slow ones)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_action_steps", type=int, default=None, help="actions executed per prediction")
    ap.add_argument("--temporal_ensemble", type=float, default=None, help="e.g. 0.01: re-plan every step and blend")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    try:  # corporate networks: use the OS certificate store for Hugging Face downloads (tokenizers etc.)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    runner = PolicyRunner(args.checkpoint, args.device, args.n_action_steps, args.temporal_ensemble)
    print("loaded", json.dumps(runner.info()), flush=True)
    if args.selftest:
        selftest(runner)
        return

    with Listener((args.host, args.port), authkey=AUTHKEY) as listener:
        print(f"serving on {args.host}:{args.port}", flush=True)
        while True:
            conn = listener.accept()
            print("client connected", listener.last_accepted, flush=True)
            try:
                while True:
                    msg = conn.recv()
                    cmd = msg.get("cmd")
                    if cmd == "act":
                        conn.send(runner.act(msg["obs"]))
                    elif cmd == "reset":
                        runner.reset()
                        conn.send({"ok": True})
                    elif cmd == "info":
                        conn.send(runner.info())
                    elif cmd == "close":
                        break
            except (EOFError, ConnectionResetError):
                pass
            finally:
                conn.close()
                print("client disconnected", flush=True)


if __name__ == "__main__":
    main()
