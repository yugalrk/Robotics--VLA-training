"""Serve an openpi pi0.5 checkpoint (JAX / Orbax) to the FR3 Duo Isaac Sim scripts. Runs in WSL, in the openpi venv.

    cd ~/openpi
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \\
        /mnt/c/Users/User/Desktop/Hack2026/C2_Teach-the-robot_VLA/serving/fr3duo_pi05_server.py \\
        --checkpoint "/mnt/c/Users/User/Desktop/Hack2026/C2_Teach-the-robot_VLA/trained checkpoint/fr3-duo-checkpoint-6a5f7c87-e14a-4dd1-a047-dc5659fa8327/checkpoint"

Same HTTP protocol as fr3duo_policy_server.py (GET /info, POST /reset, POST /act with an .npz observation), so
fr3duo_isaac_client.py and fr3duo_isaac_eval.py work unchanged. policy_type is "pi05-openpi"; the sim scripts
play its actions at 30 Hz (every control step), the rate of the handover dataset it was trained on.

The organisers' openpi training config is not part of the checkpoint, so the policy is rebuilt from openpi's
standard pi0.5 pieces: Pi0Config(pi05=True), quantile normalization with the checkpoint's assets/fr3_duo norm
stats, 224x224 images, prompt tokenized with the discretized state, state/actions padded to 32 dims. Camera
mapping: external -> base_0_rgb, left wrist -> left_wrist_0_rgb, right wrist -> right_wrist_0_rgb.
Each inference predicts --horizon actions; the first --replan of them are served one per /act call.
"""
import argparse
import io
import json
import pathlib
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

STATE_DIM = ACTION_DIM = 16
CAMERAS = {  # FR3 Duo feature -> pi0.5 image slot
    "observation.images.external_camera": "base_0_rgb",
    "observation.images.left_wrist_camera": "left_wrist_0_rgb",
    "observation.images.right_wrist_camera": "right_wrist_0_rgb",
}
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."


def build_policy(checkpoint, horizon):
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as _model
    from openpi.models import pi0_config
    from openpi.models import tokenizer as _tokenizer
    from openpi.policies import policy as _policy
    from openpi.training import checkpoints as _checkpoints

    class FR3Inputs(transforms.DataTransformFn):
        def __call__(self, data):
            img = {slot: np.asarray(data[key], np.uint8) for key, slot in CAMERAS.items()}
            out = {"state": np.asarray(data["observation.state"], np.float32),
                   "image": img, "image_mask": {slot: np.True_ for slot in img}}
            if "prompt" in data:
                out["prompt"] = data["prompt"]
            return out

    class FR3Outputs(transforms.DataTransformFn):
        def __call__(self, data):
            return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}

    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=horizon)
    ckpt = pathlib.Path(checkpoint)
    model = cfg.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))
    norm_stats = _checkpoints.load_norm_stats(ckpt / "assets", "fr3_duo")
    return _policy.Policy(
        model,
        transforms=[FR3Inputs(), transforms.Normalize(norm_stats, use_quantiles=True),
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(cfg.max_token_len),
                                              discrete_state_input=cfg.discrete_state_input),
                    transforms.PadStatesAndActions(cfg.action_dim)],
        output_transforms=[transforms.Unnormalize(norm_stats, use_quantiles=True), FR3Outputs()],
    )


class ChunkedPolicy:
    fake = False
    policy_type = "pi05-openpi"
    input_features = {"observation.state": [STATE_DIM]} | {k: [3, 224, 224] for k in CAMERAS}
    output_features = {"action": [ACTION_DIM]}

    def __init__(self, checkpoint, horizon, replan):
        self.checkpoint, self.replan = checkpoint, replan
        self.policy = build_policy(checkpoint, horizon)
        self.queue = []

    def reset(self):
        self.queue = []

    def act(self, obs, task):
        if not self.queue:
            out = self.policy.infer({**obs, "prompt": task})
            self.queue = list(out["actions"][: self.replan])
        return np.asarray(self.queue.pop(0), np.float32)


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
            self._reply(200, {"checkpoint": policy.checkpoint, "policy_type": policy.policy_type,
                              "input_features": policy.input_features, "output_features": policy.output_features,
                              "fake": policy.fake})

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
                task = str(arrays.pop("task", TASK))
                missing = [k for k in policy.input_features if k not in arrays]
                if missing:
                    return self._reply(400, {"error": f"observation is missing {missing}"})
                start = time.perf_counter()
                action = policy.act({k: arrays[k] for k in policy.input_features}, task)
                self._reply(200, {"action": action.tolist(), "latency_ms": (time.perf_counter() - start) * 1000,
                                  "fake": False})
            except Exception as exc:
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="openpi checkpoint folder (has params/ and assets/fr3_duo/)")
    ap.add_argument("--horizon", type=int, default=50, help="actions per inference (pi0.5 default 50)")
    ap.add_argument("--replan", type=int, default=25, help="actions served from each chunk before re-inferring")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    policy = ChunkedPolicy(args.checkpoint, args.horizon, args.replan)
    print(f"pi0.5 from {args.checkpoint} | horizon {args.horizon}, replan every {args.replan} actions (30 Hz)", flush=True)
    print(f"serving on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    HTTPServer((args.host, args.port), make_handler(policy)).serve_forever()


if __name__ == "__main__":
    main()
