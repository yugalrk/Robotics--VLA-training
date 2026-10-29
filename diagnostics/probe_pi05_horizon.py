"""Find the input setup the organisers' pi0.5 checkpoint was trained with: score transform variants against
recorded handover actions (eval_results/pi05_probe_frames.npz). Runs in WSL in the openpi venv."""
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.training import checkpoints as _checkpoints

P = pathlib.Path("/mnt/c/Users/User/Desktop/Hack2026/C2_Teach-the-robot_VLA")
CKPT = P / "trained checkpoint/fr3-duo-checkpoint-6a5f7c87-e14a-4dd1-a047-dc5659fa8327/checkpoint"
F = np.load(P / "eval_results/pi05_probe_frames.npz")
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
ARM = [*range(7), *range(8, 15)]
N = 25

cfg = pi0_config.Pi0Config(pi05=True, action_horizon=50)
model = cfg.load(_model.restore_params(CKPT / "params", dtype=jnp.bfloat16))
norm = _checkpoints.load_norm_stats(CKPT / "assets", "fr3_duo")
noise = np.asarray(jax.random.normal(jax.random.key(0), (50, 32)), np.float32)


class Inputs(transforms.DataTransformFn):
    def __init__(self, order):
        self.order = order

    def __call__(self, d):
        img = {"base_0_rgb": d[self.order[0]], "left_wrist_0_rgb": d[self.order[1]], "right_wrist_0_rgb": d[self.order[2]]}
        return {"state": d["state"], "image": img, "image_mask": {k: np.True_ for k in img}, "prompt": d["prompt"]}


class Outputs(transforms.DataTransformFn):
    def __call__(self, d):
        return {"actions": np.asarray(d["actions"][:, :16])}


def policy(quantile=True, discrete=True, order=("ext", "lw", "rw")):
    return _policy.Policy(model, transforms=[
        Inputs(order), transforms.Normalize(norm, use_quantiles=quantile), transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(cfg.max_token_len), discrete_state_input=discrete),
        transforms.PadStatesAndActions(32)],
        output_transforms=[transforms.Unnormalize(norm, use_quantiles=quantile), Outputs()])


def score(name, pol, prompt=TASK):
    abs_err, delta_err, hold = [], [], []
    for i in range(len(F["state"])):
        obs = {"state": F["state"][i], "ext": F["ext"][i], "lw": F["lw"][i], "rw": F["rw"][i], "prompt": prompt}
        a = pol.infer(obs, noise=noise)["actions"][:N]
        ref = F["actions"][i][:N]
        abs_err.append(np.abs(a[:, ARM] - ref[:, ARM]).mean())
        delta_err.append(np.abs((F["state"][i] + a)[:, ARM] - ref[:, ARM]).mean())
        hold.append(np.abs(F["state"][i][ARM] - ref[:, ARM]).mean())
    print(f"{name:42s} absolute {np.mean(abs_err):.4f} | as delta {np.mean(delta_err):.4f} | hold-pose {np.mean(hold):.4f}", flush=True)


for H in (10, 16, 20, 25, 30, 50):
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=H)
    model = cfg.load(_model.restore_params(CKPT / "params", dtype=jnp.bfloat16))
    noise = np.asarray(jax.random.normal(jax.random.key(0), (H, 32)), np.float32)
    N = min(H, 10)
    for q in (True, False):
        score(f"horizon {H}, {'quantile' if q else 'z-score'} (first {N})", policy(quantile=q))