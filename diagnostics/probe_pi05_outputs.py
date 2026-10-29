"""Does the checkpoint react to its input? Compare predictions across frames and with the norm-stat midpoint."""
import json
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
ARM = [*range(7), *range(8, 15)]

meta = json.loads((CKPT / "train_state/_METADATA").read_text())
steps = [k for k in meta.get("tree_metadata", {}) if "step" in k]
print("train_state entries with 'step':", steps[:5])

cfg = pi0_config.Pi0Config(pi05=True, action_horizon=50)
params = _model.restore_params(CKPT / "params", dtype=jnp.bfloat16)
flat = jax.tree_util.tree_leaves_with_path(params)
print("param leaves:", len(flat), "| total params (M):", round(sum(int(np.prod(v.shape)) for _, v in flat) / 1e6))
names = sorted({"/".join(str(getattr(k, "key", k)) for k in path[:3]) for path, _ in flat})
print("top-level groups:", names[:20])
model = cfg.load(params)
norm = _checkpoints.load_norm_stats(CKPT / "assets", "fr3_duo")


class Inputs(transforms.DataTransformFn):
    def __call__(self, d):
        img = {"base_0_rgb": d["ext"], "left_wrist_0_rgb": d["lw"], "right_wrist_0_rgb": d["rw"]}
        return {"state": d["state"], "image": img, "image_mask": {k: np.True_ for k in img}, "prompt": d["prompt"]}


pol = _policy.Policy(model, transforms=[
    Inputs(), transforms.Normalize(norm, use_quantiles=True), transforms.ResizeImages(224, 224),
    transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(cfg.max_token_len), discrete_state_input=True),
    transforms.PadStatesAndActions(32)], output_transforms=[transforms.Unnormalize(norm, use_quantiles=True)])
mid = (np.array(norm["actions"].q01) + np.array(norm["actions"].q99)) / 2
preds = []
for i in range(len(F["state"])):
    out = pol.infer({"state": F["state"][i], "ext": F["ext"][i], "lw": F["lw"][i], "rw": F["rw"][i],
                     "prompt": "Hand the box from the left gripper to the right gripper and place it on the serving tray."})
    a = np.asarray(out["actions"])[:25, :16]
    preds.append(a[0])
    if i in (0, 2, 4):
        print(f"frame {tuple(F['where'][i])}: pred[0] L-arm {np.round(a[0, :7], 2)} | recorded {np.round(F['actions'][i][0, :7], 2)} "
              f"| state {np.round(F['state'][i][:7], 2)}")
preds = np.array(preds)
print("spread of first predicted action across 12 different frames (std, rad):", np.round(preds[:, ARM].std(0).mean(), 4),
      "| spread of the recorded ones:", np.round(F["actions"][:, 0][:, ARM].std(0).mean(), 4))
print("mean |pred - quantile midpoint|:", np.round(np.abs(preds[:, ARM] - mid[ARM]).mean(), 4),
      "| mean |recorded - midpoint|:", np.round(np.abs(F["actions"][:, 0][:, ARM] - mid[ARM]).mean(), 4))
