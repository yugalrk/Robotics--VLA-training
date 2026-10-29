"""Curriculum training of a VLA (SmolVLA / pi0 / pi0.5) on sub-task-labelled FR3 Duo data.

Teaches simple skills first, then harder ones, then the complete long tasks:

  stage 1  reach_grasp        pick up an object with one arm             ("Pick up the box ...", cube picks)
  stage 2  transport_place    carry / put an object somewhere, one arm   ("Place the box on the serving tray.", cube put/stack)
  stage 3  bimanual           two-arm coordination                       ("Hand the box over to the right gripper.")
  (skills are assigned from the sub-task text, see SKILL_RULES; works with 3- or 5-step handover labels)
  stage 4  full_task          the complete instruction, whole episode    (both datasets)

Each stage samples `new_skill_weight` (default 70%) from the skills it introduces and the rest from earlier
skills (rehearsal, so earlier skills are not forgotten). Stage lengths are fractions of --steps. Change the
plan with --curriculum my_plan.json (same structure as DEFAULT_CURRICULUM) or train without curriculum
(all skills mixed from the start, the baseline to compare against) with --no_curriculum.

Data: the *_subtasks datasets written by data_prep/make_subtasks.py (per-frame `subtask_index` + meta/subtasks.parquet).
- policy runs at 6 Hz: cube data is 6 fps; handover (30 fps) uses every 5th frame, chunks spaced 1/6 s
- sub-task samples: the instruction is the sub-task; actions after the sub-task's end are held at the end pose
  (`--subtask_tail hold`, the robot learns to stop when the step is done) or masked out (`mask`)
- full-task samples: the complete instruction (cube instructions name the colours)
- cube gripper commands are shifted to the frame where the gripper actually starts moving (they lag 2-3
  frames in the recording); disable with --no_gripper_lag_fix
- optional instruction rewording (--paraphrase_prob) so the policy does not depend on one exact sentence
- episodes flagged by data_prep/make_subtasks.py QA (artifacts/subtask_qa/*_qa.json) are excluded
- held out: handover episodes in configs/splits.json["val"], cube episodes with index % 10 == 9

Usage:
    python training/train_curriculum.py --policy smolvla --steps 20000 --batch 32 --out outputs/curriculum_smolvla
    python training/train_curriculum.py --policy pi05 --steps 30000 --batch 32 --bf16 --out outputs/curriculum_pi05
    python training/train_curriculum.py --policy smolvla --no_curriculum --out outputs/mixed_smolvla     # baseline
    python training/train_curriculum.py --data handover ...       # handover only (no cube data)

Outputs in --out: <stage>/ checkpoint at the end of every stage (LeRobot format, works with serving/serve_policy.py),
final/, curriculum_log.json (plan, per-stage sample counts, validation by dataset x sub-task/full), norm stats.
pi0 / pi0.5 need a Hugging Face login with access to google/paligemma-3b-pt-224 (tokenizer).
"""
import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "handover": {"root": ROOT / "Datasets/fr3_handover84_subtasks", "stride": 5, "qa": ROOT / "artifacts/subtask_qa/handover_qa.json"},
    "cubes": {"root": ROOT / "Datasets/fr3_rand500_subtasks", "stride": 1, "qa": ROOT / "artifacts/subtask_qa/cubes_qa.json"},
}
SKILL_RULES = [  # sub-task text -> skill (first match wins); covers 3- and 5-step handover labels and cube labels
    (r"^(Hand the box over|Grasp the box with the right|Release the box)", "bimanual"),
    (r"^Pick up", "reach_grasp"),
    (r"^(Bring|Place|Put|Stack)", "transport_place"),
]


def skill_of(text):
    for pat, sk in SKILL_RULES:
        if re.match(pat, text):
            return sk
    return "transport_place"


SKILLS = ["reach_grasp", "transport_place", "bimanual", "full_task"]
DEFAULT_CURRICULUM = {
    "stages": [
        {"name": "stage1_reach_grasp", "frac": 0.15, "skills": ["reach_grasp"]},
        {"name": "stage2_transport_place", "frac": 0.20, "skills": ["reach_grasp", "transport_place"]},
        {"name": "stage3_bimanual", "frac": 0.20, "skills": ["reach_grasp", "transport_place", "bimanual"]},
        {"name": "stage4_full_task", "frac": 0.45, "skills": SKILLS},
    ],
    "new_skill_weight": 0.7,
}
CAMS = ["observation.images.external_camera", "observation.images.left_wrist_camera",
        "observation.images.right_wrist_camera"]
ARM = [*range(7), *range(8, 15)]
GRIP = [7, 15]
HZ = 6


# ------------------------------------------------------------------ instruction rewording
PARAPHRASES = [  # (regex on the original text, alternative templates using the regex groups)
    (r"^Pick up the box with the left gripper\.$", ["Grasp the box with the left hand.", "Use the left gripper to pick up the box."]),
    (r"^Bring the box to the handover position\.$", ["Carry the box to the handover point.", "Move the box to the middle for the handover."]),
    (r"^Grasp the box with the right gripper\.$", ["Take the box with the right hand.", "Close the right gripper on the box."]),
    (r"^Release the box from the left gripper\.$", ["Let go of the box with the left hand.", "Open the left gripper and release the box."]),
    (r"^Hand the box over to the right gripper\.$", ["Pass the box to the right hand.", "Give the box to the right gripper."]),
    (r"^Place the box on the serving tray\.$", ["Put the box on the tray.", "Set the box down on the serving tray."]),
    (r"^Place the box on the serving tray with the right gripper\.$", ["Put the box on the tray with the right hand.", "Set the box down on the serving tray."]),
    (r"^Hand the box from the left gripper to the right gripper and place it on the serving tray\.$",
     ["Pass the box from the left hand to the right hand, then put it on the tray.",
      "Hand over the box from the left to the right gripper and set it on the serving tray."]),
    (r"^Pick up the (\w+) cube with the (left|right) gripper\.$", ["Grab the {0} cube using the {1} hand.", "Use the {1} gripper to pick up the {0} cube."]),
    (r"^Put the (\w+) cube on (.+)\.$", ["Place the {0} cube on top of {1}.", "Set the {0} cube down on {1}."]),
    (r"^Stack the (\w+) cube on top of the (\w+) cube\.$", ["Put the {0} cube on the {1} cube.", "Place the {0} cube on top of the {1} cube."]),
    (r"^Stack the (\w+) cube on (.+), then put the (\w+) cube on top\.$",
     ["Build a tower: the {0} cube on {1}, then the {2} cube on top.", "Put the {0} cube on {1} and finish with the {2} cube on top."]),
]


def reword(text, rng):
    for pat, alts in PARAPHRASES:
        m = re.match(pat, text)
        if m:
            return rng.choice(alts).format(*m.groups())
    return text


# ------------------------------------------------------------------ data preparation
def fix_gripper_lag(S, A):
    """Move each gripper command flip to the frame where the measured gripper starts moving.
    Arm actions in the cube data are next-frame positions, so the command for a change that is visible at
    frame t+1 belongs to frame t. Returns corrected actions and the list of shifts (frames)."""
    A = A.copy()
    shifts = []
    for g in GRIP:
        cmd = A[:, g]
        flips = np.flatnonzero(np.diff(cmd) != 0) + 1
        for f in flips:
            new_val = cmd[f]
            direction = 1.0 if new_val > cmd[f - 1] else -1.0
            t = f
            while t > max(0, f - 8) and (S[t, g] - S[t - 1, g]) * direction > 0.02:  # real motion is 0.03-0.08/frame; slower drift is the object settling
                t -= 1
            if t < f:
                A[t:f, g] = new_val
                shifts.append(int(f - t))
    return A, shifts


def load_tables(name, split_eps, args):
    cfg = DATASETS[name]
    root = cfg["root"]
    files = sorted(glob.glob(str(root / "data/*/*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["index", "episode_index", "frame_index", "observation.state",
                                                "action", "subtask_index", "task_index"]) for f in files])
    df = df.sort_values("index").reset_index(drop=True)
    sub_txt = pd.read_parquet(root / "meta/subtasks.parquet")
    sub_txt = {int(i): t for t, i in zip(sub_txt.index, sub_txt.subtask_index)}
    task_txt = pd.read_parquet(root / "meta/tasks.parquet")
    task_txt = {int(i): t for t, i in zip(task_txt.index, task_txt.task_index)}
    flagged = set()
    if cfg["qa"].exists():
        flagged = {int(k) for k in json.loads(cfg["qa"].read_text())["problems"]}
    S = np.stack(df["observation.state"]).astype(np.float32)
    A = np.stack(df["action"]).astype(np.float32)
    n = len(df)
    ep = df.episode_index.values
    fi = df.frame_index.values
    sub = df.subtask_index.values
    ep_start = np.zeros(n, np.int64)
    ep_end = np.zeros(n, np.int64)
    sub_end = np.zeros(n, np.int64)
    sub_order = np.zeros(n, np.int64)
    shifts = []
    for e, g in df.groupby("episode_index").indices.items():
        g = np.sort(g)
        ep_start[g], ep_end[g] = g[0], g[-1] + 1
        runs = np.flatnonzero(np.r_[True, np.diff(sub[g]) != 0])  # start of each sub-task run
        ends = np.r_[runs[1:], len(g)]
        for k, (s, t) in enumerate(zip(runs, ends)):
            sub_order[g[s:t]] = k
            sub_end[g[s:t]] = g[0] + t
        if name == "cubes" and not args.no_gripper_lag_fix:
            A[g], sh = fix_gripper_lag(S[g], A[g])
            shifts += sh
    keep_ep = np.isin(ep, list(split_eps)) & ~np.isin(ep, list(flagged))
    on_grid = fi % cfg["stride"] == 0
    rows = np.flatnonzero(keep_ep & on_grid)
    skill = np.array([skill_of(sub_txt[int(sub[r])]) for r in rows])
    tab = {"name": name, "root": root, "stride": cfg["stride"], "S": S, "A": A, "ep": ep, "abs_index": df["index"].values,
           "ep_start": ep_start, "ep_end": ep_end, "sub_end": sub_end, "sub_text": [sub_txt[int(i)] for i in sub],
           "full_text": [task_txt[int(i)] for i in df.task_index.values], "rows": rows, "skill": skill,
           "flagged": sorted(flagged), "gripper_shifts": shifts}
    return tab


class CurriculumData(Dataset):
    """Items are (table, row, mode); mode 'sub' = sub-task instruction, 'full' = complete instruction."""

    def __init__(self, tables, chunk, args, seed=0):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.tables, self.chunk, self.args = tables, chunk, args
        self.lr = [LeRobotDataset(f"local/{t['root'].name}", root=t["root"], video_backend="pyav") for t in tables]
        items, groups = [], []
        for ti, t in enumerate(tables):
            for r, sk in zip(t["rows"], t["skill"]):
                items.append((ti, r, 0))
                groups.append(f"{t['name']}:{sk}")
                items.append((ti, r, 1))
                groups.append(f"{t['name']}:full_task")
        self.items = np.array(items, np.int64)
        self.groups = np.array(groups)
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def weights(self, skills, new_skills, new_weight):
        """Per-item sampling weights for a stage: groups of new skills share new_weight, earlier skills the rest,
        each (dataset, skill) group gets an equal share of its part."""
        old = [s for s in skills if s not in new_skills]
        if new_skills and old:
            parts = [(new_skills, new_weight), (old, 1.0 - new_weight)]
        else:  # first stage, or a stage that adds nothing new: all its skills equally
            parts = [(skills, 1.0)]
        w = np.zeros(len(self.items))
        present = np.unique(self.groups)
        for sel, part in parts:
            gs = [g for g in present if g.split(":")[1] in sel]
            for g in gs:
                m = self.groups == g
                w[m] = part / len(gs) / m.sum()
        return w

    def __getitem__(self, i):
        ti, r, mode = self.items[i]
        t = self.tables[ti]
        rng = np.random.default_rng()
        item = self.lr[ti][int(t["abs_index"][r])]
        idx = r + t["stride"] * np.arange(self.chunk)
        if mode == 0:  # sub-task: hold (or mask) after the sub-task end
            last = t["sub_end"][r] - 1
            pad = np.zeros(self.chunk, bool) if self.args.subtask_tail == "hold" else idx > last
            idx = np.minimum(idx, last)
            text = t["sub_text"][r]
        else:  # full task: pad after the episode end
            last = t["ep_end"][r] - 1
            pad = idx > last
            idx = np.minimum(idx, last)
            text = t["full_text"][r]
        if self.args.paraphrase_prob > 0 and rng.random() < self.args.paraphrase_prob:
            text = reword(text, rng)
        out = {c: item[c] for c in CAMS}
        out["observation.state"] = torch.from_numpy(t["S"][r])
        out["action"] = torch.from_numpy(t["A"][idx])
        out["action_is_pad"] = torch.from_numpy(pad)
        out["task"] = text
        return out


# ------------------------------------------------------------------ stats + policy
def norm_stats(tables):
    """Equal-weight pooled stats over the training rows of each dataset (mean/std/min/max + quantiles for pi0.5)."""
    rng = np.random.default_rng(0)
    out = {}
    for key, arr in [("observation.state", "S"), ("action", "A")]:
        n = min(len(t["rows"]) for t in tables)
        pooled = np.concatenate([t[arr][rng.choice(t["rows"], n, replace=False)] for t in tables]).astype(np.float64)
        st = {"mean": pooled.mean(0), "std": np.maximum(pooled.std(0), 1e-6), "min": pooled.min(0), "max": pooled.max(0)}
        for q in [1, 10, 50, 90, 99]:
            st[f"q{q:02d}"] = np.percentile(pooled, q, axis=0)
        out[key] = {k: torch.tensor(v, dtype=torch.float32) for k, v in st.items()}
    for c in CAMS:
        out[c] = {"mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)}
    return out


BASES = {"smolvla": "lerobot/smolvla_base", "pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_base"}


# ------------------------------------------------------------------ LoRA
class LoRALinear(torch.nn.Module):
    """y = base(x) + (alpha / r) * B(A(dropout(x))); base frozen, B starts at zero (no change at step 0)."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base, self.scale = base, alpha / r
        for p in base.parameters():
            p.requires_grad = False
        dev = base.weight.device
        self.lora_A = torch.nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.drop = torch.nn.Dropout(dropout)

    @property
    def weight(self):  # SmolVLA reads q_proj.weight.dtype
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        A, B = self.lora_A.to(x.dtype), self.lora_B.to(x.dtype)  # bf16 VLM weights at inference; autocast in training
        return y + (self.scale * (self.drop(x) @ A.t() @ B.t())).to(y.dtype)

    def delta(self):
        return (self.scale * self.lora_B @ self.lora_A).to(self.base.weight.dtype)


def lora_scope(policy, args):
    """Module whose Linear layers get LoRA: the VLM (vision encoder, connector, language layers) or everything."""
    if args.policy != "smolvla":
        raise SystemExit("--lora_rank is implemented for --policy smolvla only")
    vwe = policy.model.vlm_with_expert
    return [vwe.vlm] if args.lora_targets == "vlm" else [vwe.vlm, vwe.lm_expert]


def add_lora(policy, args):
    """Wrap Linear layers in LoRA. With targets 'vlm' the VLM base weights are frozen and only its adapters
    train, the action expert and projections train fully; with 'all' the expert is LoRA too."""
    for scope in lora_scope(policy, args):
        for p in scope.parameters():
            p.requires_grad = False
        for name, mod in list(scope.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear) and cname != "lm_head":
                    setattr(mod, cname, LoRALinear(child, args.lora_rank, args.lora_alpha or 2 * args.lora_rank,
                                                   args.lora_dropout))


def lora_modules(policy):
    """(parent, child name, LoRALinear) for every adapter."""
    return [(mod, cname, child) for mod in policy.modules() for cname, child in mod.named_children()
            if isinstance(child, LoRALinear)]


class merged_lora:
    """Context: LoRA folded into plain Linear layers (a standard checkpoint for save_pretrained), restored after."""

    def __init__(self, policy):
        self.policy = policy

    def __enter__(self):
        self.saved = []
        with torch.no_grad():
            for mod, cname, lo in lora_modules(self.policy):
                self.saved.append((mod, cname, lo, lo.base.weight.detach().cpu().clone()))
                lo.base.weight += lo.delta()
                setattr(mod, cname, lo.base)

    def __exit__(self, *exc):
        with torch.no_grad():
            for mod, cname, lo, w in self.saved:
                lo.base.weight.copy_(w.to(lo.base.weight.device))
                setattr(mod, cname, lo)


def build_policy(args, stats):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    base = args.base or BASES[args.policy]
    cfg = PreTrainedConfig.from_pretrained(base)
    cfg.input_features = {**{c: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for c in CAMS},
                          "observation.state": PolicyFeature(FeatureType.STATE, (16,))}
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (16,))}
    cfg.chunk_size, cfg.n_action_steps = args.chunk, args.chunk
    cfg.device, cfg.push_to_hub = "cuda", False
    cfg.scheduler_warmup_steps, cfg.scheduler_decay_steps = max(1, args.steps // 20), args.steps
    if args.lr:
        cfg.optimizer_lr = args.lr
    for kv in args.override:  # e.g. --override dtype=bfloat16 gradient_checkpointing=true train_expert_only=true
        k, v = kv.split("=", 1)
        setattr(cfg, k, json.loads(v) if v.lower() in ("true", "false") or v.replace(".", "").replace("-", "").isdigit() else v)
    if args.lora_rank:  # the VLM stays in train mode (adapter dropout); add_lora freezes its base weights
        cfg.train_expert_only, cfg.freeze_vision_encoder = False, False
    policy = get_policy_class(cfg.type).from_pretrained(base, config=cfg).to("cuda")
    if args.lora_rank:
        add_lora(policy, args)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    optim = cfg.get_optimizer_preset().build(policy.get_optim_params())
    sched = cfg.get_scheduler_preset().build(optim, num_training_steps=args.steps) if cfg.get_scheduler_preset() else None
    return policy, pre, post, optim, sched, cfg


# ------------------------------------------------------------------ validation
@torch.no_grad()
def validate(policy, pre, post, val_sets):
    """Error of the chunk predicted at inference time (no access to the demo actions), per dataset x mode:
    masked L1 in normalized units, and mean arm-joint error in rad (via the policy's own unnormalizer)."""
    policy.eval()
    res = {}
    for name, batches in val_sets.items():
        e_norm, e_rad = [], []
        for b in batches:
            raw_act = b["action"].clone()
            bb = pre({k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()})
            pred = policy.predict_action_chunk(bb)
            valid = ~bb["action_is_pad"]
            e_norm.append(((pred - bb["action"]).abs() * valid.unsqueeze(-1)).sum().item()
                          / (valid.sum().item() * pred.shape[-1]))
            pr = post(pred).float().cpu()
            e_rad.append((pr - raw_act).abs()[..., ARM][valid.cpu()].mean().item())
        res[name] = {"l1_norm": round(float(np.mean(e_norm)), 4), "arm_rad": round(float(np.mean(e_rad)), 4)}
    policy.train()
    return res


def val_batches(val_data, batch, n_batches, seed=0):
    """Fixed validation batches per dataset x mode (sub / full)."""
    rng = np.random.default_rng(seed)
    out = {}
    for ti, t in enumerate(val_data.tables):
        for mode, label in [(0, "sub"), (1, "full")]:
            idx = np.flatnonzero((val_data.items[:, 0] == ti) & (val_data.items[:, 2] == mode))
            pick = rng.choice(idx, min(len(idx), batch * n_batches), replace=False)
            dl = DataLoader(torch.utils.data.Subset(val_data, pick.tolist()), batch_size=batch, num_workers=0)
            out[f"{t['name']}/{label}"] = list(dl)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=list(BASES), default="smolvla")
    ap.add_argument("--base", default=None, help="override the pretrained checkpoint (HF repo id or local dir)")
    ap.add_argument("--data", choices=["both", "handover", "cubes"], default="both")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step (effective batch = batch * grad_accum)")
    ap.add_argument("--chunk", type=int, default=20, help="actions per prediction (20 @ 6 Hz = 3.3 s)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--curriculum", default=None, help="JSON file with a custom plan (see DEFAULT_CURRICULUM)")
    ap.add_argument("--no_curriculum", action="store_true", help="baseline: all skills mixed for the whole run")
    ap.add_argument("--subtask_tail", choices=["hold", "mask"], default="hold")
    ap.add_argument("--paraphrase_prob", type=float, default=0.3)
    ap.add_argument("--no_gripper_lag_fix", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (Ampere+ GPUs)")
    ap.add_argument("--override", nargs="*", default=[], help="policy config overrides key=value")
    ap.add_argument("--lora_rank", type=int, default=0, help="LoRA rank for SmolVLA (0 = off); checkpoints are saved merged")
    ap.add_argument("--lora_alpha", type=float, default=None, help="LoRA scale numerator (default 2 * rank)")
    ap.add_argument("--lora_dropout", type=float, default=0.0, help="> 0 keeps a dropout copy of every adapted input (memory)")
    ap.add_argument("--lora_targets", choices=["vlm", "all"], default="vlm",
                    help="vlm: adapters on the VLM, action expert trained fully; all: adapters on the expert too")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500, help="write <out>/resume_state.pt every N steps (0 = off)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/resume_state.pt if it exists")
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_batches", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/curriculum")
    args = ap.parse_args()

    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    torch.manual_seed(0)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    plan = json.loads(Path(args.curriculum).read_text()) if args.curriculum else DEFAULT_CURRICULUM
    if args.no_curriculum:
        plan = {"stages": [{"name": "all_skills_mixed", "frac": 1.0, "skills": SKILLS}], "new_skill_weight": 1.0}
    fr = np.array([s["frac"] for s in plan["stages"]])
    stage_steps = np.floor(fr / fr.sum() * args.steps).astype(int)
    stage_steps[-1] = args.steps - stage_steps[:-1].sum()

    splits = json.loads((ROOT / "configs" / "splits.json").read_text())
    names = ["handover", "cubes"] if args.data == "both" else [args.data]
    eps = {}
    for n in names:
        all_eps = sorted(pd.read_parquet(glob.glob(str(DATASETS[n]["root"] / "meta/episodes/*/*.parquet"))[0]).episode_index)
        val = set(splits["val"]) if n == "handover" else {e for e in all_eps if e % 10 == 9}
        eps[n] = ([e for e in all_eps if e not in val], sorted(val))
    print("loading tables ...", flush=True)
    train_tabs = [load_tables(n, eps[n][0], args) for n in names]
    val_tabs = [load_tables(n, eps[n][1], args) for n in names]
    for t in train_tabs:
        sh = np.array(t["gripper_shifts"])
        print(f"  {t['name']}: {len(t['rows'])} train frames @6 Hz, excluded flagged eps {t['flagged']}"
              + (f", gripper flips shifted earlier: {len(sh)} (by {np.bincount(sh).nonzero()[0].tolist()} frames)" if len(sh) else ""),
              flush=True)

    train = CurriculumData(train_tabs, args.chunk, args)
    val = CurriculumData(val_tabs, args.chunk, argparse.Namespace(**{**vars(args), "paraphrase_prob": 0.0}))
    vsets = val_batches(val, args.batch, args.val_batches)
    stats = norm_stats(train_tabs)
    torch.save(stats, out / "norm_stats.pt")
    policy, pre, post, optim, sched, cfg = build_policy(args, stats)
    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"{args.policy}: {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M params, {n_tr / 1e6:.0f}M trainable", flush=True)

    log = {"args": vars(args), "plan": plan, "stage_steps": stage_steps.tolist(), "stages": [], "val": []}

    def save(tag):
        d = out / tag
        if args.lora_rank:
            d.mkdir(parents=True, exist_ok=True)
            torch.save({n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad},
                       d / "trainable_params.pt")  # adapters + fully trained parts, to resume or inspect
            with merged_lora(policy):
                policy.save_pretrained(d)
        else:
            policy.save_pretrained(d)
        pre.save_pretrained(d)
        post.save_pretrained(d)
        (out / "curriculum_log.json").write_text(json.dumps(log, indent=1))
        print(f"saved {d}", flush=True)

    resume_file = out / "resume_state.pt"

    def save_resume():
        """Trainable weights + optimizer/scheduler + step, written atomically (a crash mid-write keeps the old one)."""
        tmp = resume_file.with_suffix(".tmp")
        torch.save({"step": step, "log": log, "optim": optim.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "params": {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad}}, tmp)
        tmp.replace(resume_file)
        print(f"resume state saved at step {step}", flush=True)

    step = 0
    if args.resume and resume_file.exists():
        st = torch.load(resume_file, map_location="cpu", weights_only=False)
        unexpected = policy.load_state_dict(st["params"], strict=False).unexpected_keys  # frozen weights are not stored
        assert not unexpected, f"resume state does not match the model: {unexpected[:5]}"
        optim.load_state_dict(st["optim"])
        if sched is not None and st["sched"] is not None:
            sched.load_state_dict(st["sched"])
        step, log = st["step"], st["log"]
        del st
        print(f"resumed from {resume_file} at step {step}", flush=True)
    elif args.resume:
        print(f"--resume: no {resume_file}, starting from scratch", flush=True)
    step0, t0 = step, time.time()
    stage_end = np.cumsum(stage_steps)
    seen = []
    for si, (stage, n_steps) in enumerate(zip(plan["stages"], stage_steps)):
        new = [s for s in stage["skills"] if s not in seen]
        seen += new
        if step >= stage_end[si]:  # finished before the resume (its checkpoint is already saved)
            continue
        w = train.weights(stage["skills"], new, plan.get("new_skill_weight", 0.7))
        mix = {str(g): round(float(w[train.groups == g].sum()), 3) for g in np.unique(train.groups) if w[train.groups == g].sum() > 0}
        print(f"\n=== {stage['name']}: {n_steps} steps, new skills {new}, sample mix {mix}", flush=True)
        if not any(s["name"] == stage["name"] for s in log["stages"]):
            log["stages"].append({"name": stage["name"], "steps": int(n_steps), "new_skills": new, "sample_mix": mix})
        todo = int(stage_end[si] - step)
        sampler = WeightedRandomSampler(torch.tensor(w), num_samples=todo * args.batch * args.grad_accum,
                                        replacement=True)
        dl = DataLoader(train, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=False)
        policy.train()
        optim.zero_grad(set_to_none=True)
        for micro, batch in enumerate(dl, 1):
            batch = pre(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                loss, _ = policy.forward(batch)
            (loss / args.grad_accum).backward()
            if micro % args.grad_accum:
                continue
            step += 1
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()
            if step % args.log_every == 0:
                rate = (step - step0) / (time.time() - t0)
                print(f"[{stage['name']}] step {step}/{args.steps} loss {loss.item():.4f} grad {gn:.2f} "
                      f"lr {optim.param_groups[0]['lr']:.1e} | {rate:.2f} it/s, eta {(args.steps - step) / rate / 60:.0f} min, "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            if step % args.val_every == 0:
                v = validate(policy, pre, post, vsets)
                log["val"].append({"step": step, "stage": stage["name"], **v})
                print(f"VAL {step}: " + " | ".join(f"{k} {x['l1_norm']} ({x['arm_rad']} rad)" for k, x in v.items()), flush=True)
            if args.save_every and step % args.save_every == 0 and step < stage_end[si]:
                save_resume()
        v = validate(policy, pre, post, vsets)
        log["val"].append({"step": step, "stage": stage["name"], "end_of_stage": True, **v})
        print(f"VAL end of {stage['name']}: " + " | ".join(f"{k} {x['l1_norm']} ({x['arm_rad']} rad)" for k, x in v.items()), flush=True)
        save(stage["name"])
        if args.save_every:
            save_resume()
    save("final")
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
