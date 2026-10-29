"""Short ACT training run on BOTH datasets (handover 84 eps + cube stacking 490 eps).

- Both are used at 6 Hz: rand500 is recorded at 6 fps; the 30 fps handover data is
  sampled every 5th frame and its action chunks are spaced 1/6 s apart.
- Batches are drawn 50/50 from the two datasets (handover would otherwise be ~18% of samples).
- Normalization stats for state/action are recomputed from the training episodes of both
  sets (rand500's own stats.json only covers 41 of its 490 episodes). Cameras use ImageNet stats.
- Held out: handover val split from configs/splits.json, rand500 every 10th episode.

Usage:
    python training/train_act_mixed.py --steps 6000 --out outputs/act_mixed
Checkpoints are written to <out>/step_XXXXXX (LeRobot format: policy + pre/post processors).
"""
import argparse
import glob
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.data import DATASET_ROOT as HANDOVER_ROOT, SPLITS_FILE

ROOT = Path(__file__).resolve().parents[1]
CUBES_ROOT = ROOT / "Datasets/fr3_duo_rand500/fr3_duo_rand500"
FPS = 6
CAMS = ["observation.images.external_camera", "observation.images.left_wrist_camera",
        "observation.images.right_wrist_camera"]
KEEP = CAMS + ["observation.state", "action", "action_is_pad", "task"]  # task text: used by SmolVLA, ignored by ACT
IMAGENET = {"mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)}


def splits():
    h = json.loads(SPLITS_FILE.read_text())
    cube_eps = sorted(pd.read_parquet(glob.glob(str(CUBES_ROOT / "meta/episodes/*/*.parquet"))[0]).episode_index)
    return {
        "handover_train": h["train"], "handover_val": h["val"],
        "cubes_train": [e for e in cube_eps if e % 10 != 9], "cubes_val": [e for e in cube_eps if e % 10 == 9],
    }


class Keep(Dataset):
    """Keep only the keys both datasets share (and that ACT uses)."""

    def __init__(self, ds, indices=None):
        self.ds, self.idx = ds, indices

    def __len__(self):
        return len(self.idx) if self.idx is not None else len(self.ds)

    def __getitem__(self, i):
        item = self.ds[self.idx[i] if self.idx is not None else i]
        return {k: item[k] for k in KEEP}


def load(root, repo_id, episodes, chunk):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id, root=root, episodes=episodes, video_backend="pyav",
                          delta_timestamps={"action": [i / FPS for i in range(chunk)]})


def build_sets(sp, chunk, split):
    h = load(HANDOVER_ROOT, "local/fr3_handover84", sp[f"handover_{split}"], chunk)
    fi = np.asarray(h.hf_dataset.with_format(None)["frame_index"])
    h = Keep(h, np.flatnonzero(fi % 5 == 0))  # 30 fps -> 6 fps
    c = Keep(load(CUBES_ROOT, "local/fr3_rand500", sp[f"cubes_{split}"], chunk))
    return h, c


def vec_stats(sp, data="both"):
    """Equal-weight pooled stats of state/action over the training episodes of the datasets used."""
    def read(root, eps, step):
        df = pd.concat([pd.read_parquet(f, columns=["observation.state", "action", "episode_index", "frame_index"])
                        for f in sorted(glob.glob(str(root / "data/*/*.parquet")))])
        df = df[df.episode_index.isin(eps) & (df.frame_index % step == 0)]
        return {k: np.stack(df[k]).astype(np.float64) for k in ["observation.state", "action"]}

    parts = [read(HANDOVER_ROOT, sp["handover_train"], 5)]
    if data == "both":
        parts.append(read(CUBES_ROOT, sp["cubes_train"], 1))
    out = {}
    for k in ["observation.state", "action"]:
        mean = np.mean([p[k].mean(0) for p in parts], 0)
        sq = np.mean([(p[k] ** 2).mean(0) for p in parts], 0)
        std = np.sqrt(np.maximum(sq - mean ** 2, 1e-8))
        out[k] = {"mean": torch.tensor(mean, dtype=torch.float32), "std": torch.tensor(std, dtype=torch.float32),
                  "min": torch.tensor(np.min([p[k].min(0) for p in parts], 0), dtype=torch.float32),
                  "max": torch.tensor(np.max([p[k].max(0) for p in parts], 0), dtype=torch.float32)}
    for cam in CAMS:
        out[cam] = {k: v.clone() for k, v in IMAGENET.items()}
    return out


@torch.no_grad()
def val_l1(policy, preprocessor, loader, n_batches):
    """Mean L1 error (normalized units) of the action chunk predicted at inference time
    (no access to the ground-truth actions) vs the demonstration."""
    policy.eval()
    errs = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = preprocessor(batch)
        pred = policy.predict_action_chunk(batch)
        valid = ~batch["action_is_pad"].unsqueeze(-1)
        errs.append(((pred - batch["action"]).abs() * valid).sum().item() / (valid.sum().item() * pred.shape[-1]))
    policy.train()
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20, help="actions per prediction (20 @ 6 Hz = 3.3 s)")
    ap.add_argument("--save_every", type=int, default=3000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/act_mixed")
    ap.add_argument("--data", choices=["both", "handover"], default="both",
                    help="'handover' = baseline without the cube data (val still reports both)")
    ap.add_argument("--policy", choices=["act", "smolvla"], default="act",
                    help="smolvla = fine-tune pretrained lerobot/smolvla_base (VLM frozen, action expert trained)")
    args = ap.parse_args()

    import truststore
    truststore.inject_into_ssl()
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(0)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    sp = splits()
    (out / "splits.json").write_text(json.dumps(sp))

    h_train, c_train = build_sets(sp, args.chunk, "train")
    h_val, c_val = build_sets(sp, args.chunk, "val")
    print(f"train samples: handover {len(h_train)}, cubes {len(c_train)} | val: {len(h_val)}, {len(c_val)}", flush=True)

    if args.data == "both":
        train = ConcatDataset([h_train, c_train])
        w = np.r_[np.full(len(h_train), 0.5 / len(h_train)), np.full(len(c_train), 0.5 / len(c_train))]
    else:
        train, w = h_train, np.full(len(h_train), 1.0 / len(h_train))
    sampler = WeightedRandomSampler(torch.tensor(w), num_samples=args.steps * args.batch, replacement=True)
    dl = DataLoader(train, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loaders = {name: DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                                    generator=torch.Generator().manual_seed(0))
                   for name, ds in [("handover", h_val), ("cubes", c_val)]}

    in_feats = {**{c: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for c in CAMS},
                "observation.state": PolicyFeature(FeatureType.STATE, (16,))}
    out_feats = {"action": PolicyFeature(FeatureType.ACTION, (16,))}
    stats = vec_stats(sp, args.data)
    scheduler = None
    if args.policy == "act":
        cfg = ACTConfig(input_features=in_feats, output_features=out_feats,
                        chunk_size=args.chunk, n_action_steps=args.chunk, device="cuda", push_to_hub=False)
        policy = ACTPolicy(cfg).to("cuda")
        optim = cfg.get_optimizer_preset().build(policy.get_optim_params())
    else:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        base = "lerobot/smolvla_base"
        cfg = PreTrainedConfig.from_pretrained(base)
        cfg.input_features, cfg.output_features = in_feats, out_feats
        cfg.chunk_size, cfg.n_action_steps = args.chunk, args.chunk
        cfg.device, cfg.push_to_hub = "cuda", False
        # short run: scale SmolVLA's warmup (1k) / cosine decay (30k) to the number of steps
        cfg.scheduler_warmup_steps, cfg.scheduler_decay_steps = max(1, args.steps // 10), args.steps
        policy = SmolVLAPolicy.from_pretrained(base, config=cfg).to("cuda")
        optim = cfg.get_optimizer_preset().build(policy.get_optim_params())
        scheduler = cfg.get_scheduler_preset().build(optim, num_training_steps=args.steps)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"{args.policy}: params {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M "
          f"(trainable {n_train / 1e6:.1f}M) | steps {args.steps} | batch {args.batch}", flush=True)

    def save(step):
        d = out / f"step_{step:06d}"
        policy.save_pretrained(d)
        preprocessor.save_pretrained(d)
        postprocessor.save_pretrained(d)
        (out / "last.txt").write_text(d.name)
        print(f"saved {d}", flush=True)

    policy.train()
    t0 = time.time()
    log = []
    for step, batch in enumerate(dl, start=1):
        batch = preprocessor(batch)
        loss, info = policy.forward(batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optim.step()
        if scheduler is not None:
            scheduler.step()
        if step % 100 == 0:
            rate = step / (time.time() - t0)
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"step {step}/{args.steps} loss {loss.item():.3f} grad {gn:.1f} lr {optim.param_groups[0]['lr']:.1e} "
                  f"| {rate:.2f} it/s, eta {(args.steps - step) / rate / 60:.1f} min, peak mem {mem:.1f} GB", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            v = {k: val_l1(policy, preprocessor, l, 10) for k, l in val_loaders.items()}
            log.append({"step": step, "train_loss": loss.item(), **{f"val_l1_{k}": x for k, x in v.items()}})
            (out / "val_log.json").write_text(json.dumps(log, indent=1))
            print(f"VAL step {step}: " + ", ".join(f"{k} {x:.3f}" for k, x in v.items()), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(step)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
