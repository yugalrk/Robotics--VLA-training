"""Train ACT on the FR3 Duo handover dataset (train split from configs/splits.json).

Usage:
    python training/train_act.py --steps=20000 --output_dir=outputs/act_fr3 [any other lerobot-train flag]

Wraps `lerobot-train` with two Windows fixes:
- symlinks need admin rights on Windows, so the `checkpoints/last` link is replaced by `checkpoints/last.txt`
- corporate SSL: truststore is injected so pretrained backbone weights can download
"""
import json
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()

from lerobot.scripts import lerobot_train  # noqa: E402
from lerobot.utils import train_utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.data import DATASET_ROOT, REPO_ID, SPLITS_FILE  # noqa: E402


def update_last_checkpoint(checkpoint_dir: Path) -> None:
    (checkpoint_dir.parent / "last.txt").write_text(checkpoint_dir.name)


train_utils.update_last_checkpoint = update_last_checkpoint
lerobot_train.update_last_checkpoint = update_last_checkpoint

DEFAULTS = {
    "--dataset.repo_id": REPO_ID,
    "--dataset.root": str(DATASET_ROOT),
    "--dataset.episodes": json.dumps(json.loads(SPLITS_FILE.read_text())["train"]).replace(" ", ""),
    "--dataset.video_backend": "pyav",
    "--policy.type": "act",
    "--policy.device": "cuda",
    "--policy.push_to_hub": "false",
    "--output_dir": "outputs/act_fr3",
    "--job_name": "act_fr3_handover",
    "--batch_size": "16",
    "--steps": "20000",
    "--save_freq": "5000",
    "--log_freq": "100",
    "--num_workers": "4",
    "--wandb.enable": "false",
}

if __name__ == "__main__":
    given = {a.split("=", 1)[0] for a in sys.argv[1:]}
    sys.argv = [sys.argv[0]] + [f"{k}={v}" for k, v in DEFAULTS.items() if k not in given] + sys.argv[1:]
    lerobot_train.main()
