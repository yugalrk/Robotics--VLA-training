"""One SmolVLA inference, end to end: the inputs it gets and the full action chunk it returns.

    .venv\\Scripts\\python.exe diagnostics/show_io.py [--checkpoint outputs/curriculum_smolvla_expert10k/final] [--frame 2]

Uses a recorded handover frame from eval_results/pi05_probe_frames.npz (held-out episode 0, frame 330 = 11.0 s by
default). Writes eval_results/io_example_inputs.png (the 3 camera images) and eval_results/io_example.json, and
prints the state, the instruction and the predicted 20 x 16 chunk next to what the demonstrator did.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.policy_runner import CAMS, PolicyRunner

ROOT = Path(__file__).resolve().parents[1]
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
NAMES = [f"L_j{i}" for i in range(1, 8)] + ["L_grip"] + [f"R_j{i}" for i in range(1, 8)] + ["R_grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/curriculum_smolvla_expert10k/final")
    ap.add_argument("--frame", type=int, default=2, help="index into pi05_probe_frames.npz (2 = episode 0, frame 330)")
    args = ap.parse_args()
    F = np.load(ROOT / "eval_results" / "pi05_probe_frames.npz")
    ep, t = map(int, F["where"][args.frame])
    imgs = {CAMS[0]: F["ext"][args.frame], CAMS[1]: F["lw"][args.frame], CAMS[2]: F["rw"][args.frame]}
    state = F["state"][args.frame].astype(np.float32)
    demo = F["actions"][args.frame][::5][:20]  # recorded 30 fps actions at the model's 6 Hz spacing

    torch.manual_seed(0)
    runner = PolicyRunner(str(ROOT / args.checkpoint))
    batch = {c: torch.from_numpy(imgs[c]).permute(2, 0, 1).float() / 255.0 for c in CAMS}
    batch["observation.state"] = torch.from_numpy(state)
    batch["task"] = TASK
    with torch.no_grad():
        chunk = runner.post(runner.policy.predict_action_chunk(runner.pre(batch))).squeeze(0).float().cpu().numpy()

    strip = Image.new("RGB", (3 * 224, 244), "white")
    d = ImageDraw.Draw(strip)
    for i, (c, label) in enumerate(zip(CAMS, ["head (external)", "left wrist", "right wrist"])):
        strip.paste(Image.fromarray(imgs[c]), (i * 224, 20))
        d.text((i * 224 + 4, 4), label, fill="black")
    strip.save(ROOT / "eval_results" / "io_example_inputs.png")
    out = {"checkpoint": args.checkpoint, "episode": ep, "frame": t, "time_s": t / 30, "instruction": TASK,
           "input_state": dict(zip(NAMES, state.round(4).tolist())), "images": "eval_results/io_example_inputs.png",
           "output_chunk_20x16": chunk.round(4).tolist(), "demo_actions_20x16": demo.round(4).tolist(), "columns": NAMES}
    (ROOT / "eval_results" / "io_example.json").write_text(json.dumps(out, indent=1))

    np.set_printoptions(suppress=True, linewidth=200)
    print(f"INPUT  episode {ep}, frame {t} ({t / 30:.1f} s)   instruction: {TASK}")
    print("  state:", " ".join(f"{n}={v:+.3f}" for n, v in zip(NAMES, state)))
    print("OUTPUT chunk: 20 actions x 16 values, one action per 1/6 s (3.3 s)")
    print("  step  t(+s) | " + " ".join(f"{n:>7s}" for n in NAMES[:8]) + " |  R_grip | |model-demo| L arm")
    for k in range(20):
        err = f"{np.abs(chunk[k, :7] - demo[k, :7]).mean():.4f}" if k < len(demo) else "  (no demo stored)"
        print(f"  {k + 1:4d}  {(k + 1) / 6:5.2f} | " + " ".join(f"{v:+7.3f}" for v in chunk[k, :8]) + f" | {chunk[k, 15]:+7.3f} | {err}")
    print("  (right-arm joints barely move in this chunk: max change "
          f"{np.abs(chunk[:, 8:15] - state[8:15]).max():.3f} rad)")


if __name__ == "__main__":
    main()
