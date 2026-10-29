"""Side-by-side: training camera frames vs the Isaac Sim cameras (from isaac/eval_isaac.py's replay GIF) at the same
moments of the demo. Writes eval_results/camera_domain_gap.png and prints simple image statistics."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fr3duo.data import load_dataset  # noqa: E402

CAMS = ["observation.images.external_camera", "observation.images.left_wrist_camera", "observation.images.right_wrist_camera"]
gif = Image.open(ROOT / "eval_results" / "replay_headless" / "replay_head_left_right.gif")
sim = []
try:
    while True:
        sim.append(np.array(gif.convert("RGB")))
        gif.seek(gif.tell() + 1)
except EOFError:
    pass
# replay GIF: one frame every 5 recorded frames (30 fps), each frame = head | left wrist | right wrist (224 px each)
ds = load_dataset(episodes=[0])
times = [0, 150, 300]  # 0 s, 5 s (approach), 10 s (just before the grasp)
rows = []
for t in times:
    it = ds[t]
    train = np.concatenate([(it[c].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in CAMS], axis=1)
    s = sim[min(t // 5, len(sim) - 1)]
    rows.append((t, train, s))
    for name, img in [("train", train), ("sim", s)]:
        print(f"t={t / 30:4.1f}s {name:5s} mean brightness {img.mean():6.1f}  per-camera means {[round(float(img[:, i * 224:(i + 1) * 224].mean()), 1) for i in range(3)]}")
W, H = rows[0][1].shape[1], 224
canvas = Image.new("RGB", (W, len(rows) * (2 * H + 30)), "white")
d = ImageDraw.Draw(canvas)
for k, (t, train, s) in enumerate(rows):
    y = k * (2 * H + 30)
    d.text((4, y + 2), f"t = {t / 30:.0f} s   top: TRAINING data   bottom: ISAAC SIM (same recorded actions)   cameras: head | left wrist | right wrist", fill="black")
    canvas.paste(Image.fromarray(train), (0, y + 15))
    canvas.paste(Image.fromarray(s[:, :W]), (0, y + 15 + H))
out = ROOT / "eval_results" / "camera_domain_gap.png"
canvas.save(out)
print("saved", out)
