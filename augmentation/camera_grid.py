"""One camera -> table grid -> trajectory LUT: camera assets for the box-grid player (drag the box in the image).

    .venv\\Scripts\\python.exe augmentation\\camera_grid.py

1. Camera model (augmentation/head_camera_model.py): scene head-camera pose + recorder intrinsics (fx = fy = 112 px).
2. Box detection in the real head-camera frame: bright, low-saturation blob near the expected spot on the dark cart;
   brightness-weighted (sub-pixel) centroid.
3. Centroid pixel -> plane at the box's mid-height -> table (X, Z) -> offset from the recorded box spot (the
   recorded grasp point, forward kinematics). One-shot calibration: in this real frame the box IS at the recorded
   spot, so the measured offset is the constant bias (box centroid vs grasp point) subtracted afterwards.
4. Assets for the UI: the frame with the box removed (background), the box as an RGBA sprite, its centroid inside
   the sprite, and its distance from the camera (the UI scales the sprite with distance).
Self-test: the box sprite is pasted (sub-pixel, 8x) at known shifts and detected again -> must return the same cell.
Writes Augmented/ui/camera/{background.png, box_sprite.png, camera.json}.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fr3duo.head_camera import load, pixel_to_plane, world_to_pixel  # noqa: E402

AUG = ROOT / "Datasets/fr3_duo_handover_84_ep/Augmented"
CAMDIR = AUG / "ui/camera"
FRAME = "head_t01.0.png"
CART_Y = 0.829     # cart top (box resting height)
BOX_MID_Y = 0.88   # back-projection plane for the box centroid (the calibration bias absorbs the rest)
UP = 8


def ensure_frame(t_s=1.0, episode=0):
    """The real head-camera frame at t = 1.0 s of a recorded episode (box at its recorded spot). Extracted from the
    dataset video if it is not there yet, so the build works from a fresh checkout."""
    out = CAMDIR / FRAME
    if out.exists():
        return
    import av
    import pandas as pd
    ds = ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84"
    e = pd.read_parquet(next((ds / "meta/episodes").rglob("*.parquet"))).set_index("episode_index").loc[episode]
    k = "videos/observation.images.external_camera"
    f = ds / f"videos/observation.images.external_camera/chunk-{e[k + '/chunk_index']:03d}/file-{e[k + '/file_index']:03d}.mp4"
    ts = e[k + "/from_timestamp"] + t_s
    with av.open(str(f)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0, ts - 1) / s.time_base), stream=s)
        for fr in c.decode(s):
            if fr.time >= ts - 1e-3:
                CAMDIR.mkdir(parents=True, exist_ok=True)
                Image.fromarray(fr.to_ndarray(format="rgb24")).save(out)
                print("extracted", out.name, "from episode", episode, "at", t_s, "s")
                return


def detect_box(img, expect_uv, radius=16):
    a = img.astype(float)
    m = (a.mean(2) > 150) & ((a.max(2) - a.min(2)) < 40)
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    seeds = np.argwhere(m & ((xx - expect_uv[0]) ** 2 + (yy - expect_uv[1]) ** 2 < radius ** 2))
    if len(seeds) == 0:
        return None, None
    s = seeds[np.argmin(((seeds - expect_uv[::-1]) ** 2).sum(1))]
    blob, stack = np.zeros_like(m), [tuple(s)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < h and 0 <= x < w and m[y, x] and not blob[y, x] and (x - expect_uv[0]) ** 2 + (y - expect_uv[1]) ** 2 < (3 * radius) ** 2:
            blob[y, x] = True
            stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    ys, xs = np.nonzero(blob)
    wgt = a.mean(2)[ys, xs] - 140
    return blob, np.array([(xs * wgt).sum() / wgt.sum(), (ys * wgt).sum() / wgt.sum()])


def offset_cm(cam, centroid, ref_xz, bias=(0.0, 0.0)):
    P = pixel_to_plane(cam, centroid, BOX_MID_Y)[0]
    return (P[[0, 2]] - ref_xz) * 100 - np.array(bias)


def main():
    cam = load()
    idx = json.loads((AUG / "ui/index.json").read_text())
    c0 = next(c for c in idx["cells"] if c["box_dx_mm"] == 0 and c["box_dz_mm"] == 0)
    ref = np.array([c0["grasp_x"], c0["grasp_z"]])
    ensure_frame()
    img = np.array(Image.open(CAMDIR / FRAME).convert("RGB"))
    ref_uv = world_to_pixel(cam, np.array([[ref[0], BOX_MID_Y, ref[1]]]))[0][0]
    blob, c = detect_box(img, ref_uv)
    bias = offset_cm(cam, c, ref)
    print(f"real frame: box centroid {c.round(2).tolist()} px -> raw offset {bias.round(2).tolist()} cm = calibration bias")

    # background: box (+1 px) filled with the surrounding cart colour; sprite: the box with alpha
    ys, xs = np.nonzero(blob)
    pad = np.zeros_like(blob)
    for dy in (-1, 0, 1):
        for dxp in (-1, 0, 1):
            pad[np.clip(ys + dy, 0, 223), np.clip(xs + dxp, 0, 223)] = True
    ring = np.zeros_like(blob)
    ring[ys.min() - 4:ys.max() + 5, xs.min() - 4:xs.max() + 5] = True
    ring &= ~pad
    bg = img.copy()
    bg[pad] = np.median(img[ring], axis=0).astype(np.uint8)
    y0, y1, x0, x1 = ys.min() - 1, ys.max() + 2, xs.min() - 1, xs.max() + 2
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    rgba[..., :3] = img[y0:y1, x0:x1]
    rgba[..., 3] = (pad[y0:y1, x0:x1] * 255).astype(np.uint8)
    Image.fromarray(bg).save(CAMDIR / "background.png")
    Image.fromarray(rgba, "RGBA").save(CAMDIR / "box_sprite.png")
    depth = float(world_to_pixel(cam, np.array([[ref[0] + bias[0] / 100, BOX_MID_Y, ref[1] + bias[1] / 100]]))[1][0])

    # self-test: paste the sprite at known shifts (sub-pixel), detect, compare cells
    big = lambda a, m=Image.BICUBIC: np.array(Image.fromarray(a).resize((224 * UP, 224 * UP), m))
    img_up, bg_up, m_up = big(img), big(bg), big((pad * 255).astype(np.uint8), Image.NEAREST) > 127
    by, bx = np.nonzero(m_up)
    cells_ok = {(cc["box_dx_mm"] / 10, cc["box_dz_mm"] / 10) for cc in idx["cells"] if cc["status"] == "ok"}
    tests, hits = [(5, 0), (-5, 5), (-20, 10), (-40, -10), (5, -15), (-60, 20), (0, 20)], 0
    for dx, dz in tests:
        p0 = np.array([[ref[0] + bias[0] / 100, BOX_MID_Y, ref[1] + bias[1] / 100]])
        p1 = p0 + np.array([[dx / 100, 0, dz / 100]])
        sh = world_to_pixel(cam, p1)[0][0] - world_to_pixel(cam, p0)[0][0]
        syn_up = bg_up.copy()
        syn_up[np.clip(by + int(round(sh[1] * UP)), 0, 224 * UP - 1), np.clip(bx + int(round(sh[0] * UP)), 0, 224 * UP - 1)] = img_up[by, bx]
        syn = np.array(Image.fromarray(syn_up).resize((224, 224), Image.BOX))
        _, cc = detect_box(syn, ref_uv + sh)
        if cc is None:
            print(f"  self-test box moved ({dx:+d}, {dz:+d}) cm = {sh.round(1).tolist()} px -> not detected "
                  "(the unscaled patch is pasted far from where it was cut; the UI rescales it with distance)")
            continue
        off = offset_cm(cam, cc, ref, bias)
        step = idx.get("step_mm", 10) / 10
        cell = tuple(int(v) for v in np.round(off / step) * step)
        hits += cell == (dx, dz)
        print(f"  self-test box moved ({dx:+d}, {dz:+d}) cm = {sh.round(1).tolist()} px -> detected {off.round(2).tolist()} cm -> cell {cell} "
              f"{'OK' if cell == (dx, dz) else 'MISMATCH'}{'' if (dx, dz) in cells_ok else ' (cell not feasible)'}")
    print(f"self-test: {hits}/{len(tests)} cells recovered (note: shifts move the sprite without perspective rescaling)")
    out = {"camera": cam, "box_mid_y": BOX_MID_Y, "cart_y": CART_Y, "bias_cm": bias.round(3).tolist(), "ref_xz": ref.tolist(),
           "ref_uv": ref_uv.round(2).tolist(), "box_centroid_px": c.round(2).tolist(), "box_depth": round(depth, 4),
           "sprite": {"x0": int(x0), "y0": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}, "frame": FRAME,
           "px_per_cm": round(float(np.linalg.norm(world_to_pixel(cam, np.array([[ref[0] + 0.01, BOX_MID_Y, ref[1]]]))[0][0] - ref_uv)), 2)}
    (CAMDIR / "camera.json").write_text(json.dumps(out))
    for old in CAMDIR.glob("synthetic_*.png"):
        old.unlink()
    print("wrote", CAMDIR)


if __name__ == "__main__":
    main()
