"""Label every rand500 episode with its cube colours and stacking order.

Each episode: left arm picks cube A and puts it on base cube B, then right arm picks cube C and stacks it on top.
- A and C: read from the wrist camera at the moment the gripper closes (the cube fills the bottom-centre).
- B: read from the head camera's first frame (saturated pixels near the cubes that are not A or C).
- Phase boundaries (frame indices) come from the gripper open/close commands.

Output: artifacts/rand500_labels.csv (one row per episode).

Usage:
    python data_prep/label_rand500_colors.py [--limit N]
"""
import argparse
import colorsys
import glob
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "Datasets/fr3_duo_rand500/fr3_duo_rand500"
OUT = ROOT / "artifacts" / "rand500_labels.csv"
FPS = 6


def color_name(rgb):
    r, g, b = (np.asarray(rgb, float) / 255).tolist()
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if s < 0.25:
        return "white" if v > 0.75 else "black" if v < 0.2 else "gray"
    if (h < 40 or h >= 345) and v < 0.5 and s < 0.55:
        return "brown"
    for lim, name in [(15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (200, "cyan"),
                      (255, "blue"), (290, "purple"), (345, "pink"), (361, "red")]:
        if h < lim:
            return name


def hue(rgb):
    r, g, b = (np.asarray(rgb, float) / 255).tolist()
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def hue_dist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


class Videos:
    def __init__(self):
        self.E = pd.read_parquet(glob.glob(str(B / "meta/episodes/*/*.parquet"))[0]).set_index("episode_index")

    def frames(self, ep, cam, idx):
        k = f"videos/observation.images.{cam}"
        e = self.E.loc[ep]
        f = B / f"videos/observation.images.{cam}/chunk-{e[k + '/chunk_index']:03d}/file-{e[k + '/file_index']:03d}.mp4"
        t0 = e[k + "/from_timestamp"]
        want = sorted(t0 + i / FPS for i in idx)
        out = []
        with av.open(str(f)) as c:
            s = c.streams.video[0]
            c.seek(int(max(0, want[0] - 1) / s.time_base), stream=s)
            for fr in c.decode(s):
                if want and fr.time >= want[0] - 1e-3:
                    out.append(fr.to_ndarray(format="rgb24"))
                    want.pop(0)
                if not want:
                    break
        return out


def held_cube_rgb(img):
    """Median colour of the cube between the fingers (bottom-centre of the wrist image)."""
    patch = img[150:205, 95:130].reshape(-1, 3)
    return np.median(patch, 0)


def base_cube_rgb(img, img_end, picked_hues):
    """Find the base cube in the head camera: a small coloured blob that is in the same place
    in the first and last frame (the two picked cubes move away; the robot is at home in both frames)."""
    from scipy.ndimage import median_filter

    crop = img[110:220, 30:210].astype(float)
    end = img_end[110:220, 30:210].astype(float)
    background = median_filter(crop, size=(21, 21, 1))  # cubes (~10 px) vanish, floor/shadows stay
    local = np.linalg.norm(crop - background, axis=2).reshape(-1)
    unchanged = (np.linalg.norm(crop - end, axis=2) < 30).reshape(-1)
    region = crop.reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*(p / 255)) for p in region])
    bg = np.array([colorsys.rgb_to_hsv(*(p / 255)) for p in background.reshape(-1, 3)])
    dh = np.abs(hsv[:, 0] - bg[:, 0]) * 360
    dh = np.minimum(dh, 360 - dh)
    not_shadow = (dh > 25) | (hsv[:, 1] - bg[:, 1] > 0.3)  # shadows keep the floor's hue and saturation
    cand = (local > 40) & unchanged & not_shadow & (hsv[:, 1] > 0.3) & (hsv[:, 2] > 0.12)
    if picked_hues:
        hd = np.array([min(hue_dist(h * 360, ph) for ph in picked_hues) for h in hsv[:, 0]])
        cand &= hd > 25
    if cand.sum() < 8:
        return None, int(cand.sum())
    return np.median(region[cand], 0), int(cand.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(f, columns=["action", "episode_index"])
                    for f in sorted(glob.glob(str(B / "data/chunk-000/*.parquet")))])
    vids = Videos()
    eps = sorted(df.episode_index.unique())[: args.limit]
    rows = []
    for n, ep in enumerate(eps):
        a = np.stack(df[df.episode_index == ep]["action"])
        cl = int(np.argmax(a[:, 7] < .5)); ol = cl + int(np.argmax(a[cl:, 7] > .5))
        cr = int(np.argmax(a[:, 15] < .5)); orr = cr + int(np.argmax(a[cr:, 15] > .5))
        lw = vids.frames(ep, "left_wrist_camera", [cl + 1])[0]
        rw = vids.frames(ep, "right_wrist_camera", [cr + 1])[0]
        head, head_end = vids.frames(ep, "external_camera", [0, len(a) - 1])
        A, C = held_cube_rgb(lw), held_cube_rgb(rw)
        Brgb, npx = base_cube_rgb(head, head_end, [hue(A), hue(C)])
        rows.append(dict(
            episode_index=int(ep), length=len(a),
            left_close=cl, left_open=ol, right_close=cr, right_open=orr,
            cube_left=color_name(A), cube_base=color_name(Brgb) if Brgb is not None else "unknown",
            cube_right=color_name(C), base_pixels=npx,
            rgb_left=A.astype(int).tolist(), rgb_right=C.astype(int).tolist(),
            rgb_base=Brgb.astype(int).tolist() if Brgb is not None else None,
        ))
        if n % 50 == 0:
            print(n, rows[-1]["cube_left"], "on", rows[-1]["cube_base"], "then", rows[-1]["cube_right"], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} episodes)")


if __name__ == "__main__":
    main()
