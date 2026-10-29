"""Add per-frame sub-task labels to the handover and cube datasets (LeRobot-native, no video re-encoding).

How boundaries are found (robot signals only, no vision):
- An arm "moves" when its 7 joints change faster than 0.02 rad/s; otherwise it is still.
- Events: gripper close / open (start of the command ramp) and the first arm motion after an event.
- Each boundary is placed in the middle of the still period right before the event that starts the next
  sub-task, so a few frames of error change nothing the model sees.

Sub-tasks
- handover (30 fps), default --handover_subtasks 3:
      pick up the box (left) | hand the box over to the right gripper | place the box on the tray (right)
  --handover_subtasks 5 splits the handover into: bring to handover point | grasp (right) | release (left)
  (the 5 fine steps are always found and checked; the 3-step version merges the middle three)
- cubes (6 fps):    pick A (left) | put A on B | pick C (right) | stack C on top
  Cube colours come from artifacts/rand500_labels.csv (data_prep/label_rand500_colors.py).

Quality checks (every episode): boundaries in order and inside a still period, each sub-task contains its
expected gripper event, gripper state at each boundary makes sense (holding / released), sub-task durations
not outliers (> max(4 MAD, 30%) from the median). Flagged episodes are listed and shown in the review images.

Outputs
- Datasets/<name>_subtasks/  : copy of the dataset with `subtask_index` per frame + meta/subtasks.parquet
                               (videos hard-linked, not copied). Cubes: complete-task text now names the colours.
- artifacts/subtask_qa/<name>_segments.csv, <name>_qa.json, <name>_boundaries_*.png

Usage:
    python data_prep/make_subtasks.py               # both datasets
    python data_prep/make_subtasks.py --only cubes  # or handover
"""
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "artifacts" / "subtask_qa"
SRC = {
    "handover": ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84",
    "cubes": ROOT / "Datasets/fr3_duo_rand500/fr3_duo_rand500",
}
DST = {
    "handover": ROOT / "Datasets/fr3_handover84_subtasks",
    "cubes": ROOT / "Datasets/fr3_rand500_subtasks",
}
L_ARM, R_ARM, L_G, R_G = slice(0, 7), slice(8, 15), 7, 15
STILL_RAD_S = 0.02


# ---------------------------------------------------------------- signals
def moving(S, arm, fps):
    """True where the arm moves; motion must last >= 0.1 s so settling jitter is ignored."""
    v = np.abs(np.diff(S[:, arm], axis=0)).sum(1) * fps  # rad/s
    m = np.r_[v > STILL_RAD_S, False]
    k = max(2, fps // 10)
    sustained = np.array([m[t:t + k].all() for t in range(len(m))])
    out = sustained.copy()
    for t in np.flatnonzero(sustained):  # extend each sustained run back over its own frames
        out[t:t + k] = True
    return out & m


def first(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return int(start + idx[0]) if len(idx) else None


def ramp_start(a, t, direction):
    """Walk back from t to where the gripper command started changing (direction -1 closing, +1 opening)."""
    while t > 0 and (a[t] - a[t - 1]) * direction > 1e-6:
        t -= 1
    return t


def close_onset(a, g):
    t = first(a[:, g] < 0.25)
    return None if t is None else ramp_start(a[:, g], t, -1)


def open_onset(a, g, after):
    t = first(np.r_[np.diff(a[:, g]) > 1e-4, False], after)
    return t


def grip_settled(S, g, onset, fps):
    """First frame after a close/open command where the measured gripper has stopped moving (for 0.2 s)."""
    if onset is None:
        return None
    d = np.r_[np.abs(np.diff(S[:, g])), 0.0]
    k = max(2, fps // 5)
    for t in range(onset + 1, len(S) - k):
        if (d[t:t + k] < 0.005).all() and abs(S[t, g] - S[onset, g]) > 0.05:
            return t
    return None


def still_mid_before(t, mv_l, mv_r):
    """Middle of the still period (both arms still) that ends at event t."""
    s = t
    while s > 0 and not mv_l[s - 1] and not mv_r[s - 1]:
        s -= 1
    return (s + t) // 2, t - s


# ---------------------------------------------------------------- segmentation per dataset
def segment(name, S, A, fps):
    mv_l, mv_r = moving(S, L_ARM, fps), moving(S, R_ARM, fps)
    lc, rc = close_onset(A, L_G), close_onset(A, R_G)
    if lc is None or rc is None:
        return None, {"left_close": lc, "right_close": rc}, None
    # a sub-task that ends with a grasp ends at the first motion after the gripper has *finished* closing
    lc_done, rc_done = grip_settled(S, L_G, lc, fps) or lc, grip_settled(S, R_G, rc, fps) or rc
    lo, ro = open_onset(A, L_G, lc + 1), open_onset(A, R_G, rc + 1)
    ev = {"left_close": lc, "right_close": rc, "left_open": lo, "right_open": ro}
    if name == "handover":
        events = [first(mv_l, lc_done), first(mv_r, lc_done), lo, first(mv_r, rc_done)]  # starts of sub-tasks 2..5
    else:
        lo_done = grip_settled(S, L_G, lo, fps) or lo
        events = [first(mv_l, lc_done), first(mv_r, lo_done), first(mv_r, rc_done)]  # starts of sub-tasks 2..4
    bounds, still = [], []
    for e in events:
        if e is None:
            return None, ev, None
        b, n = still_mid_before(e, mv_l, mv_r)
        bounds.append(b)
        still.append(n)
    return [0, *bounds, len(S)], ev, still


def check(name, b, ev, still, S, A, fps):
    """Return list of problems for one episode (empty = OK)."""
    p = []
    if b is None:
        return ["an event was not found"]
    if any(x >= y for x, y in zip(b, b[1:])):
        return [f"boundaries out of order {b}"]
    if any(n < 2 for n in still):
        p.append(f"boundary not in a still period (still frames {still})")
    inside = lambda t, k: t is not None and b[k] <= t < b[k + 1]  # noqa: E731
    expect = {"handover": [("left_close", 0), ("right_close", 2), ("left_open", 3), ("right_open", 4)],
              "cubes": [("left_close", 0), ("left_open", 1), ("right_close", 2), ("right_open", 3)]}[name]
    for e, k in expect:
        if not inside(ev[e], k):
            p.append(f"{e} (frame {ev[e]}) not inside sub-task {k}")
    # gripper command at the boundaries: holding = command below 0.25
    hold = lambda g, t: A[min(t, len(A) - 1), g] < 0.25  # noqa: E731
    if not hold(L_G, b[1]):
        p.append("left gripper not holding at end of pick")
    if name == "handover":
        if not (hold(L_G, b[3]) and hold(R_G, b[3])):
            p.append("both grippers should hold at the handover")
        if hold(L_G, b[4]) or not hold(R_G, b[4]):
            p.append("after release only the right gripper should hold")
    else:
        if hold(L_G, b[2]):
            p.append("left gripper should have released before the right arm starts")
        if not hold(R_G, b[3]):
            p.append("right gripper not holding at end of its pick")
    return p


# ---------------------------------------------------------------- texts
def texts(name, row=None, handover_subtasks=3):
    if name == "handover":
        full = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
        if handover_subtasks == 3:
            return full, ["Pick up the box with the left gripper.",
                          "Hand the box over to the right gripper.",
                          "Place the box on the serving tray."]
        subs = ["Pick up the box with the left gripper.",
                "Bring the box to the handover position.",
                "Grasp the box with the right gripper.",
                "Release the box from the left gripper.",
                "Place the box on the serving tray with the right gripper."]
        return full, subs
    A, Bc, C = row.cube_left, row.cube_base, row.cube_right
    base = f"the {Bc} cube" if Bc != "unknown" else "the other cube"
    full = f"Stack the {A} cube on {base}, then put the {C} cube on top."
    subs = [f"Pick up the {A} cube with the left gripper.",
            f"Put the {A} cube on {base}.",
            f"Pick up the {C} cube with the right gripper.",
            f"Stack the {C} cube on top of the {A} cube."]
    return full, subs


# ---------------------------------------------------------------- review images
def boundary_sheet(name, src, seg, episodes, out_png, fps):
    import av
    from PIL import Image, ImageDraw
    E = pd.read_parquet(glob.glob(str(src / "meta/episodes/*/*.parquet"))[0]).set_index("episode_index")
    cams = ["external_camera", "left_wrist_camera", "right_wrist_camera"]
    rows, labels = [], []
    for ep in episodes:
        b = seg[seg.episode_index == ep].sort_values("subtask_order")
        frames = b.start_frame.tolist()[1:]  # boundaries only
        tiles = []
        for t in frames:
            col = []
            for cam in cams:
                k = f"videos/observation.images.{cam}"
                e = E.loc[ep]
                f = src / f"videos/observation.images.{cam}/chunk-{e[k + '/chunk_index']:03d}/file-{e[k + '/file_index']:03d}.mp4"
                ts = e[k + "/from_timestamp"] + t / fps
                with av.open(str(f)) as c:
                    s = c.streams.video[0]
                    c.seek(int(max(0, ts - 1) / s.time_base), stream=s)
                    for fr in c.decode(s):
                        if fr.time >= ts - 1e-3:
                            col.append(np.asarray(fr.to_ndarray(format="rgb24"))[::2, ::2])
                            break
            tiles.append(np.concatenate(col, 0))
        rows.append(np.concatenate(tiles, 1))
        labels.append((ep, b.text.tolist()))
    img = Image.fromarray(np.concatenate(rows, 0))
    d = ImageDraw.Draw(img)
    h = rows[0].shape[0]
    for r, (ep, txt) in enumerate(labels):
        for j in range(len(txt) - 1):
            d.rectangle([j * 112, r * h, j * 112 + 111, r * h + 22], fill=(0, 0, 0))
            d.text((j * 112 + 2, r * h + 1), f"ep{ep} -> {j + 2}", fill=(255, 255, 0))
            d.text((j * 112 + 2, r * h + 11), txt[j + 1][:22], fill=(255, 255, 255))
    img.save(out_png)


# ---------------------------------------------------------------- dataset writing
def link_tree(src, dst):
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        t = dst / rel
        if f.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        else:
            t.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(f, t)
            except OSError:
                shutil.copy2(f, t)


def build(name, args):
    src, dst = SRC[name], DST[name]
    files = sorted(glob.glob(str(src / "data/*/*.parquet")))
    data = {f: pd.read_parquet(f) for f in files}
    df = pd.concat(data.values(), ignore_index=True)
    info = json.loads((src / "meta/info.json").read_text())
    fps = info["fps"]
    colors = pd.read_csv(ROOT / "artifacts" / "rand500_labels.csv").set_index("episode_index") if name == "cubes" else None

    seg_rows, problems, ep_task = [], {}, {}
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        S, A = np.stack(g["observation.state"]), np.stack(g["action"])
        b, ev, still = segment(name, S, A, fps)
        p = check(name, b, ev, still, S, A, fps)  # always checks the 5 fine steps (validates every event)
        if name == "handover" and args.handover_subtasks == 3 and b is not None and not p:
            # pick | hand over (bring + right grasp + left release) | place
            b = [b[0], b[1], b[4], b[5]]
        full, subs = texts(name, colors.loc[ep] if colors is not None else None, args.handover_subtasks)
        ep_task[ep] = full
        if p:
            problems[int(ep)] = p
        if b is None:
            continue
        for k in range(len(subs)):
            seg_rows.append(dict(episode_index=int(ep), subtask_order=k, start_frame=b[k], end_frame=b[k + 1],
                                 duration_s=round((b[k + 1] - b[k]) / fps, 2), text=subs[k],
                                 flagged=int(ep) in problems))
    seg = pd.DataFrame(seg_rows)

    # duration outliers per sub-task position
    for k, g in seg.groupby("subtask_order"):
        med = g.duration_s.median()
        mad = (g.duration_s - med).abs().median() or 0.1
        tol = max(4 * mad, 0.3 * med)  # demos are very uniform, so 4 MAD alone would flag ~1 s variations
        for _, r in g[(g.duration_s - med).abs() > tol].iterrows():
            problems.setdefault(int(r.episode_index), []).append(
                f"sub-task {k} duration {r.duration_s}s vs median {med}s")
    seg["flagged"] = seg.episode_index.isin(problems.keys())

    QA.mkdir(exist_ok=True)
    seg.to_csv(QA / f"{name}_segments.csv", index=False)
    summary = {
        "episodes": int(df.episode_index.nunique()), "flagged": len(problems),
        "duration_s_by_subtask": {int(k): {"median": float(g.duration_s.median()), "min": float(g.duration_s.min()),
                                           "max": float(g.duration_s.max())} for k, g in seg.groupby("subtask_order")},
        "problems": {str(k): v for k, v in sorted(problems.items())},
    }
    (QA / f"{name}_qa.json").write_text(json.dumps(summary, indent=1))
    print(f"[{name}] {summary['episodes']} episodes, {len(problems)} flagged")
    for k, v in summary["duration_s_by_subtask"].items():
        print(f"   sub-task {k}: median {v['median']}s (min {v['min']}, max {v['max']})  e.g. '{seg[seg.subtask_order == k].text.iloc[0]}'")
    for k, v in list(problems.items())[:8]:
        print(f"   flagged ep {k}: {v}")

    # review images: 8 random OK episodes + up to 8 flagged
    rng = np.random.default_rng(0)
    ok = sorted(set(seg.episode_index) - set(problems))
    sample = sorted(rng.choice(ok, min(8 if name == "cubes" else 3, len(ok)), replace=False).tolist()) if ok else []
    if sample:
        boundary_sheet(name, src, seg, sample, QA / f"{name}_boundaries_sample.png", fps)
    bad = [e for e in sorted(problems) if e in set(seg.episode_index)][:8]
    if bad:
        boundary_sheet(name, src, seg, bad, QA / f"{name}_boundaries_flagged.png", fps)

    if args.no_write:
        return
    # ---- write the new dataset (flagged episodes keep their labels; drop them at training time if wanted)
    if dst.exists():
        shutil.rmtree(dst)
    link_tree(src / "videos", dst / "videos")
    shutil.copytree(src / "meta", dst / "meta")
    for extra in ["norm_stats.json"]:
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)

    sub_texts = sorted(seg.text.unique())
    sub_idx = {t: i for i, t in enumerate(sub_texts)}
    pd.DataFrame({"subtask_index": range(len(sub_texts))}, index=pd.Index(sub_texts, name="subtask")) \
        .to_parquet(dst / "meta/subtasks.parquet")
    task_texts = sorted(set(ep_task.values()))
    task_idx = {t: i for i, t in enumerate(task_texts)}
    pd.DataFrame({"task_index": range(len(task_texts))}, index=pd.Index(task_texts, name="task")) \
        .to_parquet(dst / "meta/tasks.parquet")

    lookup = {}
    for r in seg.itertuples():
        lookup[(r.episode_index, r.subtask_order)] = (r.start_frame, r.end_frame, sub_idx[r.text])
    for f, d in data.items():
        sub = np.full(len(d), -1, np.int64)
        for ep, g in d.groupby("episode_index"):
            fi = g.frame_index.values
            for k in range(len(seg.subtask_order.unique())):
                if (ep, k) in lookup:
                    s, e, i = lookup[(ep, k)]
                    sub[g.index[(fi >= s) & (fi < e)] - d.index[0]] = i
        if (sub < 0).any():  # episodes that could not be segmented: label with their first sub-task text
            print(f"   warning: {(sub < 0).sum()} frames without sub-task in {Path(f).name}")
            sub[sub < 0] = 0
        d = d.copy()
        d["subtask_index"] = sub
        d["task_index"] = d.episode_index.map(lambda e: task_idx[ep_task[e]]).astype(np.int64)
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out)

    ep_files = glob.glob(str(dst / "meta/episodes/*/*.parquet"))
    for f in ep_files:
        e = pd.read_parquet(f)
        e["tasks"] = e.episode_index.map(lambda x: [ep_task[x]])
        e.to_parquet(f)
    info["features"]["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
    info["total_tasks"] = len(task_texts)
    info["splits"] = {"train": f"0:{info['total_episodes']}"}  # rand500 had '0:41'
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"   wrote {dst}  ({len(task_texts)} task texts, {len(sub_texts)} sub-task texts)")


def verify(name):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f"local/{DST[name].name}", root=DST[name], episodes=[1], video_backend="pyav")
    seg = pd.read_csv(QA / f"{name}_segments.csv")
    seg = seg[seg.episode_index == 1]
    print(f"[{name}] verify episode 1 via LeRobotDataset:")
    for r in seg.itertuples():
        mid = (r.start_frame + r.end_frame) // 2
        item = ds[mid]
        ok = item["subtask"] == r.text
        print(f"   frame {mid:4d}: task='{item['task'][:60]}' | subtask='{item['subtask']}' {'OK' if ok else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["handover", "cubes"])
    ap.add_argument("--no_write", action="store_true", help="only run segmentation + QA")
    ap.add_argument("--handover_subtasks", type=int, choices=[3, 5], default=3,
                    help="3 = pick | hand over to right gripper | place on tray;  "
                         "5 = pick | bring to handover point | right grasp | left release | place")
    args = ap.parse_args()
    for n in ([args.only] if args.only else ["handover", "cubes"]):
        build(n, args)
        if not args.no_write:
            verify(n)


if __name__ == "__main__":
    main()
