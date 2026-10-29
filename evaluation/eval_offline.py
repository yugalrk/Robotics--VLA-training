"""Offline evaluation of trained policies on the held-out handover episodes (no simulator needed).

The policy sees the recorded cameras + joint state of each held-out episode at 6 Hz and its outputs are
compared with what the demonstrator did. Because the robot never acts on the predictions, errors cannot
compound here — this measures prediction quality; closed-loop success needs the simulator (isaac/eval_isaac.py).

Metrics
1. action error (rad) — executed actions vs demo: overall, per arm/joint, per sub-task, p50/p95/p99
2. horizon curve — error of the full predicted chunk vs steps ahead (1..chunk)
3. gripper events — for every close/release in the demo: hit within ±1 s?, timing error (s)
4. DTW distance per sub-task — motion-shape similarity to the expert, tolerant to timing differences
5. smoothness — size of command jumps, within a chunk vs at re-plan boundaries
6. latency — normal step vs re-plan step
7. robustness — noisy joint readings, darker / brighter images, missing / reworded instruction

Usage:
    python evaluation/eval_offline.py                                # all checkpoints in MODELS
    python evaluation/eval_offline.py --models smolvla_handover/step_003000 --episodes 3
Writes eval_results/offline/<model>.json and eval_results/offline/summary.md
"""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the fr3duo package
from fr3duo.data import SPLITS_FILE, load_dataset
from fr3duo.policy_runner import CAMS, PolicyRunner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/offline"
MODELS = ["act_mixed/step_006000", "act_handover/step_006000", "smolvla_handover/step_003000"]
ARM = [*range(7), *range(8, 15)]
GRIP = [7, 15]
STRIDE = 5  # 30 fps data -> 6 Hz policy steps
HZ = 6
TASK = "Hand the box from the left gripper to the right gripper and place it on the serving tray."
PARAPHRASE = "Pass the box from the left hand to the right hand, then put it on the tray."


# ------------------------------------------------------------------ data
def load_episodes(eps):
    """Decode every 5th frame of each held-out episode once (uint8 images, state, action)."""
    out = {}
    for ep in eps:
        ds = load_dataset(episodes=[ep])
        n = len(ds)
        frames = []
        for k in range(0, n, STRIDE):
            s = ds[k]
            frames.append({**{c: (s[c].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in CAMS},
                           "state": s["observation.state"].numpy(), "action": s["action"].numpy()})
        out[ep] = frames
        print(f"  loaded ep {ep}: {len(frames)} steps", flush=True)
    return out


def subtask_of_step(n_steps):
    seg = pd.read_csv(ROOT / "artifacts/subtask_qa/handover_segments.csv")
    seg = seg[seg.episode_index == seg.episode_index.iloc[0]]  # identical for all handover episodes
    lab = np.zeros(n_steps, int)
    for r in seg.itertuples():
        lab[(np.arange(n_steps) * STRIDE >= r.start_frame) & (np.arange(n_steps) * STRIDE < r.end_frame)] = r.subtask_order
    return lab, seg.sort_values("subtask_order").text.tolist()


# ------------------------------------------------------------------ helpers
def obs_of(f, task, state_noise=0.0, img_scale=1.0, rng=None):
    o = {}
    for c in CAMS:
        img = f[c]
        if img_scale != 1.0:
            img = np.clip(img.astype(np.float32) * img_scale, 0, 255).astype(np.uint8)
        o[c] = img
    st = f["state"].copy()
    if state_noise:
        st[ARM] += rng.normal(0, state_noise, len(ARM)).astype(np.float32)
    o["observation.state"] = st
    o["task"] = task
    return o


def run_episode(runner, frames, **kw):
    runner.reset()
    acts, lat, replan = [], [], []
    for f in frames:
        out = runner.act(obs_of(f, **kw))
        acts.append(out["action"])
        lat.append(out["latency_ms"])
        replan.append(out["replanned"])
    return np.array(acts), np.array(lat), np.array(replan)


@torch.no_grad()
def predict_chunk(runner, f, task):
    """Full predicted chunk (unnormalized) for one observation, without touching the action queue."""
    o = obs_of(f, task)
    batch = {k: torch.from_numpy(o[k]).permute(2, 0, 1).float() / 255.0 for k in CAMS}
    batch["observation.state"] = torch.as_tensor(o["observation.state"])
    batch["task"] = task
    chunk = runner.policy.predict_action_chunk(runner.pre(batch))
    return runner.post(chunk).squeeze(0).cpu().numpy()  # (chunk, 16)


def gripper_events(a, g):
    closed = a[:, g] < 0.25
    t = np.flatnonzero(np.diff(closed.astype(int)) != 0) + 1
    return [(int(i), "close" if closed[i] else "open") for i in t]


def event_match(pred, gt, tol_steps=HZ):
    res = []
    for t, kind in gt:
        cand = [p for p, k in pred if k == kind and abs(p - t) <= tol_steps]
        res.append({"t_demo_s": round(t / HZ, 2), "kind": kind, "hit": bool(cand),
                    "timing_error_s": round(min(cand, key=lambda p: abs(p - t)) / HZ - t / HZ, 2) if cand else None})
    return res


def dtw(a, b):
    d, path = fastdtw(a, b, dist=lambda x, y: float(np.abs(x - y).mean()))
    return d / len(path)


# ------------------------------------------------------------------ evaluation of one model
def evaluate(name, data, args):
    runner = PolicyRunner(ROOT / "outputs" / name)
    chunk = runner.info()["chunk_size"]
    eps = list(data)
    n = len(data[eps[0]])
    sub, sub_txt = subtask_of_step(n)
    rng = np.random.default_rng(0)

    # warm-up (first call compiles / allocates)
    run_episode(runner, data[eps[0]][:3], task=TASK)

    errs, lat_all, rep_all, gr_events, dtw_rows, jumps_in, jumps_rep = [], [], [], [], [], [], []
    for ep in eps:
        acts, lat, rep = run_episode(runner, data[ep], task=TASK)
        gt = np.stack([f["action"] for f in data[ep]])
        errs.append(np.abs(acts - gt))
        lat_all += lat.tolist()
        rep_all += rep.tolist()
        for g, arm in [(7, "left"), (15, "right")]:
            for e in event_match(gripper_events(acts, g), gripper_events(gt, g)):
                gr_events.append({"episode": ep, "gripper": arm, **e})
        for k in range(len(sub_txt)):
            m = sub == k
            dtw_rows.append({"episode": ep, "subtask": k, "dtw_rad": dtw(acts[m][:, ARM], gt[m][:, ARM])})
        da = np.abs(np.diff(acts[:, ARM], axis=0)).max(1)
        jumps_rep += da[rep[1:]].tolist()
        jumps_in += da[~rep[1:]].tolist()
    E = np.concatenate(errs)
    sub_all = np.concatenate([sub] * len(eps))

    # horizon curve on a subsample of steps (full chunks are slow for SmolVLA)
    hz_err = np.zeros(chunk)
    hz_cnt = np.zeros(chunk)
    for ep in eps[: args.horizon_eps]:
        gt = np.stack([f["action"] for f in data[ep]])
        for t in range(0, n - 1, 8):
            pc = predict_chunk(runner, data[ep][t], TASK)
            h = min(chunk, n - t)
            hz_err[:h] += np.abs(pc[:h] - gt[t:t + h])[:, ARM].mean(1)
            hz_cnt[:h] += 1
    horizon = (hz_err / np.maximum(hz_cnt, 1)).round(4).tolist()

    # robustness on the first 3 held-out episodes
    variants = {"baseline": {}, "state_noise_0.01rad": {"state_noise": 0.01}, "state_noise_0.05rad": {"state_noise": 0.05}, "images_darker_x0.6": {"img_scale": 0.6},
                "images_brighter_x1.4": {"img_scale": 1.4}, "no_instruction": {"task": ""},
                "reworded_instruction": {"task": PARAPHRASE}}
    robust = {}
    for v, kw in variants.items():
        kw = {"task": TASK, **kw}
        e = []
        for ep in eps[:3]:
            acts, _, _ = run_episode(runner, data[ep], rng=rng, **kw)
            gt = np.stack([f["action"] for f in data[ep]])
            e.append(np.abs(acts - gt)[:, ARM].mean())
        robust[v] = round(float(np.mean(e)), 4)

    lat_all, rep_all = np.array(lat_all), np.array(rep_all)
    ge = pd.DataFrame(gr_events)
    dt = pd.DataFrame(dtw_rows)
    res = {
        "model": name, "type": runner.info()["type"], "episodes": eps, "steps_per_episode": n, "chunk": chunk,
        "action_error_rad": {"arm_mean": round(float(E[:, ARM].mean()), 4),
                             "arm_p50": round(float(np.percentile(E[:, ARM], 50)), 4),
                             "arm_p95": round(float(np.percentile(E[:, ARM], 95)), 4),
                             "arm_p99": round(float(np.percentile(E[:, ARM], 99)), 4),
                             "left_arm": round(float(E[:, 0:7].mean()), 4), "right_arm": round(float(E[:, 8:15].mean()), 4),
                             "per_joint_left": E[:, 0:7].mean(0).round(4).tolist(),
                             "per_joint_right": E[:, 8:15].mean(0).round(4).tolist(),
                             "gripper_mean": round(float(E[:, GRIP].mean()), 4)},
        "per_subtask": [{"subtask": k, "text": sub_txt[k], "arm_err_rad": round(float(E[sub_all == k][:, ARM].mean()), 4),
                         "gripper_err": round(float(E[sub_all == k][:, GRIP].mean()), 4),
                         "dtw_rad": round(float(dt[dt.subtask == k].dtw_rad.mean()), 4)} for k in range(len(sub_txt))],
        "horizon_arm_err_rad": horizon,
        "gripper_events": {"hit_rate": round(float(ge.hit.mean()), 3), "n": len(ge),
                           "mean_abs_timing_error_s": round(float(ge.timing_error_s.dropna().abs().mean()), 3) if ge.hit.any() else None,
                           "by_event": ge.groupby(["gripper", "kind"]).agg(hit_rate=("hit", "mean"),
                                                                          timing_s=("timing_error_s", lambda x: x.dropna().mean()))
                           .round(3).reset_index().to_dict("records")},
        "smoothness_max_joint_jump_rad": {"within_chunk_p95": round(float(np.percentile(jumps_in, 95)), 4),
                                          "at_replan_p95": round(float(np.percentile(jumps_rep, 95)), 4) if jumps_rep else None,
                                          "demo_p95": round(float(np.percentile(np.abs(np.diff(np.stack([f['action'] for f in data[eps[0]]])[:, ARM], axis=0)).max(1), 95)), 4)},
        "latency_ms": {"step_median": round(float(np.median(lat_all[~rep_all])), 1),
                       "replan_median": round(float(np.median(lat_all[rep_all])), 1),
                       "replan_p95": round(float(np.percentile(lat_all[rep_all], 95)), 1),
                       "budget_at_6hz": round(1000 / HZ, 1)},
        "robustness_arm_err_rad": robust,
    }
    del runner
    torch.cuda.empty_cache()
    return res


def summary_md(results):
    names = [r["model"].split("/")[0] for r in results]
    L = ["# Offline evaluation — held-out handover episodes", "",
         f"Episodes: {results[0]['episodes']} (unseen lighting), {results[0]['steps_per_episode']} steps each at 6 Hz. "
         "Teacher-forced: the policy sees the recorded observations, so errors do not compound.", "",
         "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    row = lambda label, f: L.append(f"| {label} | " + " | ".join(str(f(r)) for r in results) + " |")  # noqa: E731
    row("Arm error, mean (rad)", lambda r: r["action_error_rad"]["arm_mean"])
    row("Arm error, p99 (rad)", lambda r: r["action_error_rad"]["arm_p99"])
    row("Left / right arm (rad)", lambda r: f"{r['action_error_rad']['left_arm']} / {r['action_error_rad']['right_arm']}")
    row("Gripper error", lambda r: r["action_error_rad"]["gripper_mean"])
    for k in range(len(results[0]["per_subtask"])):
        row(f"Sub-task {k + 1}: {results[0]['per_subtask'][k]['text'][:38]} — err / DTW",
            lambda r, k=k: f"{r['per_subtask'][k]['arm_err_rad']} / {r['per_subtask'][k]['dtw_rad']}")
    row("Horizon err: 1 step / 10 / last (rad)", lambda r: f"{r['horizon_arm_err_rad'][0]} / {r['horizon_arm_err_rad'][min(9, len(r['horizon_arm_err_rad']) - 1)]} / {r['horizon_arm_err_rad'][-1]}")
    row("Gripper events hit (±1 s)", lambda r: f"{r['gripper_events']['hit_rate']:.0%} of {r['gripper_events']['n']}")
    row("Gripper timing error (s)", lambda r: r["gripper_events"]["mean_abs_timing_error_s"])
    row("Jump within chunk / at re-plan, p95 (rad; demo for reference)", lambda r: f"{r['smoothness_max_joint_jump_rad']['within_chunk_p95']} / {r['smoothness_max_joint_jump_rad']['at_replan_p95']} (demo {r['smoothness_max_joint_jump_rad']['demo_p95']})")
    row("Latency step / re-plan median (ms)", lambda r: f"{r['latency_ms']['step_median']} / {r['latency_ms']['replan_median']}")
    for v in results[0]["robustness_arm_err_rad"]:
        row(f"Robustness: {v} (rad)", lambda r, v=v: r["robustness_arm_err_rad"][v])
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--episodes", type=int, default=10, help="number of held-out episodes (max 10)")
    ap.add_argument("--horizon_eps", type=int, default=2)
    args = ap.parse_args()
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    val = json.loads(SPLITS_FILE.read_text())["val"]
    order = [46, 78, 8] + [e for e in val if e not in (46, 78, 8)]  # darkest, brightest, random first
    print("loading held-out episodes ...", flush=True)
    data = load_episodes(order[: args.episodes])
    results = []
    for m in args.models:
        t0 = time.time()
        r = evaluate(m, data, args)
        (OUT / f"{m.split('/')[0]}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"{m}: arm {r['action_error_rad']['arm_mean']} rad, gripper events {r['gripper_events']['hit_rate']:.0%}, "
              f"robust {r['robustness_arm_err_rad']}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        results.append(r)
    # summary over every evaluated model on disk (so models can be evaluated in separate runs)
    results = [json.loads((OUT / f"{m.split('/')[0]}.json").read_text(encoding="utf-8"))
               for m in MODELS + [m for m in args.models if m not in MODELS] if (OUT / f"{m.split('/')[0]}.json").exists()]
    (OUT / "summary.md").write_text(summary_md(results), encoding="utf-8")
    print(summary_md(results).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
