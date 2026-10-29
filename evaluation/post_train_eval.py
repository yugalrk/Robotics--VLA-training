"""Post-training evaluation pipeline: offline eval + online (Isaac Sim) eval -> one verdict: DEPLOY or MORE TRAINING.

    python evaluation/post_train_eval.py --checkpoint outputs/curriculum_smolvla_expert10k/final
    python evaluation/post_train_eval.py --checkpoint outputs/<run>/final --reference eval_results/offline/<deployed>.json
    python evaluation/post_train_eval.py --checkpoint ... --sim skip                     # offline only
    python evaluation/post_train_eval.py --checkpoint ... --sim eval_results/isaac_<time>/results.json   # use a Script Editor run

Steps
1. Offline: evaluation/eval_offline.py on the 10 held-out handover episodes (accuracy, sub-tasks, gripper events, robustness).
2. Online: headless Isaac Sim (isaac/eval_isaac.py, needs no clicks).
   a. sanity check: replay the recorded demo in the scene. Only if the demo itself succeeds does a policy result
      mean anything; otherwise the sim result is UNVERIFIED (scene problem, not a model problem).
   b. serve the checkpoint (serving/serve_policy.py, port 5555) and run --episodes closed-loop episodes with held-out lighting.
3. Gates (configs/deploy_gates.json) -> verdict + reasons + what to do next.

Verdicts
  DEPLOY             offline gates pass and the sim success rate clears the bar
  MORE TRAINING      an offline gate fails, or the sim (verified) fails
  HOLD: SIM UNVERIFIED  offline passes, but the scene cannot reproduce the demo, so closed-loop success is unknown
Writes eval_results/pipeline/<run>_<time>/summary.md and verdict.json.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ISAAC = Path(r"C:\Users\User\Downloads\isaac-sim-standalone-6.2.0-windows-x86_64")
USD = ROOT / "Environment/Environment_Hackathon_2026/Environment_Hackathon_2026/Hackathon_FR3Duo_flattened.usd"
GATES = json.loads((ROOT / "configs" / "deploy_gates.json").read_text())


def run(cmd, log, timeout):
    print("$", " ".join(map(str, cmd)), flush=True)
    with open(log, "w") as f:
        return subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


# ------------------------------------------------------------------ 1. offline
def offline_eval(ckpt, out):
    rel = ckpt.resolve().relative_to((ROOT / "outputs").resolve()).as_posix()
    code = run([sys.executable, "evaluation/eval_offline.py", "--models", rel, "--episodes", "10"], out / "offline.log", 4 * 3600)
    res = ROOT / "eval_results/offline" / f"{rel.split('/')[0]}.json"
    if code or not res.exists():
        raise SystemExit(f"offline eval failed (exit {code}), see {out / 'offline.log'}")
    return json.loads(res.read_text())


# ------------------------------------------------------------------ 2. online
def isaac(args_, log, timeout=3 * 3600):
    return run([ISAAC / "python.bat", "isaac/eval_isaac.py", "--usd", USD, *args_], log, timeout)


def sim_sanity(out):
    """Replay the recorded demo once per scene file (cached by the scene's modification time)."""
    cache = ROOT / "eval_results/pipeline/sim_sanity.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("usd_mtime") == USD.stat().st_mtime:
            return c
    rdir = out / "sim_replay"
    isaac(["--mode", "replay", "--out", rdir], out / "sim_replay.log", 1800)
    rep = rdir / "replay.json"
    r = json.loads(rep.read_text()) if rep.exists() else {"success": False, "error": "replay did not finish, see sim_replay.log"}
    c = {"usd_mtime": USD.stat().st_mtime, **r}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(c, indent=1))
    return c


def wait_port(port, proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(2)
    return False


def online_eval(ckpt, out, episodes):
    sanity = sim_sanity(out)
    if not sanity.get("success"):
        return {"verified": False, "sanity": sanity}
    server = subprocess.Popen([sys.executable, "serving/serve_policy.py", "--checkpoint", str(ckpt), "--port", "5555"],
                              cwd=ROOT, stdout=open(out / "serve_policy.log", "w"), stderr=subprocess.STDOUT)
    try:
        if not wait_port(5555, server):
            raise SystemExit(f"policy server did not start, see {out / 'serve_policy.log'}")
        sdir = out / "sim_policy"
        isaac(["--mode", "policy", "--server", "127.0.0.1:5555", "--episodes", episodes, "--out", sdir], out / "sim_policy.log")
        return {"verified": True, "sanity": sanity, **json.loads((sdir / "results.json").read_text())}
    finally:
        server.terminate()


# ------------------------------------------------------------------ 3. gates
def gate(rows, name, value, ok, limit, blocking=True):
    rows.append({"gate": name, "value": value, "limit": limit, "pass": bool(ok), "blocking": blocking})


def decide(off, sim, ref):
    g, rows = GATES["offline"], []
    b, w = g["block"], g["warn"]
    e = off["action_error_rad"]
    gate(rows, "arm error, mean (rad)", e["arm_mean"], e["arm_mean"] <= b["arm_error_mean_max"], f"<= {b['arm_error_mean_max']}")
    gate(rows, "arm error, p99 (rad)", e["arm_p99"], e["arm_p99"] <= b["arm_error_p99_max"], f"<= {b['arm_error_p99_max']}")
    worst = max(off["per_subtask"], key=lambda s: s["arm_err_rad"])
    gate(rows, f"worst sub-task: {worst['text'][:40]}", worst["arm_err_rad"],
         worst["arm_err_rad"] <= b["worst_subtask_error_max"], f"<= {b['worst_subtask_error_max']}")
    hit = off["gripper_events"]["hit_rate"]
    gate(rows, "gripper events caught within 1 s", hit, hit >= b["gripper_event_hit_rate_min"], f">= {b['gripper_event_hit_rate_min']}")
    rob = off["robustness_arm_err_rad"]
    for k in ["images_darker_x0.6", "images_brighter_x1.4", "state_noise_0.01rad", "reworded_instruction"]:
        ratio = rob[k] / rob["baseline"]
        gate(rows, f"robustness: {k} (x baseline)", round(ratio, 2), ratio <= b["robustness_ratio_max"], f"<= {b['robustness_ratio_max']}")
    if ref:
        ratio = e["arm_mean"] / ref["action_error_rad"]["arm_mean"]
        gate(rows, f"vs reference {ref['model']} (x its arm error)", round(ratio, 2),
             ratio <= b["regression_vs_reference_max"], f"<= {b['regression_vs_reference_max']}")
    lat = off["latency_ms"]["replan_median"]
    gate(rows, "re-plan latency (ms, real-robot budget at 6 Hz)", lat, lat <= w["replan_latency_budget_ms"],
         f"<= {w['replan_latency_budget_ms']}", blocking=False)
    sm = off["smoothness_max_joint_jump_rad"]
    gate(rows, "jump at re-plan (x demo jumps)", round(sm["at_replan_p95"] / sm["demo_p95"], 2),
         sm["at_replan_p95"] / sm["demo_p95"] <= w["replan_jump_vs_demo_max"], f"<= {w['replan_jump_vs_demo_max']}", blocking=False)
    te = off["gripper_events"]["mean_abs_timing_error_s"]
    gate(rows, "gripper timing error (s)", te, te <= w["gripper_timing_error_s_max"], f"<= {w['gripper_timing_error_s_max']}", blocking=False)
    offline_ok = all(r["pass"] for r in rows if r["blocking"])

    sim_rows, sim_state = [], "skipped"
    if sim is not None:
        if not sim.get("verified"):
            sim_state = "unverified"
        else:
            sb = GATES["online"]["block"]
            sr = sim["summary"]["success_rate"]
            med = float(np.median([x["stage"] for x in sim["episodes"]]))
            gate(sim_rows, "sim success rate", sr, sr >= sb["success_rate_min"], f">= {sb['success_rate_min']}")
            gate(sim_rows, "sim median stage (0-4)", med, med >= sb["median_stage_min"], f">= {sb['median_stage_min']}")
            sim_state = "pass" if all(r["pass"] for r in sim_rows) else "fail"

    todo = []
    if not offline_ok:
        verdict = "MORE TRAINING"
        for r in rows:
            if r["blocking"] and not r["pass"]:
                if r["gate"].startswith("worst sub-task") or r["gate"].startswith("arm error"):
                    todo.append(f"{r['gate']} = {r['value']}: train longer or give that sub-task's skill more weight in the curriculum")
                elif r["gate"].startswith("robustness"):
                    todo.append(f"{r['gate']} = {r['value']}: add that variation to training (augmentation / rewording)")
                elif r["gate"].startswith("gripper"):
                    todo.append(f"{r['gate']} = {r['value']}: check gripper labels / timing in the data")
                elif r["gate"].startswith("vs reference"):
                    todo.append(f"{r['gate']}: worse than the deployed model, keep the deployed one")
    elif sim_state == "fail":
        verdict = "MORE TRAINING"
        todo.append("accurate offline but fails closed loop: errors compound when the robot acts on its own predictions; "
                    "add recovery data (DAgger-style corrections) or noise injection during training")
    elif sim_state in ("unverified", "skipped"):
        verdict = "HOLD: SIM UNVERIFIED"
        todo.append("offline gates pass; closed-loop success is unknown because "
                    + ("the scene cannot reproduce the recorded demo (fix the scene: box pose / size) and re-run"
                       if sim_state == "unverified" else "the sim eval was skipped"))
    else:
        verdict = "DEPLOY"
    todo += [f"note: {r['gate']} = {r['value']} (limit {r['limit']})" for r in rows if not r["blocking"] and not r["pass"]]
    return verdict, rows, sim_rows, sim_state, todo


def summary_md(ckpt, verdict, rows, sim_rows, sim_state, sim, todo, off):
    L = [f"# Post-training evaluation: `{ckpt}`", "", f"## Verdict: **{verdict}**", ""]
    L += [f"- {t}" for t in todo] + [""]
    L += ["## Offline (10 held-out episodes, teacher-forced at 6 Hz)", "", "| Gate | Value | Limit | |", "|---|---|---|---|"]
    L += [f"| {r['gate']} | {r['value']} | {r['limit']} | {'pass' if r['pass'] else ('FAIL' if r['blocking'] else 'warn')} |" for r in rows]
    L += ["", "| Sub-task | Arm error (rad) |", "|---|---|"]
    L += [f"| {s['text']} | {s['arm_err_rad']} |" for s in off["per_subtask"]]
    L += ["", f"## Online (Isaac Sim): {sim_state}", ""]
    if sim is None:
        L.append("Skipped.")
    elif not sim.get("verified"):
        s = sim["sanity"]
        L.append(f"Sanity check failed: replaying the recorded demo in this scene does not succeed "
                 f"(stage {s.get('stage_name', '?')}, box max lift {s.get('max_lift_mm', '?')} mm). "
                 "Any policy result in this scene would reflect the scene, not the model, so the policy episodes were not run.")
    else:
        L += ["| Gate | Value | Limit | |", "|---|---|---|---|"]
        L += [f"| {r['gate']} | {r['value']} | {r['limit']} | {'pass' if r['pass'] else 'FAIL'} |" for r in sim_rows]
        L += ["", "| Episode | Stage | Lift (mm) | Box to demo end (mm) |", "|---|---|---|---|"]
        L += [f"| {x['episode']} | {x['stage']} {x['stage_name']} | {x['max_lift_mm']} | {x['dist_to_demo_final_mm']} |" for x in sim["episodes"]]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="checkpoint folder under outputs/, e.g. outputs/<run>/final")
    ap.add_argument("--reference", default=None, help="offline JSON of the currently deployed model (regression gate)")
    ap.add_argument("--sim", default="auto", help="auto (headless Isaac Sim), skip, or a results.json from a Script Editor run")
    ap.add_argument("--episodes", type=int, default=GATES["online"]["episodes"])
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    out = ROOT / "eval_results/pipeline" / f"{ckpt.parent.name}_{ckpt.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/3] offline eval ...", flush=True)
    off = offline_eval(ckpt, out)
    print("[2/3] online eval ...", flush=True)
    if args.sim == "skip":
        sim = None
    elif args.sim == "auto":
        sim = online_eval(ckpt, out, args.episodes)
    else:
        r = json.loads(Path(args.sim).read_text())
        sim = {"verified": True, "sanity": {"note": "taken from " + args.sim}, **r}
    ref = json.loads(Path(args.reference).read_text()) if args.reference else None
    print("[3/3] gates ...", flush=True)
    verdict, rows, sim_rows, sim_state, todo = decide(off, sim, ref)
    md = summary_md(ckpt, verdict, rows, sim_rows, sim_state, sim, todo, off)
    (out / "summary.md").write_text(md, encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps({"checkpoint": str(ckpt), "verdict": verdict, "todo": todo,
                                                  "offline_gates": rows, "online_gates": sim_rows, "online_state": sim_state}, indent=1))
    print("\n" + md)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
