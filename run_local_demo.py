"""Start the whole local demo with one command: builds what is missing, then serves every page on localhost.

    .venv\\Scripts\\python.exe run_local_demo.py                      # build if needed + start everything
    .venv\\Scripts\\python.exe run_local_demo.py --rebuild            # rebuild all generated pages first
    .venv\\Scripts\\python.exe run_local_demo.py --serve outputs/smolvla_handover/step_003000   # + a policy on :8090
    .venv\\Scripts\\python.exe run_local_demo.py --serve trajectory   # + the recorded handover as lookup policy on :8090

Ports
  8770  presentation: the whole project as slides, with every demo below embedded live
  8765  pipeline UI: runs, evaluate, verdicts, deploy / stop / rollback the policy server
  8779  handover explorer website (data, robot model, camera -> grid -> trajectory lookup)
  8780  sub-task splitting page
  8778  box-grid player (drag the box, see the retargeted trajectory)
  8777  results browser: eval_results/ (XYZ view, MuJoCo videos + traces, offline eval summaries)
  8090  policy server for Isaac Sim / clients (only with --serve)
Ctrl+C stops everything.
"""
import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
AUG = ROOT / "Datasets/fr3_duo_handover_84_ep/Augmented"
sys.stdout.reconfigure(line_buffering=True)

# (what it is, file that shows it is built, command that builds it)
BUILDS = [
    ("box-position grid (136 retargeted trajectories, ~6 min)", AUG / "ui/index.json", ["augmentation/augment_box_grid.py"]),
    ("camera -> grid assets", AUG / "ui/camera/camera.json", ["augmentation/camera_grid.py"]),
    ("handover explorer website", ROOT / "build/site/index.html", ["ui/demo/build_site.py"]),
    ("sub-task splitting page", ROOT / "build/subtask_demo/index.html", ["ui/demo/build_subtask_demo.py"]),
    ("video-synced gripper XYZ view", ROOT / "eval_results/xyz_verify/index.html", ["ui/demo/make_xyz_verify.py"]),
    ("health-check frames for deploy", ROOT / "eval_results/pi05_probe_frames.npz", ["diagnostics/export_probe_frames.py"]),
]
NEEDS = [ROOT / "Datasets/fr3_duo_handover_84_ep/fr3_duo_handover_v9_night_84/meta/info.json",
         ROOT / "Datasets/fr3_handover84_subtasks/meta/info.json"]


def build(rebuild):
    missing = [p for p in NEEDS if not p.exists()]
    if missing:
        print("missing input data (copy the datasets / run data_prep/make_subtasks.py first):")
        for p in missing:
            print("  ", p.relative_to(ROOT))
        sys.exit(1)
    for name, marker, cmd in BUILDS:
        if marker.exists() and not rebuild:
            print(f"  ok      {name}")
            continue
        print(f"  build   {name} ...", flush=True)
        r = subprocess.run([PY, *cmd], cwd=ROOT)
        if r.returncode != 0:
            print(f"  FAILED  {name} (exit {r.returncode}); continuing with the rest")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild", action="store_true", help="rebuild all generated pages")
    ap.add_argument("--serve", default=None, help="checkpoint folder, or 'trajectory', to serve a policy on :8090")
    ap.add_argument("--no_browser", action="store_true")
    args = ap.parse_args()

    print("building demo pages (skips what exists):")
    build(args.rebuild)

    servers = [
        ("presentation", 8770, [PY, "ui/demo/serve_static.py", "--dir", "ui/presentation", "--port", "8770"]),
        ("pipeline UI", 8765, [PY, "ui/pipeline/pipeline_ui.py"]),
        ("handover explorer", 8779, [PY, "ui/demo/serve_static.py", "--dir", "build/site", "--port", "8779"]),
        ("sub-task splitting", 8780, [PY, "ui/demo/serve_static.py", "--dir", "build/subtask_demo", "--port", "8780"]),
        ("box-grid player", 8778, [PY, "ui/demo/serve_static.py", "--dir", str(AUG), "--port", "8778"]),
        ("results browser", 8777, [PY, "ui/demo/serve_static.py", "--dir", "eval_results", "--port", "8777"]),
    ]
    if args.serve:
        src = ["--trajectory"] if args.serve == "trajectory" else ["--checkpoint", args.serve]
        servers.append(("policy server", 8090, [PY, "serving/fr3duo_policy_server.py", *src, "--port", "8090"]))

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    procs = []
    for name, port, cmd in servers:
        log = open(logs / f"local_demo_{port}.log", "w")
        procs.append((name, port, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
    time.sleep(3)

    paths = {8778: "/ui/", 8777: "/xyz_verify/", 8090: "/info"}
    print("\nrunning (logs in logs/local_demo_<port>.log), Ctrl+C to stop:")
    for name, port, p in procs:
        state = "up" if p.poll() is None else f"EXITED (see logs/local_demo_{port}.log)"
        print(f"  {name:20s} http://127.0.0.1:{port}{paths.get(port, '/')}   {state}")
    if not args.no_browser:
        webbrowser.open("http://127.0.0.1:8770/")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for _, _, p in procs:
            p.terminate()
        print("stopped")


if __name__ == "__main__":
    main()
