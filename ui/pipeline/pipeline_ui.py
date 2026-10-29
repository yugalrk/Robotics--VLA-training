"""Local web UI for the post-training pipeline: watch training runs, trigger evaluations, read verdicts, promote.

    .venv\\Scripts\\python.exe ui\\pipeline\\pipeline_ui.py   # then open http://127.0.0.1:8765

- Runs: every folder in outputs/ with its status (training / done / stopped), progress and latest validation.
- Evaluate: starts evaluation/post_train_eval.py (offline eval + Isaac Sim eval + gates) on a checkpoint; one job at a time,
  the rest wait in a queue; the log streams into the page.
- Auto-evaluate (off by default): when a run's final/ checkpoint appears, it is queued automatically.
- Verdicts: DEPLOY / MORE TRAINING / HOLD from eval_results/pipeline/*/verdict.json, with the full summary.
- Deploy (Promote): loads the checkpoint on staging port 8091, health-checks it on a recorded frame, and only
  then replaces whatever serves port 8090 (the port the Isaac Sim scripts use); on failure the old server keeps
  serving. Recorded in deploy/production.json; its offline result becomes the regression reference.
- Serving: what is on port 8090 now (also servers started outside the UI), Stop serving (frees the port),
  Start (serve the deployed model again) and Rollback (deploy the previous model).
"""
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
PIPE = ROOT / "eval_results" / "pipeline"
JOBS_DIR = PIPE / "jobs"
PROD = ROOT / "deploy" / "production.json"
PAGE = Path(__file__).with_name("pipeline_ui.html")
STEP = re.compile(r"^\[(\w+)\] step (\d+)/(\d+) loss ([\d.]+).*?eta (\d+) min")
VAL = re.compile(r"^VAL (\d+|end of \w+): (.*)")


# ------------------------------------------------------------------ runs
def training_processes():
    """outputs/<run> folders that a running training/train_curriculum.py writes to."""
    busy = set()
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info.get("cmdline") or []
        if any("train_curriculum.py" in c for c in cmd) and "--out" in cmd:
            busy.add(Path(cmd[cmd.index("--out") + 1]).name)
    return busy


def run_info(d, busy):
    log = d / "train.log"
    info = {"name": d.name, "checkpoints": sorted(p.name for p in d.iterdir() if (p / "model.safetensors").exists()),
            "has_final": (d / "final" / "model.safetensors").exists(), "step": None, "steps": None, "eta_min": None,
            "loss": None, "val": None, "updated": None}
    if log.exists():
        info["updated"] = log.stat().st_mtime
        for line in log.read_text(errors="replace").splitlines()[-400:]:
            if m := STEP.match(line):
                info.update(step=int(m[2]), steps=int(m[3]), loss=float(m[4]), eta_min=int(m[5]))
            elif m := VAL.match(line):
                info["val"] = {k: float(r) for k, r in re.findall(r"([\w/]+) [\d.]+ \(([\d.]+) rad\)", m[2])}
    info["status"] = "training" if d.name in busy else ("done" if info["has_final"] else ("stopped" if log.exists() else "imported"))
    return info


def pipeline_results():
    out = []
    for d in sorted(PIPE.glob("*/verdict.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        v = json.loads(d.read_text())
        out.append({"dir": d.parent.name, "checkpoint": v["checkpoint"].replace("\\", "/"), "verdict": v["verdict"],
                    "todo": v.get("todo", []), "time": d.stat().st_mtime, "online": v.get("online_state")})
    return out


# ------------------------------------------------------------------ jobs
class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue, self.done, self.current, self.auto = [], [], None, False
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self.worker, daemon=True).start()
        threading.Thread(target=self.watcher, daemon=True).start()

    def add(self, checkpoint, sim="auto", reference=None, reason="manual"):
        with self.lock:
            if any(j["checkpoint"] == checkpoint for j in self.queue) or (self.current and self.current["checkpoint"] == checkpoint):
                return None
            job = {"id": time.strftime("%Y%m%d-%H%M%S") + f"-{len(self.done) + len(self.queue)}", "checkpoint": checkpoint,
                   "sim": sim, "reference": reference or default_reference(), "reason": reason, "state": "queued"}
            self.queue.append(job)
            return job

    def cancel(self, job_id):
        with self.lock:
            self.queue = [j for j in self.queue if j["id"] != job_id]
            if self.current and self.current["id"] == job_id and self.current.get("proc"):
                for c in psutil.Process(self.current["proc"].pid).children(recursive=True):
                    c.kill()
                self.current["proc"].kill()
                self.current["state"] = "cancelled"

    def worker(self):
        while True:
            with self.lock:
                job = self.queue.pop(0) if (self.queue and self.current is None) else None
                if job:
                    self.current = job
            if not job:
                time.sleep(2)
                continue
            cmd = [sys.executable, "-u", "evaluation/post_train_eval.py", "--checkpoint", job["checkpoint"], "--sim", job["sim"]]
            if job["reference"]:
                cmd += ["--reference", job["reference"]]
            job.update(state="running", started=time.time(), log=str(JOBS_DIR / f"{job['id']}.log"))
            with open(job["log"], "w") as f:
                f.write("$ " + " ".join(cmd) + "\n")
                f.flush()
                job["proc"] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
                code = job["proc"].wait()
            text = Path(job["log"]).read_text(errors="replace")
            m = re.search(r"-> (.*pipeline[\\/][^\s]+)", text)
            job.update(state=job["state"] if job["state"] == "cancelled" else ("finished" if code == 0 else "failed"),
                       exit=code, ended=time.time(), result_dir=Path(m[1]).name if m else None)
            job.pop("proc", None)
            with self.lock:
                self.done.insert(0, job)
                self.current = None

    def watcher(self):
        """Auto-evaluate: queue final/ checkpoints that have no pipeline result yet."""
        while True:
            time.sleep(20)
            if not self.auto:
                continue
            evaluated = {r["checkpoint"] for r in pipeline_results()} | {j["checkpoint"] for j in self.done}
            busy = training_processes()
            for d in OUTPUTS.iterdir():
                ck = f"outputs/{d.name}/final"
                if d.name not in busy and (d / "final" / "model.safetensors").exists() and ck not in evaluated:
                    self.add(ck, reason="auto: training finished")

    def snapshot(self):
        with self.lock:
            strip = lambda j: {k: v for k, v in j.items() if k != "proc"}
            return {"auto": self.auto, "current": strip(self.current) if self.current else None,
                    "queue": [strip(j) for j in self.queue], "done": [strip(j) for j in self.done[:20]]}


def production():
    return json.loads(PROD.read_text()) if PROD.exists() else None


# ------------------------------------------------------------------ serving (port 8090 = what Isaac Sim uses)
SERVE_PORT, STAGE_PORT = 8090, 8091
PROBE = ROOT / "eval_results" / "pi05_probe_frames.npz"  # real recorded frames for the health check
SERVER_SCRIPT = ROOT / "serving" / "fr3duo_policy_server.py"


def http_json(url, data=None, timeout=5):
    import urllib.request
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def port_owners(port):
    procs = []
    for c in psutil.net_connections("tcp"):
        if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
            try:
                procs.append(psutil.Process(c.pid))
            except psutil.NoSuchProcess:
                pass
    return procs


def kill_port(port):
    for p in port_owners(port):
        names = " ".join(p.cmdline())
        if "wslrelay" in p.name().lower() or "wsl" in p.name().lower():  # a server inside WSL (the pi0.5 server)
            subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "pkill", "-f", "fr3duo_pi05_server.py"], timeout=30)
            continue
        for q in [*p.children(recursive=True), p]:
            try:
                q.kill()
            except psutil.NoSuchProcess:
                pass
        # the venv launcher and its child both hold the port; kill the launcher's parent python too
        if "fr3duo_policy_server" in names:
            try:
                parent = p.parent()
                if parent and "fr3duo_policy_server" in " ".join(parent.cmdline()):
                    parent.kill()
            except psutil.NoSuchProcess:
                pass
    for _ in range(30):
        if not port_owners(port):
            return True
        time.sleep(0.5)
    return False


def health_check(port):
    """/info must pass, then one /act with a real recorded frame: 16 finite values."""
    import io
    import numpy as np
    info = http_json(f"http://127.0.0.1:{port}/info")
    real = PROBE.exists()
    if real:
        f = np.load(PROBE)
        obs = {"observation.state": f["state"][0].astype(np.float32), "observation.images.external_camera": f["ext"][0],
               "observation.images.left_wrist_camera": f["lw"][0], "observation.images.right_wrist_camera": f["rw"][0]}
    else:
        obs = {"observation.state": np.zeros(16, np.float32),
               **{f"observation.images.{c}": np.full((224, 224, 3), 128, np.uint8)
                  for c in ["external_camera", "left_wrist_camera", "right_wrist_camera"]}}
    buf = io.BytesIO()
    np.savez(buf, task=np.array("Hand the box from the left gripper to the right gripper and place it on the serving tray."), **obs)
    http_json(f"http://127.0.0.1:{port}/reset", b"", timeout=30)
    t0 = time.time()
    act = http_json(f"http://127.0.0.1:{port}/act", buf.getvalue(), timeout=120)
    a = np.asarray(act["action"], float)
    ok = a.shape == (16,) and np.all(np.isfinite(a)) and not info.get("fake")
    return {"ok": bool(ok), "policy_type": info.get("policy_type"), "first_call_ms": round((time.time() - t0) * 1000),
            "real_frame": real, "action_shape": list(a.shape), "fake": info.get("fake")}


class Serving:
    def __init__(self):
        self.lock = threading.Lock()
        self.busy, self.message, self.started_at = None, "", None

    def status(self):
        owners = port_owners(SERVE_PORT)
        # progress while an action runs, errors until the next action; the steady state comes from the port itself
        msg = self.message if self.busy or "failed" in self.message or "could not" in self.message else ""
        st = {"port": SERVE_PORT, "live": bool(owners), "busy": self.busy, "message": msg,
              "pid": owners[0].pid if owners else None, "checkpoint": None, "policy_type": None,
              "since": owners[0].create_time() if owners else None}
        if owners:
            try:
                info = http_json(f"http://127.0.0.1:{SERVE_PORT}/info", timeout=2)
                st.update(checkpoint=str(info.get("checkpoint", "")).replace("\\", "/"), policy_type=info.get("policy_type"),
                          fake=info.get("fake"), n_action_steps=info.get("n_action_steps"), chunk_size=info.get("chunk_size"), grasp_extend_s=info.get("grasp_extend_s"))
            except Exception:
                st["message"] = st["message"] or "port is held but /info does not answer (starting?)"
        return st

    def _spawn(self, checkpoint, port, n_action_steps=None, grasp_extend=None):
        """Start the server outside this process's tree (Windows WMI), so a deployed model keeps serving when the
        UI is restarted or closed. Output goes to deploy/server_<port>.log."""
        log = ROOT / "deploy" / f"server_{port}.log"
        log.parent.mkdir(exist_ok=True)
        nas = (f" --n_action_steps {int(n_action_steps)}" if n_action_steps else "") + (f" --grasp_extend {float(grasp_extend)}" if grasp_extend else "")
        cmdline = f'cmd /c ""{sys.executable}" "{SERVER_SCRIPT}" --checkpoint "{checkpoint}" --port {port}{nas} > "{log}" 2>&1"'
        ps = (f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
              f"@{{CommandLine='{cmdline.replace(chr(39), chr(39) * 2)}'; CurrentDirectory='{ROOT}'}}; $r.ReturnValue")
        rv = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.strip()
        if rv != "0":
            raise RuntimeError(f"could not start the server process (WMI return {rv or 'none'})")
        for _ in range(180):  # up to 3 min to load the model
            try:
                http_json(f"http://127.0.0.1:{port}/info", timeout=2)
                return
            except Exception:
                time.sleep(1)
        text = log.read_text(errors="replace") if log.exists() else ""
        kill_port(port)
        raise RuntimeError("server did not answer within 3 min: " + text[-400:])

    def _run(self, name, fn):
        with self.lock:
            if self.busy:
                return False
            self.busy = name
        def go():
            try:
                fn()
            except Exception as e:
                self.message = f"{name} failed: {type(e).__name__}: {e}"
            finally:
                self.busy = None
        threading.Thread(target=go, daemon=True).start()
        return True

    def deploy(self, checkpoint, verdict=None, result_dir=None, n_action_steps=None, grasp_extend=None):
        """Stage on 8091 + health check; only then replace whatever serves 8090. On failure the old server stays."""
        def fn():
            self.message = f"loading {checkpoint} on staging port {STAGE_PORT} ..."
            kill_port(STAGE_PORT)
            self._spawn(checkpoint, STAGE_PORT, n_action_steps, grasp_extend)
            try:
                self.message = "health check on a recorded frame ..."
                hc = health_check(STAGE_PORT)
                if not hc["ok"]:
                    raise RuntimeError(f"health check failed: {hc}")
            finally:
                kill_port(STAGE_PORT)
            self.message = f"health check passed ({hc['first_call_ms']} ms); switching port {SERVE_PORT} ..."
            if not kill_port(SERVE_PORT):
                raise RuntimeError(f"could not free port {SERVE_PORT}")
            self._spawn(checkpoint, SERVE_PORT, n_action_steps, grasp_extend)
            prev = production()
            rec = {"checkpoint": checkpoint, "verdict": verdict, "result_dir": result_dir, "health_check": hc,
                   "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "port": SERVE_PORT, "n_action_steps": n_action_steps, "grasp_extend": grasp_extend,
                   "previous": {k: prev[k] for k in ("checkpoint", "promoted_at", "n_action_steps", "grasp_extend") if k in prev} if prev else None}
            PROD.parent.mkdir(exist_ok=True)
            PROD.write_text(json.dumps(rec, indent=1))
            self.message = f"deployed {checkpoint} on http://127.0.0.1:{SERVE_PORT}"
        return self._run("deploy", fn)

    def stop(self):
        def fn():
            self.message = f"stopping the server on port {SERVE_PORT} ..."
            self.message = f"port {SERVE_PORT} is free: nothing is served" if kill_port(SERVE_PORT) else f"could not stop the server on {SERVE_PORT}"
        return self._run("stop", fn)

    def start(self, checkpoint, n_action_steps=None, grasp_extend=None):
        """Serve a checkpoint on 8090 without changing what is recorded as deployed (e.g. after a stop)."""
        def fn():
            self.message = f"starting {checkpoint} on port {SERVE_PORT} ..."
            kill_port(SERVE_PORT)
            self._spawn(checkpoint, SERVE_PORT, n_action_steps, grasp_extend)
            self.message = f"serving {checkpoint} on http://127.0.0.1:{SERVE_PORT}"
        return self._run("start", fn)


SERVING = Serving()


def default_reference():
    p = production()
    if p:
        ref = ROOT / "eval_results" / "offline" / f"{Path(p['checkpoint']).parent.name}.json"
        return str(ref.relative_to(ROOT)) if ref.exists() else None
    return None


JOBS = Jobs()


# ------------------------------------------------------------------ http
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        q = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        if path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html")
        if path == "/api/state":
            busy = training_processes()
            runs = [run_info(d, busy) for d in sorted(OUTPUTS.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True) if d.is_dir()]
            refs = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "eval_results" / "offline").glob("*.json"))
            return self._send(200, {"runs": runs, "results": pipeline_results(), "jobs": JOBS.snapshot(), "serving": SERVING.status(),
                                    "production": production(), "references": refs, "default_reference": default_reference()})
        if path == "/api/log":
            job = next((j for j in [JOBS.current, *JOBS.done] if j and j["id"] == q.get("id")), None)
            text = Path(job["log"]).read_text(errors="replace")[-20000:] if job and job.get("log") else ""
            return self._send(200, text, "text/plain")
        if path == "/api/summary":
            f = PIPE / Path(q.get("dir", "")).name / "summary.md"
            return self._send(200 if f.exists() else 404, f.read_text(encoding="utf-8") if f.exists() else "not found", "text/plain")
        self._send(404, {"error": "unknown path"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/api/evaluate":
            ck = body["checkpoint"]
            if not (ROOT / ck / "model.safetensors").exists():
                return self._send(400, {"error": f"no checkpoint at {ck}"})
            job = JOBS.add(ck, body.get("sim", "auto"), body.get("reference") or None)
            return self._send(200, {"job": {k: v for k, v in job.items()} if job else None, "note": None if job else "already queued"})
        if self.path == "/api/cancel":
            JOBS.cancel(body["id"])
            return self._send(200, {"ok": True})
        if self.path == "/api/auto":
            JOBS.auto = bool(body.get("enabled"))
            return self._send(200, {"auto": JOBS.auto})
        if self.path == "/api/promote":  # deploy: stage + health check, then serve on 8090
            ok = SERVING.deploy(body["checkpoint"], body.get("verdict"), body.get("dir"), body.get("n_action_steps"), body.get("grasp_extend"))
            return self._send(200 if ok else 409, {"started": ok})
        if self.path == "/api/serve/stop":
            return self._send(200, {"started": SERVING.stop()})
        if self.path == "/api/serve/start":
            ck = body.get("checkpoint") or (production() or {}).get("checkpoint")
            if not ck:
                return self._send(400, {"error": "nothing deployed yet"})
            nas = body.get("n_action_steps", (production() or {}).get("n_action_steps"))
            ge = body.get("grasp_extend", (production() or {}).get("grasp_extend"))
            return self._send(200, {"started": SERVING.start(ck, nas, ge)})
        if self.path == "/api/serve/rollback":
            prev = ((production() or {}).get("previous") or {}).get("checkpoint")
            if not prev:
                return self._send(400, {"error": "no previous deployment"})
            return self._send(200, {"started": SERVING.deploy(prev, "rollback", n_action_steps=((production() or {}).get("previous") or {}).get("n_action_steps"), grasp_extend=((production() or {}).get("previous") or {}).get("grasp_extend"))})
        self._send(404, {"error": "unknown path"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("pipeline UI on http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
