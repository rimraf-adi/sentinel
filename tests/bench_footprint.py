#!/usr/bin/env python3
"""Benchmark Sentinel's own footprint (not training runs): daemon RAM and CPU while idle, with the
dashboard connected, under API load, and while watching a job; plus CLI command cost.

Usage: python3 tests/bench_footprint.py [path/to/sentinel] [--seconds 10]
RSS comes from `ps` (an independent measurement); CPU is the delta of the process's cumulative CPU time
over each phase, so it is an average over the phase, not a decayed or lifetime figure. Unix only.
"""
import base64, json, os, platform, re, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SECS = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--seconds=")), 10))
BIN = os.path.abspath(ARGS[0] if ARGS else os.path.join(os.path.dirname(__file__), "..", "target", "release", "sentinel"))
PORT = 2598


def ps(pid, field):
    out = subprocess.run(["ps", "-o", f"{field}=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return out


def rss_mb(pid):
    v = ps(pid, "rss")
    return int(v) / 1024 if v else 0.0


def cpu_seconds(pid):
    """Cumulative CPU time (user+system) of a process: [[dd-]hh:]mm:ss(.cc)."""
    t = ps(pid, "time")
    if not t:
        return 0.0
    days = 0
    if "-" in t:
        d, t = t.split("-"); days = int(d)
    parts = [float(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def get(path):
    return urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5).read()


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


class WsClient(threading.Thread):
    """A dashboard stand-in: opens /ws/live and drains frames."""
    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.create_connection(("127.0.0.1", PORT), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /ws/live HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        self.sock.settimeout(2)
        self.bytes = 0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                d = self.sock.recv(65536)
                if not d:
                    break
                self.bytes += len(d)
            except socket.timeout:
                continue
            except OSError:
                break


def phase(name, pid, secs, work=None):
    """Run `work` (callable, called repeatedly) for `secs`; sample RSS every 0.25s."""
    c0, t0 = cpu_seconds(pid), time.time()
    rss = []
    while time.time() - t0 < secs:
        if work:
            work()
        rss.append(rss_mb(pid))
        time.sleep(0.25)
    wall = time.time() - t0
    return {"phase": name, "rss_avg": sum(rss) / len(rss), "rss_max": max(rss), "cpu": (cpu_seconds(pid) - c0) / wall * 100}


def timed_cli(args, cwd):
    """Wall time and peak RSS of one CLI invocation (macOS `time -l`, Linux `time -v`)."""
    flag, pat = (["-l"], r"(\d+)\s+maximum resident set size") if platform.system() == "Darwin" else (["-v"], r"Maximum resident set size \(kbytes\): (\d+)")
    t = time.time()
    r = subprocess.run(["/usr/bin/time", *flag, BIN, *args], cwd=cwd, capture_output=True, text=True)
    wall = time.time() - t
    m = re.search(pat, r.stderr)
    peak = int(m.group(1)) / (1024 * 1024 if platform.system() == "Darwin" else 1024) if m else float("nan")
    return wall, peak


def main():
    if not os.path.exists(BIN):
        sys.exit(f"binary not found: {BIN} (build with `cargo build --release -p sentinel-cli`)")
    proj = tempfile.mkdtemp(prefix="sentinel-bench-")
    subprocess.run([BIN, "init", "."], cwd=proj, capture_output=True)
    env = {k: v for k, v in os.environ.items() if k != "SENTINEL_SDK_PATH"}
    daemon = subprocess.Popen([BIN, "dev", "--port", str(PORT)], cwd=proj, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results, clients = [], []
    try:
        for _ in range(60):
            try:
                get("/api/health"); break
            except Exception:
                time.sleep(0.2)
        else:
            sys.exit("daemon did not start")
        time.sleep(2)  # let startup settle (GPU probe, first samples)
        pid = daemon.pid
        startup = rss_mb(pid)

        results.append(phase("idle, no clients", pid, SECS))

        c = WsClient(); c.start(); clients.append(c)
        results.append(phase("dashboard open (1 WebSocket)", pid, SECS))

        for _ in range(2):
            c = WsClient(); c.start(); clients.append(c)
        n = {"i": 0}
        def poll():  # what 3 open dashboard tabs poll
            n["i"] += 1
            for p in ("/api/jobs", "/api/repo/status", "/api/datasets") if n["i"] % 8 == 0 else ():
                get(p)
        results.append(phase("3 dashboards + polling", pid, SECS, poll))

        def hammer():
            for _ in range(12):
                get("/api/vitals"); get("/api/jobs")
        results.append(phase("heavy API (~100 req/s)", pid, SECS, hammer))

        job = post("/api/jobs", {"id": "00000000-0000-4000-8000-000000000001", "name": "bench-sleeper", "run_mode": "command",
                                  "command": "sleep 40", "parameters": {}, "env_vars": {}, "gpu_ids": [], "working_dir": ".", "created_at": "2026-01-01T00:00:00Z"})
        results.append(phase("watching 1 running job", pid, SECS))
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/api/jobs/{job['spec']['id']}/stop", b"", method="POST"), timeout=5)
        except Exception:
            pass

        history = json.loads(get("/api/vitals/history"))
        wall_s, peak_s = timed_cli(["status", "--port", str(PORT + 1)], proj)   # no daemon on that port: local sampling
        wall_d, peak_d = timed_cli(["status", "--port", str(PORT)], proj)
        wall_g, peak_g = timed_cli(["gpus"], proj)
        wall_doc, peak_doc = timed_cli(["doctor", "--port", str(PORT + 1)], proj)
    finally:
        for c in clients:
            c.stop = True
        daemon.terminate()
        try: daemon.wait(5)
        except Exception: daemon.kill()
        shutil.rmtree(proj, ignore_errors=True)

    print(f"\nSentinel footprint  ({platform.system()} {platform.machine()}, binary {os.path.getsize(BIN)/1048576:.1f} MB, {SECS:.0f}s per phase)\n")
    print(f"daemon RSS right after startup: {startup:.1f} MB\n")
    print(f"{'phase':<34}{'RSS avg':>10}{'RSS max':>10}{'CPU avg':>10}")
    for r in results:
        print(f"{r['phase']:<34}{r['rss_avg']:>8.1f}MB{r['rss_max']:>8.1f}MB{r['cpu']:>8.2f}%")
    print(f"\nhistory ring: {len(history)} samples held\n")
    print(f"{'CLI command':<34}{'wall':>10}{'peak RSS':>10}")
    for name, w, p in (("status (local sampling)", wall_s, peak_s), ("status (via daemon)", wall_d, peak_d), ("gpus", wall_g, peak_g), ("doctor", wall_doc, peak_doc)):
        print(f"{name:<34}{w:>8.2f}s {p:>8.1f}MB")


if __name__ == "__main__":
    main()
