"""
The terminal tool for the live demo.

    python -m demo up [--agent rl|rule|off] [--chaos] [--level hard] [--no-browser]
    python -m demo status | watch
    python -m demo crash win-gpu-1          # really kills that node's process
    python -m demo slow mac-m2 0.4 --for 60 # 40% of normal speed for 60 s
    python -m demo net win-gpu-0 0.2        # 20% of normal link speed, until healed
    python -m demo heal win-gpu-1           # clear faults; restart the process if it is down
    python -m demo load +10 | -5 | auto | manual
    python -m demo chaos on [--level hard --seed 7] | off
    python -m demo agent rl | rule | off
    python -m demo down
    python -m demo node --id win-gpu-0 --controller <mac-ip>:8000 --no-emulate   (on a real laptop)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import webbrowser
from typing import Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .config import NODE_IDS, ROOT

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
PIDS = os.path.join(RUNS, "pids.json")
STATUS_STYLE = {"training": "green", "joining": "cyan", "waiting": "yellow", "idle": "white",
                "removed": "bright_black", "dead": "bold red", "starting": "bright_black"}


def parse(argv) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=argparse.SUPPRESS,
                        help="controller address (default: the one `up` started, or 127.0.0.1:8000)")
    p = argparse.ArgumentParser(prog="python -m demo", description="Self-healing training demo", parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *a, **k: _add(*a, parents=[common], **k)
    up = sub.add_parser("up", help="start the controller and three nodes, open the web page")
    up.add_argument("--agent", default="rl", choices=["rl", "rule", "off"])
    up.add_argument("--chaos", action="store_true", help="replay a simulator fault scenario")
    up.add_argument("--level", default="realistic", choices=["easy", "realistic", "hard", "chaos"])
    up.add_argument("--seed", type=int, default=3)
    up.add_argument("--scenario-steps", type=int, default=450,
                    help="squeeze the scenario's faults into this many training steps")
    up.add_argument("--port", type=int, default=8000)
    up.add_argument("--manual-load", action="store_true")
    up.add_argument("--no-spawn", action="store_true", help="nodes run on other machines")
    up.add_argument("--no-browser", action="store_true")
    sub.add_parser("down", help="stop everything")
    sub.add_parser("status", help="one table of the cluster")
    sub.add_parser("watch", help="live table (Ctrl+C to stop)")
    for kind in ("crash", "heal"):
        c = sub.add_parser(kind)
        c.add_argument("node", choices=NODE_IDS)
    for kind, what in (("slow", "speed"), ("net", "link speed")):
        c = sub.add_parser(kind, help=f"cut a node's {what} to a fraction of normal")
        c.add_argument("node", choices=NODE_IDS)
        c.add_argument("factor", type=float)
        c.add_argument("--for", dest="duration", type=float, help="seconds (default: until healed)")
    ld = sub.add_parser("load", help="change the workload")
    ld.add_argument("change", help="+N, -N, auto or manual")
    ch = sub.add_parser("chaos")
    ch.add_argument("on_off", choices=["on", "off"])
    ch.add_argument("--level", choices=["easy", "realistic", "hard", "chaos"])
    ch.add_argument("--seed", type=int)
    ag = sub.add_parser("agent")
    ag.add_argument("mode", choices=["rl", "rule", "off"])
    nd = sub.add_parser("node", help="run one node (for the real laptops)")
    nd.add_argument("--id", required=True, choices=NODE_IDS)
    nd.add_argument("--controller", required=True)
    nd.add_argument("--no-emulate", action="store_true")
    return p.parse_args(argv)


def request_for(a: argparse.Namespace) -> Tuple[str, str, Optional[dict]]:
    if a.cmd in ("status", "watch"):
        return "GET", "/api/state", None
    if a.cmd in ("crash", "heal"):
        return "POST", "/api/fault", {"node": a.node, "kind": a.cmd}
    if a.cmd in ("slow", "net"):
        return "POST", "/api/fault", {"node": a.node, "kind": a.cmd, "factor": a.factor, "duration": a.duration}
    if a.cmd == "load":
        if a.change in ("auto", "manual"):
            return "POST", "/api/load", {"auto": a.change == "auto"}
        return "POST", "/api/load", {"delta": int(a.change)}
    if a.cmd == "chaos":
        body = {"on": a.on_off == "on"}
        if a.level:
            body["level"] = a.level
        if a.seed is not None:
            body["seed"] = a.seed
        return "POST", "/api/chaos", body
    if a.cmd == "agent":
        return "POST", "/api/agent", {"mode": a.mode}
    if a.cmd == "down":
        return "POST", "/api/shutdown", None
    raise ValueError(a.cmd)


def render_status(s: dict) -> Group:
    head = Text.assemble(
        ("epoch ", "bold"), (str(s["epoch"]), "bold"), "  ·  ", (f"step {s['step']}", "bold"),
        f"  ·  group {s['group']}  ·  workload {s['B']} seq/step"
        + ("" if s.get("auto_load", True) else " (manual)"),
        "  ·  ", (f"agent {s.get('agent', '?')}", "bold magenta"),
        f"  ·  scenario faults {'on (' + s.get('level', '') + ')' if s.get('chaos') else 'off'}")
    t = Table(show_edge=False, header_style="bold")
    for col in ("node", "status", "rank", "speed", "share", "seqs", "link", "faults", "heartbeat"):
        t.add_column(col, no_wrap=col in ("node", "rank", "speed", "share", "seqs", "link"))
    for n in s["nodes"]:
        injected = []
        if n.get("throttle", 1) < 0.999:
            injected.append(f"slow {n['throttle']:.0%}")
        if n.get("bw_factor", 1) < 0.999:
            injected.append(f"net {n['bw_factor']:.0%}")
        if n.get("compensated"):
            injected.append("lighter mode")
        reboot = s.get("reboot_in", {}).get(n["id"])
        status = n["status"] + (f", restarts in {reboot:.0f} s" if reboot else "")
        t.add_row(n["id"], Text(status, style=STATUS_STYLE.get(n["status"], "")),
                  "" if n.get("rank") is None else str(n["rank"]),
                  "" if n.get("speed_ratio") is None else f"{n['speed_ratio']:.0%}",
                  f"{n['share']:.0%}", f"{n['count']}/{n['cap']}", f"{n.get('bandwidth', 1):.0%}",
                  ", ".join(injected), "" if n.get("hb_age") is None else f"{n['hb_age']:.1f}s ago")
    hist = s.get("history") or []
    last = hist[-1] if hist else None
    tail = Text(f"loss {last['loss']:.3f}  ·  {last['samples_s']:.1f} sequences/s" if last and last.get("loss")
                else "waiting for the first step", style="cyan")
    ev = Table.grid(padding=(0, 1))
    for e in (s.get("events") or [])[-8:]:
        ev.add_row(Text(time.strftime("%H:%M:%S", time.localtime(e["t"])), style="bright_black"),
                   Text(e["text"], style="bold" if e["kind"] in ("decision", "fault") else ""))
    return Group(head, t, tail, Text("recent events", style="bold"), ev)


def _url(a: argparse.Namespace) -> str:
    url = getattr(a, "url", None)
    if url:
        return url if url.startswith("http") else "http://" + url
    try:
        with open(PIDS) as f:
            return f"http://127.0.0.1:{json.load(f)['port']}"
    except (OSError, ValueError, KeyError):
        return "http://127.0.0.1:8000"


def _call(url: str, method: str, path: str, body: Optional[dict]):
    import httpx
    r = httpx.request(method, url + path, json=body, timeout=10)
    if r.status_code >= 400:
        raise SystemExit(f"error: {r.json().get('detail', r.text)}")
    return r.json()


def up(a: argparse.Namespace, con: Console) -> None:
    os.makedirs(RUNS, exist_ok=True)
    cmd = [sys.executable, "-m", "demo.controller", "--port", str(a.port), "--agent", a.agent,
           "--level", a.level, "--seed", str(a.seed), "--scenario-steps", str(a.scenario_steps)]
    cmd += ["--chaos"] * a.chaos + ["--manual-load"] * a.manual_load + ["--no-spawn"] * a.no_spawn
    log = open(os.path.join(RUNS, "controller.log"), "a")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with open(PIDS, "w") as f:
        json.dump({"controller": p.pid, "port": a.port}, f)
    url = f"http://127.0.0.1:{a.port}"
    import httpx
    for _ in range(60):
        try:
            httpx.get(url + "/api/state", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.5)
    else:
        raise SystemExit(f"controller did not start; see {os.path.join(RUNS, 'controller.log')}")
    con.print(f"[bold green]up[/] controller pid {p.pid} at {url}  (agent {a.agent}"
              f"{', scenario faults ' + a.level if a.chaos else ''})")
    con.print("try: [bold]python -m demo watch[/], [bold]python -m demo crash win-gpu-1[/], "
              "[bold]python -m demo down[/]")
    if not a.no_browser:
        webbrowser.open(url)


def main(argv=None) -> None:
    a = parse(sys.argv[1:] if argv is None else argv)
    con = Console()
    if a.cmd == "up":
        return up(a, con)
    if a.cmd == "node":
        from .node import NodeClient
        return NodeClient(a.id, a.controller, emulate=not a.no_emulate).run()
    url = _url(a)
    method, path, body = request_for(a)
    if a.cmd == "watch":
        with Live(render_status(_call(url, method, path, body)), console=con, refresh_per_second=4) as live:
            try:
                while True:
                    time.sleep(0.5)
                    live.update(render_status(_call(url, method, path, body)))
            except KeyboardInterrupt:
                return
    if a.cmd == "down":
        try:
            _call(url, method, path, body)
        except Exception:
            pass
        try:
            os.remove(PIDS)
        except OSError:
            pass
        con.print("[bold]down[/]")
        return
    out = _call(url, method, path, body)
    if a.cmd == "status":
        con.print(render_status(out))
    else:
        con.print(f"[green]ok[/] {json.dumps(out)}")
