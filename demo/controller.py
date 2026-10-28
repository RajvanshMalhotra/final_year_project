"""
The controller (backend): one process that the nodes, the CLI and the web page connect to.

    python -m demo.controller --agent rl --chaos            (usually started by `python -m demo up`)

It owns the gloo rendezvous store, the cluster state (demo/state.py), the workload, the faults,
and the agent loop: every training step (or every second while training is stalled) it turns the
live measurements into the agent's observation, asks the agent, and carries out its decision.
Everything that happens is written to demo/runs/<time>.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import json
import os
import random
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set

import torch.distributed as dist
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from simulator import Action

from .agent_adapter import AgentRunner, apply_action, build_observation, describe
from .chaos import ChaosPlayer, FaultBook, FaultCmd, Workload
from .config import NODE_IDS, ROOT, DemoConfig, load_config
from .state import ClusterState

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
STATIC = os.path.join(HERE, "static")
TICK_S = 0.1
UI_EVERY_S = 0.25
STALL_DECIDE_S = 1.0


class NodeLauncher:
    """Starts, kills (SIGKILL = a real crash) and restarts node processes on this machine."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.procs: Dict[str, subprocess.Popen] = {}

    def start(self, nid: str) -> None:
        log = open(os.path.join(RUNS, f"node-{nid}.log"), "a")
        self.procs[nid] = subprocess.Popen(
            [sys.executable, "-m", "demo.node", "--id", nid, "--controller", f"127.0.0.1:{self.port}"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    def kill(self, nid: str) -> bool:
        p = self.procs.get(nid)
        if p and p.poll() is None:
            p.send_signal(signal.SIGKILL)
            return True
        return False

    def running(self, nid: str) -> bool:
        p = self.procs.get(nid)
        return bool(p and p.poll() is None)

    def stop_all(self) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


class Controller:
    def __init__(self, config: DemoConfig, port: int = 8000, store_port: int = 29600,
                 agent_mode: str = "rl", model_path: Optional[str] = None, chaos: bool = False,
                 auto_load: bool = True, spawn_nodes: bool = True, seed: int = 0) -> None:
        self.cfg, self.port, self.store_port = config, port, store_port
        self.state = ClusterState(config)
        self.model_path = model_path or os.path.join(ROOT, "models", "v2_seed1.zip")
        self.runner = AgentRunner(agent_mode, self.model_path)
        self.rng = random.Random(seed)
        self.workload = Workload(config, self.rng, auto=auto_load, now=time.time())
        self.chaos_on = chaos
        self.player = ChaosPlayer(config.scenario_level, config.scenario_seed, config.scenario_steps)
        self.book = FaultBook()
        self.manual: Dict[str, dict] = {}            # node -> {"throttle"/"bandwidth": f, "until": t}
        self.sent_factors: Dict[str, tuple] = {}
        self.reboot_at: Dict[str, Optional[float]] = {}
        self.spawn = spawn_nodes
        self.launcher = NodeLauncher(port) if spawn_nodes else None
        self.nodes_ws: Dict[str, WebSocket] = {}
        self.ui: Set[WebSocket] = set()
        self.tick = 0
        self.last_decided_step = -1
        self.last_decision_t = 0.0
        self.decisions: List[dict] = []
        self.checks: Dict[int, Dict[str, str]] = {}
        os.makedirs(RUNS, exist_ok=True)
        self.log_path = os.path.join(RUNS, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl")
        self._log = open(self.log_path, "a")
        self.store = None
        self.started = time.time()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.store = dist.TCPStore("0.0.0.0", self.store_port, is_master=True, wait_for_workers=False,
                                   timeout=datetime.timedelta(seconds=60))
        self.state.log(time.time(), "start", f"controller up; agent {self.runner.mode}; "
                                             f"chaos {'on' if self.chaos_on else 'off'}")
        if self.launcher:
            for nid in NODE_IDS:
                self.launcher.start(nid)

    def stop(self) -> None:
        if self.launcher:
            self.launcher.stop_all()
        self._flush_events()
        self._log.close()

    # ------------------------------------------------------------------ messaging
    async def send(self, nid: str, msg: dict) -> None:
        ws = self.nodes_ws.get(nid)
        if ws is None:
            return
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            pass

    async def send_all(self, msgs) -> None:
        for nid, m in msgs:
            await self.send(nid, m)

    def _flush_events(self) -> None:
        for e in self.state.new_events:
            self._log.write(json.dumps(e) + "\n")
        self.state.new_events.clear()
        self._log.flush()

    # ------------------------------------------------------------------ node connection
    async def node_session(self, ws: WebSocket, nid: str) -> None:
        if nid not in NODE_IDS:
            await ws.close(code=4004)
            return
        await ws.accept()
        self.nodes_ws[nid] = ws
        spec = self.cfg.node(nid)
        try:
            while True:
                m = json.loads(await ws.receive_text())
                now, t = time.time(), m.get("type")
                if t == "hello":
                    self.state.on_hello(nid, m["pid"], m.get("has_state", False), m.get("step", -1), now)
                    self.sent_factors.pop(nid, None)
                    self.reboot_at.pop(nid, None)
                    await ws.send_text(json.dumps({
                        "type": "welcome", "speed_factor": spec.speed_factor, "cap": spec.cap,
                        "sec_per_seq": self.cfg.sec_per_seq, "link_s": self.cfg.link_s,
                        "ctx": self.cfg.ctx, "lr": self.cfg.lr, "data_seed": self.cfg.data_seed,
                        "store_port": self.store_port}))
                elif t == "hb":
                    self.state.on_heartbeat(nid, now)
                elif t == "report":
                    self.state.on_heartbeat(nid, now)
                    self.state.on_report(nid, m, now)
                    if "checksum" in m:
                        self._check(nid, m["step"], m["checksum"], now)
                elif t == "joined":
                    self.state.on_joined(nid, m["epoch"], m["checksum"], m["step"], now)
                elif t == "left":
                    self.state.on_left(nid, m["epoch"], m["reason"], m["step"], now)
                elif t == "probe":
                    self.state.on_heartbeat(nid, now)
                    self.state.on_probe(nid, m["sec_per_seq"], now)
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            pass
        finally:
            if self.nodes_ws.get(nid) is ws:
                del self.nodes_ws[nid]
                self.state.on_disconnect(nid, time.time())

    def _check(self, nid: str, step: int, checksum: str, now: float) -> None:
        got = self.checks.setdefault(step, {})
        got[nid] = checksum
        members = set(self.state.members)
        if members and members <= set(got):
            same = len({got[m] for m in members}) == 1
            self.state.log(now, "checksum", f"step {step}: weights on {len(members)} nodes "
                                            + ("identical" if same else "DIFFER"),
                           identical=same, checksum=checksum if same else None)
            del self.checks[step]
        for s in [s for s in self.checks if s < step - 200]:
            del self.checks[s]

    # ------------------------------------------------------------------ faults
    async def inject(self, cmd: FaultCmd, now: float, source: str, duration_s: Optional[float] = None) -> None:
        nid = cmd.node
        if cmd.kind == "crash":
            ok = self.launcher.kill(nid) if self.launcher else False
            if not ok:
                await self.send(nid, {"type": "die"})
            if source == "chaos":
                step_s = self._step_seconds()
                self.reboot_at[nid] = (now + max(self.cfg.reboot_s[0], cmd.duration_steps * step_s)
                                       if cmd.duration_steps else None)
            else:
                self.reboot_at[nid] = now + self.rng.uniform(*self.cfg.reboot_s)
            when = self.reboot_at[nid]
            self.state.log(now, "fault", f"CRASH {nid} (process killed, {source})"
                           + (f"; restarts in {when - now:.0f} s" if when else "; stays down"), nid,
                           fault="crash", source=source)
        elif cmd.kind in ("slow", "net"):
            if source == "chaos":
                self.book.add(cmd, self.state.step)
            else:
                key = "throttle" if cmd.kind == "slow" else "bandwidth"
                self.manual.setdefault(nid, {})[key] = (cmd.factor, now + duration_s if duration_s else None)
            what = "SLOWDOWN" if cmd.kind == "slow" else "SLOW NETWORK"
            self.state.log(now, "fault", f"{what} {nid}: {cmd.factor:.0%} of normal ({source})", nid,
                           fault=cmd.kind, factor=cmd.factor, source=source)

    async def heal(self, nid: str, now: float) -> None:
        self.book.heal(nid)
        self.manual.pop(nid, None)
        if self.launcher and not self.launcher.running(nid):
            self.launcher.start(nid)
            self.state.log(now, "reboot", f"{nid} restarted (manual heal)", nid)
        self.reboot_at.pop(nid, None)
        self.state.log(now, "heal", f"{nid}: faults cleared", nid)

    def _step_seconds(self) -> float:
        h = list(self.state.history)[-20:]
        walls = sorted(x["wall_s"] for x in h if x["wall_s"])
        return walls[len(walls) // 2] if walls else 0.3

    async def _apply_factors(self, now: float) -> None:
        f = self.book.factors(self.state.step)
        for nid in NODE_IDS:
            thr, bw = f.get(nid, (1.0, 1.0))
            for key, (val, until) in list(self.manual.get(nid, {}).items()):
                if until and now >= until:
                    del self.manual[nid][key]
                    self.state.log(now, "fault_end", f"{nid}: manual {key} fault over", nid)
                    continue
                thr, bw = (thr * val, bw) if key == "throttle" else (thr, bw * val)
            if self.sent_factors.get(nid) != (round(thr, 3), round(bw, 3)) and nid in self.nodes_ws:
                await self.send(nid, {"type": "throttle", "factor": thr})
                await self.send(nid, {"type": "bandwidth", "factor": bw})
                self.sent_factors[nid] = (round(thr, 3), round(bw, 3))
            n = self.state.nodes[nid]
            n.throttle, n.bw_factor = thr, bw

    # ------------------------------------------------------------------ agent
    def set_agent(self, mode: str, now: float) -> None:
        self.runner = AgentRunner(mode, self.model_path)
        self.state.log(now, "agent_mode", f"agent switched to {mode}")

    async def _agent_tick(self, now: float) -> None:
        stepped = self.state.step > self.last_decided_step
        if not stepped and now - self.last_decision_t < STALL_DECIDE_S:
            return
        if self.state.epoch == 0:
            return
        self.last_decided_step, self.last_decision_t = self.state.step, now
        self.tick += 1
        obs = build_observation(self.state, now, self.tick)
        action, target, info = self.runner.decide(obs)
        if action == Action.WAIT:
            if info.get("note") and "skipped rebalance" not in info["note"]:
                self.state.log(now, "agent_note", info["note"])
            return
        msgs = apply_action(action, target, obs, self.state, now)
        await self.send_all(msgs)
        m = obs.machines
        d = {"t": round(now, 3), "step": self.state.step, "mode": self.runner.mode,
             "action": action.name, "target": target, "text": describe(action, target),
             "seen": {nid: {"speed": round(m[nid].measured_speed / m[nid].normal_speed, 2)
                            if m[nid].normal_speed else None,
                            "hb_age": round(min(m[nid].alive_signal_age, 99), 1)} for nid in NODE_IDS}}
        self.decisions.append(d)
        self.decisions = self.decisions[-200:]
        self.state.log(now, "decision", f"agent ({self.runner.mode}): {describe(action, target)}",
                       target, action=action.name)

    # ------------------------------------------------------------------ main loop
    async def loop(self) -> None:
        last_ui = 0.0
        while True:
            now = time.time()
            try:
                await self.send_all(self.state.maybe_form_epoch(now))
                nb = self.workload.tick(now)
                if nb is not None:
                    await self.send_all(self.state.set_B(nb, now))
                if self.chaos_on and self.state.epoch:
                    for cmd in self.player.due(self.state.step):
                        await self.inject(cmd, now, "chaos")
                await self._apply_factors(now)
                for nid, when in list(self.reboot_at.items()):
                    if when and now >= when and self.launcher and not self.launcher.running(nid):
                        self.launcher.start(nid)
                        self.reboot_at[nid] = None
                        self.state.log(now, "reboot", f"{nid} restarted after its crash", nid)
                await self._agent_tick(now)
                if now - last_ui >= UI_EVERY_S:
                    await self.broadcast_ui(now)
                    last_ui = now
                self._flush_events()
            except Exception as e:      # the demo must keep running; log and carry on
                self.state.log(now, "error", f"controller loop: {e!r}")
            await asyncio.sleep(TICK_S)

    def snapshot(self, now: float, history: int = 600, events: int = 80) -> dict:
        s = self.state.snapshot(now)
        s.update({
            "agent": self.runner.mode, "chaos": self.chaos_on, "auto_load": self.workload.auto,
            "level": self.cfg.scenario_level, "scenario_seed": self.cfg.scenario_seed,
            "chaos_left": self.player.remaining, "uptime": round(now - self.started, 1),
            "history": list(self.state.history)[-history:], "events": list(self.state.events)[-events:],
            "decisions": self.decisions[-30:], "citations": self.cfg.citations,
            "reboot_in": {k: round(v - now, 1) for k, v in self.reboot_at.items() if v},
            "log_file": self.log_path,
        })
        return s

    async def broadcast_ui(self, now: float) -> None:
        if not self.ui:
            return
        msg = json.dumps(self.snapshot(now, history=600, events=80))
        for ws in list(self.ui):
            try:
                await ws.send_text(msg)
            except Exception:
                self.ui.discard(ws)


def create_app(ctl: Controller) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        ctl.start()
        task = asyncio.create_task(ctl.loop())
        yield
        task.cancel()
        ctl.stop()

    app = FastAPI(title="Self-healing training demo", lifespan=lifespan)

    @app.websocket("/ws/node/{nid}")
    async def ws_node(ws: WebSocket, nid: str) -> None:
        await ctl.node_session(ws, nid)

    @app.websocket("/ws/ui")
    async def ws_ui(ws: WebSocket) -> None:
        await ws.accept()
        ctl.ui.add(ws)
        try:
            await ws.send_text(json.dumps(ctl.snapshot(time.time())))
            while True:
                await ws.receive_text()
        except Exception:
            ctl.ui.discard(ws)

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(ctl.snapshot(time.time(), history=200, events=40))

    def _node(nid: Optional[str]) -> str:
        if nid not in NODE_IDS:
            raise HTTPException(400, f"unknown node {nid!r}; use one of {', '.join(NODE_IDS)}")
        return nid

    @app.post("/api/load")
    async def api_load(body: dict) -> dict:
        now = time.time()
        if "auto" in body:
            ctl.workload.set_auto(bool(body["auto"]), now)
            ctl.state.log(now, "workload", f"automatic workload growth {'on' if body['auto'] else 'off'}")
        if "delta" in body:
            try:
                delta = int(body["delta"])
            except (TypeError, ValueError):
                raise HTTPException(400, "delta must be a whole number")
            await ctl.send_all(ctl.state.set_B(ctl.workload.bump(delta), now))
        return {"B": ctl.state.B, "auto": ctl.workload.auto}

    @app.post("/api/fault")
    async def api_fault(body: dict) -> dict:
        now = time.time()
        nid, kind = _node(body.get("node")), body.get("kind")
        if kind == "heal":
            await ctl.heal(nid, now)
            return {"ok": True}
        if kind == "crash":
            await ctl.inject(FaultCmd("crash", nid), now, "manual")
            return {"ok": True}
        if kind not in ("slow", "net"):
            raise HTTPException(400, "kind must be crash, slow, net or heal")
        try:
            factor = float(body.get("factor", 0.4))
            duration = float(body["duration"]) if body.get("duration") is not None else None
        except (TypeError, ValueError):
            raise HTTPException(400, "factor and duration must be numbers")
        if not 0.0 < factor <= 1.0:
            raise HTTPException(400, "factor must be between 0 and 1 (fraction of normal speed)")
        if duration is not None and duration <= 0:
            raise HTTPException(400, "duration must be positive seconds")
        await ctl.inject(FaultCmd(kind, nid, factor=factor), now, "manual", duration_s=duration)
        return {"ok": True}

    @app.post("/api/chaos")
    async def api_chaos(body: dict) -> dict:
        now = time.time()
        ctl.chaos_on = bool(body.get("on", True))
        if "level" in body or "seed" in body:
            level = body.get("level", ctl.cfg.scenario_level)
            if level not in ("easy", "realistic", "hard", "chaos"):
                raise HTTPException(400, "level must be easy, realistic, hard or chaos")
            ctl.cfg.scenario_level, ctl.cfg.scenario_seed = level, int(body.get("seed", ctl.cfg.scenario_seed))
            ctl.player = ChaosPlayer(level, ctl.cfg.scenario_seed, ctl.cfg.scenario_steps)
            # Replay from the current step onward, so the schedule lines up with the live run.
            ctl.player.due(ctl.state.step)
        ctl.state.log(now, "chaos", f"scenario faults {'on' if ctl.chaos_on else 'off'} "
                                    f"({ctl.cfg.scenario_level}, seed {ctl.cfg.scenario_seed})")
        return {"chaos": ctl.chaos_on}

    @app.post("/api/agent")
    async def api_agent(body: dict) -> dict:
        mode = body.get("mode")
        if mode not in ("rl", "rule", "off"):
            raise HTTPException(400, "mode must be rl, rule or off")
        ctl.set_agent(mode, time.time())
        return {"agent": mode}

    @app.post("/api/shutdown")
    async def api_shutdown() -> dict:
        os.kill(os.getpid(), signal.SIGINT)
        return {"ok": True}

    if os.path.isdir(STATIC):
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(os.path.join(STATIC, "index.html"))

    return app


def main() -> None:
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--store-port", type=int, default=29600)
    p.add_argument("--agent", default="rl", choices=["rl", "rule", "off"])
    p.add_argument("--model", help="RL model zip (default models/v2_seed1.zip)")
    p.add_argument("--chaos", action="store_true", help="replay a simulator fault scenario")
    p.add_argument("--level", default="realistic", choices=["easy", "realistic", "hard", "chaos"])
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--manual-load", action="store_true", help="do not grow the workload automatically")
    p.add_argument("--no-spawn", action="store_true", help="nodes run elsewhere (the 3 real laptops)")
    p.add_argument("--calibration", default=os.path.join(ROOT, "calibration.json"))
    a = p.parse_args()
    cfg = load_config(a.calibration)
    cfg.scenario_level, cfg.scenario_seed = a.level, a.seed
    ctl = Controller(cfg, port=a.port, store_port=a.store_port, agent_mode=a.agent, model_path=a.model,
                     chaos=a.chaos, auto_load=not a.manual_load, spawn_nodes=not a.no_spawn)
    uvicorn.run(create_app(ctl), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
