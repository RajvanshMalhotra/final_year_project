"""
The controller's picture of the cluster, as plain logic (no network, no clock): every input
comes in as a method call with `now`, every instruction for a node goes out as a returned list
of (node_id, message). That keeps it unit-testable; controller.py does the I/O.

Epochs: a period with fixed membership and one gloo group. The membership we *want* is every
node the agent has not evicted. When the running group differs from that (EVICT / READMIT),
rank 0 is told to end the epoch; once nobody is training any more and every wanted node is
alive and waiting, a new epoch is formed. If a wanted node is dead, nothing forms: training
stalls until the agent evicts it or it comes back - the same rule as the simulator.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from simulator import CRASH_SIGNAL_AGE_S

from .config import NODE_IDS, DemoConfig, NodeSpec

Msg = Tuple[str, dict]
WARMUP = (5, 15)   # a node's own reports #5..#15 define its normal speed


@dataclass
class NodeInfo:
    spec: NodeSpec
    connected: bool = False
    ever_up: bool = False
    pid: Optional[int] = None
    last_hb: Optional[float] = None
    phase: str = "offline"           # offline | idle | joining | training | left
    has_state: bool = False
    held_step: int = -1
    evicted: bool = False
    compensated: bool = False
    throttle: float = 1.0            # injected fault, for display (the node applies it)
    bw_factor: float = 1.0           # injected fault, for display
    n_reports: int = 0
    warm_sps: List[float] = field(default_factory=list)
    warm_net: List[float] = field(default_factory=list)
    normal_sps: Optional[float] = None
    base_net: Optional[float] = None
    last_sps: Optional[float] = None
    ema_sps: Optional[float] = None
    last_net: Optional[float] = None
    last_comm: Optional[float] = None
    count: int = 0
    checksum: Optional[str] = None


class ClusterState:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.nodes: Dict[str, NodeInfo] = {n.id: NodeInfo(n) for n in config.nodes}
        self.epoch = 0
        self.members: List[str] = []
        self.group = "none"              # none | forming | running | stopping | stalled | idle
        self.B = config.b_start
        self.shares: Dict[str, float] = {}
        self.version = 0
        self.step = 0
        self.last_step_at: Optional[float] = None
        self._joined: Dict[str, str] = {}
        self._stop_sent = False
        self.history: Deque[dict] = deque(maxlen=3000)
        self.events: Deque[dict] = deque(maxlen=400)
        self.new_events: List[dict] = []

    # ------------------------------------------------------------------ helpers
    def log(self, now: float, kind: str, text: str, node: Optional[str] = None, **extra) -> None:
        e = {"t": round(now, 3), "kind": kind, "text": text, "node": node, **extra}
        self.events.append(e)
        self.new_events.append(e)

    def alive(self, nid: str, now: float) -> bool:
        n = self.nodes[nid]
        return n.connected and n.last_hb is not None and now - n.last_hb <= CRASH_SIGNAL_AGE_S

    def wanted(self) -> List[str]:
        return [nid for nid in NODE_IDS if not self.nodes[nid].evicted]

    def rank0(self) -> Optional[str]:
        return self.members[0] if self.members else None

    def _busy(self, now: float) -> List[str]:
        return [m for m in self.members
                if self.alive(m, now) and self.nodes[m].phase in ("joining", "training")]

    def _config_msg(self) -> dict:
        return {"type": "config", "version": self.version, "B": self.B, "shares": dict(self.shares)}

    def _speed(self, nid: str) -> float:
        n = self.nodes[nid]
        if n.last_sps:
            return 1.0 / n.last_sps
        return n.spec.speed_factor

    def _norm_shares(self, members: List[str]) -> Dict[str, float]:
        if members and all(self.shares.get(m, 0) > 0 for m in members):
            raw = {m: self.shares[m] for m in members}
        else:
            raw = {m: self._speed(m) for m in members}
        total = sum(raw.values()) or 1.0
        return {m: v / total for m, v in raw.items()}

    # ------------------------------------------------------------------ node inputs
    def on_hello(self, nid: str, pid: int, has_state: bool, step: int, now: float) -> None:
        n = self.nodes[nid]
        n.connected, n.ever_up, n.pid, n.last_hb = True, True, pid, now
        n.phase, n.has_state, n.held_step = "idle", has_state, step
        n.compensated, n.throttle, n.bw_factor = False, 1.0, 1.0
        n.last_net = n.last_comm = None          # a fresh process: old link readings no longer apply
        self.log(now, "node_up", f"{nid} is up (pid {pid})", nid)

    def on_heartbeat(self, nid: str, now: float) -> None:
        self.nodes[nid].last_hb = now

    def on_disconnect(self, nid: str, now: float) -> None:
        n = self.nodes[nid]
        if n.connected:
            n.connected, n.phase = False, "offline"
            self.log(now, "node_down", f"{nid} lost connection", nid)

    def on_joined(self, nid: str, epoch: int, checksum: str, step: int, now: float) -> None:
        if epoch != self.epoch or nid not in self.members:
            return
        n = self.nodes[nid]
        n.phase, n.checksum, n.has_state, n.held_step = "training", checksum, True, step
        self._joined[nid] = checksum
        if set(self._joined) == set(self.members):
            self.group = "running"
            same = len(set(self._joined.values())) == 1
            self.log(now, "epoch_running",
                     f"epoch {epoch}: {len(self.members)} nodes training, weights "
                     + ("identical" if same else "DIFFERENT"),
                     checksum=checksum if same else None, identical=same)

    def on_left(self, nid: str, epoch: int, reason: str, step: int, now: float) -> None:
        if epoch != self.epoch or nid not in self.members:
            return
        n = self.nodes[nid]
        n.phase, n.held_step = "left", step
        n.has_state = n.has_state or step >= 0
        if reason != "end_epoch" and self.group in ("running", "forming"):
            self.group = "stalled"
            self.log(now, "group_failed", f"{nid}: network group failed ({reason})", nid)
        if not self._busy(now):
            self.group = "idle" if self.group == "stopping" else "stalled"

    def on_report(self, nid: str, rep: dict, now: float) -> None:
        if rep.get("epoch") != self.epoch:
            return
        n = self.nodes[nid]
        n.n_reports += 1
        n.count = rep.get("count", 0)
        n.held_step = rep["step"]
        sps, net = rep.get("sec_per_seq"), rep.get("net_s")
        if sps:
            self._speed_sample(n, sps)
            if WARMUP[0] <= n.n_reports <= WARMUP[1] and n.normal_sps is None:
                n.warm_sps.append(sps)
                if net is not None:
                    n.warm_net.append(net)
                if len(n.warm_sps) >= WARMUP[1] - WARMUP[0] + 1:
                    n.normal_sps = statistics.median(n.warm_sps)
                    n.base_net = statistics.median(n.warm_net) if n.warm_net else None
        if net is not None:
            n.last_net = net
        n.last_comm = rep.get("comm_s")
        if rep["step"] > self.step:
            self.step = rep["step"]
            self.last_step_at = now
        if nid == self.rank0():
            wall = rep.get("wall_s") or 0.0
            self.history.append({"t": round(now, 3), "step": rep["step"],
                                 "loss": rep.get("gloss", rep.get("loss")), "B": rep.get("B", self.B),
                                 "wall_s": wall, "samples_s": rep.get("B", self.B) / wall if wall else 0.0,
                                 "members": len(self.members)})

    def on_probe(self, nid: str, sec_per_seq: float, now: float, net_s: Optional[float] = None) -> None:
        """An idle (evicted) node benchmarks itself and its link, so we can tell when it is healthy again."""
        self._speed_sample(self.nodes[nid], sec_per_seq)
        if net_s is not None:
            self.nodes[nid].last_net = net_s

    @staticmethod
    def _speed_sample(n: NodeInfo, sps: float) -> None:
        n.last_sps = sps
        n.ema_sps = sps if n.ema_sps is None else 0.3 * sps + 0.7 * n.ema_sps

    # ------------------------------------------------------------------ decisions
    def maybe_form_epoch(self, now: float) -> List[Msg]:
        out = self._stop_if_needed(now)
        if self.group == "running" and any(not self.alive(m, now) for m in self.members):
            self.group = "stalled"
            dead = [m for m in self.members if not self.alive(m, now)]
            self.log(now, "stall", f"training stalled: {', '.join(dead)} not responding")
        if self.group == "stopping" and not self._busy(now):
            self.group = "idle"
        if self.group in ("forming", "running", "stopping") or self._busy(now):
            return out
        wanted = self.wanted()
        ready = [w for w in wanted if self.alive(w, now) and self.nodes[w].phase in ("idle", "left")]
        if not wanted or len(ready) < len(wanted):
            return out
        return out + self._begin_epoch(wanted, now)

    def _begin_epoch(self, members: List[str], now: float) -> List[Msg]:
        holders = [m for m in members if self.nodes[m].has_state]
        first = max(holders, key=lambda m: (self.nodes[m].held_step, -NODE_IDS.index(m))) \
            if holders else members[0]
        self.members = [first] + [m for m in members if m != first]
        self.epoch += 1
        self.shares = self._norm_shares(self.members)
        self.version += 1
        self.group, self._joined, self._stop_sent = "forming", {}, False
        for m in self.members:
            self.nodes[m].phase = "joining"
        self.log(now, "epoch", f"epoch {self.epoch}: forming group of {len(self.members)} "
                               f"({', '.join(self.members)}); {first} sends the model")
        return [(m, {"type": "epoch", "epoch": self.epoch, "rank": r, "world": len(self.members),
                     "members": list(self.members), "shares": dict(self.shares), "B": self.B,
                     "version": self.version})
                for r, m in enumerate(self.members)]

    def _stop_if_needed(self, now: float) -> List[Msg]:
        if self.group not in ("running", "forming") or self._stop_sent:
            return []
        if set(self.wanted()) == set(self.members):
            return []
        self.group, self._stop_sent = "stopping", True
        r0 = self.rank0()
        if r0 and self.alive(r0, now):
            return [(r0, {"type": "end_epoch", "epoch": self.epoch})]
        return []

    def evict(self, nid: str, now: float) -> List[Msg]:
        self.nodes[nid].evicted = True
        self.log(now, "evict", f"remove {nid} from the group", nid)
        return self._stop_if_needed(now)

    def readmit(self, nid: str, now: float) -> List[Msg]:
        self.nodes[nid].evicted = False
        self.log(now, "readmit", f"bring {nid} back into the group", nid)
        return self._stop_if_needed(now)

    def rebalance(self, shares: Dict[str, float], now: float) -> List[Msg]:
        members = self.members or self.wanted()
        total = sum(max(0.0, shares.get(m, 0.0)) for m in members) or 1.0
        self.shares = {m: max(0.0, shares.get(m, 0.0)) / total for m in members}
        self.version += 1
        self.log(now, "rebalance", "new split: " + ", ".join(
            f"{m} {v:.0%}" for m, v in self.shares.items()))
        return self._to_rank0(self._config_msg())

    def set_B(self, B: int, now: float) -> List[Msg]:
        self.B = int(B)
        self.version += 1
        self.log(now, "workload", f"workload: {self.B} sequences per step", B=self.B)
        return self._to_rank0(self._config_msg())

    def compensate(self, nid: str, now: float) -> List[Msg]:
        self.nodes[nid].compensated = True
        self.log(now, "compensate", f"{nid} switches to lighter mode", nid)
        return [(nid, {"type": "compensate", "on": True})]

    def _to_rank0(self, msg: dict) -> List[Msg]:
        r0 = self.rank0()
        if self.group in ("running", "forming") and r0:
            return [(r0, msg)]
        return []

    # ------------------------------------------------------------------ views
    def node_view(self, nid: str, now: float) -> dict:
        n = self.nodes[nid]
        b = max(self.B, 1)
        measured = 1.0 / (b * n.last_sps) if n.last_sps else 0.0
        normal = 1.0 / (b * n.normal_sps) if n.normal_sps else measured
        bw = 1.0
        if n.base_net and n.last_net:
            bw = min(1.0, n.base_net / n.last_net)
        in_group = nid in self.members and not n.evicted
        return {
            "normal_speed": normal, "measured_speed": measured,
            "share": self.shares.get(nid, 0.0) if in_group else 0.0,
            "evicted": n.evicted, "compensated": n.compensated, "bandwidth": bw,
            "heartbeat_age": (now - n.last_hb) if n.last_hb is not None else 1e9,
            "is_active": (not n.evicted) and self.alive(nid, now),
        }

    def status(self, nid: str, now: float) -> str:
        n = self.nodes[nid]
        if not n.ever_up:
            return "starting"
        if not self.alive(nid, now):
            return "dead"
        if n.evicted:
            return "removed"
        if nid in self.members and n.phase == "training" and self.group == "running":
            return "training"
        if n.phase == "joining":
            return "joining"
        if self.group in ("stalled", "stopping") and nid in self.members:
            return "waiting"
        return "idle"

    def snapshot(self, now: float) -> dict:
        nodes = []
        for nid in NODE_IDS:
            n, v = self.nodes[nid], self.node_view(nid, now)
            nodes.append({
                "id": nid, "label": n.spec.label, "status": self.status(nid, now),
                "speed_factor": n.spec.speed_factor, "source": n.spec.source, "cap": n.spec.cap,
                "pid": n.pid, "share": v["share"], "count": n.count,
                "speed_ratio": (v["measured_speed"] / v["normal_speed"]) if v["normal_speed"] else None,
                "bandwidth": v["bandwidth"], "compensated": n.compensated,
                "throttle": n.throttle, "bw_factor": n.bw_factor,
                "hb_age": round(v["heartbeat_age"], 2) if n.last_hb is not None else None,
                "held_step": n.held_step, "checksum": n.checksum, "evicted": n.evicted,
                "rank": self.members.index(nid) if nid in self.members else None,
            })
        return {"t": round(now, 3), "epoch": self.epoch, "group": self.group,
                "members": list(self.members), "step": self.step, "B": self.B,
                "version": self.version, "nodes": nodes}
