"""
One training node: a real process that trains the tiny GPT together with the others.

    python -m demo.node --id win-gpu-0 --controller 127.0.0.1:8000             (one-Mac demo)
    python -m demo.node --id win-gpu-0 --controller 192.168.1.10:8000 --no-emulate   (real laptop)

Parts:
  Trainer   the model, the data, gradient accumulation and one shared AdamW on a flat vector
            (identical maths on every node, so the copies stay identical)
  Emulator  only in the one-Mac demo: makes this node as slow as the hardware it stands for,
            and applies injected slowdowns / slow links (see demo/config.py for every value)
  run_epoch one period of fixed membership: join the gloo group, receive the state from rank 0,
            then train step by step until told to leave or the group fails
  NodeClient the control connection to the controller (hello, heartbeats every 0.5 s, reports)
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from . import tinygpt
from .shares import split_counts

HEARTBEAT_S = 0.5
GROUP_TIMEOUT_S = 5.0
CHECKSUM_EVERY = 25
COMPENSATE_SPEEDUP = 1.25   # same as simulator.COMPENSATE_SPEEDUP


# ----------------------------------------------------------------------------- training maths
class FlatAdamW:
    """AdamW over one flat parameter vector. Same code and inputs on every node => same result."""

    def __init__(self, n: int, lr: float, betas=(0.9, 0.99), eps=1e-8, wd=0.01) -> None:
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, wd
        self.m, self.v, self.t = torch.zeros(n), torch.zeros(n), 0

    def step(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        self.t += 1
        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * (mhat / (vhat.sqrt() + self.eps) + self.wd * p)


class Trainer:
    def __init__(self, ctx: int = 64, lr: float = 3e-3, data_seed: int = 0, text: Optional[str] = None) -> None:
        torch.manual_seed(0)
        self.model = tinygpt.TinyGPT(ctx=ctx)
        self.ctx, self.data_seed = ctx, data_seed
        self.data = tinygpt.encode(text if text is not None else tinygpt.load_text())
        self.n = sum(p.numel() for p in self.model.parameters())
        self.opt = FlatAdamW(self.n, lr)
        self.step = 0                      # global training step this node's state is at

    def params(self) -> torch.Tensor:
        return torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()

    def compute(self, B: int, counts: List[int], rank: int, cap: int) -> Tuple[Optional[float], int, float]:
        """Gradient of this node's slice of the global batch for self.step. Returns (loss, count, s)."""
        t0 = time.perf_counter()
        x, y = tinygpt.batch_for_step(self.data, self.step, B, self.ctx, self.data_seed)
        lo = sum(counts[:rank])
        n = counts[rank]
        loss = tinygpt.accumulate_grads(self.model, x[lo:lo + n], y[lo:lo + n], cap)
        return loss, n, time.perf_counter() - t0

    def pack(self, loss: Optional[float], count: int, B: int) -> torch.Tensor:
        """[grads * count/B, loss * count, count]: one all-reduce gives the full-batch gradient and loss."""
        g = tinygpt.flat_grads(self.model) * (count / B if B else 0.0)
        return torch.cat([g, torch.tensor([(loss or 0.0) * count, float(count)])])

    def apply(self, summed: torch.Tensor) -> Optional[float]:
        g, loss_sum, count = summed[:-2], float(summed[-2]), float(summed[-1])
        norm = float(g.norm())
        if norm > 1.0:                      # gradient clipping, same on every node
            g = g / norm
        new = self.opt.step(self.params(), g)
        torch.nn.utils.vector_to_parameters(new, self.model.parameters())
        self.step += 1
        return loss_sum / count if count else None

    def state(self) -> torch.Tensor:
        return torch.cat([self.params(), self.opt.m, self.opt.v,
                          torch.tensor([float(self.opt.t), float(self.step)])])

    def load_state(self, s: torch.Tensor) -> None:
        n = self.n
        torch.nn.utils.vector_to_parameters(s[:n].clone(), self.model.parameters())
        self.opt.m, self.opt.v = s[n:2 * n].clone(), s[2 * n:3 * n].clone()
        self.opt.t, self.step = int(s[3 * n]), int(s[3 * n + 1])

    def checksum(self) -> str:
        return tinygpt.checksum(self.model)


# ----------------------------------------------------------------------------- emulation
@dataclass
class Emulator:
    """Makes one shared Mac look like different hardware. Disabled on real laptops."""
    enabled: bool = True
    speed_factor: float = 1.0      # 1.0 = fastest node (config.NodeSpec.speed_factor)
    sec_per_seq: float = 0.05      # config.DemoConfig.sec_per_seq
    link_s: float = 0.05           # config.DemoConfig.link_s
    throttle: float = 1.0          # injected slowdown (1 = none)
    bandwidth: float = 1.0         # injected slow link (1 = none)
    compensated: bool = False

    def effective_speed(self) -> float:
        return self.speed_factor * self.throttle * (COMPENSATE_SPEEDUP if self.compensated else 1.0)

    def compute_wait(self, count: int, compute_s: float) -> float:
        """Extra seconds so that compute takes count * sec_per_seq / speed in total."""
        if not self.enabled or count <= 0:
            return 0.0
        return max(0.0, count * self.sec_per_seq / max(self.effective_speed(), 1e-3) - compute_s)

    def link_wait(self) -> float:
        return self.link_s / max(self.bandwidth, 1e-3) if self.enabled else 0.0


# ----------------------------------------------------------------------------- one epoch
@dataclass
class Inbox:
    """What the control connection has received; read by the training loop between steps."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    epochs: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    end_epoch: int = -1
    config: Optional[dict] = None


def make_store(host: str, port: int) -> dist.TCPStore:
    return dist.TCPStore(host, port, is_master=False, timeout=datetime.timedelta(seconds=30),
                         use_libuv=platform.system() != "Windows")


def run_epoch(tr: Trainer, emu: Emulator, msg: dict, store: dist.Store, inbox: Inbox,
              send: Callable[[dict], None], cap: int, sleep=time.sleep) -> Tuple[str, int]:
    """Train inside one gloo group until rank 0 says leave (-> 'end_epoch') or a collective fails."""
    epoch, rank, world = msg["epoch"], msg["rank"], msg["world"]
    members = msg["members"]
    B, shares, version = msg["B"], msg["shares"], msg["version"]
    try:
        dist.init_process_group("gloo", store=dist.PrefixStore(f"e{epoch}", store), rank=rank,
                                world_size=world, timeout=datetime.timedelta(seconds=GROUP_TIMEOUT_S))
    except Exception as e:
        send({"type": "left", "epoch": epoch, "reason": f"join failed: {type(e).__name__}", "step": tr.step})
        return "group_failed", tr.step
    try:
        state = tr.state() if rank == 0 else torch.zeros(3 * tr.n + 2)
        dist.broadcast(state, src=0)
        if rank != 0:
            tr.load_state(state)
        send({"type": "joined", "epoch": epoch, "checksum": tr.checksum(), "step": tr.step})
        while True:
            # 1. rank 0 tells everyone the settings for this step (so all switch together)
            ctrl = torch.zeros(3 + world, dtype=torch.long)
            if rank == 0:
                with inbox.lock:
                    leave = inbox.end_epoch >= epoch
                    cfg = inbox.config
                if cfg and cfg["version"] > version:
                    version, B, shares = cfg["version"], cfg["B"], cfg["shares"]
                counts = split_counts(B, shares, members)
                ctrl[:3] = torch.tensor([version, B, int(leave)])
                ctrl[3:] = torch.tensor([counts[m] for m in members])
            dist.broadcast(ctrl, src=0)
            version, B, leave = int(ctrl[0]), int(ctrl[1]), bool(ctrl[2])
            counts = [int(c) for c in ctrl[3:]]
            if leave:
                dist.barrier()
                send({"type": "left", "epoch": epoch, "reason": "end_epoch", "step": tr.step})
                return "end_epoch", tr.step
            # 2. compute my slice, then the emulated hardware and link
            t0 = time.perf_counter()
            loss, count, compute_s = tr.compute(B, counts, rank, cap)
            emu_s = emu.compute_wait(count, compute_s)
            if emu_s:
                sleep(emu_s)
            t1 = time.perf_counter()
            link = emu.link_wait()
            if link:
                sleep(link)
            net_s = time.perf_counter() - t1
            # 3. average with everyone, 4. the same optimizer step on every node
            buf = tr.pack(loss, count, B)
            t2 = time.perf_counter()
            dist.all_reduce(buf)
            comm_s = time.perf_counter() - t2
            gloss = tr.apply(buf)
            rep = {"type": "report", "epoch": epoch, "step": tr.step, "loss": loss, "gloss": gloss,
                   "count": count, "B": B, "compute_s": compute_s, "emu_s": emu_s,
                   "sec_per_seq": (compute_s + emu_s) / count if count else None,
                   "net_s": net_s, "comm_s": comm_s, "wall_s": time.perf_counter() - t0}
            if tr.step % CHECKSUM_EVERY == 0:
                rep["checksum"] = tr.checksum()
            send(rep)
    except Exception as e:   # a member died or timed out: discard this step, wait for a new epoch
        send({"type": "left", "epoch": epoch, "reason": f"group failed: {type(e).__name__}", "step": tr.step})
        return "group_failed", tr.step
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def probe(tr: Trainer, emu: Emulator, sleep=time.sleep) -> Tuple[float, float]:
    """Idle health check: time two sequences and the link (with emulation) without changing the model."""
    x, y = tinygpt.batch_for_step(tr.data, 10**6 + tr.step, 2, tr.ctx, tr.data_seed)
    t0 = time.perf_counter()
    tinygpt.accumulate_grads(tr.model, x, y, 2)
    compute_s = time.perf_counter() - t0
    wait = emu.compute_wait(2, compute_s)
    if wait:
        sleep(wait)
    tr.model.zero_grad(set_to_none=False)
    t0 = time.perf_counter()
    link = emu.link_wait()
    if link:
        sleep(link)
    return (compute_s + wait) / 2, time.perf_counter() - t0


# ----------------------------------------------------------------------------- control client
class NodeClient:
    def __init__(self, node_id: str, controller: str, emulate: bool, threads: int = 2) -> None:
        self.id, self.controller, self.emulate = node_id, controller, emulate
        torch.set_num_threads(threads)
        self.inbox = Inbox()
        self.emu = Emulator(enabled=emulate)
        self._send_lock = threading.Lock()
        self._ws = None
        self._stop = threading.Event()
        self.welcome: Optional[dict] = None
        self._welcomed = threading.Event()

    def send(self, msg: dict) -> None:
        try:
            with self._send_lock:
                self._ws.send(json.dumps(msg))
        except Exception:
            self._stop.set()     # controller gone: shut down

    def _receive(self) -> None:
        try:
            for raw in self._ws:
                m = json.loads(raw)
                t = m.get("type")
                if t == "welcome":
                    self.welcome = m
                    self._welcomed.set()
                elif t == "epoch":
                    self.inbox.epochs.put(m)
                elif t == "end_epoch":
                    with self.inbox.lock:
                        self.inbox.end_epoch = max(self.inbox.end_epoch, m["epoch"])
                elif t == "config":
                    with self.inbox.lock:
                        if not self.inbox.config or m["version"] > self.inbox.config["version"]:
                            self.inbox.config = m
                elif t == "throttle":
                    self.emu.throttle = float(m["factor"])
                elif t == "bandwidth":
                    self.emu.bandwidth = float(m["factor"])
                elif t == "compensate":
                    self.emu.compensated = bool(m["on"])
                elif t == "die":           # injected crash on a real laptop (no SIGKILL across machines)
                    os._exit(137)
                elif t == "shutdown":
                    break
        except Exception:
            pass
        self._stop.set()

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            self.send({"type": "hb", "t": time.time()})

    def run(self) -> None:
        from websockets.sync.client import connect
        host = self.controller.split(":")[0]
        self._ws = connect(f"ws://{self.controller}/ws/node/{self.id}", open_timeout=30, max_size=None)
        threading.Thread(target=self._receive, daemon=True).start()
        self.send({"type": "hello", "pid": os.getpid(), "has_state": False, "step": -1})
        if not self._welcomed.wait(30):
            raise SystemExit("controller did not answer")
        w = self.welcome
        self.emu.speed_factor, self.emu.sec_per_seq, self.emu.link_s = \
            w["speed_factor"], w["sec_per_seq"], w["link_s"]
        tr = Trainer(ctx=w["ctx"], lr=w["lr"], data_seed=w["data_seed"])
        store = make_store(host, w["store_port"])
        threading.Thread(target=self._heartbeat, daemon=True).start()
        last_probe = 0.0
        epoch_seen = 0
        while not self._stop.is_set():
            try:
                msg = self.inbox.epochs.get(timeout=0.2)
            except queue.Empty:
                if time.time() - last_probe >= 1.0:      # idle: report health so we can be readmitted
                    sps, net_s = probe(tr, self.emu)
                    self.send({"type": "probe", "sec_per_seq": sps, "net_s": net_s})
                    last_probe = time.time()
                continue
            if msg["epoch"] <= epoch_seen:
                continue                                  # stale
            epoch_seen = msg["epoch"]
            run_epoch(tr, self.emu, msg, store, self.inbox, self.send, w["cap"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--id", required=True)
    p.add_argument("--controller", default="127.0.0.1:8000")
    p.add_argument("--no-emulate", action="store_true", help="real hardware: no emulated speed or faults")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    NodeClient(a.id, a.controller, emulate=not a.no_emulate, threads=a.threads).run()


if __name__ == "__main__":
    main()
