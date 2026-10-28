"""
Workload growth and fault replay.

ChaosPlayer replays a simulator scenario (scenarios.make_scenario, same seed and level) on the
real nodes at the same training steps, so one fault schedule can be run in the simulator and in
the live system and the two compared. FaultBook keeps the faults that are active on each node
and combines overlapping ones (two 50% slowdowns = 25% speed), like the simulator.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scenarios import make_scenario

from .config import DemoConfig

KINDS = {"crash": "crash", "slowdown": "slow", "network": "net"}


@dataclass
class FaultCmd:
    kind: str                      # crash | slow | net
    node: str
    factor: float = 1.0            # remaining speed / bandwidth fraction (slow, net)
    duration_steps: int = 0        # 0 = until healed (crash: permanent)
    ramp_steps: int = 0            # slowdown grows over this many steps (thermal)


class Workload:
    """Global batch B: grows by a random 2-6 every 20-40 s when auto is on, up to b_max."""

    def __init__(self, config: DemoConfig, rng: random.Random, auto: bool = True, now: float = 0.0):
        self.cfg, self.rng, self.auto = config, rng, auto
        self.B = config.b_start
        self._next = now + rng.uniform(*config.b_every_s)

    def set_auto(self, on: bool, now: float) -> None:
        self.auto = on
        self._next = now + self.rng.uniform(*self.cfg.b_every_s)

    def tick(self, now: float) -> Optional[int]:
        if not self.auto or now < self._next or self.B >= self.cfg.b_max:
            return None
        self._next = now + self.rng.uniform(*self.cfg.b_every_s)
        self.B = min(self.cfg.b_max, self.B + self.rng.randint(*self.cfg.b_step))
        return self.B

    def bump(self, delta: int) -> int:
        self.B = max(1, min(self.cfg.b_max, self.B + int(delta)))
        return self.B


class ChaosPlayer:
    def __init__(self, level: str, seed: int, total_steps: int = 1000) -> None:
        self.level, self.seed = level, seed
        self._events = sorted(make_scenario(seed, level, total_steps).events, key=lambda e: e.step)
        self._i = 0

    def due(self, step: int) -> List[FaultCmd]:
        out = []
        while self._i < len(self._events) and self._events[self._i].step <= step:
            e = self._events[self._i]
            self._i += 1
            kind = KINDS[e.kind]
            if kind == "crash":
                out.append(FaultCmd("crash", e.machine_id, duration_steps=e.duration_steps))
            else:
                out.append(FaultCmd(kind, e.machine_id, factor=1.0 - e.slowdown_fraction,
                                    duration_steps=max(1, e.duration_steps),
                                    ramp_steps=getattr(e, "ramp_steps", 0) or 0))
        return out

    @property
    def remaining(self) -> int:
        return len(self._events) - self._i


class FaultBook:
    """Active slow / net faults per node -> (throttle, bandwidth) multipliers at a given step."""

    def __init__(self) -> None:
        self._active: List[Tuple[FaultCmd, int]] = []

    def add(self, cmd: FaultCmd, step: int) -> None:
        if cmd.kind in ("slow", "net"):
            self._active.append((cmd, step))

    def heal(self, node: str) -> None:
        self._active = [(c, s) for c, s in self._active if c.node != node]

    def factors(self, step: int) -> Dict[str, Tuple[float, float]]:
        self._active = [(c, s) for c, s in self._active
                        if not c.duration_steps or step < s + c.duration_steps]
        out: Dict[str, Tuple[float, float]] = {}
        for c, s in self._active:
            if step < s:
                continue
            f = c.factor
            if c.ramp_steps:
                f = 1.0 - (1.0 - f) * min(1.0, (step - s + 1) / c.ramp_steps)
            thr, bw = out.get(c.node, (1.0, 1.0))
            out[c.node] = (thr * f, bw) if c.kind == "slow" else (thr, bw * f)
        return out
