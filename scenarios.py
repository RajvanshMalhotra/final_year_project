"""
scenarios.py
============
Fault scenarios at four difficulty levels, for training and for trying to break
the agents.

  easy       the original mix (1 permanent crash, 2-3 slowdowns of 10/60/300
             steps), exact measurements, no network cost. Same as
             simulator.RandomFaultGenerator.
  realistic  what we expect on three laptops on Wi-Fi: 5% measurement noise,
             occasional false-alarm spikes, all-reduce time, the 8 GB Mac capped
             at 30% of a batch, crashes that often come back, slowdowns with
             heavy-tailed durations (most short, some very long), gradual thermal
             ramps, network slowdowns, and "flapping" (a machine going slow/fast
             in bursts).
  hard       more of everything, overlapping faults, a Wi-Fi-wide slowdown that
             hits every machine at once, 10% noise.
  chaos      fault storms: several crashes (some at the same time), constant
             overlapping slowdowns, 15% noise. Meant to break things.

Durations are drawn from a log-normal distribution: the agent cannot know how
long a fault will last, which is exactly the wait-or-act problem (N1).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List

from simulator import FaultEvent, FaultSchedule, RandomFaultGenerator, SimulatedCluster

DIFFICULTIES = ("easy", "realistic", "hard", "chaos")
MACHINE_IDS  = ["mac-m2", "win-gpu-0", "win-gpu-1"]


@dataclass
class _Level:
    noise_std:      float
    spike_prob:     float
    comm_s:         float
    mac_cap:        float
    crashes:        tuple      # (min, max) per episode
    transient_prob: float      # chance a crash comes back
    crash_median:   float      # median ticks down (log-normal)
    slowdowns:      tuple
    slow_frac:      tuple      # (min, max) speed lost
    slow_median:    float
    ramp_prob:      float
    networks:       tuple
    net_frac:       tuple
    flap_bursts:    tuple
    wifi_wide_prob: float      # one network fault hitting every machine
    paired_crash_prob: float   # two machines crash on the same tick


LEVELS: Dict[str, _Level] = {
    "realistic": _Level(0.05, 0.01, 0.03, 0.30, (1, 2), 0.5, 30, (3, 5), (0.2, 0.6), 40, 0.4,
                        (1, 2), (0.5, 0.9), (0, 1), 0.0, 0.0),
    "hard":      _Level(0.10, 0.03, 0.05, 0.25, (2, 3), 0.6, 40, (5, 8), (0.2, 0.7), 40, 0.5,
                        (2, 4), (0.5, 0.95), (1, 2), 0.5, 0.0),
    "chaos":     _Level(0.15, 0.05, 0.08, 0.20, (3, 5), 0.7, 40, (8, 14), (0.2, 0.85), 30, 0.5,
                        (4, 6), (0.5, 0.95), (2, 4), 1.0, 0.5),
}


@dataclass
class Scenario:
    seed:           int
    difficulty:     str
    schedule:       FaultSchedule
    cluster_kwargs: Dict = field(default_factory=dict)

    @property
    def events(self) -> List[FaultEvent]:
        return self.schedule.events

    def make_cluster(self) -> SimulatedCluster:
        return SimulatedCluster(seed=self.seed, **self.cluster_kwargs)

    @property
    def normal_speeds(self) -> Dict[str, float]:
        return {mid: m.normal_speed for mid, m in self.make_cluster().machines.items()}


def _lognormal_ticks(rng: random.Random, median: float, sigma: float,
                     lo: int = 2, hi: int = 600) -> int:
    return int(min(hi, max(lo, round(rng.lognormvariate(math.log(median), sigma)))))


def _calibrated(kwargs: Dict, difficulty: str, calibration) -> Dict:
    """Measured speeds everywhere; measured all-reduce time scaled by the level's stress
    (hard/chaos keep their extra network cost relative to realistic). Easy has no comm."""
    if calibration is None:
        return kwargs
    out = dict(kwargs, normal_speeds=dict(calibration.speeds))
    if difficulty != "easy":
        out["comm_s"] = calibration.comm_s * LEVELS[difficulty].comm_s / LEVELS["realistic"].comm_s
    return out


def make_scenario(seed: int, difficulty: str = "realistic", total_steps: int = 1000,
                  calibration=None) -> Scenario:
    """calibration: a calibration.Calibration (measured laptop numbers), or None for the
    built-in estimates. The fault schedule is the same either way."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
    if difficulty == "easy":
        gen = RandomFaultGenerator(seed=seed, total_steps=total_steps)
        return Scenario(seed, difficulty, gen.as_schedule(), _calibrated({}, difficulty, calibration))

    L   = LEVELS[difficulty]
    rng = random.Random(f"{difficulty}-{seed}")
    T   = total_steps
    ev: List[FaultEvent] = []

    def start() -> int:
        return rng.randint(int(T * 0.03), int(T * 0.90))

    # Crashes. At most 2 permanent ones, so some machine can always come back.
    permanent = 0
    for _ in range(rng.randint(*L.crashes)):
        transient = rng.random() < L.transient_prob or permanent >= 2
        permanent += not transient
        dur = _lognormal_ticks(rng, L.crash_median, 1.0) if transient else 0
        s, mid = start(), rng.choice(MACHINE_IDS)
        ev.append(FaultEvent(s, mid, "crash", duration_steps=dur))
        if rng.random() < L.paired_crash_prob:
            other = rng.choice([m for m in MACHINE_IDS if m != mid])
            ev.append(FaultEvent(s, other, "crash",
                                 duration_steps=_lognormal_ticks(rng, L.crash_median, 1.0)))

    # Slowdowns: heavy-tailed durations, some ramp up gradually (thermal throttling).
    for _ in range(rng.randint(*L.slowdowns)):
        dur  = _lognormal_ticks(rng, L.slow_median, 1.2)
        ramp = rng.randint(5, 40) if rng.random() < L.ramp_prob else 0
        ev.append(FaultEvent(start(), rng.choice(MACHINE_IDS), "slowdown",
                             slowdown_fraction=rng.uniform(*L.slow_frac),
                             duration_steps=dur, ramp_steps=min(ramp, dur)))

    # Network slowdowns on one machine, and possibly the whole Wi-Fi at once.
    for _ in range(rng.randint(*L.networks)):
        ev.append(FaultEvent(start(), rng.choice(MACHINE_IDS), "network",
                             slowdown_fraction=rng.uniform(*L.net_frac),
                             duration_steps=_lognormal_ticks(rng, 30, 1.0)))
    if rng.random() < L.wifi_wide_prob:
        s, frac, dur = start(), rng.uniform(0.5, 0.8), _lognormal_ticks(rng, 40, 0.8)
        for mid in MACHINE_IDS:
            ev.append(FaultEvent(s, mid, "network", slowdown_fraction=frac, duration_steps=dur))

    # Flapping: one machine goes slow/fast in short bursts (worst case for acting at once).
    for _ in range(rng.randint(*L.flap_bursts)):
        mid, s = rng.choice(MACHINE_IDS), start()
        for _ in range(rng.randint(4, 8)):
            dur = rng.randint(2, 6)
            ev.append(FaultEvent(s, mid, "slowdown", slowdown_fraction=rng.uniform(0.3, 0.6),
                                 duration_steps=dur))
            s += dur + rng.randint(5, 15)

    kwargs = dict(noise_std=L.noise_std, spike_prob=L.spike_prob, comm_s=L.comm_s,
                  max_share={"mac-m2": L.mac_cap})
    return Scenario(seed, difficulty, FaultSchedule(ev), _calibrated(kwargs, difficulty, calibration))
