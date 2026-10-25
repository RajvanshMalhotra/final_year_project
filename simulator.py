"""
simulator.py
============
Self-Healing Mixed-GPU Training – Part B: Simulator & Agent Loop
-----------------------------------------------------------------
Days 1-2 : Core data structures, SimulatedCluster, agent loop, simple baselines.
Days 3-7 : RandomFaultGenerator, OraclePolicy, TimeDelayedPolicy, ClusterEnv
            (RL wrapper), ExperimentRunner, multi-seed sweeps, matplotlib plots.

Model of time (fixed 2026-10-25)
--------------------------------
* A *tick* is one attempt at a training step. Ticks never go backwards; faults
  are scheduled on ticks, so a rollback cannot replay them.
* `train_step` is real training progress. A stalled tick does not advance it and
  RESTART rewinds it to the last checkpoint, so it is the goodput numerator.
* Every action except WAIT pauses the whole ring for ACTION_OVERHEAD_S seconds.
  An action that would change nothing (e.g. REBALANCE on already balanced
  shares) is free, so repeating it is harmless.
* A crash is only visible once the machine's heartbeat is older than
  CRASH_SIGNAL_AGE_S. Until then it still looks alive with its last speed.
* Only machines in the ring (not evicted) are billed.

External deps (optional)
------------------------
  numpy      – multi-seed statistics (mean / std)
  matplotlib – PNG plot generation
All other code uses only the Python standard library.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# Optional scientific dependencies – degrade gracefully if absent.
try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    np = None          # type: ignore
    _NUMPY_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless – saves PNGs without a display
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    plt = None         # type: ignore
    _MPL_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# §1  DATA STRUCTURES & INTERFACES
# ═══════════════════════════════════════════════════════════════════════════════

class Action(Enum):
    """Every action the self-healing agent may take at each step."""
    WAIT        = auto()   # Do nothing; let the cluster continue as-is.
    REBALANCE   = auto()   # Re-split the batch in proportion to current speeds.
    COMPENSATE  = auto()   # Switch a slow node to a lighter kernel, then re-split.
    EVICT       = auto()   # Remove a node from the ring (it stays powered on).
    READMIT     = auto()   # Re-add a previously evicted node that is alive again.
    RESTART     = auto()   # Relaunch on the survivors from the last checkpoint.
    CHECKPOINT  = auto()   # Persist current model weights to durable storage.


# A policy returns an Action, or (Action, target machine id) to pick the target.
PolicyOutput = Union[Action, Tuple[Action, Optional[str]]]

# Seconds the whole ring pauses while an action runs.
ACTION_OVERHEAD_S: Dict[Action, float] = {
    Action.WAIT:        0.0,
    Action.REBALANCE:   0.5,    # move data shards between workers
    Action.COMPENSATE:  2.0,    # load a lighter kernel on one worker, then re-split
    Action.EVICT:       3.0,    # re-form the all-reduce ring without the worker
    Action.READMIT:     5.0,    # copy current weights to the returning worker
    Action.RESTART:     30.0,   # relaunch the job and reload the last checkpoint
    Action.CHECKPOINT:  0.5,    # write adapter + optimizer state
}

COMPENSATE_SPEEDUP:   float = 1.25   # a lighter kernel wins back part of a degraded worker's speed
STALL_TICK_S:         float = 1.0    # time lost per tick while the ring waits on a crashed worker
HEARTBEAT_INTERVAL_S: float = 0.5    # simulated seconds between heartbeats
CRASH_SIGNAL_AGE_S:   float = 3.0    # heartbeat older than this -> machine is presumed dead
DEAD_SPEED_THRESHOLD: float = 0.01   # steps/s; below this -> machine is presumed dead
SLOW_RATIO_THRESHOLD: float = 0.85   # below 85% of its own normal speed -> "slow"
SHARE_TOLERANCE:      float = 0.02   # shares further than this from ideal -> "imbalanced"


@dataclass
class MachineObs:
    """Per-machine slice of the observation the agent receives."""
    alive_signal_age: float   # Seconds since the last heartbeat was received.
    normal_speed:     float   # Baseline speed in "steps/second" under ideal conditions.
    measured_speed:   float   # Last measured speed (stale for a machine that went silent).
    share:            float   # Fraction of each batch this machine currently owns (0-1).
    code_version:     str     # Running code version (used to detect stale replicas).
    price:            float   # Hourly cost in USD for this machine.
    is_active:        bool = True    # In the ring and not presumed dead.
    evicted:          bool = False   # We removed it from the ring (it may still be alive).
    compensated:      bool = False   # Running the lighter COMPENSATE kernel.

    @property
    def active(self) -> bool:
        return self.is_active


@dataclass
class Observation:
    """Global snapshot the agent receives at every step."""
    time:       float                     # Wall-clock simulation time (seconds).
    step:       int                       # Tick index (monotonic; never rewinds).
    machines:   Dict[str, MachineObs]     # machine_id -> MachineObs
    train_step: int = 0                   # Training progress (rewinds on RESTART).


@dataclass
class EpisodeResult:
    """Metrics produced at the end of one simulation episode."""
    policy_name:      str
    total_steps:      int      # Ticks run.
    useful_steps:     int      # Net training progress at the end (after rollbacks).
    total_time_s:     float    # Total simulated time in seconds.
    total_cost_usd:   float    # Sum of price * time over machines while in the ring.
    goodput_per_hr:   float    # useful_steps / (total_time_s / 3600)
    goodput_per_usd:  float    # useful_steps / total_cost_usd
    total_wasted_time_s: float # Stalls + action overheads + time of rolled-back steps.
    work_lost:        int      # Rolled-back steps (due to RESTART actions).
    avg_decision_ms:  float    # Mean policy decision latency in milliseconds.
    decisions:        Dict[str, int] = field(default_factory=dict)  # Action -> count

    def __str__(self) -> str:
        lines = [
            f"  Policy            : {self.policy_name}",
            f"  Total steps       : {self.total_steps}",
            f"  Useful steps      : {self.useful_steps}",
            f"  Goodput / hr      : {self.goodput_per_hr:>10.2f} steps/hr",
            f"  Goodput / $       : {self.goodput_per_usd:>10.2f} steps/$",
            f"  Wasted time       : {self.total_wasted_time_s:>10.2f} s",
            f"  Work lost         : {self.work_lost:>10d} steps",
            f"  Avg decision time : {self.avg_decision_ms:>10.4f} ms",
            f"  Decision counts   : {self.decisions}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# §1b  SHARED HELPERS (used by the cluster, the policies, the Oracle and the MPC)
# ═══════════════════════════════════════════════════════════════════════════════

def proportional_shares(speeds: Dict[str, float]) -> Dict[str, float]:
    """b_i = s_i / sum(s): every worker finishes its part of the batch at the same time."""
    total = sum(speeds.values())
    if total <= 0:
        return {mid: 0.0 for mid in speeds}
    return {mid: s / total for mid, s in speeds.items()}


def step_time(speeds: Dict[str, float], shares: Dict[str, float]) -> float:
    """Synchronous all-reduce: the step lasts as long as the slowest worker's part."""
    times = [shares.get(mid, 0.0) / s for mid, s in speeds.items()
             if s > 0 and shares.get(mid, 0.0) > 0]
    return max(times) if times else math.inf


def speed_ratio(mobs: MachineObs) -> float:
    return mobs.measured_speed / mobs.normal_speed if mobs.normal_speed > 0 else 0.0


def is_dead(mobs: MachineObs) -> bool:
    """Presumed crashed and still in the ring. An evicted machine is no longer our problem."""
    return (not mobs.evicted
            and (mobs.measured_speed < DEAD_SPEED_THRESHOLD
                 or mobs.alive_signal_age > CRASH_SIGNAL_AGE_S))


def is_slow(mobs: MachineObs) -> bool:
    return mobs.is_active and speed_ratio(mobs) < SLOW_RATIO_THRESHOLD


def is_readmittable(mobs: MachineObs) -> bool:
    """Evicted, still sending heartbeats, and back to (near) normal speed."""
    return (mobs.evicted
            and mobs.alive_signal_age <= CRASH_SIGNAL_AGE_S
            and speed_ratio(mobs) >= SLOW_RATIO_THRESHOLD)


def _shares_differ(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-9) -> bool:
    keys = set(a) | set(b)
    return any(abs(a.get(k, 0.0) - b.get(k, 0.0)) > tol for k in keys)


def shares_imbalanced(obs: Observation, tol: float = SHARE_TOLERANCE) -> bool:
    """True when the current split is noticeably off from speed-proportional."""
    speeds = {mid: m.measured_speed for mid, m in obs.machines.items() if m.is_active}
    shares = {mid: m.share for mid, m in obs.machines.items() if m.is_active}
    return _shares_differ(proportional_shares(speeds), shares, tol)


def default_target(obs: Observation, action: Action) -> Optional[str]:
    """The machine an action applies to when the policy did not name one."""
    machines = obs.machines
    active = [mid for mid, m in machines.items() if m.is_active]
    if action == Action.EVICT:
        dead = [mid for mid, m in machines.items() if is_dead(m)]
        if dead:
            return dead[0]
        if len(active) > 1:
            return min(active, key=lambda mid: speed_ratio(machines[mid]))
    elif action == Action.COMPENSATE:
        slow = [mid for mid in active
                if not machines[mid].compensated and speed_ratio(machines[mid]) < 1.0]
        if slow:
            return min(slow, key=lambda mid: speed_ratio(machines[mid]))
    elif action == Action.READMIT:
        for mid, m in machines.items():
            if is_readmittable(m):
                return mid
    return None


def plan_action(
    obs: Observation,
    action: Action,
    target: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Predict (overhead_s, step_time_s afterwards) of an action, from the observation
    alone, using the same rules as SimulatedCluster.apply_action. step_time is
    math.inf when the ring would still be stalled by a dead machine.
    """
    if target is None:
        target = default_target(obs, action)
    machines = obs.machines
    in_ring  = {mid for mid, m in machines.items() if not m.evicted}
    dead     = {mid for mid in in_ring if is_dead(machines[mid])}
    speeds   = {mid: machines[mid].measured_speed for mid in in_ring - dead}
    shares   = {mid: machines[mid].share for mid in in_ring - dead}
    overhead = 0.0

    if action == Action.REBALANCE:
        new = proportional_shares(speeds)
        if _shares_differ(new, shares):
            shares, overhead = new, ACTION_OVERHEAD_S[action]

    elif action == Action.COMPENSATE:
        m = machines.get(target) if target else None
        if (m is not None and target in speeds and not m.compensated
                and m.measured_speed < m.normal_speed):
            speeds[target] *= COMPENSATE_SPEEDUP
            shares, overhead = proportional_shares(speeds), ACTION_OVERHEAD_S[action]

    elif action == Action.EVICT:
        if target in dead:
            dead.discard(target)
            shares, overhead = proportional_shares(speeds), ACTION_OVERHEAD_S[action]
        elif target in speeds and len(speeds) > 1:
            speeds.pop(target)
            shares, overhead = proportional_shares(speeds), ACTION_OVERHEAD_S[action]

    elif action == Action.READMIT:
        m = machines.get(target) if target else None
        if m is not None and m.evicted and m.alive_signal_age <= CRASH_SIGNAL_AGE_S:
            speeds[target] = m.measured_speed
            shares, overhead = proportional_shares(speeds), ACTION_OVERHEAD_S[action]

    elif action == Action.RESTART:
        dead = set()
        shares, overhead = proportional_shares(speeds), ACTION_OVERHEAD_S[action]

    elif action == Action.CHECKPOINT:
        overhead = ACTION_OVERHEAD_S[action]

    if dead:
        return overhead, math.inf
    return overhead, step_time(speeds, shares)


def _unpack(output: PolicyOutput) -> Tuple[Action, Optional[str]]:
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


# ═══════════════════════════════════════════════════════════════════════════════
# §2  FAULT SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FaultEvent:
    """A single fault injected into the simulation."""
    step:       int    # Tick at which the fault is injected.
    machine_id: str    # Target machine.
    kind:       str    # "crash" | "slowdown"
    # For "slowdown" only:
    slowdown_fraction: float = 0.40   # Fraction by which speed is reduced.
    duration_steps:    int   = 20     # How many ticks the slowdown lasts.


class FaultSchedule:
    """
    Immutable list of faults.  The simulator queries this at every step.
    """

    def __init__(self, events: Optional[List[FaultEvent]] = None):
        self._events: List[FaultEvent] = sorted(events or [], key=lambda e: e.step)

    def events_at(self, step: int) -> List[FaultEvent]:
        return [e for e in self._events if e.step == step]

    @property
    def events(self) -> List[FaultEvent]:
        return list(self._events)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  SIMULATED CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _MachineState:
    """Internal mutable state for one machine."""
    machine_id:      str
    label:           str           # Human-readable label (e.g. "Mac-M2").
    normal_speed:    float         # steps / simulated-second under normal load.
    measured_speed:  float         # Current true speed (0 once crashed).
    share:           float         # Fraction of each batch owned by this machine.
    price_per_hr:    float         # Hourly USD cost while in the ring.
    code_version:    str = "v1.0"
    alive:           bool = True   # False once crashed (only a crash sets this).
    evicted:         bool = False  # Our decision; independent of `alive`.
    compensated:     bool = False
    # Heartbeat tracking
    last_heartbeat:  float = 0.0   # Simulated time of last heartbeat.
    last_seen_speed: float = 0.0   # What observers still see after a crash.
    # Slowdown tracking
    slowdown_steps_left: int = 0

    @property
    def is_active(self) -> bool:
        return self.alive and not self.evicted and self.measured_speed > 0

    @property
    def active(self) -> bool:
        return self.is_active


class SimulatedCluster:
    """
    Simulates a 3-machine heterogeneous cluster (1 Mac M2 + 2 Windows NVIDIA).

    Time model
    ----------
    Each machine takes `m.share / m.measured_speed` simulated seconds to process its share.
    The bottleneck is the slowest machine in the ring (All-reduce barrier). A crashed
    machine still in the ring stalls every tick for STALL_TICK_S.
    """

    HEARTBEAT_INTERVAL_S: float = HEARTBEAT_INTERVAL_S
    CRASH_SIGNAL_AGE:     float = CRASH_SIGNAL_AGE_S

    def __init__(self, seed: int = 42, checkpoint_every: int = 100, verbose: bool = False):
        rng = random.Random(seed)
        self._sim_time:   float = 0.0
        self._tick:       int   = 0
        self._train_step: int   = 0
        self._cost_usd:   float = 0.0
        self._checkpoint_every = checkpoint_every
        self._verbose = verbose
        self._fired: set = set()          # ids of fault events already injected

        # Bookkeeping for metrics
        self.stall_time_s:    float = 0.0
        self.overhead_time_s: float = 0.0
        self.lost_time_s:     float = 0.0  # time spent on steps later rolled back
        self.work_lost:       int   = 0
        self._time_since_ckpt: float = 0.0

        # Initialise the three machines
        self._machines: Dict[str, _MachineState] = {
            "mac-m2": _MachineState(
                machine_id    = "mac-m2",
                label         = "Mac M2 (MLX)",
                normal_speed  = 1.2,        # steps/sim-second – slowest
                measured_speed= 1.2,
                share         = 0.30,
                price_per_hr  = 0.00,       # Personal laptop -> $0 marginal cost
            ),
            "win-gpu-0": _MachineState(
                machine_id    = "win-gpu-0",
                label         = "Windows NVIDIA #0",
                normal_speed  = 3.5,
                measured_speed= 3.5,
                share         = 0.35,
                price_per_hr  = 0.50,       # Electricity / cloud cost estimate
            ),
            "win-gpu-1": _MachineState(
                machine_id    = "win-gpu-1",
                label         = "Windows NVIDIA #1",
                normal_speed  = 3.5,
                measured_speed= 3.5,
                share         = 0.35,
                price_per_hr  = 0.50,
            ),
        }

        # Stagger initial heartbeats
        for m in self._machines.values():
            m.last_heartbeat = -rng.uniform(0, self.HEARTBEAT_INTERVAL_S)
            m.last_seen_speed = m.measured_speed

        self._last_checkpoint_step: int = 0
        self._rebalance_shares()   # Distribute initial shares proportional to capacity

    # Public helpers

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def step(self) -> int:
        """Tick index (monotonic)."""
        return self._tick

    @property
    def train_step(self) -> int:
        """Net training progress (rewinds on RESTART)."""
        return self._train_step

    @property
    def wasted_time_s(self) -> float:
        return self.stall_time_s + self.overhead_time_s + self.lost_time_s

    @property
    def machines(self) -> Dict[str, _MachineState]:
        return self._machines

    @property
    def machine_list(self) -> List[_MachineState]:
        return list(self._machines.values())

    def get_machine(self, identifier: Union[str, int]) -> _MachineState:
        if isinstance(identifier, int):
            keys = list(self._machines.keys())
            if 1 <= identifier <= len(keys):
                return self._machines[keys[identifier - 1]]
            elif 0 <= identifier < len(keys):
                return self._machines[keys[identifier]]
        return self._machines[str(identifier)]

    def inject_slowdown(
        self,
        machine_id: str,
        slowdown_fraction: float = 0.50,
        duration_steps: int = 20,
    ) -> None:
        """Helper to inject a slowdown fault directly."""
        self._inject_fault(FaultEvent(
            step=self._tick, machine_id=machine_id, kind="slowdown",
            slowdown_fraction=slowdown_fraction, duration_steps=duration_steps,
        ))

    def inject_crash(self, machine_id: str) -> None:
        """Helper to inject a crash fault directly."""
        self._inject_fault(FaultEvent(step=self._tick, machine_id=machine_id, kind="crash"))

    def fire_due_faults(self, fault_schedule: FaultSchedule) -> None:
        """Inject every scheduled fault whose tick has come and that has not fired yet."""
        for event in fault_schedule.events:
            if event.step <= self._tick and id(event) not in self._fired:
                self._fired.add(id(event))
                self._inject_fault(event)

    def apply_action(self, action: Action, target_machine_id: Optional[str] = None) -> Dict[str, float]:
        """
        Apply an action. Returns {"overhead_s": ..., "steps_lost": ...}.
        Without a target, the same default as plan_action() is used.
        """
        if target_machine_id is None:
            target_machine_id = default_target(self.get_observation(), action)
        m = self._machines.get(target_machine_id) if target_machine_id else None
        changed    = False
        steps_lost = 0

        if action == Action.REBALANCE:
            changed = self._rebalance_shares()

        elif action == Action.COMPENSATE:
            if (m is not None and m.is_active and not m.compensated
                    and m.measured_speed < m.normal_speed):
                m.measured_speed *= COMPENSATE_SPEEDUP
                m.compensated = True
                self._rebalance_shares()
                changed = True

        elif action == Action.EVICT:
            in_ring_alive = [x for x in self._machines.values() if x.is_active]
            if m is not None and not m.evicted and (not m.alive or len(in_ring_alive) > 1):
                m.evicted = True
                m.share   = 0.0
                self._rebalance_shares()
                changed = True

        elif action == Action.READMIT:
            if m is not None and m.evicted and m.alive:
                m.evicted = False
                self._rebalance_shares()
                changed = True

        elif action == Action.RESTART:
            steps_lost = self.apply_restart()
            changed = True

        elif action == Action.CHECKPOINT:
            self.apply_checkpoint()
            changed = True

        overhead = ACTION_OVERHEAD_S[action] if changed else 0.0
        if overhead > 0:
            self._advance(overhead)
            self.overhead_time_s += overhead
        return {"overhead_s": overhead, "steps_lost": float(steps_lost)}

    def get_observation(self) -> Observation:
        obs_machines: Dict[str, MachineObs] = {}
        for mid, m in self._machines.items():
            age   = self._sim_time - m.last_heartbeat
            speed = m.measured_speed if m.alive else m.last_seen_speed
            obs_machines[mid] = MachineObs(
                alive_signal_age = age,
                normal_speed     = m.normal_speed,
                measured_speed   = speed,
                share            = m.share,
                code_version     = m.code_version,
                price            = m.price_per_hr,
                is_active        = (not m.evicted and age <= self.CRASH_SIGNAL_AGE
                                    and speed > DEAD_SPEED_THRESHOLD),
                evicted          = m.evicted,
                compensated      = m.compensated,
            )
        return Observation(time=self._sim_time, step=self._tick,
                           machines=obs_machines, train_step=self._train_step)

    def step_forward(self, fault_schedule: FaultSchedule) -> Tuple[bool, float]:
        """
        Advance one tick.

        Returns
        -------
        (step_was_useful, simulated_seconds_elapsed)
            step_was_useful  - False when cluster is stalled (e.g. unevicted crash).
            simulated_seconds - the wall time consumed by this step.
        """
        # 1. Inject faults due now (no-op if run_episode already did)
        self.fire_due_faults(fault_schedule)

        # 2. A crashed machine still in the ring freezes the all-reduce barrier
        ring = [m for m in self._machines.values() if not m.evicted]
        if any(not m.alive for m in ring) or not any(m.is_active for m in ring):
            self._advance(STALL_TICK_S)
            self.stall_time_s += STALL_TICK_S
            useful, duration = False, STALL_TICK_S
        else:
            active   = {m.machine_id: m.measured_speed for m in ring}
            shares   = {m.machine_id: m.share for m in ring}
            duration = step_time(active, shares)
            self._advance(duration)
            self._train_step      += 1
            self._time_since_ckpt += duration
            useful = True
            if (self._checkpoint_every and self._train_step % self._checkpoint_every == 0
                    and self._train_step > self._last_checkpoint_step):
                self.apply_action(Action.CHECKPOINT)

        # 3. Tick slowdown counters for the step that just ran, then move to the next tick
        for m in self._machines.values():
            if m.slowdown_steps_left > 0:
                m.slowdown_steps_left -= 1
                if m.slowdown_steps_left == 0 and m.alive:
                    m.measured_speed  = m.normal_speed   # recover
                    m.last_seen_speed = m.normal_speed
                    m.compensated     = False
        self._tick += 1

        # 4. Faults due at the new tick are visible in the next observation
        self.fire_due_faults(fault_schedule)
        return useful, duration

    # Action handlers

    def apply_evict(self, machine_id: str) -> None:
        """Remove a machine from the active training ring."""
        self.apply_action(Action.EVICT, machine_id)

    def apply_readmit(self, machine_id: str) -> None:
        """Re-admit a previously evicted machine."""
        self.apply_action(Action.READMIT, machine_id)

    def apply_rebalance(self) -> None:
        """Redistribute dataset shares among active machines."""
        self.apply_action(Action.REBALANCE)

    def apply_restart(self) -> int:
        """
        Relaunch from the last checkpoint on the machines that are still alive.

        Returns the number of steps that were lost (rolled back).
        """
        steps_lost = self._train_step - self._last_checkpoint_step
        self._train_step = self._last_checkpoint_step
        self.work_lost  += steps_lost
        self.lost_time_s += self._time_since_ckpt
        self._time_since_ckpt = 0.0
        for m in self._machines.values():
            if not m.alive:
                m.evicted = True     # relaunch without the dead machine
                m.share   = 0.0
        self._rebalance_shares()
        return steps_lost

    def apply_checkpoint(self) -> None:
        """Persist current step as a durable checkpoint."""
        self._last_checkpoint_step = self._train_step
        self._time_since_ckpt = 0.0

    def total_cost_so_far(self) -> float:
        """Cumulative USD cost of the machines while they were in the ring."""
        return self._cost_usd

    # Private helpers

    def _advance(self, seconds: float) -> None:
        """Move the clock, bill machines in the ring, and emit heartbeats."""
        self._sim_time += seconds
        self._cost_usd += sum(m.price_per_hr for m in self._machines.values()
                              if not m.evicted) * seconds / 3600.0
        for m in self._machines.values():
            if m.alive and (self._sim_time - m.last_heartbeat) >= self.HEARTBEAT_INTERVAL_S:
                m.last_heartbeat = self._sim_time

    def _inject_fault(self, event: FaultEvent) -> None:
        m = self._machines.get(event.machine_id)
        if m is None or not m.alive:
            return

        if event.kind == "crash":
            m.last_seen_speed = m.measured_speed   # observers keep the last value they saw
            m.alive           = False              # process crashed
            m.measured_speed  = 0.0                # no progress
            m.compensated     = False
            m.slowdown_steps_left = 0
            if self._verbose:
                print(f"  [FAULT] Step {self._tick}: CRASH injected -> {event.machine_id}")

        elif event.kind == "slowdown":
            frac = max(0.0, min(0.95, event.slowdown_fraction))
            m.measured_speed      = m.normal_speed * (1.0 - frac)
            m.last_seen_speed     = m.measured_speed
            m.compensated         = False
            m.slowdown_steps_left = max(m.slowdown_steps_left, event.duration_steps)
            if self._verbose:
                print(f"  [FAULT] Step {self._tick}: SLOWDOWN ({frac*100:.0f}%) injected "
                      f"-> {event.machine_id} for {event.duration_steps} steps")

    def _rebalance_shares(self) -> bool:
        """Split the batch in proportion to current speed. Returns True if shares changed."""
        active = {mid: m.measured_speed for mid, m in self._machines.items() if m.is_active}
        new    = proportional_shares(active)
        old    = {mid: m.share for mid, m in self._machines.items()}
        for mid, m in self._machines.items():
            m.share = new.get(mid, 0.0)
        return _shares_differ(old, {mid: m.share for mid, m in self._machines.items()})


# ═══════════════════════════════════════════════════════════════════════════════
# §4  BASE POLICY
# ═══════════════════════════════════════════════════════════════════════════════

class BasePolicy:
    """Abstract base class for all policies."""

    name: str = "BasePolicy"

    def decide(self, observation: Observation) -> PolicyOutput:
        raise NotImplementedError("Subclasses must implement decide().")


# ═══════════════════════════════════════════════════════════════════════════════
# §5  SIMPLE BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

class DoNothingPolicy(BasePolicy):
    """
    Baseline 1 - Always WAIT.
    Represents the naive approach of ignoring all faults.
    """
    name = "DoNothingPolicy"

    def decide(self, observation: Observation) -> PolicyOutput:
        return Action.WAIT


class AlwaysRestartPolicy(BasePolicy):
    """
    Baseline 2 - RESTART (on the survivors, from the last checkpoint) whenever a
    machine is presumed dead. The classic checkpoint-restart strategy.
    """
    name = "AlwaysRestartPolicy"

    def decide(self, observation: Observation) -> PolicyOutput:
        if any(is_dead(m) for m in observation.machines.values()):
            return Action.RESTART
        return Action.WAIT


class RebalanceOnlyPolicy(BasePolicy):
    """
    Baseline 3 - EVICT dead machines; REBALANCE as soon as the split is off
    (a machine slowed down or recovered). Reacts immediately, never waits.
    """
    name = "RebalanceOnlyPolicy"

    def decide(self, observation: Observation) -> PolicyOutput:
        if any(is_dead(m) for m in observation.machines.values()):
            return Action.EVICT
        if shares_imbalanced(observation):
            return Action.REBALANCE
        return Action.WAIT


# ═══════════════════════════════════════════════════════════════════════════════
# §6  AGENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(
    cluster:        SimulatedCluster,
    policy:         BasePolicy,
    fault_schedule: FaultSchedule,
    total_steps:    int,
) -> EpisodeResult:
    """
    Run one full training episode of `total_steps` ticks.

    Each tick: observe -> policy decides (timed) -> apply action -> one step.
    All action semantics live in SimulatedCluster.apply_action.
    """
    decision_times_ms: List[float] = []
    decision_counts:   Dict[str, int] = {a.name: 0 for a in Action}

    # Issue an initial checkpoint so RESTART has a valid rollback point.
    cluster.apply_checkpoint()
    cluster.fire_due_faults(fault_schedule)

    for _ in range(total_steps):
        obs = cluster.get_observation()

        t0 = time.perf_counter()
        action, target = _unpack(policy.decide(obs))
        decision_times_ms.append((time.perf_counter() - t0) * 1_000)
        decision_counts[action.name] += 1

        cluster.apply_action(action, target)
        cluster.step_forward(fault_schedule)

    useful_steps   = cluster.train_step
    total_time_s   = cluster.sim_time
    total_cost_usd = cluster.total_cost_so_far()

    goodput_per_hr  = (useful_steps / (total_time_s / 3600.0)) if total_time_s > 0 else 0.0
    goodput_per_usd = (useful_steps / total_cost_usd)           if total_cost_usd > 0 else float("inf")
    avg_decision_ms = (sum(decision_times_ms) / len(decision_times_ms)
                       if decision_times_ms else 0.0)

    return EpisodeResult(
        policy_name         = policy.name,
        total_steps         = total_steps,
        useful_steps        = useful_steps,
        total_time_s        = total_time_s,
        total_cost_usd      = total_cost_usd,
        goodput_per_hr      = goodput_per_hr,
        goodput_per_usd     = goodput_per_usd,
        total_wasted_time_s = cluster.wasted_time_s,
        work_lost           = cluster.work_lost,
        avg_decision_ms     = avg_decision_ms,
        decisions           = {k: v for k, v in decision_counts.items() if v > 0},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §8  DAY 3 – RANDOM FAULT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RandomFaultGenerator:
    """
    Generates a randomised FaultSchedule for a 1 000-step episode.

    Fault mix per episode
    ---------------------
    * 1 crash     (fail-stop)  – chosen machine, random step.
    * 2-3 thermal throttles (fail-slow) – varying durations:
        short  :  ~10 simulated steps
        medium : ~60 simulated steps
        long   : ~300 simulated steps
    """

    MACHINE_IDS = ["mac-m2", "win-gpu-0", "win-gpu-1"]

    def __init__(self, seed: int = 0, total_steps: int = 1000):
        self._seed        = seed
        self._total_steps = total_steps
        self._rng         = random.Random(seed)
        self._events: List[FaultEvent] = []
        self._generate()

    # Public interface matches FaultSchedule so the two are interchangeable.
    def as_schedule(self) -> FaultSchedule:
        return FaultSchedule(events=list(self._events))

    def events_at(self, step: int) -> List[FaultEvent]:
        return [e for e in self._events if e.step == step]

    @property
    def events(self) -> List[FaultEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    def _generate(self) -> None:
        rng = self._rng
        T   = self._total_steps
        machines = self.MACHINE_IDS

        # 1. One crash – placed in first 60 % of the episode
        crash_step = rng.randint(int(T * 0.10), int(T * 0.60))
        crash_machine = rng.choice(machines)
        self._events.append(FaultEvent(
            step       = crash_step,
            machine_id = crash_machine,
            kind       = "crash",
        ))

        # 2. 2-3 thermal throttles
        n_throttles = rng.randint(2, 3)
        duration_pool = [10, 60, 300]   # short / medium / long
        used_steps: set = {crash_step}

        for i in range(n_throttles):
            dur   = duration_pool[i % len(duration_pool)]
            frac  = rng.uniform(0.30, 0.50)
            # Pick a step that doesn't collide with existing events
            for _ in range(50):
                s = rng.randint(int(T * 0.05), int(T * 0.85))
                if s not in used_steps:
                    used_steps.add(s)
                    break
            m = rng.choice([mid for mid in machines if mid != crash_machine])
            self._events.append(FaultEvent(
                step              = s,
                machine_id        = m,
                kind              = "slowdown",
                slowdown_fraction = frac,
                duration_steps    = dur,
            ))

        self._events.sort(key=lambda e: e.step)


# ═══════════════════════════════════════════════════════════════════════════════
# §9  DAY 3 – SMART POLICIES
# ═══════════════════════════════════════════════════════════════════════════════

class OraclePolicy(BasePolicy):
    """
    Perfect-information planner – knows the whole fault schedule.

    Decision rules
    --------------
    * A machine has crashed (by the schedule) and is still in the ring -> EVICT it
      at once (no heartbeat timeout needed).
    * Otherwise the current situation lasts R more ticks: until the next fault
      starts or an active slowdown ends (capped at HORIZON). For every valid
      action, cost = overhead + R * step_time_after. Pick the cheapest; WAIT
      wins ties.
    """
    name = "OraclePolicy"

    HORIZON: int = 300    # cap on how far ahead one decision has to pay off

    def __init__(self, fault_generator: object):
        self._gen = fault_generator

    @property
    def events(self) -> List[FaultEvent]:
        if hasattr(self._gen, "events"):
            return self._gen.events
        elif isinstance(self._gen, list):
            return self._gen
        return []

    def _ticks_until_change(self, tick: int) -> int:
        changes = [tick + self.HORIZON]
        for e in self.events:
            if e.step > tick:
                changes.append(e.step)
            elif e.kind == "slowdown" and e.step + e.duration_steps > tick:
                changes.append(e.step + e.duration_steps)
        return max(1, min(changes) - tick)

    def decide(self, observation: Observation) -> PolicyOutput:
        tick = observation.step
        machines = observation.machines

        for e in self.events:
            if (e.kind == "crash" and e.step <= tick
                    and e.machine_id in machines and not machines[e.machine_id].evicted):
                return (Action.EVICT, e.machine_id)

        crashed = {e.machine_id for e in self.events if e.kind == "crash" and e.step <= tick}
        R = self._ticks_until_change(tick)

        candidates: List[Tuple[Action, Optional[str]]] = [(Action.WAIT, None), (Action.REBALANCE, None)]
        for mid, m in machines.items():
            if m.is_active:
                candidates.append((Action.COMPENSATE, mid))
                candidates.append((Action.EVICT, mid))
            elif m.evicted and mid not in crashed:
                candidates.append((Action.READMIT, mid))

        best, best_cost = (Action.WAIT, None), math.inf
        for action, target in candidates:
            overhead, st = plan_action(observation, action, target)
            cost = overhead + R * st
            if cost < best_cost - 1e-9:
                best, best_cost = (action, target), cost
        return best


class TimeDelayedPolicy(BasePolicy):
    """
    Waits `wait_steps` ticks after the split first becomes imbalanced (a machine
    slowed down or recovered) before acting. Dead machines are evicted at once.

    State machine
    -------------
    NORMAL   → split goes off → record tick + enter WATCHING
    WATCHING → still off after wait_steps → REBALANCE
    any time the split is fine again → reset to NORMAL
    """

    def __init__(self, wait_steps: int = 60, wait_time: Optional[int] = None):
        if wait_time is not None:
            wait_steps = wait_time
        self._wait_steps = wait_steps
        self._detect_step: Optional[int] = None   # tick at which imbalance first noticed
        self.name = f"TimeDelayedPolicy(wait={wait_steps})"

    def decide(self, observation: Observation) -> PolicyOutput:
        if any(is_dead(m) for m in observation.machines.values()):
            self._detect_step = None
            return Action.EVICT

        if not shares_imbalanced(observation):
            self._detect_step = None   # cluster balanced again
            return Action.WAIT

        if self._detect_step is None:
            self._detect_step = observation.step
            return Action.WAIT

        if observation.step - self._detect_step >= self._wait_steps:
            return Action.REBALANCE

        return Action.WAIT   # still inside patience window


# ═══════════════════════════════════════════════════════════════════════════════
# §10 DAY 4 – RL ENVIRONMENT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class ClusterEnv:
    """
    OpenAI-Gym-style wrapper around SimulatedCluster.

    Reward (per tick)
    -----------------
    r_t = Δtrain_steps − HEALTHY_RATE · Δtime_s − cost_weight · Δcost_usd

    HEALTHY_RATE is the steps/second of the healthy cluster, so a normal step
    earns about 0, a slow step or an action overhead earns less than 0, and a
    RESTART loses every rolled-back step. Summed over an episode this is
    (goodput − ideal goodput), which is what we want to maximise.

    Usage
    -----
    env = ClusterEnv(fault_generator, total_steps=1000)
    obs = env.reset(seed=42)
    while True:
        action = my_policy(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    """

    def __init__(
        self,
        fault_generator: Optional[RandomFaultGenerator] = None,
        total_steps:    int = 1000,
        cost_weight:    float = 0.0,
    ):
        if fault_generator is None:
            fault_generator = RandomFaultGenerator(seed=0, total_steps=total_steps)
        self._gen         = fault_generator
        self._total_steps = total_steps
        self._cost_weight = cost_weight
        self._cluster:    Optional[SimulatedCluster] = None
        self._schedule:   Optional[FaultSchedule]   = None
        self._steps_done: int = 0
        self._healthy_rate: float = 0.0

    def reset(self, seed: int = 0) -> Observation:
        """Reset the environment and return the first observation."""
        # Rebuild fault generator with new seed for variety
        new_gen          = RandomFaultGenerator(seed=seed, total_steps=self._total_steps)
        self._schedule   = new_gen.as_schedule()
        self._cluster    = SimulatedCluster(seed=seed)
        self._cluster.apply_checkpoint()
        self._cluster.fire_due_faults(self._schedule)
        self._steps_done = 0
        self._healthy_rate = sum(m.normal_speed for m in self._cluster.machines.values())
        return self._cluster.get_observation()

    def step(self, action: PolicyOutput) -> Tuple[Observation, float, bool, dict]:
        """
        Apply `action`, advance one tick.

        Returns
        -------
        obs    : new Observation
        reward : see class docstring
        done   : True when episode is finished
        info   : diagnostic dict
        """
        assert self._cluster is not None and self._schedule is not None, "Call reset() before step()."
        c = self._cluster
        progress0, time0, cost0 = c.train_step, c.sim_time, c.total_cost_so_far()

        act, target = _unpack(action)
        info_action = c.apply_action(act, target)
        c.step_forward(self._schedule)
        self._steps_done += 1

        reward = ((c.train_step - progress0)
                  - self._healthy_rate * (c.sim_time - time0)
                  - self._cost_weight * (c.total_cost_so_far() - cost0))

        done = bool(self._steps_done >= self._total_steps)
        obs  = c.get_observation()
        info = {"work_lost": int(info_action["steps_lost"]), "useful_steps": int(c.train_step)}
        return obs, float(reward), done, info


# ═══════════════════════════════════════════════════════════════════════════════
# §11 DAY 4 – MULTI-SEED EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SweepStats:
    """Aggregate statistics over multiple seeds."""
    policy_name:          str
    mean_goodput_hr:      float
    std_goodput_hr:       float
    mean_goodput_usd:     float
    std_goodput_usd:      float
    mean_wasted_time_s:   float
    std_wasted_time_s:    float
    mean_work_lost:       float
    std_work_lost:        float
    n_seeds:              int

    def __str__(self) -> str:
        def _fmt(m: float, s: float, unit: str = "") -> str:
            return f"{m:.2f} +/- {s:.2f}{unit}"
        return (
            f"  {self.policy_name:<30s}  "
            f"Gput/hr={_fmt(self.mean_goodput_hr, self.std_goodput_hr, ' s/hr')}  "
            f"Gput/$={_fmt(self.mean_goodput_usd, self.std_goodput_usd, ' s/$')}  "
            f"Wasted={_fmt(self.mean_wasted_time_s, self.std_wasted_time_s, 's')}  "
            f"Lost={_fmt(self.mean_work_lost, self.std_work_lost, ' steps')}"
        )


def _mean(lst: List[float]) -> float:
    return sum(lst) / len(lst)


def _std_fallback(lst: List[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    if len(lst) <= 1:
        return 0.0
    m = sum(lst) / len(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))


def _finite_gput_usd(result: EpisodeResult) -> float:
    # Replace inf (nothing billed) with a large finite number so stats work
    return (result.goodput_per_usd if result.goodput_per_usd != float("inf")
            else result.useful_steps / 1e-9)


class ExperimentRunner:
    """
    Runs a policy across `n_seeds` independent episodes and aggregates stats.

    For each seed a new `RandomFaultGenerator` and a fresh `SimulatedCluster`
    are created so every seed sees a different but reproducible fault scenario.
    """

    def __init__(self, n_seeds: int = 5, total_steps: int = 1000):
        self._n_seeds     = n_seeds
        self._total_steps = total_steps

    def run(self, policy_factory, policy_name: Optional[str] = None) -> SweepStats:
        """
        Parameters
        ----------
        policy_factory : callable(fault_gen) -> BasePolicy
            Factory that accepts the RandomFaultGenerator and returns a
            freshly constructed policy.  For stateless policies you can
            pass a lambda: e.g. ``lambda _: DoNothingPolicy()``.
        policy_name    : display name (falls back to policy.name attribute).
        """
        gput_hrs:   List[float] = []
        gput_usds:  List[float] = []
        wasted_ts:  List[float] = []
        works_lost: List[float] = []
        name = policy_name or ""

        for seed in range(self._n_seeds):
            gen      = RandomFaultGenerator(seed=seed, total_steps=self._total_steps)
            schedule = gen.as_schedule()
            cluster  = SimulatedCluster(seed=seed)
            policy   = policy_factory(gen)
            name     = name or policy.name

            result = run_episode(cluster, policy, schedule, self._total_steps)

            gput_hrs.append(result.goodput_per_hr)
            gput_usds.append(_finite_gput_usd(result))
            wasted_ts.append(result.total_wasted_time_s)
            works_lost.append(float(result.work_lost))

        return SweepStats(
            policy_name        = name,
            mean_goodput_hr    = _mean(gput_hrs),
            std_goodput_hr     = _std_fallback(gput_hrs),
            mean_goodput_usd   = _mean(gput_usds),
            std_goodput_usd    = _std_fallback(gput_usds),
            mean_wasted_time_s = _mean(wasted_ts),
            std_wasted_time_s  = _std_fallback(wasted_ts),
            mean_work_lost     = _mean(works_lost),
            std_work_lost      = _std_fallback(works_lost),
            n_seeds            = self._n_seeds,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §12 DAYS 5-6 – SWEEP PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

class _ShortBiasedFaultGen(RandomFaultGenerator):
    """Re-generates events so that a given fraction of slowdowns are 'short'."""
    def __init__(self, seed: int, total_steps: int, short_frac: float):
        self._short_frac = short_frac
        super().__init__(seed=seed, total_steps=total_steps)

    def _generate(self) -> None:
        super()._generate()
        for ev in self._events:
            if ev.kind == "slowdown":
                ev.duration_steps = 10 if self._rng.random() < self._short_frac else 300


class _OnSlowPolicy(BasePolicy):
    """Plot-2 strategy: use `action` on a slow machine; re-split / readmit once it recovers."""
    def __init__(self, action: Action):
        self._action = action
        self.name = f"{action.name}OnSlow"

    def decide(self, obs: Observation) -> PolicyOutput:
        for mid, m in obs.machines.items():
            if is_dead(m):
                return (Action.EVICT, mid)
            if is_readmittable(m):
                return (Action.READMIT, mid)
            if is_slow(m) and not (self._action == Action.COMPENSATE and m.compensated):
                return (self._action, mid)
        if shares_imbalanced(obs):
            return Action.REBALANCE
        return Action.WAIT


def run_final_evaluations(
    output_dir: str = ".",
    n_seeds:    int = 5,
    total_steps: int = 1000,
) -> None:
    """
    Generates two PNG files:

    plot1_wait_vs_act.png
        X-axis : share of slowdown events that are "short" (0 % – 100 %).
        Y-axis : average Goodput/hr.
        Lines  : RebalanceOnly (act at once), TimeDelayed(10), TimeDelayed(60), Oracle.

    plot2_evict_vs_rebalance.png
        Goodput/hr for EVICT / REBALANCE / COMPENSATE strategies when facing a
        severe 300-step slowdown.
    """

    if not _MPL_OK:
        print("[run_final_evaluations] matplotlib not installed – skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── PLOT 1: Wait vs Act ────────────────────────────────────────────────────
    short_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]   # fraction of throttles that are short

    policy_configs = [
        ("Act at once",              lambda gen: RebalanceOnlyPolicy()),
        ("Wait 10 steps, then act",  lambda gen: TimeDelayedPolicy(wait_steps=10)),
        ("Wait 60 steps, then act",  lambda gen: TimeDelayedPolicy(wait_steps=60)),
        ("Oracle (knows the future)", lambda gen: OraclePolicy(gen)),
    ]

    results_p1: Dict[str, List[float]] = {name: [] for name, _ in policy_configs}
    for sf in short_fracs:
        for pol_name, pol_factory in policy_configs:
            vals: List[float] = []
            for seed in range(n_seeds):
                gen      = _ShortBiasedFaultGen(seed=seed, total_steps=total_steps, short_frac=sf)
                result   = run_episode(SimulatedCluster(seed=seed), pol_factory(gen),
                                       gen.as_schedule(), total_steps)
                vals.append(result.goodput_per_hr)
            results_p1[pol_name].append(_mean(vals))

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]
    for idx, (pol_name, _) in enumerate(policy_configs):
        ax1.plot(
            [int(sf * 100) for sf in short_fracs],
            results_p1[pol_name],
            marker=markers[idx % len(markers)],
            linewidth=2,
            markersize=7,
            label=pol_name,
        )
    ax1.set_xlabel("Short-lived slowdown share (%)", fontsize=12)
    ax1.set_ylabel("Goodput / hr (training steps per hour)", fontsize=12)
    ax1.set_title("Wait vs Act: Effect of Slowdown Duration Mix", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig1.tight_layout()
    p1_path = os.path.join(output_dir, "plot1_wait_vs_act.png")
    fig1.savefig(p1_path, dpi=150)
    plt.close(fig1)
    print(f"  [Plot 1] Saved -> {p1_path}")

    # ── PLOT 2: Evict vs Rebalance vs Compensate (300-step slowdown) ──────────
    def severe_schedule() -> FaultSchedule:
        """One 300-step 40% slowdown on win-gpu-0 at step 50."""
        return FaultSchedule(events=[FaultEvent(
            step=50, machine_id="win-gpu-0", kind="slowdown",
            slowdown_fraction=0.40, duration_steps=300,
        )])

    p2_policies = [
        ("WAIT",       lambda: DoNothingPolicy()),
        ("EVICT",      lambda: _OnSlowPolicy(Action.EVICT)),
        ("REBALANCE",  lambda: _OnSlowPolicy(Action.REBALANCE)),
        ("COMPENSATE", lambda: _OnSlowPolicy(Action.COMPENSATE)),
    ]

    bar_means: List[float] = []
    bar_labels: List[str]  = []
    for pol_label, make in p2_policies:
        # The scenario is deterministic, so one run per strategy is exact.
        result = run_episode(SimulatedCluster(seed=0), make(), severe_schedule(), total_steps)
        bar_means.append(result.goodput_per_hr)
        bar_labels.append(pol_label)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    x_pos  = list(range(len(bar_labels)))
    colors = ["#9e9e9e", "#e07b54", "#54a8e0", "#54e09e"]
    bars = ax2.bar(
        x_pos, bar_means,
        color=colors[:len(bar_labels)],
        width=0.5, edgecolor="black", linewidth=0.8,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels, fontsize=12)
    ax2.set_ylabel("Goodput / hr (training steps per hour)", fontsize=12)
    ax2.set_title("Compensate vs Evict: 300-step 40% Slowdown", fontsize=14)
    ax2.set_ylim(0, max(bar_means) * 1.08)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars, bar_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{mean:.0f}", ha="center", va="bottom", fontsize=10)
    fig2.tight_layout()
    p2_path = os.path.join(output_dir, "plot2_evict_vs_rebalance.png")
    fig2.savefig(p2_path, dpi=150)
    plt.close(fig2)
    print(f"  [Plot 2] Saved -> {p2_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# §13 DAY 7 – ENTRY POINT  (replaces the old §7)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    TOTAL_STEPS = 1000   # Simulated training steps per episode (Day 7 uses 1000)
    N_SEEDS     = 5
    SEED        = 42
    OUT_DIR     = "."    # PNGs saved to current working directory

    # ── Day 1-2 quick smoke-test (200 steps, fixed faults) ────────────────────
    print("=" * 72)
    print("  DAY 1-2 SMOKE TEST  (200 steps, fixed fault schedule)")
    print("=" * 72)

    smoke_schedule = FaultSchedule(events=[
        FaultEvent(step=50,  machine_id="win-gpu-0", kind="crash"),
        FaultEvent(step=100, machine_id="mac-m2",    kind="slowdown",
                   slowdown_fraction=0.40, duration_steps=30),
    ])

    smoke_policies = [
        DoNothingPolicy(),
        AlwaysRestartPolicy(),
        RebalanceOnlyPolicy(),
        OraclePolicy(smoke_schedule),
    ]
    smoke_results: List[EpisodeResult] = []
    for pol in smoke_policies:
        cluster = SimulatedCluster(seed=SEED)
        r = run_episode(cluster, pol, smoke_schedule, total_steps=200)
        smoke_results.append(r)

    col_w, label_w = 24, 22
    smoke_metrics = [
        ("Useful Steps",      lambda r: str(r.useful_steps)),
        ("Goodput / hr",      lambda r: f"{r.goodput_per_hr:.1f} s/hr"),
        ("Goodput / $",       lambda r: (f"{r.goodput_per_usd:.1f} s/$"
                                         if r.goodput_per_usd != float('inf')
                                         else "inf")),
        ("Wasted Time (s)",   lambda r: f"{r.total_wasted_time_s:.2f} s"),
        ("Work Lost (steps)", lambda r: str(r.work_lost)),
    ]
    header = f"{'Metric':<{label_w}}" + "".join(f"{r.policy_name:<{col_w}}" for r in smoke_results)
    print(header)
    print("-" * (label_w + col_w * len(smoke_results)))
    for mname, mfn in smoke_metrics:
        print(f"{mname:<{label_w}}" + "".join(f"{mfn(r):<{col_w}}" for r in smoke_results))
    print()

    # ── Day 7: Multi-seed evaluation table ────────────────────────────────────
    print("=" * 72)
    print("  DAY 7 RESULTS TABLE  (5 seeds x 1000 steps, random faults)")
    print("=" * 72)
    print("  numpy available   :", _NUMPY_OK)
    print("  matplotlib avail  :", _MPL_OK)
    print()

    runner = ExperimentRunner(n_seeds=N_SEEDS, total_steps=TOTAL_STEPS)

    day7_policies = [
        ("DoNothingPolicy",       lambda _: DoNothingPolicy()),
        ("AlwaysRestartPolicy",   lambda _: AlwaysRestartPolicy()),
        ("RebalanceOnlyPolicy",   lambda _: RebalanceOnlyPolicy()),
        ("TimeDelayed(60 steps)", lambda _: TimeDelayedPolicy(wait_steps=60)),
        ("OraclePolicy",          lambda gen: OraclePolicy(gen)),
    ]

    sweep_stats: List[SweepStats] = []
    for pol_name, pol_factory in day7_policies:
        print(f"  Running: {pol_name} ...", end="", flush=True)
        stats = runner.run(pol_factory, policy_name=pol_name)
        sweep_stats.append(stats)
        print(" done.")

    print()

    hdr_cols = ["Policy", "Goodput/hr (avg+/-std)", "Goodput/$ (avg+/-std)",
                "Wasted time (s)", "Work Lost (steps)"]
    cw = [30, 26, 26, 18, 20]

    def _row(*cells) -> str:
        return "".join(f"{str(c):<{cw[i]}}" for i, c in enumerate(cells))

    print("=" * sum(cw))
    print("  DAY 7 FINAL RESULTS  (mean +/- std over 5 seeds)")
    print("=" * sum(cw))
    print(_row(*hdr_cols))
    print("-" * sum(cw))

    for st in sweep_stats:
        ghr    = f"{st.mean_goodput_hr:.1f} +/- {st.std_goodput_hr:.1f}"
        gusd   = f"{st.mean_goodput_usd:.1f} +/- {st.std_goodput_usd:.1f}"
        wasted = f"{st.mean_wasted_time_s:.1f} s"
        lost   = f"{st.mean_work_lost:.1f} +/- {st.std_work_lost:.1f}"
        print(_row(st.policy_name, ghr, gusd, wasted, lost))

    print("=" * sum(cw))

    # ── Day 5-6: Generate PNG plots ───────────────────────────────────────────
    print()
    print("=" * 72)
    print("  GENERATING MATPLOTLIB PLOTS ...")
    print("=" * 72)
    run_final_evaluations(output_dir=OUT_DIR, n_seeds=N_SEEDS, total_steps=TOTAL_STEPS)

    print()
    print("  All done.")
    if _MPL_OK:
        print("  Plots saved to:", os.path.abspath(OUT_DIR))
