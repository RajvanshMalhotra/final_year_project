"""
simulator.py
============
Self-Healing Mixed-GPU Training – Part B: Simulator & Agent Loop
-----------------------------------------------------------------
Days 1-2 : Core data structures, SimulatedCluster, agent loop, simple baselines.
Days 3-7 : RandomFaultGenerator, OraclePolicy, TimeDelayedPolicy, ClusterEnv
            (RL wrapper), ExperimentRunner, multi-seed sweeps, matplotlib plots.

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
    REBALANCE   = auto()   # Re-distribute data shares among alive machines.
    COMPENSATE  = auto()   # Speed-compensate a slow node (e.g. reduce its share).
    EVICT       = auto()   # Remove a dead / too-slow node from the ring.
    READMIT     = auto()   # Re-add a previously evicted node that has recovered.
    RESTART     = auto()   # Roll back to last checkpoint and re-start training.
    CHECKPOINT  = auto()   # Persist current model weights to durable storage.


@dataclass
class MachineObs:
    """Per-machine slice of the observation the agent receives."""
    alive_signal_age: float   # Seconds since the last heartbeat was received.
    normal_speed:     float   # Baseline speed in "steps/second" under ideal conditions.
    measured_speed:   float   # Current measured speed (may be 0 if crashed).
    share:            float   # Fraction of the dataset this machine currently owns (0-1).
    code_version:     str     # Running code version (used to detect stale replicas).
    price:            float   # Hourly cost in USD for this machine.
    is_active:        bool = True   # True if alive, not evicted, and operational.

    @property
    def active(self) -> bool:
        return self.is_active


@dataclass
class Observation:
    """Global snapshot the agent receives at every step."""
    time:    float                     # Wall-clock simulation time (seconds).
    step:    int                       # Current global training step index.
    machines: Dict[str, MachineObs]   # machine_id -> MachineObs


@dataclass
class EpisodeResult:
    """Metrics produced at the end of one simulation episode."""
    policy_name:      str
    total_steps:      int
    useful_steps:     int      # Steps that produced good gradient updates.
    total_time_s:     float    # Total simulated time in seconds.
    total_cost_usd:   float    # Sum of (machine_price * simulated_hours) over all machines.
    goodput_per_hr:   float    # useful_steps / (total_time_s / 3600)
    goodput_per_usd:  float    # useful_steps / total_cost_usd
    total_wasted_time_s: float # Simulated seconds lost to restarts / stalls.
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
# §2  FAULT SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FaultEvent:
    """A single fault injected into the simulation."""
    step:       int    # Training step at which the fault is injected.
    machine_id: str    # Target machine.
    kind:       str    # "crash" | "slowdown"
    # For "slowdown" only:
    slowdown_fraction: float = 0.40   # Fraction by which speed is reduced (0.30-0.50).
    duration_steps:    int   = 20     # How many steps the slowdown lasts.


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
    measured_speed:  float         # Current speed (may be degraded).
    share:           float         # Dataset fraction owned by this machine.
    price_per_hr:    float         # Hourly USD cost.
    code_version:    str = "v1.0"
    alive:           bool = True
    evicted:         bool = False
    # Heartbeat tracking
    last_heartbeat:  float = 0.0   # Simulated time of last heartbeat.
    # Slowdown tracking
    slowdown_steps_left: int = 0
    original_speed:      float = 0.0

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
    The bottleneck is the slowest alive, non-evicted machine (All-reduce barrier).
    """

    HEARTBEAT_INTERVAL_S: float = 0.5   # Simulated seconds between heartbeats.
    CRASH_SIGNAL_AGE:     float = 3.0   # Signal age above which a machine is "dead".

    def __init__(self, seed: int = 42):
        rng = random.Random(seed)
        self._sim_time: float = 0.0
        self._step: int = 0

        # Initialise the three machines
        self._machines: Dict[str, _MachineState] = {
            "mac-m2": _MachineState(
                machine_id    = "mac-m2",
                label         = "Mac M2 (CPU/MPS)",
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
            m.last_heartbeat = rng.uniform(0, self.HEARTBEAT_INTERVAL_S)
            m.original_speed = m.normal_speed

        self._last_checkpoint_step: int = 0
        self._rebalance_shares()   # Distribute initial shares proportional to normal capacity

    # Public helpers

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def step(self) -> int:
        return self._step

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
        self._inject_fault(
            FaultEvent(
                step=self._step,
                machine_id=machine_id,
                kind="slowdown",
                slowdown_fraction=slowdown_fraction,
                duration_steps=duration_steps,
            )
        )

    def inject_crash(self, machine_id: str) -> None:
        """Helper to inject a crash fault directly."""
        self._inject_fault(
            FaultEvent(
                step=self._step,
                machine_id=machine_id,
                kind="crash",
            )
        )

    def apply_action(self, action: Action, target_machine_id: Optional[str] = None) -> None:
        """Apply a high-level Action enum to the cluster."""
        if action == Action.WAIT:
            pass
        elif action in (Action.REBALANCE, Action.COMPENSATE):
            self.apply_rebalance()
        elif action == Action.EVICT:
            if target_machine_id:
                self.apply_evict(target_machine_id)
            else:
                for mid, m in self._machines.items():
                    if not m.alive or m.measured_speed <= 0.01:
                        self.apply_evict(mid)
                        break
        elif action == Action.RESTART:
            self.apply_restart()
        elif action == Action.CHECKPOINT:
            self.apply_checkpoint()
        elif action == Action.READMIT:
            if target_machine_id:
                self.apply_readmit(target_machine_id)
            else:
                for mid, m in self._machines.items():
                    if m.evicted and m.alive and m.measured_speed > 0:
                        self.apply_readmit(mid)
                        break

    def get_observation(self) -> Observation:
        obs_machines: Dict[str, MachineObs] = {}
        for mid, m in self._machines.items():
            age = self._sim_time - m.last_heartbeat if m.alive else 9999.0
            obs_machines[mid] = MachineObs(
                alive_signal_age = age,
                normal_speed     = m.normal_speed,
                measured_speed   = m.measured_speed if not m.evicted else 0.0,
                share            = m.share,
                code_version     = m.code_version,
                price            = m.price_per_hr,
                is_active        = m.is_active,
            )
        return Observation(time=self._sim_time, step=self._step, machines=obs_machines)

    def step_forward(self, fault_schedule: FaultSchedule) -> Tuple[bool, float]:
        """
        Advance one training step.

        Returns
        -------
        (step_was_useful, simulated_seconds_elapsed)
            step_was_useful  - False when cluster is stalled (e.g. unevicted crash).
            simulated_seconds - the wall time consumed by this step.
        """
        # 1. Inject faults scheduled for this step
        for event in fault_schedule.events_at(self._step):
            self._inject_fault(event)

        # 2. Tick slowdown counters
        for m in self._machines.values():
            if m.slowdown_steps_left > 0:
                m.slowdown_steps_left -= 1
                if m.slowdown_steps_left == 0:
                    m.measured_speed = m.original_speed   # recover

        # 3. Determine effective cluster speed (All-reduce barrier = slowest partition)
        # Any active machine that has crashed (measured_speed <= 0 or not alive)
        # stalls All-Reduce unless evicted.
        crashed_un_evicted = [
            m for m in self._machines.values()
            if not m.evicted and (not m.alive or m.measured_speed <= 0)
        ]
        active_machines = [
            m for m in self._machines.values()
            if m.alive and not m.evicted and m.measured_speed > 0
        ]

        if crashed_un_evicted or not active_machines:
            # Complete stall - cluster is frozen waiting for dead machine
            self._sim_time += 1.0   # 1 second lost
            self._step += 1
            return False, 1.0

        # In data parallelism with synchronous All-Reduce, step duration is
        # bottlenecked by the slowest machine to finish its assigned share:
        durations = [
            m.share / m.measured_speed
            for m in active_machines
        ]
        step_duration = max(durations) if durations else 1.0

        # 4. Advance simulated time and emit heartbeats
        self._sim_time += step_duration
        for m in self._machines.values():
            if m.alive and not m.evicted:
                if (self._sim_time - m.last_heartbeat) >= self.HEARTBEAT_INTERVAL_S:
                    m.last_heartbeat = self._sim_time

        self._step += 1
        return True, step_duration

    # Action handlers (called by the agent loop)

    def apply_evict(self, machine_id: str) -> None:
        """Remove a machine from the active training ring."""
        if machine_id in self._machines:
            m = self._machines[machine_id]
            m.evicted = True
            m.alive   = False
            m.share   = 0.0
            self._rebalance_shares()

    def apply_readmit(self, machine_id: str) -> None:
        """Re-admit a previously evicted machine."""
        if machine_id in self._machines:
            m = self._machines[machine_id]
            if m.evicted:
                m.evicted = False
                m.alive   = True
                self._rebalance_shares()

    def apply_rebalance(self) -> None:
        """Redistribute dataset shares among active machines."""
        self._rebalance_shares()

    def apply_restart(self) -> int:
        """
        Roll back to last checkpoint.

        Returns the number of steps that were lost (rolled back).
        """
        steps_lost = self._step - self._last_checkpoint_step
        self._step = self._last_checkpoint_step
        # Restore all non-crashed machines
        for m in self._machines.values():
            if m.alive:
                m.evicted = False
                m.measured_speed  = m.normal_speed
                m.slowdown_steps_left = 0
        self._rebalance_shares()
        return steps_lost

    def apply_checkpoint(self) -> None:
        """Persist current step as a durable checkpoint."""
        self._last_checkpoint_step = self._step

    def total_cost_so_far(self) -> float:
        """Cumulative USD cost across all machines up to current sim_time."""
        hours = self._sim_time / 3600.0
        return sum(m.price_per_hr * hours for m in self._machines.values())

    # Private helpers

    def _inject_fault(self, event: FaultEvent) -> None:
        m = self._machines.get(event.machine_id)
        if m is None:
            return

        if event.kind == "crash":
            m.alive          = False         # process crashed
            m.measured_speed = 0.0           # no progress
            m.last_heartbeat = self._sim_time - self.CRASH_SIGNAL_AGE - 0.1
            print(f"  [FAULT] Step {self._step}: CRASH injected -> {event.machine_id}")

        elif event.kind == "slowdown":
            frac = max(0.30, min(0.50, event.slowdown_fraction))
            m.original_speed      = m.measured_speed if m.measured_speed > 0 else m.normal_speed
            m.measured_speed      = m.original_speed * (1.0 - frac)
            m.slowdown_steps_left = event.duration_steps
            print(f"  [FAULT] Step {self._step}: SLOWDOWN ({frac*100:.0f}%) injected "
                  f"-> {event.machine_id} for {event.duration_steps} steps")

    def _rebalance_shares(self) -> None:
        """Redistribute dataset shares among active machines proportional to their current speed."""
        active = [m for m in self._machines.values() if m.alive and not m.evicted and m.measured_speed > 0]
        if not active:
            for m in self._machines.values():
                m.share = 0.0
            return
        total_speed = sum(m.measured_speed for m in active)
        if total_speed > 0:
            for m in active:
                m.share = m.measured_speed / total_speed
        else:
            equal_share = 1.0 / len(active)
            for m in active:
                m.share = equal_share
        # Zero out evicted machines
        for m in self._machines.values():
            if m.evicted or not m.alive or m.measured_speed <= 0:
                m.share = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# §4  BASE POLICY
# ═══════════════════════════════════════════════════════════════════════════════

class BasePolicy:
    """Abstract base class for all policies."""

    name: str = "BasePolicy"

    def decide(self, observation: Observation) -> Action:
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

    def decide(self, observation: Observation) -> Action:
        return Action.WAIT


class AlwaysRestartPolicy(BasePolicy):
    """
    Baseline 2 - RESTART whenever any machine is dead or signal age is high.
    Represents the conservative "roll back to checkpoint" strategy.
    """
    name = "AlwaysRestartPolicy"

    DEAD_SPEED_THRESHOLD:  float = 0.01   # steps/s; below this -> considered dead.
    MAX_SIGNAL_AGE_S:      float = SimulatedCluster.CRASH_SIGNAL_AGE

    def decide(self, observation: Observation) -> Action:
        for mobs in observation.machines.values():
            is_dead = (mobs.measured_speed < self.DEAD_SPEED_THRESHOLD
                       or mobs.alive_signal_age > self.MAX_SIGNAL_AGE_S)
            if is_dead:
                return Action.RESTART
        return Action.WAIT


class RebalanceOnlyPolicy(BasePolicy):
    """
    Baseline 3 - EVICT dead machines; REBALANCE slow ones.
    Represents a moderate fault-tolerant strategy without rollbacks.
    """
    name = "RebalanceOnlyPolicy"

    DEAD_SPEED_THRESHOLD:  float = 0.01
    SLOW_RATIO_THRESHOLD:  float = 0.65   # < 65% of normal speed -> "slow"
    MAX_SIGNAL_AGE_S:      float = SimulatedCluster.CRASH_SIGNAL_AGE

    def decide(self, observation: Observation) -> Action:
        has_slow = False
        for mobs in observation.machines.values():
            is_dead  = (mobs.measured_speed < self.DEAD_SPEED_THRESHOLD
                        or mobs.alive_signal_age > self.MAX_SIGNAL_AGE_S)
            is_slow  = (mobs.normal_speed > 0
                        and mobs.measured_speed / mobs.normal_speed < self.SLOW_RATIO_THRESHOLD
                        and not is_dead)
            if is_dead:
                return Action.EVICT
            if is_slow:
                has_slow = True

        if has_slow:
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
    Run one full training episode.

    Parameters
    ----------
    cluster        : Fresh (or reset) SimulatedCluster instance.
    policy         : Any BasePolicy subclass.
    fault_schedule : Pre-built FaultSchedule (shared across all baseline runs
                     so comparisons are fair - each run gets a fresh cluster).
    total_steps    : Number of simulated training steps to execute.

    Returns
    -------
    EpisodeResult with all goodput / cost / latency metrics.
    """

    useful_steps:      int   = 0
    wasted_time_s:     float = 0.0
    work_lost:         int   = 0
    decision_times_ms: List[float] = []
    decision_counts:   Dict[str, int] = {a.name: 0 for a in Action}

    # Issue an initial checkpoint so RESTART has a valid rollback point.
    cluster.apply_checkpoint()

    steps_done = 0
    while steps_done < total_steps:

        # 1. Gather observation
        obs = cluster.get_observation()

        # 2. Policy decides (timed)
        t0 = time.perf_counter()
        action = policy.decide(obs)
        decision_ms = (time.perf_counter() - t0) * 1_000
        decision_times_ms.append(decision_ms)
        decision_counts[action.name] += 1

        # 3. Apply action
        if action == Action.WAIT:
            pass   # Intentional no-op

        elif action == Action.EVICT:
            # Evict the first dead machine found in the observation
            for mid, mobs in obs.machines.items():
                is_dead = (mobs.measured_speed < RebalanceOnlyPolicy.DEAD_SPEED_THRESHOLD
                           or mobs.alive_signal_age > SimulatedCluster.CRASH_SIGNAL_AGE)
                if is_dead:
                    cluster.apply_evict(mid)
                    break   # Evict one machine per step

        elif action == Action.REBALANCE:
            cluster.apply_rebalance()

        elif action == Action.COMPENSATE:
            # COMPENSATE: reduce the share of slow machines, pushing more to fast ones
            cluster.apply_rebalance()

        elif action == Action.READMIT:
            # Re-admit any evicted machines that might have recovered
            for mid, mobs in obs.machines.items():
                if mobs.alive_signal_age < SimulatedCluster.HEARTBEAT_INTERVAL_S * 2:
                    cluster.apply_readmit(mid)

        elif action == Action.RESTART:
            lost = cluster.apply_restart()
            work_lost    += lost
            wasted_time_s += lost / 5.0   # Rough: ~5 steps/sec average speed
            steps_done += 1
            continue   # Skip step_forward for this iteration

        elif action == Action.CHECKPOINT:
            cluster.apply_checkpoint()

        # 4. Advance the cluster one step
        step_useful, step_duration = cluster.step_forward(fault_schedule)

        if step_useful:
            useful_steps += 1
        else:
            wasted_time_s += step_duration

        steps_done += 1

    # 5. Compute final metrics
    total_time_s   = cluster.sim_time
    total_cost_usd = cluster.total_cost_so_far()

    goodput_per_hr  = (useful_steps / (total_time_s / 3600.0)) if total_time_s > 0 else 0.0
    goodput_per_usd = (useful_steps / total_cost_usd)           if total_cost_usd > 0 else float("inf")
    avg_decision_ms = (sum(decision_times_ms) / len(decision_times_ms)
                       if decision_times_ms else 0.0)

    return EpisodeResult(
        policy_name         = policy.name,
        total_steps         = steps_done,
        useful_steps        = useful_steps,
        total_time_s        = total_time_s,
        total_cost_usd      = total_cost_usd,
        goodput_per_hr      = goodput_per_hr,
        goodput_per_usd     = goodput_per_usd,
        total_wasted_time_s = wasted_time_s,
        work_lost           = work_lost,
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
    Cheating oracle – peeks at the future fault schedule.

    Decision rules
    --------------
    * Upcoming crash           → EVICT (proactive)
    * Upcoming long slowdown   (>= long_threshold steps) → COMPENSATE
    * Upcoming medium slowdown → REBALANCE
    * Upcoming short slowdown  (< short_threshold steps) → WAIT
    * No upcoming fault        → WAIT

    The look-ahead window is `horizon_steps` into the future.
    """
    name = "OraclePolicy"

    SHORT_THRESHOLD:  int = 30    # steps; below this → WAIT
    LONG_THRESHOLD:   int = 120   # steps; above this → COMPENSATE
    HORIZON:          int = 50    # how many future steps to inspect

    def __init__(self, fault_generator: any):
        self._gen = fault_generator

    @property
    def events(self) -> List[FaultEvent]:
        if hasattr(self._gen, "events"):
            return self._gen.events
        elif hasattr(self._gen, "_events"):
            return self._gen._events
        elif isinstance(self._gen, list):
            return self._gen
        return []

    def decide(self, observation: Observation) -> Action:
        current_step = observation.step
        horizon_end  = current_step + self.HORIZON

        # Scan upcoming events in the look-ahead window
        for event in self.events:
            if not (current_step <= event.step < horizon_end):
                continue
            if event.kind == "crash":
                return Action.EVICT
            if event.kind == "slowdown":
                if event.duration_steps >= self.LONG_THRESHOLD:
                    return Action.COMPENSATE
                if event.duration_steps >= self.SHORT_THRESHOLD:
                    return Action.REBALANCE
                # Short slowdown → ride it out
                return Action.WAIT

        return Action.WAIT


class TimeDelayedPolicy(BasePolicy):
    """
    Waits `wait_steps` (or `wait_time`) after first detecting a slowdown before acting.

    State machine
    -------------
    NORMAL  → sees a slow machine → record step + enter WATCHING
    WATCHING→ if slowdown persists past wait_steps → enter ACTING → REBALANCE
    ACTING  → keeps returning REBALANCE until the cluster looks normal again
             → then reset to NORMAL
    """

    DEAD_SPEED_THRESHOLD: float = 0.01
    SLOW_RATIO_THRESHOLD: float = 0.65

    def __init__(self, wait_steps: int = 60, wait_time: Optional[int] = None):
        if wait_time is not None:
            wait_steps = wait_time
        self._wait_steps     = wait_steps
        self._wait_time      = wait_steps
        self._detect_step: Optional[int] = None   # step at which slowdown first noticed
        self._detect_time: Optional[float] = None # sim_time at which slowdown first noticed
        self.name = f"TimeDelayedPolicy(wait={wait_steps})"

    def decide(self, observation: Observation) -> Action:
        # Detect whether any machine is currently slow or dead
        slow_or_dead = False
        dead         = False
        for mobs in observation.machines.values():
            if mobs.measured_speed < self.DEAD_SPEED_THRESHOLD or mobs.alive_signal_age > SimulatedCluster.CRASH_SIGNAL_AGE:
                dead = slow_or_dead = True
            elif (mobs.normal_speed > 0
                  and mobs.measured_speed / mobs.normal_speed < self.SLOW_RATIO_THRESHOLD):
                slow_or_dead = True

        if dead:
            self._detect_step = None
            self._detect_time = None
            return Action.EVICT

        if not slow_or_dead:
            self._detect_step = None   # cluster recovered
            self._detect_time = None
            return Action.WAIT

        # First detection of slowdown
        if self._detect_step is None:
            self._detect_step = observation.step
            self._detect_time = observation.time
            return Action.WAIT

        # Check whether wait window has elapsed (supports both steps and time)
        steps_waiting = observation.step - self._detect_step
        time_waiting = observation.time - (self._detect_time if self._detect_time is not None else 0.0)
        effective_waiting = max(steps_waiting, time_waiting)
        if effective_waiting >= self._wait_steps:
            return Action.REBALANCE

        return Action.WAIT   # still inside patience window


# ═══════════════════════════════════════════════════════════════════════════════
# §10 DAY 4 – RL ENVIRONMENT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class ClusterEnv:
    """
    OpenAI-Gym-style wrapper around SimulatedCluster.

    Reward
    ------
    r_t = goodput_per_usd(t) - goodput_per_usd(t-1)

    A positive reward means the action improved efficiency per dollar.
    A zero reward (WAIT on a healthy cluster) is neutral.
    A negative reward means the action degraded efficiency.

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
    ):
        if fault_generator is None:
            fault_generator = RandomFaultGenerator(seed=0, total_steps=total_steps)
        self._gen         = fault_generator
        self._total_steps = total_steps
        self._cluster:    Optional[SimulatedCluster] = None
        self._schedule:   Optional[FaultSchedule]   = None
        self._steps_done: int   = 0
        self._prev_gput_usd: float = 0.0
        self._useful_steps:  int   = 0

    def reset(self, seed: int = 0) -> Observation:
        """Reset the environment and return the first observation."""
        # Rebuild fault generator with new seed for variety
        new_gen          = RandomFaultGenerator(seed=seed, total_steps=self._total_steps)
        self._schedule   = new_gen.as_schedule()
        self._cluster    = SimulatedCluster(seed=seed)
        self._cluster.apply_checkpoint()
        self._steps_done = 0
        self._prev_gput_usd = 0.0
        self._useful_steps  = 0
        return self._cluster.get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Apply `action`, advance one step.

        Returns
        -------
        obs    : new Observation
        reward : delta goodput-per-dollar
        done   : True when episode is finished
        info   : diagnostic dict
        """
        assert self._cluster is not None, "Call reset() before step()."

        # Apply action (same logic as run_episode)
        work_lost = 0
        if action == Action.EVICT:
            obs_now = self._cluster.get_observation()
            for mid, mobs in obs_now.machines.items():
                if (mobs.measured_speed < 0.01
                        or mobs.alive_signal_age > SimulatedCluster.CRASH_SIGNAL_AGE):
                    self._cluster.apply_evict(mid)
                    break
        elif action in (Action.REBALANCE, Action.COMPENSATE):
            self._cluster.apply_rebalance()
        elif action == Action.RESTART:
            work_lost = self._cluster.apply_restart()
        elif action == Action.CHECKPOINT:
            self._cluster.apply_checkpoint()
        elif action == Action.READMIT:
            obs_now = self._cluster.get_observation()
            for mid, mobs in obs_now.machines.items():
                if mobs.alive_signal_age < SimulatedCluster.HEARTBEAT_INTERVAL_S * 2:
                    self._cluster.apply_readmit(mid)
        # WAIT → no-op

        # Advance simulation
        assert self._schedule is not None
        useful, _ = self._cluster.step_forward(self._schedule)
        if useful:
            self._useful_steps += 1
        self._steps_done += 1

        # Compute reward = delta goodput/$
        cost = self._cluster.total_cost_so_far()
        cur_gput_usd = (self._useful_steps / cost) if cost > 0 else 0.0
        reward = float(cur_gput_usd - self._prev_gput_usd)
        self._prev_gput_usd = cur_gput_usd

        done = bool(self._steps_done >= self._total_steps)
        obs  = self._cluster.get_observation()
        info = {"work_lost": int(work_lost), "useful_steps": int(self._useful_steps)}
        return obs, reward, done, info


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
        gput_hrs:    List[float] = []
        gput_usds:   List[float] = []
        wasted_ts:   List[float] = []
        works_lost:  List[float] = []

        for seed in range(self._n_seeds):
            gen      = RandomFaultGenerator(seed=seed, total_steps=self._total_steps)
            schedule = gen.as_schedule()
            cluster  = SimulatedCluster(seed=seed)
            policy   = policy_factory(gen)

            result = run_episode(cluster, policy, schedule, self._total_steps)

            gput_hrs.append(result.goodput_per_hr)
            # Replace inf with a large finite number so stats work
            gput_usds.append(
                result.goodput_per_usd if result.goodput_per_usd != float("inf")
                else result.useful_steps / 1e-9
            )
            wasted_ts.append(result.total_wasted_time_s)
            works_lost.append(float(result.work_lost))

        name = policy_name or ""

        if _NUMPY_OK:
            def _mean(lst): return float(np.mean(lst))
            def _std(lst):  return float(np.std(lst, ddof=1)) if len(lst) > 1 else 0.0
        else:
            def _mean(lst): return sum(lst) / len(lst)
            def _std(lst):
                if len(lst) <= 1: return 0.0
                m = sum(lst) / len(lst)
                return math.sqrt(sum((x - m)**2 for x in lst) / (len(lst) - 1))

        return SweepStats(
            policy_name        = name,
            mean_goodput_hr    = _mean(gput_hrs),
            std_goodput_hr     = _std(gput_hrs),
            mean_goodput_usd   = _mean(gput_usds),
            std_goodput_usd    = _std(gput_usds),
            mean_wasted_time_s = _mean(wasted_ts),
            std_wasted_time_s  = _std(wasted_ts),
            mean_work_lost     = _mean(works_lost),
            std_work_lost      = _std(works_lost),
            n_seeds            = self._n_seeds,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §12 DAYS 5-6 – SWEEP PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _std_fallback(lst: List[float]) -> float:
    """Pure-Python population std-dev (ddof=1)."""
    if len(lst) <= 1:
        return 0.0
    m = sum(lst) / len(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))


def run_final_evaluations(
    output_dir: str = ".",
    n_seeds:    int = 5,
    total_steps: int = 1000,
) -> None:
    """
    Generates two PNG files:

    plot1_wait_vs_act.png
        X-axis : share of slowdown events that are "short" (0 % – 100 %).
        Y-axis : average Goodput/$.
        Lines  : TimeDelayedPolicy(10), TimeDelayedPolicy(60), OraclePolicy.

    plot2_evict_vs_rebalance.png
        Compares Goodput/$ for EVICT / REBALANCE / COMPENSATE strategies when
        facing a severe 300-step slowdown.
    """

    if not _MPL_OK:
        print("[run_final_evaluations] matplotlib not installed – skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── PLOT 1: Wait vs Act ────────────────────────────────────────────────────
    short_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]   # fraction of throttles that are short

    policy_configs = [
        ("TimeDelayed(10s)",  lambda gen: TimeDelayedPolicy(wait_steps=10)),
        ("TimeDelayed(60s)",  lambda gen: TimeDelayedPolicy(wait_steps=60)),
        ("OraclePolicy",      lambda gen: OraclePolicy(gen)),
    ]

    # We approximate "short-fraction" by controlling the duration_steps
    # distribution inside a custom fault generator variant.
    class _ShortBiasedFaultGen(RandomFaultGenerator):
        """Re-generates events so that a given fraction of slowdowns are 'short'."""
        def __init__(self, seed: int, total_steps: int, short_frac: float):
            self._short_frac  = short_frac
            super().__init__(seed=seed, total_steps=total_steps)

        def _generate(self) -> None:
            # Call parent generator first to populate crash event.
            super()._generate()
            # Rewrite the slowdown durations according to the short_frac.
            for ev in self._events:
                if ev.kind == "slowdown":
                    if self._rng.random() < self._short_frac:
                        ev.duration_steps = 10    # short
                    else:
                        ev.duration_steps = 300   # long

    results_p1: Dict[str, List[float]] = {name: [] for name, _ in policy_configs}

    for sf in short_fracs:
        for pol_name, pol_factory in policy_configs:
            vals: List[float] = []
            for seed in range(n_seeds):
                gen      = _ShortBiasedFaultGen(seed=seed, total_steps=total_steps,
                                                short_frac=sf)
                schedule = gen.as_schedule()
                cluster  = SimulatedCluster(seed=seed)
                policy   = pol_factory(gen)
                result   = run_episode(cluster, policy, schedule, total_steps)
                gput = (result.goodput_per_usd
                        if result.goodput_per_usd != float("inf")
                        else result.useful_steps / 1e-9)
                vals.append(gput)
            mean_gput = sum(vals) / len(vals)
            results_p1[pol_name].append(mean_gput)

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
    ax1.set_ylabel("Goodput / $ (steps per dollar)", fontsize=12)
    ax1.set_title("Wait vs Act: Effect of Slowdown Duration Mix", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig1.tight_layout()
    p1_path = os.path.join(output_dir, "plot1_wait_vs_act.png")
    fig1.savefig(p1_path, dpi=150)
    plt.close(fig1)
    print(f"  [Plot 1] Saved -> {p1_path}")

    # ── PLOT 2: Evict vs Rebalance vs Compensate (300-step slowdown) ──────────
    class _Severe300SlowdownSchedule(FaultSchedule):
        """A fixed schedule with one 300-step 40% slowdown on win-gpu-0 at step 50."""
        def __init__(self):
            super().__init__(events=[
                FaultEvent(
                    step=50, machine_id="win-gpu-0",
                    kind="slowdown",
                    slowdown_fraction=0.40,
                    duration_steps=300,
                ),
            ])

    class _EvictOnSlowPolicy(BasePolicy):
        name = "EvictOnSlow"
        SLOW_RATIO = 0.65
        def decide(self, obs: Observation) -> Action:
            for mobs in obs.machines.values():
                if (mobs.normal_speed > 0
                        and mobs.measured_speed / mobs.normal_speed < self.SLOW_RATIO
                        and mobs.measured_speed > 0.01):
                    return Action.EVICT
            return Action.WAIT

    class _RebalanceOnSlowPolicy(BasePolicy):
        name = "RebalanceOnSlow"
        SLOW_RATIO = 0.65
        def decide(self, obs: Observation) -> Action:
            for mobs in obs.machines.values():
                if (mobs.normal_speed > 0
                        and mobs.measured_speed / mobs.normal_speed < self.SLOW_RATIO
                        and mobs.measured_speed > 0.01):
                    return Action.REBALANCE
            return Action.WAIT

    class _CompensateOnSlowPolicy(BasePolicy):
        name = "CompensateOnSlow"
        SLOW_RATIO = 0.65
        def decide(self, obs: Observation) -> Action:
            for mobs in obs.machines.values():
                if (mobs.normal_speed > 0
                        and mobs.measured_speed / mobs.normal_speed < self.SLOW_RATIO
                        and mobs.measured_speed > 0.01):
                    return Action.COMPENSATE
            return Action.WAIT

    p2_policies = [
        ("EVICT",       _EvictOnSlowPolicy()),
        ("REBALANCE",   _RebalanceOnSlowPolicy()),
        ("COMPENSATE",  _CompensateOnSlowPolicy()),
    ]

    bar_means: List[float] = []
    bar_stds:  List[float] = []
    bar_labels: List[str]  = []

    for pol_label, policy in p2_policies:
        vals = []
        for seed in range(n_seeds):
            sched   = _Severe300SlowdownSchedule()
            cluster = SimulatedCluster(seed=seed)
            result  = run_episode(cluster, policy, sched, total_steps)
            gput = (result.goodput_per_usd
                    if result.goodput_per_usd != float("inf")
                    else result.useful_steps / 1e-9)
            vals.append(gput)
        bar_means.append(sum(vals) / len(vals))
        bar_stds.append(_std_fallback(vals))
        bar_labels.append(pol_label)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    x_pos  = list(range(len(bar_labels)))
    colors = ["#e07b54", "#54a8e0", "#54e09e"]
    bars = ax2.bar(
        x_pos, bar_means, yerr=bar_stds,
        color=colors[:len(bar_labels)],
        capsize=6, width=0.5, edgecolor="black", linewidth=0.8,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels, fontsize=12)
    ax2.set_ylabel("Goodput / $ (steps per dollar)", fontsize=12)
    ax2.set_title("Compensate vs Evict: 300-step 40% Slowdown", fontsize=14)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars, bar_means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_stds) * 0.05,
            f"{mean:.0f}",
            ha="center", va="bottom", fontsize=10,
        )
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
        ("TimeDelayed(60s)",      lambda _: TimeDelayedPolicy(wait_steps=60)),
        ("OraclePolicy",          lambda gen: OraclePolicy(gen)),
    ]

    sweep_stats: List[SweepStats] = []
    for pol_name, pol_factory in day7_policies:
        print(f"  Running: {pol_name} ...", end="", flush=True)
        stats = runner.run(pol_factory, policy_name=pol_name)
        sweep_stats.append(stats)
        print(" done.")

    print()

    # Pretty-print Day 7 table
    hdr_cols = ["Policy", "Goodput/hr (avg+/-std)", "Goodput/$ (avg+/-std)",
                "Cost/episode ($)", "Work Lost (steps)"]
    cw = [30, 26, 26, 18, 20]

    def _row(*cells) -> str:
        return "".join(f"{str(c):<{cw[i]}}" for i, c in enumerate(cells))

    print("=" * sum(cw))
    print("  DAY 7 FINAL RESULTS  (mean +/- std over 5 seeds)")
    print("=" * sum(cw))
    print(_row(*hdr_cols))
    print("-" * sum(cw))

    for st in sweep_stats:
        ghr  = f"{st.mean_goodput_hr:.1f} +/- {st.std_goodput_hr:.1f}"
        gusd = f"{st.mean_goodput_usd:.1f} +/- {st.std_goodput_usd:.1f}"
        # Approximate cost: from total_time & machine prices – not stored in
        # SweepStats, so we note "see EpisodeResult" for detailed per-run cost.
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

