"""
stress_test.py
==============
Try to break the simulator and the agents.

1. fuzz()       random (often nonsensical) actions on chaos scenarios; after every
                tick, check invariants that must always hold. Any violation is a bug.
2. benchmark()  every policy x every difficulty x many seeds, scored as % of the
                Oracle's goodput. Shows where each policy breaks (worst episode,
                episodes below 80% of the Oracle).

    python stress_test.py            # fuzz + benchmark, prints the tables
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from scenarios import DIFFICULTIES, Scenario, make_scenario
from simulator import (
    STALL_TICK_S, Action, AlwaysRestartPolicy, BasePolicy, DoNothingPolicy, OraclePolicy,
    RebalanceOnlyPolicy, SimulatedCluster, TimeDelayedPolicy, plan_action, run_episode,
)

EPS = 1e-9


# =============================================================================
# 1. FUZZING WITH INVARIANTS
# =============================================================================

def check_invariants(c: SimulatedCluster, prev: Dict[str, float]) -> List[str]:
    """Rules that must hold after every tick. Returns a list of violations."""
    bad: List[str] = []
    if c.sim_time < prev["time"] - EPS:
        bad.append(f"time went backwards {prev['time']} -> {c.sim_time}")
    if c.total_cost_so_far() < prev["cost"] - EPS:
        bad.append("cost decreased")
    if not (0 - EPS <= c.train_step <= c.step + EPS):
        bad.append(f"progress {c.train_step} outside [0, ticks={c.step}]")
    ring_share = 0.0
    for mid, m in c.machines.items():
        if m.share < -EPS or m.share > m.max_share + EPS:
            bad.append(f"{mid} share {m.share} outside [0, cap {m.max_share}]")
        if m.evicted and m.share != 0.0:
            bad.append(f"{mid} evicted but share {m.share}")
        if not m.evicted:
            ring_share += m.share
        for name in ("measured_speed", "share", "bandwidth", "last_heartbeat"):
            if not math.isfinite(getattr(m, name)):
                bad.append(f"{mid}.{name} not finite")
        if m.alive and m.measured_speed <= 0:
            bad.append(f"{mid} alive with speed {m.measured_speed}")
    if ring_share > 1 + 1e-6:
        bad.append(f"shares in ring sum to {ring_share}")
    if any(m.is_active for m in c.machines.values()) and \
            all(m.alive for m in c.machines.values() if not m.evicted) and ring_share <= 0:
        bad.append("healthy ring with no work assigned")
    obs = c.get_observation()
    for mid, m in obs.machines.items():
        if not all(math.isfinite(v) for v in (m.measured_speed, m.alive_signal_age, m.share)):
            bad.append(f"observation of {mid} not finite")
    return bad


def fuzz(n_episodes: int = 30, difficulty: str = "chaos", ticks: int = 400,
         seed: int = 0) -> List[str]:
    """Random actions with random (sometimes invalid) targets. Returns violations."""
    rng = random.Random(seed)
    targets = [None, "mac-m2", "win-gpu-0", "win-gpu-1", "no-such-machine"]
    violations: List[str] = []
    for ep in range(n_episodes):
        sc = make_scenario(1000 + ep, difficulty, total_steps=ticks)
        c  = sc.make_cluster()
        c.fire_due_faults(sc.schedule)
        for t in range(ticks):
            prev   = {"time": c.sim_time, "cost": c.total_cost_so_far()}
            action = rng.choice(list(Action)) if rng.random() < 0.3 else Action.WAIT
            target = rng.choice(targets)
            truth  = c.get_observation().truth
            try:
                overhead, predicted = plan_action(truth, action, target)
                info = c.apply_action(action, target)
                useful, duration = c.step_forward(sc.schedule)
            except Exception as e:  # any exception is a bug
                violations.append(f"ep {ep} tick {t}: {action.name}({target}) raised {e!r}")
                break
            if target != "no-such-machine" and action != Action.CHECKPOINT:
                expect = predicted if math.isfinite(predicted) else STALL_TICK_S
                if abs(expect - duration) > 1e-9 or abs(overhead - info["overhead_s"]) > 1e-9:
                    violations.append(
                        f"ep {ep} tick {t}: plan said ({overhead:.3f}, {expect:.4f}) for "
                        f"{action.name}({target}), cluster did ({info['overhead_s']:.3f}, {duration:.4f})")
            violations += [f"ep {ep} tick {t}: {v}" for v in check_invariants(c, prev)]
    return violations


# =============================================================================
# 2. POLICY BENCHMARK ACROSS DIFFICULTY LEVELS
# =============================================================================

PolicyFactory = Callable[[Scenario], BasePolicy]


def default_policies() -> Dict[str, PolicyFactory]:
    from agent_brain import ClusterDetector, DecisionModel, HybridAgentPolicy, MPCFallback, NORMAL_SPEEDS
    return {
        "Do nothing":        lambda sc: DoNothingPolicy(),
        "Restart on crash":  lambda sc: AlwaysRestartPolicy(),
        "Act at once":       lambda sc: RebalanceOnlyPolicy(),
        "Wait 10, then act": lambda sc: TimeDelayedPolicy(10),
        "Wait 60, then act": lambda sc: TimeDelayedPolicy(60),
        "MPC only":          lambda sc: HybridAgentPolicy(DecisionModel(), MPCFallback(),
                                                          ClusterDetector(NORMAL_SPEEDS)),
    }


def benchmark(
    policies:     Dict[str, PolicyFactory],
    difficulties: Tuple[str, ...] = DIFFICULTIES,
    seeds:        range = range(20),
    total_steps:  int = 1000,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """result[difficulty][policy] = {mean_pct, worst_pct, below80, goodput, ms}"""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for d in difficulties:
        oracle = {}
        for s in seeds:
            sc = make_scenario(s, d, total_steps)
            oracle[s] = run_episode(sc.make_cluster(), OraclePolicy(sc.schedule),
                                    sc.schedule, total_steps).goodput_per_hr
        out[d] = {"Oracle": {"mean_pct": 100.0, "worst_pct": 100.0, "below80": 0,
                             "goodput": float(np.mean(list(oracle.values()))), "ms": 0.0}}
        for name, make in policies.items():
            pct, gput, ms = [], [], []
            for s in seeds:
                sc = make_scenario(s, d, total_steps)
                r  = run_episode(sc.make_cluster(), make(sc), sc.schedule, total_steps)
                pct.append(100.0 * r.goodput_per_hr / oracle[s] if oracle[s] > 0 else 0.0)
                gput.append(r.goodput_per_hr)
                ms.append(r.avg_decision_ms)
            out[d][name] = {"mean_pct": float(np.mean(pct)), "worst_pct": float(np.min(pct)),
                            "below80": int(np.sum(np.array(pct) < 80)),
                            "goodput": float(np.mean(gput)), "ms": float(np.mean(ms))}
    return out


def print_benchmark(res: Dict[str, Dict[str, Dict[str, float]]], n_seeds: int) -> None:
    for d, rows in res.items():
        print(f"\n  [{d}]  {n_seeds} scenarios, goodput as % of the Oracle")
        print(f"  {'Policy':<22} {'mean %':>8} {'worst %':>8} {'<80% eps':>9} {'goodput/hr':>11} {'ms':>7}")
        for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean_pct"]):
            print(f"  {name:<22} {r['mean_pct']:>8.1f} {r['worst_pct']:>8.1f} "
                  f"{r['below80']:>9d} {r['goodput']:>11.0f} {r['ms']:>7.3f}")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 72 + "\n  1. FUZZING (random actions on chaos scenarios)\n" + "=" * 72)
    v = fuzz()
    print(f"  {len(v)} invariant violations")
    for line in v[:20]:
        print("   ", line)

    print("\n" + "=" * 72 + "\n  2. BENCHMARK (every policy x every difficulty)\n" + "=" * 72)
    res = benchmark(default_policies())
    print_benchmark(res, 20)
    print(f"\n  done in {time.time() - t0:.0f}s")
