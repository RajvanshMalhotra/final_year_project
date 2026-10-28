"""
Connects the agent (trained in the simulator, unchanged) to the real cluster:

  real measurements (ClusterState) --build_observation--> simulator.Observation
  Observation --AgentRunner.decide--> (Action, target)      # RL model, simple rule, or nothing
  (Action, target) --apply_action--> ClusterState changes + messages for the nodes

Nothing tells the agent what fault happened: it sees the same kinds of numbers as in the
simulator - speeds, heartbeat ages, shares, link speed - measured from the running nodes.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from simulator import (
    COMPENSATE_SPEEDUP, Action, MachineObs, Observation, ReactivePolicy, default_target,
    plan_action_full, proportional_shares,
)

from .config import NODE_IDS
from .state import ClusterState, Msg

PRICES = {"mac-m2": 0.0, "win-gpu-0": 0.5, "win-gpu-1": 0.5}   # same as the simulator
MODES = ("rl", "rule", "off")


def comm_constant(state: ClusterState) -> float:
    """
    The simulator's comm_s: all-reduce time = comm_s * 2(n-1)/n at full bandwidth.
    Measured: the median healthy link time of the nodes (their warm-up), converted for n = 3.
    """
    base = sorted(n.base_net for n in state.nodes.values() if n.base_net)
    if not base:
        return 0.0
    n = len(NODE_IDS)
    return base[len(base) // 2] * n / (2 * (n - 1))


def build_observation(state: ClusterState, now: float, tick: int) -> Observation:
    machines: Dict[str, MachineObs] = {}
    for nid in NODE_IDS:
        v = state.node_view(nid, now)
        age = v["heartbeat_age"]
        machines[nid] = MachineObs(
            alive_signal_age=age, normal_speed=v["normal_speed"], measured_speed=v["measured_speed"],
            share=v["share"], code_version="v1", price=PRICES[nid],
            # Same rule as the simulator: only heartbeats and speed say whether a node is alive.
            is_active=(not v["evicted"]) and age <= 3.0 and v["measured_speed"] > 0.01,
            evicted=v["evicted"], compensated=v["compensated"], bandwidth=v["bandwidth"],
            max_share=1.0,
        )
    return Observation(time=now, step=tick, machines=machines, train_step=float(state.step),
                       comm_s=comm_constant(state))


class AgentRunner:
    """mode 'rl' = trained MaskablePPO model, 'rule' = ReactivePolicy, 'off' = always WAIT."""

    def __init__(self, mode: str = "rule", model_path: Optional[str] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.policy = None
        if mode == "rl":
            from rl_agent import MODEL_PATH, load_policy
            self.policy = load_policy(model_path or MODEL_PATH)(None)
        elif mode == "rule":
            self.policy = ReactivePolicy()

    def decide(self, obs: Observation) -> Tuple[Action, Optional[str], dict]:
        if self.policy is None:
            return Action.WAIT, None, {}
        try:
            out = self.policy.decide(obs)
        except Exception as e:   # a broken policy must never take the cluster down
            return Action.WAIT, None, {"note": f"policy error: {e!r}"}
        action, target = (out if isinstance(out, tuple) else (out, None))
        if action == Action.WAIT:
            return action, None, {}
        if target is None:
            target = default_target(obs, action)
        overhead, _, _ = plan_action_full(obs, action, target)
        if overhead <= 0:
            return Action.WAIT, None, {"note": f"ignored {action.name} {target or ''}: changes nothing"}
        if action not in (Action.REBALANCE, Action.COMPENSATE, Action.EVICT, Action.READMIT):
            return Action.WAIT, None, {"note": f"ignored {action.name}: not supported in the demo"}
        return action, target, {}


def _resplit(obs: Observation, drop: Optional[str] = None, add: Optional[str] = None,
             boost: Optional[str] = None) -> Dict[str, float]:
    """Speed-proportional shares over the nodes that will be in the group (simulator rule)."""
    m = obs.machines
    in_ring = [mid for mid in NODE_IDS if not m[mid].evicted and mid != drop]
    if add:
        in_ring.append(add)
    speeds = {mid: m[mid].measured_speed for mid in in_ring
              if m[mid].alive_signal_age <= 3.0 and m[mid].measured_speed > 0.01}
    if boost in speeds:
        speeds[boost] *= COMPENSATE_SPEEDUP
    return proportional_shares(speeds)


def apply_action(action: Action, target: Optional[str], obs: Observation, state: ClusterState,
                 now: float) -> List[Msg]:
    """Carry out a (validated) action on the real cluster. Returns messages for the nodes."""
    if action == Action.REBALANCE:
        return state.rebalance(_resplit(obs), now)
    if action == Action.COMPENSATE and target:
        return state.compensate(target, now) + state.rebalance(_resplit(obs, boost=target), now)
    if action == Action.EVICT and target:
        state.shares = {k: v for k, v in _resplit(obs, drop=target).items() if v > 0}
        return state.evict(target, now)
    if action == Action.READMIT and target:
        state.shares = {k: v for k, v in _resplit(obs, add=target).items() if v > 0}
        return state.readmit(target, now)
    return []


def describe(action: Action, target: Optional[str]) -> str:
    words = {Action.WAIT: "wait", Action.REBALANCE: "re-split the work",
             Action.COMPENSATE: "switch to lighter mode", Action.EVICT: "remove",
             Action.READMIT: "bring back"}
    w = words.get(action, action.name.lower())
    return f"{w} {target}" if target else w


def finite(x: float) -> Optional[float]:
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None
