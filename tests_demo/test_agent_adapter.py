import os

from demo.agent_adapter import AgentRunner, apply_action, build_observation
from demo.config import load_config
from demo.state import ClusterState
from simulator import Action

IDS = ["mac-m2", "win-gpu-0", "win-gpu-1"]
SPS = {"mac-m2": 0.146, "win-gpu-0": 0.05, "win-gpu-1": 0.05}


def _running(now=0.0):
    s = ClusterState(load_config("calibration.json"))
    for nid in IDS:
        s.on_hello(nid, pid=1, has_state=False, step=-1, now=now)
    for nid, m in s.maybe_form_epoch(now):
        s.on_joined(nid, m["epoch"], "c", 0, now)
    for step in range(1, 17):
        for nid in IDS:
            s.on_heartbeat(nid, now + step * 0.1)
            s.on_report(nid, {"epoch": s.epoch, "step": step, "loss": 4.0, "count": 2,
                              "sec_per_seq": SPS[nid], "net_s": 0.05, "comm_s": 0.06,
                              "wall_s": 0.4, "B": 6}, now + step * 0.1)
    return s


def test_observation_has_three_machines_and_dead_node_age():
    s = _running()
    s.on_disconnect("win-gpu-0", 1.7)
    obs = build_observation(s, now=6.0, tick=1)
    assert set(obs.machines) == set(IDS)
    assert obs.machines["win-gpu-0"].alive_signal_age > 3
    assert not obs.machines["win-gpu-0"].is_active
    assert obs.machines["mac-m2"].normal_speed > 0
    assert abs(sum(m.share for m in obs.machines.values()) - 1.0) < 1e-6
    assert 0 < obs.comm_s < 1


def test_rule_mode_evicts_dead_node():
    s = _running()
    s.on_disconnect("win-gpu-0", 1.7)
    for nid in ("mac-m2", "win-gpu-1"):
        s.on_heartbeat(nid, 6.0)
    action, target, _ = AgentRunner("rule").decide(build_observation(s, 6.0, 1))
    assert (action, target) == (Action.EVICT, "win-gpu-0")


def test_off_mode_always_waits():
    s = _running()
    s.on_disconnect("win-gpu-0", 1.7)
    action, target, _ = AgentRunner("off").decide(build_observation(s, 6.0, 1))
    assert action == Action.WAIT and target is None


def test_adapter_invalid_action_is_wait():
    class Bad:
        def decide(self, obs):
            return (Action.READMIT, "mac-m2")       # mac-m2 is not evicted

    class Boom:
        def decide(self, obs):
            raise RuntimeError("model broke")

    s = _running()
    obs = build_observation(s, 1.7, 1)
    for policy in (Bad(), Boom()):
        runner = AgentRunner("off")
        runner.policy = policy
        action, target, info = runner.decide(obs)
        assert action == Action.WAIT and info.get("note")


def test_apply_evict_and_rebalance_produce_operations():
    s = _running()
    s.on_disconnect("win-gpu-0", 1.7)
    for nid in ("mac-m2", "win-gpu-1"):
        s.on_heartbeat(nid, 6.0)
    obs = build_observation(s, 6.0, 1)
    apply_action(Action.EVICT, "win-gpu-0", obs, s, now=6.0)
    assert s.nodes["win-gpu-0"].evicted
    assert set(k for k, v in s.shares.items() if v > 0) == {"mac-m2", "win-gpu-1"}

    s2 = _running()
    s2.shares = {"mac-m2": 1 / 3, "win-gpu-0": 1 / 3, "win-gpu-1": 1 / 3}
    obs2 = build_observation(s2, 1.7, 1)
    msgs = apply_action(Action.REBALANCE, None, obs2, s2, now=1.7)
    assert msgs and msgs[0][1]["type"] == "config"
    assert s2.shares["mac-m2"] < 0.2


def test_rl_mode_loads_model_and_acts():
    path = os.path.join("models", "v2_seed1.zip")
    runner = AgentRunner("rl", model_path=path)
    s = _running()
    action, target, info = runner.decide(build_observation(s, 1.7, 1))
    assert isinstance(action, Action)


def test_rebalance_that_barely_changes_shares_becomes_wait():
    class AlwaysRebalance:
        def decide(self, obs):
            return Action.REBALANCE

    s = _running()
    speeds = {nid: 1 / SPS[nid] for nid in IDS}
    total = sum(speeds.values())
    s.shares = {nid: v / total * (1.01 if nid == "mac-m2" else 1.0) for nid, v in speeds.items()}
    runner = AgentRunner("off")
    runner.policy = AlwaysRebalance()
    action, _, info = runner.decide(build_observation(s, 1.7, 1))
    assert action == Action.WAIT and "2%" in info["note"]
    s.shares = {nid: 1 / 3 for nid in IDS}
    action, _, _ = runner.decide(build_observation(s, 1.7, 1))
    assert action == Action.REBALANCE
