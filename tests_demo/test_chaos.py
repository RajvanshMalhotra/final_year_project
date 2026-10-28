import random

from demo.chaos import ChaosPlayer, FaultBook, FaultCmd, Workload
from demo.config import load_config
from scenarios import make_scenario


def test_workload_grows_in_jumps_and_never_passes_max():
    cfg = load_config("calibration.json")
    w = Workload(cfg, random.Random(0), auto=True, now=0.0)
    B, seen = cfg.b_start, []
    for t in range(0, 3000):
        nb = w.tick(float(t))
        if nb is not None:
            assert nb > B and nb - B <= cfg.b_step[1]
            B = nb
            seen.append(t)
    assert B == cfg.b_max
    assert seen and all(b - a >= cfg.b_every_s[0] - 1 for a, b in zip(seen, seen[1:]))


def test_workload_manual_mode_and_bump():
    cfg = load_config("calibration.json")
    w = Workload(cfg, random.Random(0), auto=False, now=0.0)
    assert all(w.tick(float(t)) is None for t in range(200))
    assert w.bump(+10) == cfg.b_start + 10
    assert w.bump(-1000) == 1
    assert w.bump(+1000) == cfg.b_max


def test_chaos_player_replays_the_scenario_at_the_same_steps():
    sc = make_scenario(3, "realistic", 1000)
    p = ChaosPlayer("realistic", 3, 1000)
    fired = []
    for step in range(1000):
        fired += [(step, c) for c in p.due(step)]
    assert len(fired) == len(sc.events)
    for (step, c), e in zip(fired, sorted(sc.events, key=lambda e: e.step)):
        assert step == e.step and c.node == e.machine_id
        assert c.kind == {"crash": "crash", "slowdown": "slow", "network": "net"}[e.kind]
        if c.kind != "crash":
            assert 0 < c.factor < 1 and c.duration_steps > 0


def test_faultbook_multiplies_overlapping_faults_and_expires():
    b = FaultBook()
    b.add(FaultCmd("slow", "mac-m2", factor=0.5, duration_steps=10), step=0)
    b.add(FaultCmd("slow", "mac-m2", factor=0.5, duration_steps=5), step=2)
    b.add(FaultCmd("net", "win-gpu-0", factor=0.2, duration_steps=5), step=0)
    assert b.factors(3)["mac-m2"] == (0.25, 1.0)
    assert b.factors(3)["win-gpu-0"] == (1.0, 0.2)
    assert b.factors(8)["mac-m2"] == (0.5, 1.0)
    assert b.factors(20) == {}


def test_faultbook_ramp_grows_slowdown_gradually():
    b = FaultBook()
    b.add(FaultCmd("slow", "mac-m2", factor=0.4, duration_steps=50, ramp_steps=10), step=0)
    early, late = b.factors(1)["mac-m2"][0], b.factors(20)["mac-m2"][0]
    assert 0.4 < early < 1.0 and abs(late - 0.4) < 1e-9


def test_faultbook_heal_clears_a_node():
    b = FaultBook()
    b.add(FaultCmd("slow", "mac-m2", factor=0.5, duration_steps=100), step=0)
    b.heal("mac-m2")
    assert b.factors(1) == {}
