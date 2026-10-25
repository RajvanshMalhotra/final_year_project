"""
test_realism.py
===============
Tests for the realistic / hard simulator features:
noisy measurements, false-alarm spikes, all-reduce communication time, network
faults, memory caps (Mac 8 GB), transient crashes that come back, gradual
(ramping) slowdowns, and the scenario generator with difficulty levels.

    pytest test_realism.py -v
"""

from __future__ import annotations

import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import (
    ACTION_OVERHEAD_S,
    Action,
    FaultEvent,
    FaultSchedule,
    OraclePolicy,
    SimulatedCluster,
    plan_action,
    proportional_shares,
    run_episode,
)
from scenarios import DIFFICULTIES, make_scenario


def _run(c: SimulatedCluster, n: int, sched: FaultSchedule = None) -> None:
    for _ in range(n):
        c.step_forward(sched or FaultSchedule())


class TestNoise(unittest.TestCase):

    def test_observed_speed_is_noisy_but_unbiased(self):
        c = SimulatedCluster(seed=0, noise_std=0.05)
        readings = []
        for _ in range(2000):
            readings.append(c.get_observation().machines["win-gpu-0"].measured_speed)
            _run(c, 1)
        self.assertGreater(statistics.pstdev(readings), 0.05 * 3.5 * 0.8)
        self.assertAlmostEqual(statistics.mean(readings), 3.5, delta=0.02 * 3.5)

    def test_truth_is_noise_free(self):
        c = SimulatedCluster(seed=0, noise_std=0.2)
        self.assertEqual(c.get_observation().truth.machines["win-gpu-0"].measured_speed, 3.5)

    def test_no_noise_by_default(self):
        c = SimulatedCluster(seed=0)
        self.assertEqual(c.get_observation().machines["win-gpu-0"].measured_speed, 3.5)

    def test_spikes_are_false_alarms(self):
        c = SimulatedCluster(seed=0, spike_prob=1.0)
        self.assertAlmostEqual(c.get_observation().machines["win-gpu-0"].measured_speed, 3.5 * 0.5)
        self.assertEqual(c.machines["win-gpu-0"].measured_speed, 3.5)   # truth unchanged


class TestCommunication(unittest.TestCase):

    def test_allreduce_time_is_added_to_each_step(self):
        _, plain = SimulatedCluster(seed=0).step_forward(FaultSchedule())
        _, with_comm = SimulatedCluster(seed=0, comm_s=0.03).step_forward(FaultSchedule())
        self.assertAlmostEqual(with_comm - plain, 0.03 * 2 * 2 / 3)   # ring: 2(n-1)/n

    def test_network_fault_slows_the_whole_ring(self):
        sched = FaultSchedule([FaultEvent(step=0, machine_id="win-gpu-0", kind="network",
                                          slowdown_fraction=0.75, duration_steps=50)])
        c = SimulatedCluster(seed=0, comm_s=0.03)
        _, slow = c.step_forward(sched)
        self.assertAlmostEqual(slow - 1 / 8.2, 0.03 * 4 / 3 / 0.25, places=6)
        self.assertAlmostEqual(c.get_observation().machines["win-gpu-0"].bandwidth, 0.25)

    def test_evicting_the_network_slow_machine_helps(self):
        sched = FaultSchedule([FaultEvent(step=0, machine_id="win-gpu-0", kind="network",
                                          slowdown_fraction=0.9, duration_steps=50)])
        c = SimulatedCluster(seed=0, comm_s=0.03)
        _, before = c.step_forward(sched)
        c.apply_action(Action.EVICT, "win-gpu-0")
        _, after = c.step_forward(sched)
        self.assertLess(after, before)

    def test_single_worker_has_no_comm(self):
        c = SimulatedCluster(seed=0, comm_s=0.03)
        c.apply_action(Action.EVICT, "win-gpu-0")
        c.apply_action(Action.EVICT, "win-gpu-1")
        _, t = c.step_forward(FaultSchedule())
        self.assertAlmostEqual(t, 1 / 1.2)


class TestMemoryCaps(unittest.TestCase):

    def test_water_filling_respects_caps(self):
        shares = proportional_shares({"a": 1.0, "b": 1.0}, caps={"a": 0.2})
        self.assertAlmostEqual(shares["a"], 0.2)
        self.assertAlmostEqual(shares["b"], 0.8)

    def test_capped_machine_alone_does_part_of_the_batch(self):
        c = SimulatedCluster(seed=0, max_share={"mac-m2": 0.3})
        c.apply_action(Action.EVICT, "win-gpu-0")
        c.apply_action(Action.EVICT, "win-gpu-1")
        self.assertAlmostEqual(c.machines["mac-m2"].share, 0.3)
        p0 = c.train_step
        _run(c, 10)
        self.assertAlmostEqual(c.train_step - p0, 3.0)   # 30% of a batch per step


class TestTransientCrash(unittest.TestCase):

    def _sched(self, recover_after: int) -> FaultSchedule:
        return FaultSchedule([FaultEvent(step=5, machine_id="win-gpu-0", kind="crash",
                                         duration_steps=recover_after)])

    def test_machine_comes_back_and_rejoins(self):
        c = SimulatedCluster(seed=0, checkpoint_every=0)
        _run(c, 30, self._sched(10))
        m = c.machines["win-gpu-0"]
        self.assertTrue(m.alive and m.is_active)
        self.assertAlmostEqual(c.stall_time_s, 10.0)            # stalled while down
        self.assertGreaterEqual(c.overhead_time_s, ACTION_OVERHEAD_S[Action.READMIT])

    def test_evicted_machine_waits_for_readmit_after_recovery(self):
        c = SimulatedCluster(seed=0)
        sched = self._sched(10)
        _run(c, 6, sched)
        c.apply_action(Action.EVICT, "win-gpu-0")
        _run(c, 20, sched)
        self.assertFalse(c.machines["win-gpu-0"].is_active)
        c.apply_action(Action.READMIT, "win-gpu-0")
        self.assertTrue(c.machines["win-gpu-0"].is_active)

    def test_permanent_crash_by_default(self):
        c = SimulatedCluster(seed=0)
        _run(c, 100, FaultSchedule([FaultEvent(step=5, machine_id="win-gpu-0", kind="crash")]))
        self.assertFalse(c.machines["win-gpu-0"].alive)


class TestRampSlowdown(unittest.TestCase):

    def test_speed_declines_gradually(self):
        sched = FaultSchedule([FaultEvent(step=0, machine_id="win-gpu-0", kind="slowdown",
                                          slowdown_fraction=0.4, duration_steps=100,
                                          ramp_steps=10)])
        c = SimulatedCluster(seed=0)
        speeds = []
        for _ in range(12):
            _run(c, 1, sched)
            speeds.append(c.machines["win-gpu-0"].measured_speed)
        self.assertTrue(all(a >= b for a, b in zip(speeds, speeds[1:])))
        self.assertGreater(speeds[2], 3.5 * 0.6 + 1e-6)
        self.assertAlmostEqual(speeds[-1], 3.5 * 0.6)


class TestPlannerStillMatches(unittest.TestCase):

    def test_plan_matches_cluster_with_comm_caps_and_network(self):
        sched = FaultSchedule([
            FaultEvent(step=0, machine_id="win-gpu-1", kind="network",
                       slowdown_fraction=0.5, duration_steps=300),
            FaultEvent(step=0, machine_id="win-gpu-0", kind="slowdown",
                       slowdown_fraction=0.4, duration_steps=300),
        ])
        for action in (Action.WAIT, Action.REBALANCE, Action.COMPENSATE, Action.EVICT):
            for target in ("win-gpu-0", "win-gpu-1"):
                c = SimulatedCluster(seed=0, comm_s=0.03, max_share={"mac-m2": 0.3})
                c.fire_due_faults(sched)
                overhead, predicted = plan_action(c.get_observation().truth, action, target)
                t0 = c.sim_time
                c.apply_action(action, target)
                _, actual = c.step_forward(sched)
                self.assertAlmostEqual(predicted, actual, places=9, msg=f"{action.name} {target}")
                self.assertAlmostEqual(overhead + actual, c.sim_time - t0, places=9)


class TestOracleRealism(unittest.TestCase):

    def test_oracle_ignores_measurement_noise(self):
        sched = FaultSchedule([FaultEvent(step=100, machine_id="win-gpu-0", kind="slowdown",
                                          slowdown_fraction=0.4, duration_steps=300)])
        clean = run_episode(SimulatedCluster(0), OraclePolicy(sched), sched, 600)
        noisy = run_episode(SimulatedCluster(0, noise_std=0.2, spike_prob=0.05),
                            OraclePolicy(sched), sched, 600)
        self.assertAlmostEqual(clean.goodput_per_hr, noisy.goodput_per_hr, places=6)

    def _first_action_after_crash(self, down_for: int) -> Action:
        sched = FaultSchedule([FaultEvent(step=10, machine_id="win-gpu-0", kind="crash",
                                          duration_steps=down_for)])
        c, oracle = SimulatedCluster(0), OraclePolicy(sched)
        _run(c, 10, sched)
        out = oracle.decide(c.get_observation())
        return out[0] if isinstance(out, tuple) else out

    def test_oracle_waits_out_a_very_short_crash(self):
        self.assertEqual(self._first_action_after_crash(1), Action.WAIT)

    def test_oracle_evicts_a_long_crash(self):
        self.assertEqual(self._first_action_after_crash(200), Action.EVICT)

    def test_oracle_readmits_after_recovery(self):
        sched = FaultSchedule([FaultEvent(step=10, machine_id="win-gpu-0", kind="crash",
                                          duration_steps=50)])
        r = run_episode(SimulatedCluster(0), OraclePolicy(sched), sched, 300)
        self.assertEqual(r.decisions.get("READMIT", 0), 1, r.decisions)


class TestScenarios(unittest.TestCase):

    def test_difficulty_levels_exist(self):
        self.assertEqual(DIFFICULTIES, ("easy", "realistic", "hard", "chaos"))

    def test_scenarios_are_deterministic(self):
        a = make_scenario(7, "hard")
        b = make_scenario(7, "hard")
        self.assertEqual([(e.step, e.machine_id, e.kind, e.duration_steps) for e in a.schedule.events],
                         [(e.step, e.machine_id, e.kind, e.duration_steps) for e in b.schedule.events])

    def test_harder_levels_have_more_faults_and_noise(self):
        n = {d: sum(len(make_scenario(s, d).schedule.events) for s in range(20)) for d in DIFFICULTIES}
        self.assertLess(n["easy"], n["realistic"])
        self.assertLess(n["realistic"], n["hard"])
        self.assertLess(n["hard"], n["chaos"])
        self.assertEqual(make_scenario(0, "easy").cluster_kwargs.get("noise_std", 0.0), 0.0)
        self.assertGreater(make_scenario(0, "hard").cluster_kwargs["noise_std"], 0.0)

    def test_realistic_mix_contains_every_fault_type(self):
        kinds = set()
        for s in range(30):
            for e in make_scenario(s, "realistic").schedule.events:
                kinds.add(e.kind if e.kind != "crash" else ("transient" if e.duration_steps else "crash"))
                if e.kind == "slowdown" and e.ramp_steps:
                    kinds.add("ramp")
        self.assertTrue({"crash", "transient", "slowdown", "ramp", "network"} <= kinds, kinds)

    def test_oracle_runs_on_every_level(self):
        for d in DIFFICULTIES:
            sc = make_scenario(0, d)
            r = run_episode(sc.make_cluster(), OraclePolicy(sc.schedule), sc.schedule, 1000)
            self.assertGreater(r.useful_steps, 0, d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
