"""
test_simulator.py
=================
Comprehensive Test Suite for the Self-Healing Training Simulator.

Verifies:
  1. Cluster Physics & Actions (REBALANCE, EVICT, step duration, workload shares).
  2. Fault Generation (Determinism across seeds, exact fault application).
  3. Advanced Policies (TimeDelayedPolicy patience window, OraclePolicy lookahead).
  4. RL Environment Wrapper (Gym interface compliance, reward logic).

Can be executed standalone:
    python test_simulator.py
or with pytest:
    pytest test_simulator.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Dict, List

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import (
    Action,
    BasePolicy,
    ClusterEnv,
    DoNothingPolicy,
    FaultEvent,
    FaultSchedule,
    MachineObs,
    Observation,
    OraclePolicy,
    RandomFaultGenerator,
    RebalanceOnlyPolicy,
    SimulatedCluster,
    TimeDelayedPolicy,
)


class TestClusterPhysics(unittest.TestCase):
    """
    §1. Test Cluster Physics & Actions
    ---------------------------------
    Verifies that workload shares, All-Reduce step barriers, and recovery actions
    behave in strict accordance with distributed machine learning physics.
    """

    def setUp(self) -> None:
        self.cluster = SimulatedCluster(seed=42)

    def test_rebalance_decreases_slow_node_share_and_speeds_up_step(self) -> None:
        """
        Test Rebalance:
        Initialize a cluster. Inject a 50% slowdown on Machine 2.
        Apply REBALANCE action.
        Assert:
          - Machine 2's share decreases.
          - Other machines' shares increase.
          - Resulting step time is faster than doing nothing (WAIT).
        """
        # Machine 2 is the second machine ("win-gpu-0")
        m2 = self.cluster.get_machine(2)
        m2_id = m2.machine_id
        initial_m2_share = m2.share

        # Record initial shares of all other machines
        initial_other_shares = {
            mid: m.share
            for mid, m in self.cluster.machines.items()
            if mid != m2_id
        }

        # Inject a 50% slowdown on Machine 2
        self.cluster.inject_slowdown(m2_id, slowdown_fraction=0.50, duration_steps=10)

        # Baseline: Compute what step time would be if doing nothing (WAIT)
        active_before = [m for m in self.cluster.machines.values() if m.is_active]
        durations_wait = [m.share / m.measured_speed for m in active_before]
        time_doing_nothing = max(durations_wait)

        # Apply REBALANCE action
        self.cluster.apply_action(Action.REBALANCE)

        # 1. Assert Machine 2's share decreases
        self.assertLess(
            m2.share,
            initial_m2_share,
            f"Machine 2 share should decrease from {initial_m2_share:.4f}, but got {m2.share:.4f}",
        )

        # 2. Assert other machines' shares increase
        for mid, m in self.cluster.machines.items():
            if mid != m2_id and m.is_active:
                self.assertGreater(
                    m.share,
                    initial_other_shares[mid],
                    f"Machine {mid} share should increase from {initial_other_shares[mid]:.4f}, but got {m.share:.4f}",
                )

        # 3. Assert resulting step time is faster than doing nothing
        active_after = [m for m in self.cluster.machines.values() if m.is_active]
        durations_rebalance = [m.share / m.measured_speed for m in active_after]
        time_after_rebalance = max(durations_rebalance)

        self.assertLess(
            time_after_rebalance,
            time_doing_nothing,
            f"Rebalanced step time ({time_after_rebalance:.4f}s) should be faster than doing nothing ({time_doing_nothing:.4f}s)",
        )

        # Also verify via step_forward
        step_useful, step_duration = self.cluster.step_forward(FaultSchedule([]))
        self.assertTrue(step_useful, "Step after rebalance must be useful.")
        self.assertLess(
            step_duration,
            time_doing_nothing,
            f"Executed step duration ({step_duration:.4f}s) should be faster than doing nothing ({time_doing_nothing:.4f}s)",
        )

    def test_evict_marks_machine_inactive_and_steps_with_remaining(self) -> None:
        """
        Test Evict:
        Initialize a cluster. Inject a crash on Machine 3.
        Apply EVICT action.
        Assert:
          - Machine 3 is marked inactive.
          - Machine 3's share goes to 0.
          - Cluster successfully takes a step using only the remaining two machines.
        """
        # Machine 3 is the third machine ("win-gpu-1")
        m3 = self.cluster.get_machine(3)
        m3_id = m3.machine_id

        # Inject crash on Machine 3
        self.cluster.inject_crash(m3_id)

        # Prior to eviction, stepping forward stalls because All-Reduce hangs on dead node
        stalled_useful, _ = self.cluster.step_forward(FaultSchedule([]))
        self.assertFalse(stalled_useful, "Cluster must stall while dead machine is un-evicted.")

        # Apply EVICT action
        self.cluster.apply_action(Action.EVICT, target_machine_id=m3_id)

        # 1. Assert Machine 3 is marked inactive
        self.assertFalse(m3.is_active, "Evicted machine must be marked inactive (is_active == False).")
        self.assertTrue(m3.evicted, "Machine 3 must be marked as evicted.")

        # Also verify in Observation snapshot
        obs = self.cluster.get_observation()
        self.assertFalse(obs.machines[m3_id].is_active, "Observation must show evicted machine as inactive.")

        # 2. Assert Machine 3's share goes to 0
        self.assertEqual(m3.share, 0.0, f"Evicted machine share must be 0.0, got {m3.share}")
        self.assertEqual(obs.machines[m3_id].share, 0.0)

        # 3. Assert cluster successfully takes a step using only remaining 2 machines
        step_useful, step_duration = self.cluster.step_forward(FaultSchedule([]))
        self.assertTrue(step_useful, "Cluster should make useful progress with remaining 2 machines.")
        self.assertGreater(step_duration, 0.0, "Step duration must be strictly positive.")

        active_machines = [m for m in self.cluster.machines.values() if m.is_active]
        self.assertEqual(len(active_machines), 2, "Only 2 machines should be active.")
        self.assertNotIn(m3, active_machines, "Machine 3 must not be in active set.")

        # Shares of active machines must sum to 1.0
        total_active_share = sum(m.share for m in active_machines)
        self.assertAlmostEqual(total_active_share, 1.0, places=5)


class TestFaultGeneration(unittest.TestCase):
    """
    §2. Test Fault Generation
    -------------------------
    Verifies reproducibility across seeds and precise application of faults.
    """

    def test_determinism_same_seed_produces_identical_sequence(self) -> None:
        """
        Test Determinism:
        Initialize two RandomFaultGenerator instances with the same seed.
        Assert they produce the exact same sequence of crashes and slowdowns.
        """
        seed = 42
        gen1 = RandomFaultGenerator(seed=seed, total_steps=1000)
        gen2 = RandomFaultGenerator(seed=seed, total_steps=1000)

        self.assertEqual(
            len(gen1.events),
            len(gen2.events),
            "Generators with same seed must produce the same number of events.",
        )

        for idx, (e1, e2) in enumerate(zip(gen1.events, gen2.events)):
            self.assertEqual(e1.step, e2.step, f"Event {idx} step mismatch.")
            self.assertEqual(e1.machine_id, e2.machine_id, f"Event {idx} machine mismatch.")
            self.assertEqual(e1.kind, e2.kind, f"Event {idx} kind mismatch.")
            self.assertEqual(e1.duration_steps, e2.duration_steps, f"Event {idx} duration mismatch.")
            self.assertAlmostEqual(e1.slowdown_fraction, e2.slowdown_fraction, places=5)

        # Verify that different seeds produce different sequences
        gen_diff = RandomFaultGenerator(seed=12345, total_steps=1000)
        events1 = [(e.step, e.machine_id, e.kind) for e in gen1.events]
        events_diff = [(e.step, e.machine_id, e.kind) for e in gen_diff.events]
        self.assertNotEqual(events1, events_diff, "Different seeds should produce different schedules.")

    def test_fault_application_slowdown_drops_speed_accurately(self) -> None:
        """
        Test Fault Application:
        Step cluster forward until a scheduled slowdown hits.
        Assert that the cluster's measured_speed for the affected machine
        drops exactly according to the schedule.
        """
        cluster = SimulatedCluster(seed=42)
        target_machine = "win-gpu-0"
        target_step = 4
        slowdown_fraction = 0.40
        duration_steps = 15

        schedule = FaultSchedule([
            FaultEvent(
                step=target_step,
                machine_id=target_machine,
                kind="slowdown",
                slowdown_fraction=slowdown_fraction,
                duration_steps=duration_steps,
            )
        ])

        normal_speed = cluster.machines[target_machine].normal_speed
        expected_slow_speed = normal_speed * (1.0 - slowdown_fraction)

        # Step forward until step 3 (before the fault hits)
        for s in range(target_step):
            self.assertAlmostEqual(
                cluster.machines[target_machine].measured_speed,
                normal_speed,
                places=4,
                msg=f"Machine speed before step {target_step} should be normal.",
            )
            cluster.step_forward(schedule)

        # At target_step (step 4), the fault is injected during step_forward
        self.assertEqual(cluster.step, target_step)
        cluster.step_forward(schedule)

        # Assert measured_speed dropped exactly according to schedule
        actual_speed = cluster.machines[target_machine].measured_speed
        self.assertAlmostEqual(
            actual_speed,
            expected_slow_speed,
            places=4,
            msg=f"Expected speed {expected_slow_speed}, got {actual_speed}",
        )

        # Also verify in Observation snapshot
        obs = cluster.get_observation()
        self.assertAlmostEqual(
            obs.machines[target_machine].measured_speed,
            expected_slow_speed,
            places=4,
        )


class TestAdvancedPolicies(unittest.TestCase):
    """
    §3. Test Advanced Policies
    --------------------------
    Verifies TimeDelayedPolicy patience window and OraclePolicy lookahead logic.
    """

    def _create_observation(
        self,
        step: int,
        sim_time: float,
        slow_machine_id: str = "win-gpu-0",
        slowdown_ratio: float = 0.50,
    ) -> Observation:
        """Helper to create realistic Observation with a degraded machine."""
        machines = {
            "mac-m2": MachineObs(
                alive_signal_age=0.1,
                normal_speed=1.2,
                measured_speed=1.2,
                share=0.15,
                code_version="v1.0",
                price=0.00,
                is_active=True,
            ),
            "win-gpu-0": MachineObs(
                alive_signal_age=0.1,
                normal_speed=3.5,
                measured_speed=3.5,
                share=0.425,
                code_version="v1.0",
                price=0.50,
                is_active=True,
            ),
            "win-gpu-1": MachineObs(
                alive_signal_age=0.1,
                normal_speed=3.5,
                measured_speed=3.5,
                share=0.425,
                code_version="v1.0",
                price=0.50,
                is_active=True,
            ),
        }
        # Apply degradation to target machine
        target = machines[slow_machine_id]
        target.measured_speed = target.normal_speed * slowdown_ratio
        return Observation(time=sim_time, step=step, machines=machines)

    def test_time_delayed_policy_waits_and_then_acts(self) -> None:
        """
        Test TimeDelayedPolicy:
        Feed observations of a slow machine to a TimeDelayedPolicy(wait_time=10).
        Assert it returns:
          - WAIT at time=0 (step=0)
          - WAIT at time=5 (step=5)
          - REBALANCE at time=11 (step=11)
        """
        policy = TimeDelayedPolicy(wait_time=10)

        # Feed observation at time=0 / step=0 -> First detection, enters WATCHING
        obs_0 = self._create_observation(step=0, sim_time=0.0)
        action_0 = policy.decide(obs_0)
        self.assertEqual(action_0, Action.WAIT, "Must return WAIT at time=0 (first detection).")

        # Feed observation at time=5 / step=5 -> Inside patience window (5 < 10)
        obs_5 = self._create_observation(step=5, sim_time=5.0)
        action_5 = policy.decide(obs_5)
        self.assertEqual(action_5, Action.WAIT, "Must return WAIT at time=5 (inside patience window).")

        # Feed observation at time=11 / step=11 -> Patience window expired (11 >= 10)
        obs_11 = self._create_observation(step=11, sim_time=11.0)
        action_11 = policy.decide(obs_11)
        self.assertEqual(
            action_11,
            Action.REBALANCE,
            "Must return REBALANCE at time=11 (patience window exceeded).",
        )

    def test_oracle_policy_short_vs_long_slowdowns(self) -> None:
        """
        Test OraclePolicy:
        Feed an observation to OraclePolicy where a short 5-second slowdown is scheduled.
        Assert it returns WAIT.
        Feed a different schedule with a 300-second slowdown.
        Assert it returns COMPENSATE or REBALANCE.
        """
        healthy_obs = Observation(
            time=0.0,
            step=0,
            machines={
                "mac-m2": MachineObs(0.1, 1.2, 1.2, 0.15, "v1.0", 0.0, True),
                "win-gpu-0": MachineObs(0.1, 3.5, 3.5, 0.425, "v1.0", 0.5, True),
                "win-gpu-1": MachineObs(0.1, 3.5, 3.5, 0.425, "v1.0", 0.5, True),
            },
        )

        # 1. Short 5-second / 5-step slowdown scheduled at step 2
        short_schedule = FaultSchedule([
            FaultEvent(
                step=2,
                machine_id="win-gpu-0",
                kind="slowdown",
                slowdown_fraction=0.40,
                duration_steps=5,
            )
        ])
        oracle_short = OraclePolicy(short_schedule)
        action_short = oracle_short.decide(healthy_obs)
        self.assertEqual(
            action_short,
            Action.WAIT,
            f"OraclePolicy should ride out short slowdowns (< 30 steps) with WAIT, got {action_short}",
        )

        # 2. Long 300-second / 300-step slowdown scheduled at step 2
        long_schedule = FaultSchedule([
            FaultEvent(
                step=2,
                machine_id="win-gpu-0",
                kind="slowdown",
                slowdown_fraction=0.40,
                duration_steps=300,
            )
        ])
        oracle_long = OraclePolicy(long_schedule)
        action_long = oracle_long.decide(healthy_obs)
        self.assertIn(
            action_long,
            [Action.COMPENSATE, Action.REBALANCE],
            f"OraclePolicy should act on long slowdowns with COMPENSATE or REBALANCE, got {action_long}",
        )

    def test_oracle_policy_proactively_evicts_on_crash(self) -> None:
        """Assert OraclePolicy returns EVICT when a crash is scheduled ahead."""
        crash_schedule = FaultSchedule([
            FaultEvent(step=5, machine_id="win-gpu-1", kind="crash")
        ])
        oracle = OraclePolicy(crash_schedule)
        obs = Observation(time=0.0, step=0, machines={})
        action = oracle.decide(obs)
        self.assertEqual(action, Action.EVICT, "OraclePolicy must return EVICT on upcoming crash.")


class TestClusterEnvRLWrapper(unittest.TestCase):
    """
    §4. Test RL Environment Wrapper
    -------------------------------
    Verifies OpenAI-Gym-style interface compliance and reward alignment.
    """

    def test_gym_interface_reset_and_step(self) -> None:
        """
        Test Gym Interface:
        Instantiate ClusterEnv. Call reset(seed=42) and assert it returns a valid initial Observation.
        Call step(Action.WAIT) and assert it returns a tuple of (Observation, float_reward, bool_done, dict_info).
        """
        env = ClusterEnv(total_steps=100)

        # 1. Reset
        obs = env.reset(seed=42)
        self.assertIsInstance(obs, Observation, "reset() must return an Observation instance.")
        self.assertIsInstance(obs.machines, dict, "Observation.machines must be a dict.")
        self.assertEqual(len(obs.machines), 3, "Initial observation must contain all 3 machines.")
        self.assertEqual(obs.step, 0, "Initial observation step must be 0.")

        # 2. Step with Action.WAIT
        step_result = env.step(Action.WAIT)
        self.assertIsInstance(step_result, tuple, "step() must return a 4-tuple.")
        self.assertEqual(len(step_result), 4, "step() must return exactly (obs, reward, done, info).")

        next_obs, reward, done, info = step_result

        # Assert types
        self.assertIsInstance(next_obs, Observation, "obs must be an Observation.")
        self.assertIsInstance(reward, float, "reward must be a float.")
        self.assertIsInstance(done, bool, "done must be a boolean.")
        self.assertIsInstance(info, dict, "info must be a dictionary.")

        # Assert info content
        self.assertIn("work_lost", info, "info dict must contain 'work_lost'.")
        self.assertIn("useful_steps", info, "info dict must contain 'useful_steps'.")

    def test_reward_logic_rebalance_reward_higher_than_wait_during_slowdown(self) -> None:
        """
        Test Reward Logic:
        Assert that the reward (delta in Goodput/$) is higher when calling
        REBALANCE during a severe slowdown compared to calling WAIT.
        """
        severe_fault = [
            FaultEvent(
                step=0,
                machine_id="win-gpu-0",
                kind="slowdown",
                slowdown_fraction=0.50,
                duration_steps=20,
            )
        ]

        # Setup two identical environments
        env_wait = ClusterEnv(total_steps=100)
        env_wait.reset(seed=42)
        env_wait._schedule = FaultSchedule(severe_fault)

        env_reb = ClusterEnv(total_steps=100)
        env_reb.reset(seed=42)
        env_reb._schedule = FaultSchedule(severe_fault)

        # Step 0: The severe slowdown hits at step 0
        env_wait.step(Action.WAIT)
        env_reb.step(Action.WAIT)

        # Step 1: Both clusters are now actively suffering the severe slowdown.
        # Scenario A: Agent continues to WAIT (no adaptation)
        _, reward_wait, _, _ = env_wait.step(Action.WAIT)

        # Scenario B: Agent takes Action.REBALANCE (adapts workload shares)
        _, reward_rebalance, _, _ = env_reb.step(Action.REBALANCE)

        # Assert REBALANCE achieves higher delta in Goodput/$
        self.assertGreater(
            reward_rebalance,
            reward_wait,
            f"REBALANCE reward ({reward_rebalance:.2f}) must exceed WAIT reward ({reward_wait:.2f}) "
            "during a severe slowdown.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §5. TEST EXECUTION & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

class SDETTestResult(unittest.TextTestResult):
    """Custom test result formatter that prints a structured SDET report."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes: List[unittest.TestCase] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


class SDETTestRunner(unittest.TextTestRunner):
    resultclass = SDETTestResult

    def run(self, test):
        print("\n" + "=" * 76)
        print("  SELF-HEALING MIXED-GPU CLUSTER SIMULATOR -- RIGOROUS SDET TEST SUITE")
        print("=" * 76)
        result = super().run(test)
        print("\n" + "=" * 76)
        print("  COMPONENT PASS / FAIL SUMMARY REPORT")
        print("=" * 76)
        print(f"  Total Test Cases Executed : {result.testsRun}")
        print(f"  Tests Passed              : {len(result.successes)}")
        print(f"  Tests Failed              : {len(result.failures)}")
        print(f"  Tests Errored             : {len(result.errors)}")
        print("-" * 76)
        status = "ALL TESTS PASSED [SUCCESS]" if result.wasSuccessful() else "TESTS FAILED [FAILURE]"
        print(f"  FINAL VERDICT             : {status}")
        print("=" * 76 + "\n")
        return result


if __name__ == "__main__":
    unittest.main(testRunner=SDETTestRunner(verbosity=2))
