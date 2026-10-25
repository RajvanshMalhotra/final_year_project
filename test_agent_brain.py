"""
test_agent_brain.py
===================
Unit tests for agent_brain.py (moved out of agent_brain.py so a bare `pytest` finds them).

    pytest test_agent_brain.py -v
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_brain import *  # noqa: F401,F403
from agent_brain import (
    ClusterDetector, DecisionModel, DiagnosedFault, HybridAgentPolicy, MachineDetector,
    MPCFallback, evaluate_detector, featurize, generate_mock_oracle_data,
    train_behavioral_cloning,
)
from simulator import Action, MachineObs, Observation


# =============================================================================

class TestMachineDetector(unittest.TestCase):
    """Unit tests for MachineDetector (Days 1-2)."""

    def test_ema_converges_to_normal_speed(self):
        """EMA must converge to ~normal_speed after 50 normal steps."""
        det = MachineDetector("m0", normal_speed=3.5)
        for s in range(50):
            det.update(s, measured_speed=3.5, alive_signal_age=0.1)
        self.assertAlmostEqual(det._ema_speed, 3.5, delta=0.05)

    def test_ema_drops_after_sustained_slowdown(self):
        """EMA must drop below baseline after a persistent speed reduction."""
        det = MachineDetector("m0", normal_speed=3.5)
        for s in range(20):
            det.update(s, 3.5, 0.1)
        baseline = det._ema_speed
        for s in range(20, 60):
            det.update(s, 3.5 * 0.5, 0.1)
        self.assertLess(det._ema_speed, baseline)

    def test_fail_stop_via_zero_speed(self):
        det = MachineDetector("m0", 3.5)
        r   = det.update(10, measured_speed=0.0, alive_signal_age=0.1)
        self.assertEqual(r.fault_kind, "fail-stop")

    def test_fail_stop_via_stale_heartbeat(self):
        det = MachineDetector("m0", 3.5)
        r   = det.update(10, measured_speed=3.5, alive_signal_age=999.0)
        self.assertEqual(r.fault_kind, "fail-stop")

    def test_fail_slow_detected_after_warmup(self):
        """
        The EMA change-point detector fires at the transition step when
        z-score spikes above threshold.  The EMA then adapts, so the
        detector correctly returns 'none' once it has re-baselined.
        This test checks that at least one step inside the slow window
        is classified as 'fail-slow'.
        """
        det = MachineDetector("m0", normal_speed=3.5)
        for s in range(30):
            det.update(s, 3.5, 0.1)
        detected_any = False
        for s in range(30, 70):
            result = det.update(s, 3.5 * 0.5, 0.1)
            if result.fault_kind == "fail-slow":
                detected_any = True
                break   # found it — no need to continue
        self.assertTrue(
            detected_any,
            "Expected at least one 'fail-slow' detection during 50%% slowdown window"
        )

    def test_no_fault_on_normal_operation(self):
        """Small normal noise must not trigger any fault."""
        det = MachineDetector("m0", 3.5)
        rng = random.Random(0)
        last = None
        for s in range(80):
            spd  = 3.5 * rng.uniform(0.92, 1.08)
            last = det.update(s, spd, 0.1)
        self.assertEqual(last.fault_kind, "none")

    def test_evaluate_detector_precision_recall_above_60pct(self):
        """EMA detector Precision and Recall must both exceed 0.60."""
        ema_r, _ = evaluate_detector(
            n_machines=3, total_steps=200,
            fault_step=40, slowdown_frac=0.50, seed=7,
        )
        self.assertGreater(ema_r.precision, 0.60,
                           f"EMA P={ema_r.precision:.3f}")
        self.assertGreater(ema_r.recall,    0.60,
                           f"EMA R={ema_r.recall:.3f}")

    def test_ema_ttn_not_much_worse_than_naive(self):
        """EMA time-to-notice must be <= naive TTN + 5 steps."""
        ema_r, naive_r = evaluate_detector(
            n_machines=3, total_steps=200,
            fault_step=40, slowdown_frac=0.50, seed=7,
        )
        self.assertLessEqual(
            ema_r.avg_time_to_notice,
            naive_r.avg_time_to_notice + 5,
            f"EMA TTN={ema_r.avg_time_to_notice:.1f} > "
            f"naive TTN={naive_r.avg_time_to_notice:.1f} + 5",
        )


class TestMPCFallback(unittest.TestCase):
    """Unit tests for MPCFallback (Day 3)."""

    NORMAL  = {"mac-m2": 1.2, "win-gpu-0": 3.5, "win-gpu-1": 3.5}
    PRICES  = {"mac-m2": 0.0, "win-gpu-0": 0.5, "win-gpu-1": 0.5}

    def _obs(self, speeds: Dict[str, float]) -> Observation:
        # Shares are the ones the cluster set while healthy (proportional to normal
        # speed), as in the simulator. Equal shares would make the healthy cluster
        # look imbalanced, and REBALANCE would then rightly win.
        total_normal = sum(self.NORMAL.get(mid, 3.5) for mid in speeds)
        machines = {}
        for mid, spd in speeds.items():
            ns = self.NORMAL.get(mid, 3.5)
            machines[mid] = MachineObs(
                alive_signal_age = 0.1 if spd > 0 else 10.0,
                normal_speed     = ns,
                measured_speed   = spd,
                share            = ns / total_normal,
                code_version     = "v1.0",
                price            = self.PRICES.get(mid, 0.5),
                is_active        = spd > 0.01,
            )
        return Observation(time=0.0, step=10, machines=machines)

    def test_rebalance_over_wait_for_permanent_50pct_slowdown(self):
        """
        REBALANCE or COMPENSATE must beat WAIT for a 50% permanent slowdown
        (fail-slow with long duration).
        """
        mpc   = MPCFallback()
        obs   = self._obs({"mac-m2": 1.2, "win-gpu-0": 3.5*0.5, "win-gpu-1": 3.5})
        fault = DiagnosedFault("win-gpu-0", "fail-slow", severity=0.50, duration_est=100)
        best, _, all_proj = mpc.decide(fault, obs)
        self.assertIn(
            best, [Action.REBALANCE, Action.COMPENSATE],
            f"Expected REBALANCE/COMPENSATE, got {best}. Proj={all_proj}",
        )
        active_score = all_proj.get("REBALANCE", 0) + all_proj.get("COMPENSATE", 0)
        self.assertGreater(active_score, all_proj.get("WAIT", 0),
                           "Active actions should outscore WAIT")

    def test_evict_chosen_for_crash(self):
        """EVICT must be the best action for a fail-stop fault."""
        mpc   = MPCFallback()
        obs   = self._obs({"mac-m2": 1.2, "win-gpu-0": 0.0, "win-gpu-1": 3.5})
        fault = DiagnosedFault("win-gpu-0", "fail-stop", severity=1.0, duration_est=999)
        best, _, _ = mpc.decide(fault, obs)
        self.assertEqual(best, Action.EVICT, f"Expected EVICT, got {best}")

    def test_wait_on_healthy_cluster(self):
        """On a healthy cluster MPC must return WAIT or CHECKPOINT."""
        mpc   = MPCFallback()
        obs   = self._obs({"mac-m2": 1.2, "win-gpu-0": 3.5, "win-gpu-1": 3.5})
        fault = DiagnosedFault("none", "none", severity=0.0, duration_est=0)
        best, _, _ = mpc.decide(fault, obs)
        self.assertIn(best, [Action.WAIT, Action.CHECKPOINT],
                      f"Expected WAIT/CHECKPOINT, got {best}")


class TestDecisionModel(unittest.TestCase):
    """Unit tests for DecisionModel + behavioral cloning (Day 4)."""

    @classmethod
    def setUpClass(cls):
        cls.obs_list, cls.act_list = generate_mock_oracle_data(n_samples=800, seed=0)
        cls.model = train_behavioral_cloning(cls.obs_list, cls.act_list, verbose=False)

    def test_model_trained_flag(self):
        self.assertTrue(self.model.is_trained)

    def test_predict_proba_sums_to_one(self):
        X    = np.vstack([featurize(o) for o in self.obs_list[:20]])
        sums = self.model.predict_proba(X).sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(20), atol=1e-5)

    def test_train_accuracy_above_2x_chance(self):
        X = np.vstack([featurize(o) for o in self.obs_list])
        y = np.array([a.name for a in self.act_list])
        preds, _ = self.model.predict(X)
        acc    = float(np.mean(np.array(preds) == y))
        chance = 1.0 / len(set(y))
        self.assertGreater(acc, chance * 2.0,
                           f"Acc={acc:.2%} not > 2x chance ({chance:.2%})")

    def test_confidence_in_unit_interval(self):
        X = np.vstack([featurize(o) for o in self.obs_list[:50]])
        _, confs = self.model.predict(X)
        self.assertTrue(np.all(confs >= 0.0))
        self.assertTrue(np.all(confs <= 1.0))


class TestHybridPolicy(unittest.TestCase):
    """Unit tests for HybridAgentPolicy (Days 5-6)."""

    def _simple_obs(self) -> Observation:
        return Observation(
            time=0.0, step=5,
            machines={
                "mac-m2":    MachineObs(0.1, 1.2, 1.2, 0.33, "v1", 0.0, True),
                "win-gpu-0": MachineObs(0.1, 3.5, 3.5, 0.33, "v1", 0.5, True),
                "win-gpu-1": MachineObs(0.1, 3.5, 3.5, 0.34, "v1", 0.5, True),
            }
        )

    def _policy(self, mock_model: Any) -> HybridAgentPolicy:
        return HybridAgentPolicy(mock_model, MPCFallback(), cluster_detector=None)

    def test_low_confidence_routes_to_mpc(self):
        """
        60% confidence (< 0.70) -> MPC must be called;
        mpc_call_count = 1, nn_call_count = 0.
        """
        m = MagicMock(spec=DecisionModel)
        m.is_trained = True
        m.predict_single.return_value = ("WAIT", 0.60)

        policy = self._policy(m)
        policy.decide(self._simple_obs())

        m.predict_single.assert_called_once()
        self.assertEqual(policy.mpc_call_count, 1)
        self.assertEqual(policy.nn_call_count,  0)

    def test_high_confidence_skips_mpc(self):
        """
        95% confidence (>= 0.70) -> NN action returned directly;
        mpc_call_count = 0, nn_call_count = 1.
        """
        m = MagicMock(spec=DecisionModel)
        m.is_trained = True
        m.predict_single.return_value = ("WAIT", 0.95)

        policy = self._policy(m)
        action = policy.decide(self._simple_obs())

        self.assertEqual(action, Action.WAIT)
        self.assertEqual(policy.mpc_call_count, 0)
        self.assertEqual(policy.nn_call_count,  1)

    def test_exact_threshold_uses_nn(self):
        """Confidence == 0.70 should use the NN (>= rule)."""
        m = MagicMock(spec=DecisionModel)
        m.is_trained = True
        m.predict_single.return_value = ("REBALANCE", 0.70)

        policy = self._policy(m)
        action = policy.decide(self._simple_obs())

        self.assertEqual(action, Action.REBALANCE)
        self.assertEqual(policy.nn_call_count,  1)
        self.assertEqual(policy.mpc_call_count, 0)

    def test_decision_latency_under_2ms(self):
        """
        Average decide() latency must be < 2.0 ms in the unit test
        environment.  The evaluation report targets < 1.0 ms on the actual
        hardware; the 2 ms threshold here accounts for CI/test overhead.
        """
        obs_tr, act_tr = generate_mock_oracle_data(n_samples=400, seed=1)
        model  = train_behavioral_cloning(obs_tr, act_tr, verbose=False)
        policy = HybridAgentPolicy(model, MPCFallback())

        test_obs, _ = generate_mock_oracle_data(n_samples=50, seed=99)
        for obs in test_obs:
            policy.decide(obs)

        # Median, not mean: one slow first call or OS hiccup in 50 samples made
        # the mean flaky (failed ~1 run in 5) without the agent being slow.
        med_ms = float(np.median(policy.decision_times))
        self.assertLess(med_ms, 2.0,
                        f"Median latency {med_ms:.4f} ms > 2.0 ms (CI threshold)")

    def test_untrained_model_always_routes_to_mpc(self):
        """An untrained model must always trigger MPC (5/5 times)."""
        policy = HybridAgentPolicy(DecisionModel(), MPCFallback())
        for _ in range(5):
            policy.decide(self._simple_obs())
        self.assertEqual(policy.mpc_call_count, 5)
        self.assertEqual(policy.nn_call_count,  0)




# =============================================================================
# REGRESSION TESTS FOR THE 2026-10-24 REVIEW
# =============================================================================

from simulator import (  # noqa: E402
    DoNothingPolicy, FaultEvent, FaultSchedule, RandomFaultGenerator,
    RebalanceOnlyPolicy, SimulatedCluster, run_episode,
)
from agent_brain import expected_calibration_error, generate_oracle_data, train_agent_model  # noqa: E402

NORMALS = {"mac-m2": 1.2, "win-gpu-0": 3.5, "win-gpu-1": 3.5}
TRAIN_SEEDS = range(100, 130)   # never overlaps the evaluation seeds 0-4


def _trained_agent() -> HybridAgentPolicy:
    if not hasattr(_trained_agent, "model"):
        _trained_agent.model = train_agent_model(TRAIN_SEEDS, total_steps=1000, dagger_rounds=1)
    return HybridAgentPolicy(_trained_agent.model, MPCFallback(), ClusterDetector(NORMALS))


class TestDetectorFixes(unittest.TestCase):

    # The EMA used to re-baseline onto the slow speed, so it said "none" mid-slowdown.
    def test_fail_slow_persists_while_machine_is_slow(self):
        det = MachineDetector("m0", normal_speed=3.5)
        for s in range(30):
            det.update(s, 3.5, 0.1)
        kinds = [det.update(s, 3.5 * 0.5, 0.1).fault_kind for s in range(30, 70)]
        self.assertTrue(all(k == "fail-slow" for k in kinds[5:]), kinds)

    def test_detector_clears_after_recovery(self):
        det = MachineDetector("m0", normal_speed=3.5)
        for s in range(30):
            det.update(s, 3.5 * 0.5, 0.1)
        for s in range(30, 45):
            r = det.update(s, 3.5, 0.1)
        self.assertEqual(r.fault_kind, "none")

    # An evicted machine is not a fault any more.
    def test_evicted_machine_is_not_reported_as_fault(self):
        c = SimulatedCluster(seed=0)
        c.apply_action(Action.EVICT, "win-gpu-0")
        dets = ClusterDetector(NORMALS).scan(c.get_observation())
        self.assertEqual({d.machine_id: d.fault_kind for d in dets}["win-gpu-0"], "none")


class TestCalibration(unittest.TestCase):

    def test_ece_is_zero_when_confidence_matches_accuracy(self):
        conf    = np.array([0.8] * 10)
        correct = np.array([1] * 8 + [0] * 2)
        self.assertAlmostEqual(expected_calibration_error(conf, correct), 0.0)

    def test_ece_of_overconfident_model(self):
        conf    = np.array([0.99] * 10)
        correct = np.array([1] * 5 + [0] * 5)
        self.assertAlmostEqual(expected_calibration_error(conf, correct), 0.49)

    # Calibration used to be fitted on the training data, so confidences were ~0.997.
    def test_model_is_calibrated_on_held_out_data(self):
        model = _trained_agent()._model
        self.assertLess(model.calibration_ece, 0.10)


class TestOracleData(unittest.TestCase):

    def test_labels_come_from_the_simulator_oracle(self):
        obs, acts = generate_oracle_data(range(3), total_steps=1000)
        self.assertEqual(len(obs), 3000)
        self.assertIn(Action.EVICT, acts)
        self.assertTrue({Action.COMPENSATE, Action.REBALANCE} & set(acts))


class TestFeaturize(unittest.TestCase):

    def test_features_see_evicted_and_compensated_flags(self):
        c = SimulatedCluster(seed=0)
        before = featurize(c.get_observation())
        c.inject_slowdown("win-gpu-0", 0.4, 100)
        c.apply_action(Action.COMPENSATE, "win-gpu-0")
        compensated = featurize(c.get_observation())
        c.apply_action(Action.EVICT, "win-gpu-0")
        evicted = featurize(c.get_observation())
        self.assertFalse(np.allclose(before, compensated))
        self.assertFalse(np.allclose(compensated, evicted))


class TestMPCFixes(TestMPCFallback):

    def test_slow_shortlist_is_wait_compensate_evict(self):
        obs   = self._obs({"mac-m2": 1.2, "win-gpu-0": 1.75, "win-gpu-1": 3.5})
        fault = DiagnosedFault("win-gpu-0", "fail-slow", severity=0.5, duration_est=100)
        _, _, proj = MPCFallback().decide(fault, obs)
        self.assertEqual(set(proj), {"WAIT", "COMPENSATE", "EVICT"})

    def test_rebalance_after_recovery(self):
        c = SimulatedCluster(seed=0)
        c.inject_slowdown("win-gpu-0", 0.4, 3)
        c.apply_action(Action.REBALANCE)
        for _ in range(5):
            c.step_forward(FaultSchedule())      # recovered, shares still skewed
        fault = DiagnosedFault("none", "none", 0.0, 0)
        best, _, _ = MPCFallback().decide(fault, c.get_observation())
        self.assertEqual(best, Action.REBALANCE)

    def test_readmit_recovered_machine(self):
        c = SimulatedCluster(seed=0)
        c.inject_slowdown("win-gpu-0", 0.4, 3)
        c.apply_action(Action.EVICT, "win-gpu-0")
        for _ in range(5):
            c.step_forward(FaultSchedule())
        fault = DiagnosedFault("none", "none", 0.0, 0)
        best, _, _ = MPCFallback().decide(fault, c.get_observation())
        self.assertEqual(best, Action.READMIT)


class TestHybridInSimulator(unittest.TestCase):

    # The detector used to be updated only on the (rare) MPC path.
    def test_detector_sees_every_decision(self):
        policy = _trained_agent()
        c = SimulatedCluster(seed=0)
        for _ in range(7):
            policy.decide(c.get_observation())
            c.step_forward(FaultSchedule())
        self.assertEqual(policy._detector._detectors["mac-m2"]._step_count, 7)

    # The agent used to spam EVICT hundreds of times per episode.
    def test_no_evict_spam(self):
        sched = FaultSchedule([FaultEvent(step=200, machine_id="win-gpu-0", kind="crash")])
        r = run_episode(SimulatedCluster(0), _trained_agent(), sched, 1000)
        self.assertLessEqual(r.decisions.get("EVICT", 0), 10, r.decisions)

    # It was never run in the simulator. It must beat doing nothing and keep up
    # with the simple reactive rule on seeds it was not trained on.
    def test_agent_competitive_on_unseen_seeds(self):
        agent, rule, idle = [], [], []
        for seed in range(5):
            sched = RandomFaultGenerator(seed=seed).as_schedule()
            agent.append(run_episode(SimulatedCluster(seed), _trained_agent(), sched, 1000).goodput_per_hr)
            rule.append(run_episode(SimulatedCluster(seed), RebalanceOnlyPolicy(), sched, 1000).goodput_per_hr)
            idle.append(run_episode(SimulatedCluster(seed), DoNothingPolicy(), sched, 1000).goodput_per_hr)
        self.assertGreater(np.mean(agent), np.mean(idle))
        self.assertGreaterEqual(np.mean(agent), 0.98 * np.mean(rule),
                                f"agent {np.mean(agent):.0f} vs rule {np.mean(rule):.0f}")


class TestShieldAndCoverage(unittest.TestCase):

    # A confident model said WAIT on a visibly dead machine and the ring stalled forever.
    def test_shield_overrules_confident_bad_action(self):
        c = SimulatedCluster(seed=0)
        c.inject_crash("win-gpu-0")
        for _ in range(5):
            c.step_forward(FaultSchedule())          # heartbeat now timed out
        m = MagicMock(spec=DecisionModel)
        m.is_trained = True
        m.predict_single.return_value = ("WAIT", 0.99)
        policy = HybridAgentPolicy(m, MPCFallback(), ClusterDetector(NORMALS))
        self.assertEqual(policy.decide(c.get_observation()), (Action.EVICT, "win-gpu-0"))

    def test_shield_keeps_a_good_model_action(self):
        m = MagicMock(spec=DecisionModel)
        m.is_trained = True
        m.predict_single.return_value = ("WAIT", 0.99)
        policy = HybridAgentPolicy(m, MPCFallback(), ClusterDetector(NORMALS))
        self.assertEqual(policy.decide(SimulatedCluster(seed=0).get_observation()), Action.WAIT)
        self.assertEqual(policy.nn_call_count, 1)

    # Training data must contain the states the agent really meets: a crash that
    # is visible (heartbeat timed out) and still needs an EVICT.
    def test_oracle_data_covers_visible_crashes(self):
        obs, acts = generate_oracle_data(range(4), total_steps=1000)
        visible_crash_evicts = sum(
            1 for o, a in zip(obs, acts)
            if a == Action.EVICT and featurize(o)[-2] == 1.0)   # any_dead feature
        # Seeds 1-3 are driven by baselines that must wait for the heartbeat
        # timeout, so each contributes the visible-crash state once (seed 0 follows
        # the Oracle mix, which usually evicts before the crash is visible).
        self.assertGreaterEqual(visible_crash_evicts, 3)
