"""
test_rl.py
==========
Tests for the gymnasium environment and the RL policy wrapper (rl_agent.py).

    pytest test_rl.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_agent import ACTIONS, GymClusterEnv, RLFeatures
from simulator import Action, SimulatedCluster


class TestGymEnv(unittest.TestCase):

    def test_passes_gymnasium_env_checker(self):
        from gymnasium.utils.env_checker import check_env
        check_env(GymClusterEnv(difficulties=("realistic",), episode_len=50), skip_render_check=True)

    def test_observation_is_finite_and_fixed_size(self):
        env = GymClusterEnv(difficulties=("chaos",), episode_len=300)
        obs, _ = env.reset(seed=3)
        rng = np.random.default_rng(0)
        for _ in range(300):
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertTrue(np.all(np.isfinite(obs)))
            valid = np.flatnonzero(env.action_masks())
            obs, _, term, trunc, _ = env.step(int(rng.choice(valid)))
            if term or trunc:
                break

    def test_wait_is_always_allowed(self):
        env = GymClusterEnv(difficulties=("hard",), episode_len=200)
        env.reset(seed=1)
        for _ in range(200):
            self.assertTrue(env.action_masks()[0])
            env.step(0)

    def test_readmit_masked_when_nothing_is_evicted(self):
        env = GymClusterEnv(difficulties=("easy",), episode_len=10)
        env.reset(seed=0)
        mask = env.action_masks()
        for i, (action, _) in enumerate(ACTIONS):
            if action == Action.READMIT:
                self.assertFalse(mask[i])

    def test_healthy_wait_earns_about_zero(self):
        env = GymClusterEnv(difficulties=("easy",), episode_len=1000)
        env.reset(seed=0)            # easy: first fault is >= 5% of the episode in
        _, reward, *_ = env.step(0)
        self.assertAlmostEqual(reward, 0.0, places=6)

    def test_overhead_is_punished(self):
        env = GymClusterEnv(difficulties=("easy",), episode_len=10)
        env.reset(seed=0)
        evict = next(i for i, (a, t) in enumerate(ACTIONS) if a == Action.EVICT and t == "mac-m2")
        _, reward, *_ = env.step(evict)
        self.assertLess(reward, -0.5)

    def test_episode_truncates(self):
        env = GymClusterEnv(difficulties=("easy",), episode_len=25)
        env.reset(seed=0)
        for t in range(25):
            *_, term, trunc, _ = env.step(0)
        self.assertTrue(trunc)
        self.assertFalse(term)


class TestFeatures(unittest.TestCase):

    def test_slow_counter_grows_during_slowdown_and_resets(self):
        c, f = SimulatedCluster(seed=0), RLFeatures()
        from simulator import FaultEvent, FaultSchedule
        sched = FaultSchedule([FaultEvent(step=5, machine_id="win-gpu-0", kind="slowdown",
                                          slowdown_fraction=0.5, duration_steps=20)])
        c.fire_due_faults(sched)
        idx = RLFeatures.MACHINES.index("win-gpu-0")
        vals = []
        for _ in range(40):
            vec = f.update(c.get_observation())
            vals.append(vec[idx * RLFeatures.PER_MACHINE + RLFeatures.SLOW_TICKS_COL])
            c.step_forward(sched)
        self.assertGreater(max(vals[5:25]), 0.1)
        self.assertEqual(vals[-1], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
