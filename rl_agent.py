"""
rl_agent.py
===========
Reinforcement-learning agent: MaskablePPO on the realistic simulator.

  GymClusterEnv  gymnasium environment around SimulatedCluster + scenarios.py
  RLFeatures     turns noisy observations into a fixed vector (with memory:
                 smoothed speeds, how long a machine has been slow, ...)
  RLPolicy       a trained model as a BasePolicy, so it runs in run_episode
                 and in stress_test.benchmark() next to every other policy

Action space (11): WAIT, REBALANCE, then COMPENSATE / EVICT / READMIT for each
of the 3 machines. Actions that would change nothing are masked out (so are
evicting the last machine and readmitting a machine that is not evicted).

Reward per tick: 0.1 * (batches done - healthy_rate * seconds used). A healthy
step earns 0, slow steps / stalls / action overheads cost, a RESTART loses its
rolled-back work. The episode return is (goodput - ideal goodput) x time.

    python rl_agent.py train --steps 1000000      # trains, saves models/ppo_cluster.zip
    python rl_agent.py eval                        # RL vs all baselines vs Oracle
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from scenarios import DIFFICULTIES, make_scenario
from simulator import (
    CRASH_SIGNAL_AGE_S, SLOW_RATIO_THRESHOLD, Action, BasePolicy, Observation, PolicyOutput,
    plan_action, proportional_shares, step_time,
)

MACHINES = ["mac-m2", "win-gpu-0", "win-gpu-1"]
ACTIONS: List[Tuple[Action, Optional[str]]] = (
    [(Action.WAIT, None), (Action.REBALANCE, None)]
    + [(a, mid) for a in (Action.COMPENSATE, Action.EVICT, Action.READMIT) for mid in MACHINES]
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ppo_cluster.zip")
VALIDATION_SEEDS = range(500, 506)   # model selection only; tests/reports use seeds 0-99
BC_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bc_hybrid.pkl")


# =============================================================================
# FEATURES (shared by the environment and the deployed policy)
# =============================================================================

class RLFeatures:
    """
    Stateful featurizer: call update(obs) once per tick.

    Per machine (9): smoothed speed ratio, raw speed ratio, heartbeat age/10,
        share, evicted, compensated, bandwidth, ticks slow so far/100 (capped 3),
        presumed dead
    Global (6): imbalance (step time with current shares / with ideal shares - 1,
        from smoothed speeds), active machines/3, ticks since our last action/100,
        all-reduce time constant x10, slowest bandwidth, any machine ready to readmit
    """
    MACHINES       = MACHINES
    PER_MACHINE    = 9
    SLOW_TICKS_COL = 7
    N_GLOBAL       = 6
    SIZE           = PER_MACHINE * len(MACHINES) + N_GLOBAL
    EMA_ALPHA      = 0.3

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ema: Dict[str, float] = {}
        self._slow: Dict[str, int] = {mid: 0 for mid in MACHINES}
        self._since_action = 0

    def note_action(self, action: Action) -> None:
        self._since_action = 0 if action != Action.WAIT else self._since_action

    def smoothed(self, mid: str) -> float:
        return self._ema.get(mid, 1.0)

    def update(self, obs: Observation) -> np.ndarray:
        vec: List[float] = []
        speeds, shares, bw, caps = {}, {}, {}, {}
        any_readmit = 0.0
        for mid in MACHINES:
            m = obs.machines[mid]
            raw  = m.measured_speed / m.normal_speed if m.normal_speed > 0 else 0.0
            prev = self._ema.get(mid, raw)
            ema  = self.EMA_ALPHA * raw + (1 - self.EMA_ALPHA) * prev
            self._ema[mid] = ema
            dead = float(not m.evicted and (m.alive_signal_age > CRASH_SIGNAL_AGE_S
                                            or m.measured_speed < 0.01))
            self._slow[mid] = self._slow[mid] + 1 if (ema < SLOW_RATIO_THRESHOLD and not dead) else 0
            if m.evicted and not dead and m.alive_signal_age <= CRASH_SIGNAL_AGE_S \
                    and ema >= SLOW_RATIO_THRESHOLD:
                any_readmit = 1.0
            vec += [ema, min(raw, 3.0), min(m.alive_signal_age, 10.0) / 10.0, m.share,
                    float(m.evicted), float(m.compensated), m.bandwidth,
                    min(self._slow[mid] / 100.0, 3.0), dead]
            caps[mid] = m.max_share
            if m.is_active:
                speeds[mid], shares[mid], bw[mid] = ema * m.normal_speed, m.share, m.bandwidth

        now   = step_time(speeds, shares, bw, obs.comm_s)
        ideal = step_time(speeds, proportional_shares(speeds, caps), bw, obs.comm_s)
        imbalance = min(now / ideal - 1.0, 5.0) if math.isfinite(now) and ideal > 0 else 5.0
        vec += [max(imbalance, 0.0), len(speeds) / 3.0, min(self._since_action / 100.0, 3.0),
                obs.comm_s * 10.0, min(bw.values()) if bw else 0.0, any_readmit]
        self._since_action += 1
        return np.asarray(vec, dtype=np.float32)


def action_mask(obs: Observation) -> np.ndarray:
    """WAIT always; any other action only if it would change something."""
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[0] = True
    for i, (action, target) in enumerate(ACTIONS[1:], start=1):
        overhead, _ = plan_action(obs, action, target)
        mask[i] = overhead > 0
    return mask


# =============================================================================
# ENVIRONMENT
# =============================================================================

class GymClusterEnv(gym.Env):
    """Each reset draws a difficulty and a scenario seed (training seeds >= 1000)."""

    metadata = {"render_modes": []}
    REWARD_SCALE = 0.1

    def __init__(
        self,
        difficulties: Sequence[str] = ("realistic", "hard", "chaos"),
        episode_len:  int = 1000,
        seeds:        Tuple[int, int] = (1000, 1_000_000),
        calibration=None,
    ):
        super().__init__()
        self.calibration  = calibration
        self.difficulties = tuple(difficulties)
        self.episode_len  = episode_len
        self.seed_range   = seeds
        self.observation_space = gym.spaces.Box(0.0, 10.0, (RLFeatures.SIZE,), np.float32)
        self.action_space      = gym.spaces.Discrete(len(ACTIONS))
        self._features = RLFeatures()
        self._cluster  = None
        self._obs: Optional[Observation] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        d  = self.difficulties[int(self.np_random.integers(len(self.difficulties)))]
        sc = make_scenario(int(self.np_random.integers(*self.seed_range)), d, self.episode_len,
                           self.calibration)
        self._scenario = sc
        self._cluster  = sc.make_cluster()
        self._cluster.apply_checkpoint()
        self._cluster.fire_due_faults(sc.schedule)
        self._rate  = 1.0 / self._cluster.healthy_step_time()
        self._ticks = 0
        self._features.reset()
        self._obs = self._cluster.get_observation()
        return self._features.update(self._obs), {"difficulty": d}

    def action_masks(self) -> np.ndarray:
        return action_mask(self._obs)

    def step(self, action_index: int):
        c = self._cluster
        action, target = ACTIONS[int(action_index)]
        p0, t0 = c.train_step, c.sim_time
        c.apply_action(action, target)
        c.step_forward(self._scenario.schedule)
        self._features.note_action(action)
        self._ticks += 1
        reward = self.REWARD_SCALE * ((c.train_step - p0) - self._rate * (c.sim_time - t0))
        self._obs = c.get_observation()
        truncated = self._ticks >= self.episode_len
        return self._features.update(self._obs), float(reward), False, truncated, {}


# =============================================================================
# DEPLOYED POLICY
# =============================================================================

class RLPolicy(BasePolicy):
    """A trained MaskablePPO model as a BasePolicy (deterministic actions)."""

    name = "RL agent (PPO)"

    def __init__(self, model) -> None:
        self._model = model
        self._features = RLFeatures()

    def decide(self, obs: Observation) -> PolicyOutput:
        x = self._features.update(obs)
        a, _ = self._model.predict(x, action_masks=action_mask(obs), deterministic=True)
        action, target = ACTIONS[int(a)]
        self._features.note_action(action)
        return (action, target) if target else action


def load_policy(path: str = MODEL_PATH):
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(path, device="cpu")
    return lambda sc: RLPolicy(model)


# =============================================================================
# TRAIN / EVAL
# =============================================================================

_RULE_CACHE: Dict[Tuple[str, int, int], float] = {}


def validation_score(make_policy, seeds: Sequence[int] = VALIDATION_SEEDS,
                     difficulties: Sequence[str] = DIFFICULTIES, calibration=None) -> float:
    """Mean goodput relative to the Reactive rule on validation scenarios (1.0 = tie)."""
    from simulator import ReactivePolicy, run_episode
    ratios = []
    for d in difficulties:
        for s in seeds:
            sc  = make_scenario(s, d, calibration=calibration)
            key = (d, s, id(calibration))
            if key not in _RULE_CACHE:
                _RULE_CACHE[key] = run_episode(sc.make_cluster(), ReactivePolicy(),
                                               sc.schedule, 1000).goodput_per_hr
            g = run_episode(sc.make_cluster(), make_policy(), sc.schedule, 1000).goodput_per_hr
            ratios.append(g / _RULE_CACHE[key])
    return float(np.mean(ratios))


def train(total_steps: int, n_envs: int = 8, path: str = MODEL_PATH, seed: int = 0,
          eval_every: int = 100_000, calibration=None) -> None:
    """
    Train on all difficulty levels with a decaying learning rate. Every
    `eval_every` steps, score the model on validation scenarios and keep the best
    checkpoint (PPO often gets worse late in training; this guards against it).
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    class KeepBest(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.best, self.next_eval = -math.inf, eval_every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self.next_eval += eval_every
                score = validation_score(lambda: RLPolicy(self.model), calibration=calibration)
                tag = ""
                if score > self.best:
                    self.best, tag = score, "  <- best, saved"
                    self.model.save(path)
                print(f"  [val] {self.num_timesteps:>9d} steps: {score:.4f} x rule{tag}", flush=True)
            return True

    venv  = DummyVecEnv([lambda: Monitor(GymClusterEnv(difficulties=DIFFICULTIES, calibration=calibration))
                         for _ in range(n_envs)])
    venv.seed(seed)
    model = MaskablePPO(
        "MlpPolicy", venv, n_steps=1024, batch_size=512, n_epochs=10, gamma=0.995,
        gae_lambda=0.95, learning_rate=lambda progress: 3e-4 * progress, ent_coef=0.01,
        clip_range=0.2, policy_kwargs=dict(net_arch=[128, 128]), seed=seed, device="cpu",
        verbose=0,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=KeepBest())
    print(f"done: best checkpoint in {path}, {total_steps} steps in {time.time() - t0:.0f}s")


def train_bc(path: str = BC_PATH, n_seeds: int = 12) -> None:
    """Imitation baseline: BC + 1 DAgger round on realistic/hard/chaos Oracle labels."""
    import pickle
    from agent_brain import train_agent_model
    t0 = time.time()
    model = train_agent_model(range(1000, 1000 + n_seeds), dagger_rounds=1, verbose=True,
                              difficulty="realistic,hard,chaos")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"saved {path} in {time.time() - t0:.0f}s")


def evaluate(path: str = MODEL_PATH, seeds: range = range(20)) -> None:
    from stress_test import benchmark, default_policies, print_benchmark
    policies = default_policies()
    if os.path.exists(BC_PATH):
        import pickle
        from agent_brain import ClusterDetector, HybridAgentPolicy, MPCFallback
        with open(BC_PATH, "rb") as f:
            bc = pickle.load(f)
        policies["BC + DAgger hybrid"] = lambda sc: HybridAgentPolicy(
            bc, MPCFallback(), ClusterDetector(sc.normal_speeds))
    policies["RL agent (PPO)"] = load_policy(path)
    print_benchmark(benchmark(policies, DIFFICULTIES, seeds), len(seeds))


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], n: int = 10000,
                        seed: int = 0) -> Tuple[float, float, float]:
    """Mean of a-b over paired scenarios, with a 95% bootstrap confidence interval."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(model_paths: Sequence[str], seeds: range = range(30),
           out_dir: str = "results", calibration=None) -> None:
    """
    Final comparison: every baseline, the BC hybrid, and each trained PPO model,
    on `seeds` unseen scenarios per difficulty. Writes results/results.json and
    prints RL mean +/- spread over training seeds, plus a paired bootstrap CI of
    (RL - best simple rule) per level.
    """
    import json
    from stress_test import benchmark, default_policies, print_benchmark
    policies = default_policies()
    if os.path.exists(BC_PATH):
        import pickle
        from agent_brain import ClusterDetector, HybridAgentPolicy, MPCFallback
        with open(BC_PATH, "rb") as f:
            bc = pickle.load(f)
        policies["BC + DAgger hybrid"] = lambda sc: HybridAgentPolicy(
            bc, MPCFallback(), ClusterDetector(sc.normal_speeds))
    rl_names = []
    for i, path in enumerate(model_paths):
        rl_names.append(f"RL seed {i}")
        policies[rl_names[-1]] = load_policy(path)

    res = benchmark(policies, DIFFICULTIES, seeds, calibration=calibration)
    print_benchmark(res, len(seeds))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("\n  RL over training seeds, and paired 95% CI of (RL - best simple rule), % of Oracle")
    for d, rows in res.items():
        rl_means = [rows[n]["mean_pct"] for n in rl_names]
        rl_worst = [rows[n]["worst_pct"] for n in rl_names]
        rules = [n for n in rows if n not in rl_names + ["Oracle", "BC + DAgger hybrid"]]
        best_rule = max(rules, key=lambda n: rows[n]["mean_pct"])
        rl_avg = np.mean([rows[n]["pcts"] for n in rl_names], axis=0)
        diff, lo, hi = paired_bootstrap_ci(rl_avg, rows[best_rule]["pcts"])
        print(f"  [{d:<9}] RL {np.mean(rl_means):5.1f} (seeds {min(rl_means):.1f}-{max(rl_means):.1f}), "
              f"worst ep {min(rl_worst):5.1f} | best rule {best_rule}: {rows[best_rule]['mean_pct']:5.1f} "
              f"| RL - rule = {diff:+.1f} [{lo:+.1f}, {hi:+.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "train-bc", "eval", "report"])
    ap.add_argument("--models", nargs="*", default=[MODEL_PATH])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--path", default=MODEL_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", help="calibration.json from calibration.py (measured laptops)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cal = None
    if a.calibration:
        from calibration import Calibration
        cal = Calibration.load(a.calibration)
    if a.mode == "train":
        train(a.steps, path=a.path, seed=a.seed, calibration=cal)
    elif a.mode == "train-bc":
        train_bc()
    elif a.mode == "report":
        report(a.models, out_dir=a.out, calibration=cal)
    else:
        evaluate(a.path)
