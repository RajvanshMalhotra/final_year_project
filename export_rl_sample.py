"""
Write one practice round of the RL agent as a CSV, so people can see what its "training data" is.

There is no fixed dataset: the agent creates its own data by playing rounds in the simulator.
Each row = one training step of the simulated cluster:
  - 33 numbers the agent sees (its input),
  - which actions were allowed (the action mask),
  - the action it chose,
  - the reward it got for that step,
  - what was really happening (the fault schedule) - the agent never sees this column.
During training (PPO) about a million such rows are generated, used to adjust the network, and
thrown away; nothing is stored.

    python export_rl_sample.py --model models/v2_seed1.zip --level realistic --seed 3
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from rl_agent import ACTIONS, MACHINES, GymClusterEnv, RLFeatures

PER_MACHINE = ["speed_smoothed", "speed_now", "heartbeat_age_div10", "share", "evicted",
               "compensated", "bandwidth", "slow_steps_div100", "presumed_dead"]
GLOBAL = ["imbalance", "active_machines_frac", "steps_since_action_div100", "allreduce_s_x10",
          "slowest_bandwidth", "readmit_ready"]


def action_name(i: int) -> str:
    a, t = ACTIONS[i]
    return a.name + (f" {t}" if t else "")


def truth(events, tick: int) -> str:
    live = []
    for e in events:
        end = e.step + e.duration_steps if e.duration_steps else 10**9
        if e.step <= tick < end:
            pct = f" -{e.slowdown_fraction:.0%}" if e.kind != "crash" else ""
            live.append(f"{e.machine_id} {e.kind}{pct}")
    return "; ".join(live) or "all healthy"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/v2_seed1.zip")
    p.add_argument("--level", default="realistic")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="results/rl_experience_sample.csv")
    a = p.parse_args()
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(a.model, device="cpu")
    env = GymClusterEnv(difficulties=(a.level,), seeds=(a.seed, a.seed + 1))
    x, _ = env.reset(seed=0)
    cols = [f"{m}.{f}" for m in MACHINES for f in PER_MACHINE] + GLOBAL
    assert len(cols) == RLFeatures.SIZE
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    total = 0.0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "really_happening_hidden_from_agent"] + cols
                   + ["allowed_actions", "chosen_action", "reward", "training_progress"])
        done = False
        while not done:
            mask = env.action_masks()
            act, _ = model.predict(x, action_masks=mask, deterministic=True)
            act = int(act)
            row = [env._ticks, truth(env._scenario.events, env._ticks)] + [f"{v:.3f}" for v in x]
            allowed = "|".join(action_name(i) for i in np.flatnonzero(mask))
            x, r, term, trunc, _ = env.step(act)
            total += r
            w.writerow(row + [allowed, action_name(act), f"{r:.4f}", f"{env._cluster.train_step:.1f}"])
            done = term or trunc
    print(f"wrote {a.out}: {env._ticks} rows, total reward {total:.2f}")


if __name__ == "__main__":
    main()
