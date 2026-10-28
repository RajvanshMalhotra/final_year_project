"""
Run the same simulator fault scenario on the live system once per agent mode and compare.

    python -m demo.compare --level hard --seed 3 --seconds 300 --modes rl rule off

Each run: fresh controller + 3 node processes, the scenario's faults replayed at the same training
steps, the same workload growth. Results go to demo/runs/compare-<time>.json and a table.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import httpx

from .config import ROOT

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def run_once(mode: str, level: str, seed: int, seconds: float, port: int, scenario_steps: int) -> dict:
    log = open(os.path.join(RUNS, f"compare-{mode}.log"), "w")
    p = subprocess.Popen([sys.executable, "-m", "demo.controller", "--port", str(port), "--store-port",
                          str(port + 1000), "--agent", mode, "--chaos", "--level", level, "--seed", str(seed),
                          "--scenario-steps", str(scenario_steps)],
                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                httpx.get(url + "/api/state", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        t0 = None
        while True:                                    # the clock starts at the first training step
            s = httpx.get(url + "/api/state", timeout=5).json()
            if s["step"] > 0 and t0 is None:
                t0 = time.time()
            if t0 and time.time() - t0 >= seconds:
                break
            time.sleep(1.0)
        s = httpx.get(url + "/api/state", timeout=5).json()
    finally:
        try:
            httpx.post(url + "/api/shutdown", timeout=5)
        except httpx.HTTPError:
            pass
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
    events = [json.loads(line) for line in open(s["log_file"])]
    hist = s["history"]
    # stalled time = gaps of more than 2 s between consecutive finished steps
    stall = 0.0
    hs = [h["t"] for h in hist]
    for a, b in zip(hs, hs[1:]):
        if b - a > 2.0:
            stall += b - a
    return {
        "mode": mode, "seconds": seconds, "steps": s["step"],
        "sequences": sum(h["B"] for h in hist),
        "final_loss": next((h["loss"] for h in reversed(hist) if h["loss"] is not None), None),
        "stalled_s": round(stall, 1),
        "decisions": sum(1 for e in events if e["kind"] == "decision"),
        "evictions": sum(1 for e in events if e["kind"] == "evict"),
        "readmits": sum(1 for e in events if e["kind"] == "readmit"),
        "faults": sum(1 for e in events if e["kind"] == "fault"),
        "members_at_end": len(s["members"]),
        "weights_checks": sum(1 for e in events if e["kind"] == "checksum"),
        "weights_all_identical": all(e.get("identical", True) for e in events if e["kind"] == "checksum"),
        "log_file": s["log_file"],
    }


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--level", default="hard")
    a.add_argument("--seed", type=int, default=3)
    a.add_argument("--seconds", type=float, default=300)
    a.add_argument("--modes", nargs="+", default=["rl", "rule", "off"])
    a.add_argument("--port", type=int, default=8800)
    a.add_argument("--scenario-steps", type=int, default=450,
                   help="squeeze all scenario faults into this many steps (a 5-minute run reaches ~550)")
    args = a.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    results = []
    for mode in args.modes:
        print(f"running {mode} for {args.seconds:.0f} s ...", flush=True)
        results.append(run_once(mode, args.level, args.seed, args.seconds, args.port, args.scenario_steps))
        print(json.dumps(results[-1]), flush=True)
        time.sleep(3)
    out = os.path.join(RUNS, f"compare-{datetime.datetime.now():%Y%m%d-%H%M%S}.json")
    with open(out, "w") as f:
        json.dump({"level": args.level, "seed": args.seed, "scenario_steps": args.scenario_steps,
                   "results": results}, f, indent=1)
    print(f"\n{'mode':6} {'steps':>6} {'sequences':>10} {'stalled s':>10} {'decisions':>10} {'loss':>6} {'nodes':>6}")
    for r in results:
        print(f"{r['mode']:6} {r['steps']:6d} {r['sequences']:10d} {r['stalled_s']:10.1f} {r['decisions']:10d} "
              f"{r['final_loss'] or 0:6.3f} {r['members_at_end']:6d}")
    print("saved", out)


if __name__ == "__main__":
    main()
