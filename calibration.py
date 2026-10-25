"""
calibration.py
==============
Set the simulator from numbers measured on our laptops, instead of guesses.

Input: the results/ folder of the mlx_test kit (github.com/RajvanshMalhotra/mlx_test),
one sub-folder per laptop:
    test0_env.json        what the machine is (macOS / NVIDIA)
    test1_train.json      LoRA fine-tuning speed (tokens/s), trainable params, memory
    test3_ring_rank*.json MLX all-reduce timings over Wi-Fi (1, 10, 50 MB)
    net_bench.json        plain TCP bandwidth + round-trip time

What it sets:
    speeds   per machine, in batches/second = tokens/s / tokens per global batch
    comm_s   the simulator's all-reduce constant: time to all-reduce the LoRA
             gradients (trainable_params x 4 bytes) over 3 machines, divided by
             the ring factor 2(n-1)/n that the simulator multiplies back in

Every number records where it came from ("measured: ..." or "assumed: ..."), so a
report can say exactly which parts are real. Missing measurements fall back to the
old assumptions (NVIDIA laptop = 3.5/1.2 x the Mac; Wi-Fi 5 MB/s, 5 ms).

Not calibrated (still assumptions): fault rates and durations, action overheads,
memory caps (with gradient accumulation any machine can take any share; the level's
Mac cap stays as a stress setting), prices.

    python calibration.py ../spikes/mlx_feasibility/results            # guess hosts
    python calibration.py RESULTS --mac my-mac --win0 pc-a --win1 pc-b --out calibration.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator import proportional_shares, step_time

MACHINES   = ["mac-m2", "win-gpu-0", "win-gpu-1"]
NVIDIA     = ["win-gpu-0", "win-gpu-1"]
OLD_RATIO  = 3.5 / 1.2          # the simulator's built-in NVIDIA : Mac speed ratio
RING_N     = len(MACHINES)
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


@dataclass
class Calibration:
    speeds:              Dict[str, float]     # batches / second
    comm_s:              float                # simulator all-reduce constant (s)
    grad_mb:             float                # LoRA gradient size, MiB (fp32)
    global_batch_tokens: int
    fits:                Dict[str, bool]      # False = training failed on that laptop
    sources:             Dict[str, str] = field(default_factory=dict)

    def cluster_kwargs(self) -> Dict:
        bad = [m for m, ok in self.fits.items() if not ok]
        if bad:
            raise ValueError(f"training failed on {bad}: the simulated cluster would be wrong; "
                             f"see sources for the error")
        return {"normal_speeds": dict(self.speeds), "comm_s": self.comm_s}

    def save(self, path: str = DEFAULT_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "Calibration":
        with open(path) as f:
            return cls(**json.load(f))


def _read(results_dir: str, host: str, name: str) -> Optional[dict]:
    path = os.path.join(results_dir, host, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def guess_hosts(results_dir: str) -> Dict[str, str]:
    """Mac = the folder whose test0_env says Darwin; NVIDIA = folders that saw nvidia-smi."""
    mac, nvidia = [], []
    for host in sorted(os.listdir(results_dir)):
        env = _read(results_dir, host, "test0_env") or {}
        if env.get("system") == "Darwin":
            mac.append(host)
        elif "nvidia_smi" in env:
            nvidia.append(host)
    hosts = {}
    if mac:
        hosts["mac-m2"] = mac[0]
    for mid, host in zip(NVIDIA, nvidia):
        hosts[mid] = host
    return hosts


def _fit_line(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Least-squares t = a + b * size."""
    x, y = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    b, a = np.polyfit(x, y, 1) if len(points) > 1 else (y[0] / x[0], 0.0)
    return float(a), float(b)


def load_calibration(
    results_dir:         str,
    hosts:               Dict[str, str],
    global_batch_tokens: int   = 2048,
    nvidia_to_mac_ratio: float = OLD_RATIO,
    assumed_link_mb_s:   float = 5.0,
    assumed_rtt_ms:      float = 5.0,
) -> Calibration:
    """hosts: simulator machine id -> mlx_test results folder name."""
    src: Dict[str, str] = {}
    speeds: Dict[str, float] = {}
    fits = {m: True for m in MACHINES}
    params: Optional[int] = None

    # 1. Speeds from Test 1.
    for mid in MACHINES:
        host = hosts.get(mid)
        r = _read(results_dir, host, "test1_train") if host else None
        if r is None:
            continue
        if not r.get("ok"):
            fits[mid] = False
            src[f"speed:{mid}"] = f"FAILED on {host}: {r.get('error', 'unknown error')}"
            continue
        speeds[mid] = r["tokens_per_s"] / global_batch_tokens
        params = params or r.get("trainable_params")
        src[f"speed:{mid}"] = (f"measured: {host}/test1_train.json, {r['tokens_per_s']} tokens/s"
                               f" ({r.get('model', '?')})")

    if not speeds:
        raise ValueError(f"no successful test1_train.json for any of {hosts} in {results_dir}")
    models = {m: _read(results_dir, hosts[m], "test1_train").get("model") for m in speeds}
    if len(set(models.values())) > 1:
        raise ValueError(f"speeds must come from the same model on every laptop, got {models}")

    measured_nv = [m for m in NVIDIA if m in speeds]
    for mid in MACHINES:
        if mid in speeds or not fits[mid]:
            continue
        if mid in NVIDIA and measured_nv:
            speeds[mid] = speeds[measured_nv[0]]
            src[f"speed:{mid}"] = f"assumed: same as {measured_nv[0]} (same laptop model)"
        elif mid in NVIDIA and "mac-m2" in speeds:
            speeds[mid] = speeds["mac-m2"] * nvidia_to_mac_ratio
            src[f"speed:{mid}"] = f"assumed: {nvidia_to_mac_ratio:.2f} x the Mac (old simulator ratio)"
        elif mid == "mac-m2":
            speeds[mid] = speeds[measured_nv[0]] / nvidia_to_mac_ratio
            src[f"speed:{mid}"] = f"assumed: NVIDIA / {nvidia_to_mac_ratio:.2f} (old simulator ratio)"
    for mid in MACHINES:            # failed machines keep a placeholder speed; cluster_kwargs refuses
        speeds.setdefault(mid, 0.0)

    # 2. All-reduce time for the LoRA gradients.
    grad_mb = (params or 0) * 4 / 2**20
    ring_factor = 2 * (RING_N - 1) / RING_N
    comm_s, why = None, ""
    for host in hosts.values():
        for path in sorted(glob.glob(os.path.join(results_dir, host, "test3_ring_rank*.json"))):
            with open(path) as f:
                r = json.load(f)
            if r.get("ok") and r.get("world_size") == RING_N and r.get("all_sum"):
                pts = [(float(k.rstrip("MB")), v["seconds"]) for k, v in r["all_sum"].items()]
                a, b = _fit_line(pts)
                comm_s = (a + b * grad_mb) / ring_factor
                why = f"measured: {os.path.relpath(path, results_dir)}, MLX all-reduce over {RING_N} laptops"
                break
        if comm_s is not None:
            break
    if comm_s is None:
        for host in hosts.values():
            r = _read(results_dir, host, "net_bench")
            if r and r.get("MB_per_s"):
                comm_s = grad_mb / r["MB_per_s"] + RING_N * r["rtt_ms_median"] / 1000 / 2
                why = (f"measured: {host}/net_bench.json, {r['MB_per_s']} MB/s, "
                       f"{r['rtt_ms_median']} ms (ring formula)")
                break
    if comm_s is None:
        comm_s = grad_mb / assumed_link_mb_s + RING_N * assumed_rtt_ms / 1000 / 2
        why = f"assumed: Wi-Fi {assumed_link_mb_s} MB/s, {assumed_rtt_ms} ms round trip"
    src["comm_s"] = why
    src["grad_mb"] = (f"measured: {params} trainable LoRA params x 4 bytes" if params
                      else "unknown: no trainable_params in test1_train.json")

    return Calibration(speeds=speeds, comm_s=comm_s, grad_mb=grad_mb,
                       global_batch_tokens=global_batch_tokens, fits=fits, sources=src)


# =============================================================================
# DOES TRAINING TOGETHER PAY OFF?
# =============================================================================

def ring_seconds_per_batch(cal: Calibration, ring: List[str]) -> float:
    """Healthy seconds per global batch if only `ring` trains (no faults, no memory caps)."""
    speeds = {m: cal.speeds[m] for m in ring}
    return step_time(speeds, proportional_shares(speeds), None, cal.comm_s)


def best_ring(cal: Calibration) -> Tuple[List[str], float]:
    """The subset of machines with the fewest seconds per batch."""
    rings = [list(r) for k in range(1, len(MACHINES) + 1) for r in itertools.combinations(MACHINES, k)]
    best = min(rings, key=lambda r: ring_seconds_per_batch(cal, r))
    return best, ring_seconds_per_batch(cal, best)


def break_even_comm_s(cal: Calibration) -> float:
    """Largest comm_s at which all 3 laptops are still as fast as the fastest one alone."""
    single  = 1 / max(cal.speeds.values())
    compute = 1 / sum(cal.speeds.values())
    return (single - compute) / (2 * (RING_N - 1) / RING_N)


def min_global_batch_tokens(cal: Calibration) -> float:
    """Smallest global batch (tokens) at which all 3 laptops beat the fastest one alone.
    Compute grows with the batch; the all-reduce does not (same gradient size)."""
    tok   = {m: s * cal.global_batch_tokens for m, s in cal.speeds.items()}     # tokens / s
    comm  = cal.comm_s * 2 * (RING_N - 1) / RING_N
    return comm / (1 / max(tok.values()) - 1 / sum(tok.values()))


def what_if(cal: Calibration, ratios=(2, 5, 10, 30), links_mb_s=(2, 5, 10, 25, 50),
            rtt_ms: float = 5.0) -> Dict[Tuple[float, float], Tuple[List[str], float]]:
    """Best ring for each (NVIDIA : Mac speed ratio, Wi-Fi MB/s), keeping the measured Mac
    speed and gradient size. Answers 'does it pay off?' before the NVIDIA numbers arrive."""
    mac = cal.speeds["mac-m2"]
    out = {}
    for ratio in ratios:
        for link in links_mb_s:
            alt = Calibration(**{**cal.__dict__,
                                 "speeds": {"mac-m2": mac, **{m: mac * ratio for m in NVIDIA}},
                                 "comm_s": cal.grad_mb / link + RING_N * rtt_ms / 1000 / 2})
            out[(ratio, link)] = best_ring(alt)
    return out


def describe_what_if(cal: Calibration) -> str:
    grid  = what_if(cal)
    links = sorted({k[1] for k in grid})
    short = {"mac-m2": "Mac", "win-gpu-0": "N", "win-gpu-1": "N"}
    lines = ["\n  What if? Best ring by NVIDIA speed (x the measured Mac) and Wi-Fi speed",
             "  (Mac+N+N = all 3, N+N = both NVIDIA, N = one NVIDIA alone; s per batch)",
             f"  {'NVIDIA / Mac':>13} " + "".join(f"{f'{l:g} MB/s':>16}" for l in links)]
    for ratio in sorted({k[0] for k in grid}):
        cells = []
        for l in links:
            ring, sec = grid[(ratio, l)]
            cells.append(f"{'+'.join(short[m] for m in ring)} {sec:.1f}".rjust(16))
        lines.append(f"  {f'{ratio:g}x':>13} " + "".join(cells))
    return "\n".join(lines)


def describe(cal: Calibration) -> str:
    step = {m: 1 / s if s > 0 else float("inf") for m, s in cal.speeds.items()}
    lines = [f"  global batch = {cal.global_batch_tokens} tokens, LoRA gradients = {cal.grad_mb:.1f} MiB",
             f"  {'machine':<11} {'batches/s':>10} {'s per batch alone':>18}  source"]
    for m in MACHINES:
        lines.append(f"  {m:<11} {cal.speeds[m]:>10.4f} {step[m]:>18.1f}  {cal.sources.get(f'speed:{m}', '')}")
    lines.append(f"  comm_s = {cal.comm_s:.3f} s (all-reduce of 3 = {cal.comm_s * 4 / 3:.3f} s)  "
                 f"{cal.sources['comm_s']}")
    lines.append("\n  Does training together pay off? (healthy cluster, seconds per batch)")
    for ring in (MACHINES, NVIDIA, [max(cal.speeds, key=cal.speeds.get)]):
        lines.append(f"    {' + '.join(ring):<32} {ring_seconds_per_batch(cal, ring):7.2f}")
    ring, sec = best_ring(cal)
    lines.append(f"    best: {' + '.join(ring)} at {sec:.2f} s per batch")
    be = break_even_comm_s(cal)
    rtt_part = RING_N * 0.005 / 2
    need = cal.grad_mb / (be - rtt_part) if be > rtt_part else float("inf")
    lines.append(f"    all 3 beat the fastest alone only if comm_s < {be:.3f} s, i.e. a link faster than "
                 f"~{need:.1f} MB/s (5 ms round trip), or a global batch > "
                 f"{min_global_batch_tokens(cal):,.0f} tokens at the current link")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--mac")
    ap.add_argument("--win0")
    ap.add_argument("--win1")
    ap.add_argument("--batch-tokens", type=int, default=2048)
    ap.add_argument("--out", default=DEFAULT_PATH)
    a = ap.parse_args()
    hosts = guess_hosts(a.results_dir)
    for mid, host in (("mac-m2", a.mac), ("win-gpu-0", a.win0), ("win-gpu-1", a.win1)):
        if host:
            hosts[mid] = host
    print(f"  hosts: {hosts}")
    cal = load_calibration(a.results_dir, hosts, a.batch_tokens)
    print(describe(cal))
    print(describe_what_if(cal))
    cal.save(a.out)
    print(f"  saved {a.out}")
