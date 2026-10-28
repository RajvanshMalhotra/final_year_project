"""
Every number the demo emulates, with where it comes from.

All three nodes share one Mac's CPU, so their *hardware* differences cannot be real here.
Those differences are emulated with the values below; each carries a source string:
"measured: ..." (our own test results), "published: ..." (a paper), or "assumed: ...".
On the three real laptops (`python -m demo node --controller <ip> --no-emulate`) none of this
is used: speeds, network and faults are whatever the hardware really does.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CALIBRATION = os.path.join(ROOT, "calibration.json")

NODE_IDS = ["mac-m2", "win-gpu-0", "win-gpu-1"]
LABELS = {"mac-m2": "Apple M2 Mac · MLX", "win-gpu-0": "Windows laptop 1 · NVIDIA",
          "win-gpu-1": "Windows laptop 2 · NVIDIA"}
# Sequences per micro-batch that fit in memory (8 GB unified Mac vs ~8 GB VRAM laptops).
CAPS = {"mac-m2": 8, "win-gpu-0": 16, "win-gpu-1": 16}
FALLBACK_FACTORS = {"mac-m2": 1 / 2.92, "win-gpu-0": 1.0, "win-gpu-1": 1.0}

CITATIONS: Dict[str, str] = {
    "speeds": "Relative node speeds = calibration.json speeds / fastest. Mac: measured with "
              "mlx_test test1_train.py (108.9 tokens/s, Qwen2.5-1.5B 4-bit). NVIDIA: assumed 2.92x "
              "the Mac until torch_train.py results from the Windows laptops arrive.",
    "slowdowns": "published: GREYHOUND (USENIX ATC '25) - production fail-slows last from <1 min to "
                 "~10 h, compute fail-slows average ~10 min; durations heavy-tailed. Replayed from "
                 "scenarios.make_scenario (log-normal durations), time-compressed for the demo.",
    "crashes": "published: Llama 3 paper (Meta, 2024) - 419 unexpected interruptions in 54 days of "
               "pre-training, most from hardware. Demo crash count/timing from scenarios.make_scenario, "
               "time-compressed; a crash is a real SIGKILL of the node process.",
    "network": "assumed: the healthy link adds 0.05 s per step, the same share of a step as the "
               "all-reduce time in the agent's training scenarios (scenarios.py realistic comm_s 0.03-0.08). "
               "Measured Wi-Fi (calibration.json: 5 MB/s assumed, 20 MB/step measured) would make it ~4 s; "
               "gloo_test.py will measure it. Network faults cut bandwidth by 50-95% (scenarios.py).",
    "workload": "input, not a measurement: the global batch grows in random jumps, like more work "
                "being submitted.",
    "pacing": "assumed: demo pacing matched to the agent's training scenarios (simulator 'realistic' "
              "level): one sequence takes 0.05 s on an 'NVIDIA' node, so a full batch on one NVIDIA node "
              "takes about as long as in the simulator (1 batch / 3.5 per s). Real compute runs; the node "
              "sleeps for the rest of its emulated time. Ratios between nodes come from 'speeds'.",
    "compensate": "assumed: lighter kernel = x1.25 speed (simulator.COMPENSATE_SPEEDUP).",
    "reboot": "assumed: a crashed laptop is back 15-40 s later (reboot + restart the program).",
}


@dataclass
class NodeSpec:
    id: str
    label: str
    speed_factor: float   # 1.0 = fastest node
    cap: int              # sequences per micro-batch
    source: str


@dataclass
class DemoConfig:
    nodes: List[NodeSpec]
    b_start: int = 6
    b_max: int = 48
    b_step: Tuple[int, int] = (2, 6)
    b_every_s: Tuple[float, float] = (20.0, 40.0)
    reboot_s: Tuple[float, float] = (15.0, 40.0)
    scenario_level: str = "realistic"
    scenario_seed: int = 3
    scenario_steps: int = 1000
    sec_per_seq: float = 0.05     # emulated compute per sequence at speed_factor 1.0, see CITATIONS["pacing"]
    link_s: float = 0.05          # emulated healthy link time per step, see CITATIONS["network"]
    ctx: int = 64
    lr: float = 3e-3
    data_seed: int = 0
    citations: Dict[str, str] = field(default_factory=lambda: dict(CITATIONS))

    def node(self, nid: str) -> NodeSpec:
        return next(n for n in self.nodes if n.id == nid)


def load_config(calibration_path: str = DEFAULT_CALIBRATION) -> DemoConfig:
    """Speed factors from calibration.json (speeds / fastest); defaults if the file is missing."""
    speeds, sources = None, {}
    if os.path.exists(calibration_path):
        with open(calibration_path) as f:
            cal = json.load(f)
        speeds, sources = cal.get("speeds"), cal.get("sources", {})
    nodes = []
    for nid in NODE_IDS:
        if speeds and nid in speeds:
            factor = speeds[nid] / max(speeds.values())
            src = sources.get(f"speed:{nid}", "assumed: calibration.json without a source")
        else:
            factor = FALLBACK_FACTORS[nid]
            src = "assumed: no calibration.json; NVIDIA = 2.92 x Mac (old simulator ratio)"
        nodes.append(NodeSpec(nid, LABELS[nid], round(factor, 4), CAPS[nid], src))
    return DemoConfig(nodes=nodes)
