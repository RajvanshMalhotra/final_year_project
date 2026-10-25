"""
test_calibration.py
===================
Tests for calibration.py: turning mlx_test result files (measured on our laptops)
into simulator settings, and saying which numbers were measured and which assumed.

    pytest test_calibration.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration import (Calibration, best_ring, break_even_comm_s, load_calibration,
                         min_global_batch_tokens, ring_seconds_per_batch, what_if)
from scenarios import LEVELS, make_scenario
from simulator import STALL_TICK_S, OraclePolicy, SimulatedCluster, run_episode

HOSTS = {"mac-m2": "mac", "win-gpu-0": "win0", "win-gpu-1": "win1"}


class _Results:
    """A throwaway mlx_test results/ folder."""

    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp()

    def put(self, host: str, name: str, data: dict) -> None:
        os.makedirs(os.path.join(self.dir, host), exist_ok=True)
        with open(os.path.join(self.dir, host, f"{name}.json"), "w") as f:
            json.dump({"host": host, **data}, f)

    def train(self, host: str, tokens_per_s: float, params: int = 3_000_000,
              model: str = "qwen-1.5b") -> None:
        self.put(host, "test1_train", {"ok": True, "tokens_per_s": tokens_per_s, "model": model,
                                        "trainable_params": params, "peak_memory_gb": 4.0})

    def cleanup(self) -> None:
        shutil.rmtree(self.dir)


class TestSpeeds(unittest.TestCase):
    def setUp(self) -> None:
        self.r = _Results()

    def tearDown(self) -> None:
        self.r.cleanup()

    def test_all_measured_speeds_are_batches_per_second(self):
        self.r.train("mac", 256)
        self.r.train("win0", 1024)
        self.r.train("win1", 512)
        cal = load_calibration(self.r.dir, HOSTS, global_batch_tokens=2048)
        self.assertAlmostEqual(cal.speeds["mac-m2"], 256 / 2048)
        self.assertAlmostEqual(cal.speeds["win-gpu-0"], 1024 / 2048)
        self.assertAlmostEqual(cal.speeds["win-gpu-1"], 512 / 2048)
        self.assertTrue(all(cal.sources[f"speed:{m}"].startswith("measured") for m in HOSTS))

    def test_missing_windows_uses_assumed_ratio_to_the_mac(self):
        self.r.train("mac", 256)
        cal = load_calibration(self.r.dir, HOSTS, global_batch_tokens=2048, nvidia_to_mac_ratio=3.0)
        self.assertAlmostEqual(cal.speeds["win-gpu-0"], 3.0 * 256 / 2048)
        self.assertTrue(cal.sources["speed:win-gpu-0"].startswith("assumed"))
        self.assertTrue(cal.sources["speed:mac-m2"].startswith("measured"))

    def test_one_windows_measured_is_copied_to_the_other(self):
        self.r.train("mac", 256)
        self.r.train("win0", 1024)
        cal = load_calibration(self.r.dir, HOSTS, global_batch_tokens=2048)
        self.assertAlmostEqual(cal.speeds["win-gpu-1"], 1024 / 2048)
        self.assertIn("same as win-gpu-0", cal.sources["speed:win-gpu-1"])

    def test_no_measurements_at_all_is_an_error(self):
        with self.assertRaises(ValueError):
            load_calibration(self.r.dir, HOSTS)

    def test_different_models_on_different_laptops_is_an_error(self):
        self.r.train("mac", 256, model="qwen-1.5b")
        self.r.train("win0", 1024, model="qwen-3b")
        with self.assertRaisesRegex(ValueError, "same model"):
            load_calibration(self.r.dir, HOSTS)

    def test_failed_training_run_is_reported_not_hidden(self):
        self.r.train("mac", 256)
        self.r.put("win0", "test1_train", {"ok": False, "error": "OutOfMemory"})
        cal = load_calibration(self.r.dir, HOSTS)
        self.assertFalse(cal.fits["win-gpu-0"])
        self.assertIn("OutOfMemory", cal.sources["speed:win-gpu-0"])
        with self.assertRaises(ValueError):
            cal.cluster_kwargs()


class TestComm(unittest.TestCase):
    def setUp(self) -> None:
        self.r = _Results()
        self.r.train("mac", 256, params=2 * 2**20 // 4)      # LoRA grads = 2 MiB in fp32

    def tearDown(self) -> None:
        self.r.cleanup()

    def test_gradient_size_from_trainable_params(self):
        cal = load_calibration(self.r.dir, HOSTS)
        self.assertAlmostEqual(cal.grad_mb, 2.0)

    def test_ring_all_reduce_timings_are_preferred(self):
        # t = 0.01 s + size / 10 MB/s, measured with 3 machines.
        self.r.put("mac", "test3_ring_rank0", {"ok": True, "world_size": 3, "all_sum": {
            f"{mb}MB": {"seconds": 0.01 + mb / 10} for mb in (1, 10, 50)}})
        self.r.put("mac", "net_bench", {"MB_per_s": 1000.0, "rtt_ms_median": 0.1})
        cal = load_calibration(self.r.dir, HOSTS)
        # The simulator multiplies comm_s by 2(n-1)/n = 4/3 for 3 machines.
        self.assertAlmostEqual(cal.comm_s * 4 / 3, 0.01 + 2.0 / 10, places=6)
        self.assertIn("test3_ring", cal.sources["comm_s"])

    def test_net_bench_when_no_ring_test(self):
        self.r.put("win0", "net_bench", {"MB_per_s": 4.0, "rtt_ms_median": 10.0})
        cal = load_calibration(self.r.dir, HOSTS)
        # Ring all-reduce over 3 machines: sends 4/3 of the gradient, and 4 one-way hops (2 round trips).
        self.assertAlmostEqual(cal.comm_s * 4 / 3, (4 / 3) * 2.0 / 4.0 + 2 * 0.010, places=6)
        self.assertIn("net_bench", cal.sources["comm_s"])

    def test_assumed_wifi_when_nothing_measured(self):
        cal = load_calibration(self.r.dir, HOSTS, assumed_link_mb_s=5.0, assumed_rtt_ms=0.0)
        self.assertAlmostEqual(cal.comm_s, 2.0 / 5.0)
        self.assertTrue(cal.sources["comm_s"].startswith("assumed"))


class TestSimulatorUsesIt(unittest.TestCase):
    def setUp(self) -> None:
        self.cal = Calibration(speeds={"mac-m2": 0.1, "win-gpu-0": 0.4, "win-gpu-1": 0.4},
                               comm_s=0.6, grad_mb=1.0, global_batch_tokens=2048,
                               fits={m: True for m in HOSTS}, sources={})

    def test_cluster_takes_measured_speeds(self):
        c = SimulatedCluster(**self.cal.cluster_kwargs())
        self.assertEqual({m: s.normal_speed for m, s in c.machines.items()}, self.cal.speeds)
        self.assertEqual(c.comm_s, 0.6)

    def test_default_cluster_unchanged(self):
        c = SimulatedCluster()
        self.assertEqual(c.machines["mac-m2"].normal_speed, 1.2)
        self.assertEqual(c.machines["win-gpu-0"].normal_speed, 3.5)

    def test_scenario_keeps_level_stress_on_top_of_measured_comm(self):
        base = LEVELS["realistic"].comm_s
        for d in ("realistic", "hard", "chaos"):
            sc = make_scenario(3, d, calibration=self.cal)
            c = sc.make_cluster()
            self.assertAlmostEqual(c.comm_s, 0.6 * LEVELS[d].comm_s / base)
            self.assertAlmostEqual(c.machines["win-gpu-0"].normal_speed, 0.4)
            self.assertEqual(c.machines["mac-m2"].max_share, LEVELS[d].mac_cap)

    def test_scenario_fault_schedule_does_not_depend_on_calibration(self):
        a = make_scenario(7, "hard").events
        b = make_scenario(7, "hard", calibration=self.cal).events
        self.assertEqual(a, b)

    def test_round_trip_json(self):
        path = os.path.join(tempfile.mkdtemp(), "cal.json")
        self.cal.save(path)
        self.assertEqual(Calibration.load(path), self.cal)


class TestStallCannotBeCheaperThanTraining(unittest.TestCase):
    """Regression: episodes are a fixed number of ticks and goodput = progress / time.
    If a stalled tick (1 s) is shorter than a training step (3-8 s on real laptops),
    freezing the ring 'ends' the episode cheaply and scores better than training
    on. The calibrated Oracle did exactly that instead of evicting a dead machine."""

    def setUp(self) -> None:
        self.cal = Calibration(speeds={"mac-m2": 0.05, "win-gpu-0": 0.15, "win-gpu-1": 0.15},
                               comm_s=0.0, grad_mb=20.0, global_batch_tokens=2048,
                               fits={m: True for m in HOSTS}, sources={})

    def test_stall_tick_lasts_at_least_one_healthy_step(self):
        c = SimulatedCluster(**self.cal.cluster_kwargs())
        self.assertAlmostEqual(c.stall_tick_s, c.healthy_step_time())
        self.assertGreater(c.stall_tick_s, STALL_TICK_S)

    def test_default_cluster_keeps_one_second_stall(self):
        self.assertEqual(SimulatedCluster().stall_tick_s, STALL_TICK_S)

    def test_oracle_evicts_permanently_dead_machine(self):
        sc = make_scenario(0, "easy", calibration=self.cal)   # win-gpu-0 dies for good at tick 532
        c = sc.make_cluster()
        run_episode(c, OraclePolicy(sc.schedule), sc.schedule, 1000)
        self.assertTrue(c.machines["win-gpu-0"].evicted)
        self.assertGreater(c.train_step, 900)


class TestDoesTheRingPayOff(unittest.TestCase):
    """With slow Wi-Fi, all-reducing the gradients can cost more than extra laptops save."""

    def setUp(self) -> None:
        self.cal = Calibration(speeds={"mac-m2": 0.1, "win-gpu-0": 0.4, "win-gpu-1": 0.4},
                               comm_s=0.6, grad_mb=1.0, global_batch_tokens=2048,
                               fits={m: True for m in HOSTS}, sources={})
        self.all3 = ["mac-m2", "win-gpu-0", "win-gpu-1"]

    def test_seconds_per_batch_per_ring(self):
        self.assertAlmostEqual(ring_seconds_per_batch(self.cal, self.all3), 1 / 0.9 + 0.6 * 4 / 3)
        self.assertAlmostEqual(ring_seconds_per_batch(self.cal, ["win-gpu-0"]), 2.5)

    def test_best_ring_can_drop_the_mac(self):
        ring, sec = best_ring(self.cal)
        self.assertEqual(sorted(ring), ["win-gpu-0", "win-gpu-1"])
        self.assertAlmostEqual(sec, 1 / 0.8 + 0.6)

    def test_break_even_comm_makes_full_ring_equal_to_best_single(self):
        c = break_even_comm_s(self.cal)
        tied = Calibration(**{**self.cal.__dict__, "comm_s": c})
        self.assertAlmostEqual(ring_seconds_per_batch(tied, self.all3), 2.5)

    def test_min_batch_makes_full_ring_equal_to_best_single(self):
        b = min_global_batch_tokens(self.cal)
        scale = self.cal.global_batch_tokens / b
        tied = Calibration(**{**self.cal.__dict__, "global_batch_tokens": b,
                              "speeds": {m: s * scale for m, s in self.cal.speeds.items()}})
        self.assertAlmostEqual(ring_seconds_per_batch(tied, self.all3),
                               ring_seconds_per_batch(tied, ["win-gpu-0"]))

    def test_what_if_grid_picks_best_ring_per_cell(self):
        grid = what_if(self.cal, ratios=(4.0, 100.0), links_mb_s=(0.1, 1000.0), rtt_ms=0.0)
        # Much faster NVIDIA on a slow link: train alone. Similar speeds on a fast link: all 3.
        self.assertEqual(grid[(100.0, 0.1)][0], ["win-gpu-0"])
        self.assertEqual(sorted(grid[(4.0, 1000.0)][0]), sorted(self.all3))


if __name__ == "__main__":
    unittest.main()
