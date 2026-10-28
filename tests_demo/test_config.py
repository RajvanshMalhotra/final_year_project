import json

from demo.config import CITATIONS, load_config


def test_factors_from_real_calibration():
    cfg = load_config("calibration.json")
    f = {n.id: n.speed_factor for n in cfg.nodes}
    assert abs(f["mac-m2"] - 0.343) < 0.01
    assert f["win-gpu-0"] == 1.0 and f["win-gpu-1"] == 1.0
    src = {n.id: n.source for n in cfg.nodes}
    assert src["mac-m2"].startswith("measured")
    assert src["win-gpu-0"].startswith("assumed")


def test_every_node_and_setting_has_a_source():
    cfg = load_config("calibration.json")
    assert all(n.source for n in cfg.nodes)
    assert {"speeds", "slowdowns", "crashes", "workload", "pacing"} <= set(CITATIONS)


def test_missing_calibration_falls_back_to_assumed(tmp_path):
    cfg = load_config(str(tmp_path / "nope.json"))
    assert all(n.source.startswith("assumed") for n in cfg.nodes)
    assert [n.id for n in cfg.nodes] == ["mac-m2", "win-gpu-0", "win-gpu-1"]


def test_calibration_with_measured_nvidia(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"speeds": {"mac-m2": 1.0, "win-gpu-0": 2.0, "win-gpu-1": 4.0},
                             "sources": {"speed:win-gpu-0": "measured: x", "speed:win-gpu-1": "measured: y",
                                         "speed:mac-m2": "measured: z"}}))
    cfg = load_config(str(p))
    f = {n.id: n.speed_factor for n in cfg.nodes}
    assert f == {"mac-m2": 0.25, "win-gpu-0": 0.5, "win-gpu-1": 1.0}
    assert all(n.source.startswith("measured") for n in cfg.nodes)
