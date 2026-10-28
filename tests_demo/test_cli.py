from rich.console import Console

from demo.cli import parse, request_for, render_status

SNAP = {
    "t": 0, "epoch": 3, "group": "running", "members": ["win-gpu-0", "mac-m2"], "step": 120, "B": 12,
    "version": 5, "agent": "rl", "chaos": True, "auto_load": True, "level": "realistic", "scenario_seed": 3,
    "chaos_left": 4, "uptime": 60.0, "reboot_in": {"win-gpu-1": 12.0},
    "nodes": [
        {"id": "mac-m2", "label": "Apple M2 Mac", "status": "training", "share": 0.25, "count": 3, "cap": 8,
         "speed_ratio": 0.98, "bandwidth": 1.0, "compensated": False, "throttle": 1.0, "bw_factor": 1.0,
         "hb_age": 0.3, "rank": 1, "pid": 11},
        {"id": "win-gpu-0", "label": "Windows 1", "status": "training", "share": 0.75, "count": 9, "cap": 16,
         "speed_ratio": 1.0, "bandwidth": 1.0, "compensated": False, "throttle": 1.0, "bw_factor": 1.0,
         "hb_age": 0.2, "rank": 0, "pid": 12},
        {"id": "win-gpu-1", "label": "Windows 2", "status": "dead", "share": 0.0, "count": 0, "cap": 16,
         "speed_ratio": None, "bandwidth": 1.0, "compensated": False, "throttle": 1.0, "bw_factor": 1.0,
         "hb_age": 7.5, "rank": None, "pid": 13}],
    "history": [{"t": 1, "step": 119, "loss": 2.31, "B": 12, "wall_s": 0.4, "samples_s": 30.0, "members": 2}],
    "events": [{"t": 1, "kind": "decision", "text": "agent (rl): remove win-gpu-1", "node": "win-gpu-1"}],
    "decisions": [],
}


def test_commands_map_to_api_calls():
    assert request_for(parse(["crash", "win-gpu-1"])) == ("POST", "/api/fault", {"node": "win-gpu-1", "kind": "crash"})
    assert request_for(parse(["slow", "mac-m2", "0.4", "--for", "30"])) == \
        ("POST", "/api/fault", {"node": "mac-m2", "kind": "slow", "factor": 0.4, "duration": 30.0})
    assert request_for(parse(["net", "win-gpu-0", "0.2"])) == \
        ("POST", "/api/fault", {"node": "win-gpu-0", "kind": "net", "factor": 0.2, "duration": None})
    assert request_for(parse(["heal", "mac-m2"])) == ("POST", "/api/fault", {"node": "mac-m2", "kind": "heal"})
    assert request_for(parse(["load", "+10"])) == ("POST", "/api/load", {"delta": 10})
    assert request_for(parse(["load", "-4"])) == ("POST", "/api/load", {"delta": -4})
    assert request_for(parse(["load", "auto"])) == ("POST", "/api/load", {"auto": True})
    assert request_for(parse(["load", "manual"])) == ("POST", "/api/load", {"auto": False})
    assert request_for(parse(["chaos", "on", "--level", "hard", "--seed", "7"])) == \
        ("POST", "/api/chaos", {"on": True, "level": "hard", "seed": 7})
    assert request_for(parse(["chaos", "off"])) == ("POST", "/api/chaos", {"on": False})
    assert request_for(parse(["agent", "rule"])) == ("POST", "/api/agent", {"mode": "rule"})
    assert request_for(parse(["status"])) == ("GET", "/api/state", None)


def test_status_table_shows_every_node_and_the_cluster_line():
    con = Console(record=True, width=140)
    con.print(render_status(SNAP))
    out = con.export_text()
    for word in ("mac-m2", "win-gpu-0", "win-gpu-1", "training", "dead", "epoch 3", "step 120",
                 "agent rl", "75%", "restarts in 12"):
        assert word in out, word


def test_url_works_before_or_after_the_command():
    assert parse(["--url", "1.2.3.4:8000", "status"]).url == "1.2.3.4:8000"
    assert parse(["down", "--url", "1.2.3.4:8000"]).url == "1.2.3.4:8000"
