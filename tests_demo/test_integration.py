"""End-to-end on localhost with real processes: controller + 3 node processes + gloo."""
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from demo.config import load_config
from demo.controller import Controller, create_app


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(pred, timeout, every=0.25):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(every)
    return None


@pytest.fixture(scope="module")
def demo():
    cfg = load_config("calibration.json")
    cfg.reboot_s = (9999.0, 9999.0)        # tests restart nodes by hand
    port, store_port = _free_port(), _free_port()
    ctl = Controller(cfg, port=port, store_port=store_port, agent_mode="off", auto_load=False)
    server = uvicorn.Server(uvicorn.Config(create_app(ctl), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10)
    assert _wait(lambda: client.get("/api/state").status_code == 200 if _up(client) else None, 30)
    yield ctl, client
    server.should_exit = True
    th.join(15)


def _up(client):
    try:
        client.get("/api/state")
        return True
    except httpx.HTTPError:
        return False


def _state(client):
    return client.get("/api/state").json()


def test_api_rejects_bad_fault(demo):
    _, c = demo
    assert c.post("/api/fault", json={"node": "nope", "kind": "crash"}).status_code == 400
    assert c.post("/api/fault", json={"node": "mac-m2", "kind": "explode"}).status_code == 400
    assert c.post("/api/fault", json={"node": "mac-m2", "kind": "slow", "factor": 3}).status_code == 400
    assert c.post("/api/agent", json={"mode": "magic"}).status_code == 400
    assert _state(c)["epoch"] >= 0          # still alive


def test_three_nodes_train_with_identical_weights(demo):
    _, c = demo
    s = _wait(lambda: (lambda s: s if s["step"] >= 30 else None)(_state(c)), 120)
    assert s, "training did not reach step 30"
    assert len(s["members"]) == 3
    checks = [e for e in s["events"] if e["kind"] == "checksum"]
    assert checks and all(e["identical"] for e in checks)
    losses = [h["loss"] for h in s["history"] if h["loss"] is not None]
    assert losses[-1] < losses[0]


def test_kill_rank0_then_evict(demo):
    ctl, c = demo
    victim = _state(c)["members"][0]
    assert c.post("/api/fault", json={"node": victim, "kind": "crash"}).json()["ok"]
    time.sleep(1.0)
    step0 = _state(c)["step"]
    time.sleep(6.0)
    s = _state(c)
    assert s["step"] <= step0 + 1 and s["group"] == "stalled"      # agent off: training stalls
    assert c.post("/api/agent", json={"mode": "rule"}).status_code == 200
    s = _wait(lambda: (lambda s: s if len(s["members"]) == 2 and s["group"] == "running"
                       and s["step"] > step0 + 3 else None)(_state(c)), 20)
    assert s, "survivors did not resume after the eviction"
    assert victim not in s["members"]
    assert any(d["action"] == "EVICT" and d["target"] == victim for d in s["decisions"])
    test_kill_rank0_then_evict.victim = victim


def test_reboot_and_readmit(demo):
    _, c = demo
    victim = test_kill_rank0_then_evict.victim
    assert c.post("/api/fault", json={"node": victim, "kind": "heal"}).json()["ok"]
    s = _wait(lambda: (lambda s: s if len(s["members"]) == 3 and s["group"] == "running"
                       else None)(_state(c)), 60)
    assert s, "restarted node was not brought back"
    step = s["step"]
    s = _wait(lambda: (lambda s: s if any(e["kind"] == "checksum" and e["identical"] and "3 nodes" in e["text"]
                                          and int(e["text"].split()[1].rstrip(":")) > step
                                          for e in s["events"]) else None)(_state(c)), 60)
    assert s, "no identical-weights check with 3 nodes after the readmit"
