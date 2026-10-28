from demo.config import load_config
from demo.state import ClusterState

IDS = ["mac-m2", "win-gpu-0", "win-gpu-1"]


def _state():
    return ClusterState(load_config("calibration.json"))


def _up(s, now=0.0, ids=IDS, has_state=False):
    for nid in ids:
        s.on_hello(nid, pid=100 + IDS.index(nid), has_state=has_state, step=-1, now=now)
        s.on_heartbeat(nid, now)


def _join_all(s, msgs, now):
    for nid, m in msgs:
        if m["type"] == "epoch":
            s.on_joined(nid, m["epoch"], checksum="abc", step=0, now=now)


def _report(s, nid, epoch, step, now, sps=0.05, count=2, loss=4.0, net=0.05, comm=0.06):
    s.on_report(nid, {"epoch": epoch, "step": step, "loss": loss, "count": count,
                      "sec_per_seq": sps, "compute_s": sps * count, "emu_s": 0.0,
                      "net_s": net, "comm_s": comm, "wall_s": sps * count + comm, "B": 6}, now)


def test_state_snapshot_without_nodes():
    snap = _state().snapshot(now=0.0)
    assert [n["id"] for n in snap["nodes"]] == IDS
    assert all(n["status"] == "starting" for n in snap["nodes"])
    assert snap["epoch"] == 0 and snap["group"] == "none"


def test_first_epoch_forms_when_all_alive():
    s = _state()
    _up(s, ids=IDS[:2])
    assert s.maybe_form_epoch(now=0.1) == []
    _up(s, ids=IDS[2:])
    msgs = s.maybe_form_epoch(now=0.2)
    epochs = {nid: m for nid, m in msgs if m["type"] == "epoch"}
    assert set(epochs) == set(IDS)
    assert sorted(m["rank"] for m in epochs.values()) == [0, 1, 2]
    assert all(m["world"] == 3 and m["epoch"] == 1 for m in epochs.values())
    assert s.group == "forming"
    _join_all(s, msgs, 0.3)
    assert s.group == "running"


def test_rank0_is_a_node_that_has_state():
    s = _state()
    _up(s, ids=["mac-m2"], has_state=False)
    _up(s, ids=["win-gpu-0", "win-gpu-1"], has_state=True)
    s.nodes["win-gpu-1"].held_step = 50
    s.nodes["win-gpu-0"].held_step = 40
    msgs = dict(s.maybe_form_epoch(now=0.0))
    assert msgs["win-gpu-1"]["rank"] == 0


def test_evict_alive_node_ends_epoch_via_rank0_then_reforms_without_it():
    s = _state()
    _up(s)
    msgs = s.maybe_form_epoch(0.0)
    _join_all(s, msgs, 0.1)
    rank0 = next(nid for nid, m in msgs if m["rank"] == 0)
    out = s.evict("win-gpu-1", now=1.0)
    assert out == [(rank0, {"type": "end_epoch", "epoch": 1})]
    assert s.group == "stopping"
    for nid in IDS:
        s.on_left(nid, epoch=1, reason="end_epoch", step=10, now=1.2)
    msgs2 = dict(s.maybe_form_epoch(1.3))
    assert set(msgs2) == {"mac-m2", "win-gpu-0"}
    assert msgs2["mac-m2"]["world"] == 2 and msgs2["mac-m2"]["epoch"] == 2
    assert s.nodes["win-gpu-1"].evicted


def test_crash_stalls_until_evict_and_survivors_reform():
    s = _state()
    _up(s)
    _join_all(s, s.maybe_form_epoch(0.0), 0.1)
    s.on_disconnect("win-gpu-0", now=2.0)        # process killed
    for nid in ("mac-m2", "win-gpu-1"):
        s.on_heartbeat(nid, 6.0)
        s.on_left(nid, epoch=1, reason="group_failed", step=20, now=6.0)
    assert s.maybe_form_epoch(6.1) == []          # dead node still wanted: stall
    assert s.group == "stalled"
    s.evict("win-gpu-0", now=6.5)
    msgs = dict(s.maybe_form_epoch(6.6))
    assert set(msgs) == {"mac-m2", "win-gpu-1"}


def test_crash_then_reboot_without_evict_resumes_with_all_three():
    s = _state()
    _up(s)
    _join_all(s, s.maybe_form_epoch(0.0), 0.1)
    s.on_disconnect("win-gpu-0", now=2.0)
    for nid in ("mac-m2", "win-gpu-1"):
        s.on_left(nid, epoch=1, reason="group_failed", step=20, now=6.0)
    s.on_hello("win-gpu-0", pid=999, has_state=False, step=-1, now=20.0)
    for nid in IDS:
        s.on_heartbeat(nid, 20.0)
    msgs = dict(s.maybe_form_epoch(20.1))
    assert set(msgs) == set(IDS)
    assert msgs["win-gpu-0"]["rank"] != 0


def test_readmit_adds_node_back():
    s = _state()
    _up(s)
    _join_all(s, s.maybe_form_epoch(0.0), 0.1)
    s.evict("win-gpu-1", 1.0)
    for nid in IDS:
        s.on_left(nid, 1, "end_epoch", 10, 1.1)
    _join_all(s, s.maybe_form_epoch(1.2), 1.3)
    out = s.readmit("win-gpu-1", 2.0)
    assert out and out[0][1]["type"] == "end_epoch"
    for nid in ("mac-m2", "win-gpu-0"):
        s.on_left(nid, 2, "end_epoch", 15, 2.1)
    msgs = dict(s.maybe_form_epoch(2.2))
    assert set(msgs) == set(IDS) and not s.nodes["win-gpu-1"].evicted


def test_stale_epoch_ignored():
    s = _state()
    _up(s)
    _join_all(s, s.maybe_form_epoch(0.0), 0.1)
    _report(s, "mac-m2", 1, 7, 0.5)
    _report(s, "mac-m2", 0, 99, 0.6)
    assert s.step == 7
    s.on_left("mac-m2", epoch=0, reason="group_failed", step=99, now=0.7)
    assert s.nodes["mac-m2"].phase == "training"


def test_speed_ema_and_normal_speed_from_reports():
    s = _state()
    _up(s)
    _join_all(s, s.maybe_form_epoch(0.0), 0.1)
    for step in range(1, 16):
        _report(s, "mac-m2", 1, step, step * 0.1, sps=0.1)
    n = s.nodes["mac-m2"]
    assert abs(n.normal_sps - 0.1) < 1e-9
    for step in range(16, 30):
        _report(s, "mac-m2", 1, step, step * 0.1, sps=0.2)
    v = s.node_view("mac-m2", now=3.0)
    assert v["measured_speed"] < 0.6 * v["normal_speed"]
    assert abs(v["normal_speed"] - 1 / (6 * 0.1)) < 1e-6


def test_heartbeat_age_marks_dead():
    s = _state()
    _up(s, now=0.0)
    assert s.alive("mac-m2", 2.9)
    assert not s.alive("mac-m2", 3.1)
    assert s.snapshot(3.1)["nodes"][0]["status"] == "dead"


def test_rebalance_and_set_B_go_to_rank0_as_config():
    s = _state()
    _up(s)
    msgs = s.maybe_form_epoch(0.0)
    _join_all(s, msgs, 0.1)
    rank0 = next(nid for nid, m in msgs if m["rank"] == 0)
    out = s.rebalance({"mac-m2": .2, "win-gpu-0": .4, "win-gpu-1": .4}, now=1.0)
    assert out[0][0] == rank0 and out[0][1]["type"] == "config"
    assert out[0][1]["shares"]["mac-m2"] == .2
    v = out[0][1]["version"]
    out2 = s.set_B(10, now=2.0)
    assert out2[0][1]["B"] == 10 and out2[0][1]["version"] == v + 1
