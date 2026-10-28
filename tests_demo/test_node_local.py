import datetime
import multiprocessing as mp
import os
import signal
import socket
import time

import torch
import torch.distributed as dist

from demo.node import Emulator, Inbox, Trainer, run_epoch

TEXT = "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer. " * 300


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_two_nodes_with_shares_equal_one_node_on_the_full_batch():
    one, a, b = Trainer(ctx=32, text=TEXT), Trainer(ctx=32, text=TEXT), Trainer(ctx=32, text=TEXT)
    B = 10
    for _ in range(3):
        l1, c1, _ = one.compute(B, [B], 0, cap=4)
        full = one.pack(l1, c1, B)
        la, ca, _ = a.compute(B, [3, 7], 0, cap=2)
        lb, cb, _ = b.compute(B, [3, 7], 1, cap=16)
        summed = a.pack(la, ca, B) + b.pack(lb, cb, B)
        # share-weighted sum of two slices == gradient and loss of the whole batch on one node
        assert torch.allclose(summed[:-2], full[:-2], atol=1e-6, rtol=1e-4)
        assert abs(float(summed[-2]) / B - l1) < 1e-5
        ga = a.apply(summed.clone())
        b.apply(summed.clone())
        one.apply(summed.clone())
        assert a.checksum() == b.checksum() == one.checksum()
    assert abs(ga - l1) < 1e-5


def test_state_roundtrip_makes_identical_copies():
    a, b = Trainer(ctx=32, text=TEXT), Trainer(ctx=32, text=TEXT)
    l, c, _ = a.compute(4, [4], 0, cap=4)
    a.apply(a.pack(l, c, 4))
    b.load_state(a.state())
    assert a.checksum() == b.checksum() and b.step == a.step == 1 and b.opt.t == 1


def test_emulator_makes_slow_nodes_slower():
    e = Emulator(speed_factor=0.5, sec_per_seq=0.01)
    assert abs(e.compute_wait(4, 0.0) - 0.08) < 1e-9
    e.throttle = 0.5
    assert abs(e.compute_wait(4, 0.0) - 0.16) < 1e-9
    e.compensated = True
    assert abs(e.compute_wait(4, 0.0) - 0.16 / 1.25) < 1e-9
    assert e.compute_wait(0, 0.0) == 0.0
    e.bandwidth = 0.25
    assert abs(e.link_wait() - e.link_s * 4) < 1e-9
    assert Emulator(enabled=False).compute_wait(4, 0.0) == 0.0


def _node(rank, world, port, steps, out):
    torch.set_num_threads(1)
    store = dist.TCPStore("127.0.0.1", port, is_master=False, timeout=datetime.timedelta(seconds=30))
    tr = Trainer(ctx=32, text=TEXT)
    if rank == 1:
        torch.manual_seed(123)
        tr = Trainer(ctx=32, text=TEXT)
        with torch.no_grad():
            for p in tr.model.parameters():
                p.add_(1.0)           # different start: rank 0's broadcast must overwrite it
    inbox, sent = Inbox(), []

    def send(m):
        sent.append(m)
        if m["type"] == "report" and m["step"] >= steps and rank == 0:
            with inbox.lock:
                inbox.end_epoch = 1

    msg = {"epoch": 1, "rank": rank, "world": world, "members": [f"n{i}" for i in range(world)],
           "B": 6, "shares": {f"n{i}": 1.0 for i in range(world)}, "version": 1}
    reason, step = run_epoch(tr, Emulator(enabled=False), msg, store, inbox, send, cap=4)
    losses = [m["gloss"] for m in sent if m["type"] == "report"]
    out.put((rank, reason, step, tr.checksum(), losses[0], losses[-1]))


def test_run_epoch_two_processes_stay_identical_and_learn():
    port = _free_port()
    master = dist.TCPStore("127.0.0.1", port, is_master=True, wait_for_workers=False)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    ps = [ctx.Process(target=_node, args=(r, 2, port, 30, out)) for r in range(2)]
    for p in ps:
        p.start()
    res = sorted(out.get(timeout=120) for _ in ps)
    for p in ps:
        p.join(10)
    assert [r[1] for r in res] == ["end_epoch", "end_epoch"]
    assert res[0][2] == res[1][2] >= 30
    assert res[0][3] == res[1][3]                    # identical weights
    assert res[0][5] < res[0][4]                     # loss went down
    del master


def _victim(port, out):
    _node(1, 2, port, 10**9, out)


def test_killed_member_makes_survivor_leave_with_group_failed():
    port = _free_port()
    master = dist.TCPStore("127.0.0.1", port, is_master=True, wait_for_workers=False)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    survivor = ctx.Process(target=_node, args=(0, 2, port, 10**9, out))
    victim = ctx.Process(target=_victim, args=(port, out))
    survivor.start(); victim.start()
    time.sleep(6)
    os.kill(victim.pid, signal.SIGKILL)
    t0 = time.time()
    rank, reason, *_ = out.get(timeout=30)
    assert rank == 0 and reason == "group_failed"
    assert time.time() - t0 < 12
    survivor.join(10)
    del master


def test_stacked_network_faults_never_make_a_step_look_like_a_crash():
    from demo.node import GROUP_TIMEOUT_S
    e = Emulator(link_s=0.05)
    e.bandwidth = 0.36 * 0.07 * 0.36          # overlapping chaos faults: about 1% of normal
    assert e.link_wait() < GROUP_TIMEOUT_S / 2
