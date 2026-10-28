# Live Cluster Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three real node processes train a tiny GPT with data parallelism, and a FastAPI controller grows the workload, injects faults and lets the trained RL agent heal the cluster live. The demo is shown through a CLI and a browser page, and it can run on one Mac or across the 3 laptops.

**Architecture:**
- **Controller** (one process): the gloo TCPStore master, a pure-logic `ClusterState`, the agent loop, REST/WebSocket for the CLI and UI, and an event log.
- **Nodes** connect to the controller over WebSocket for control. They train in gloo groups, one group per "epoch" of fixed membership. Rank 0 broadcasts the config each step, so every node switches settings at the same step.
- **Faults:** a crash is a real SIGKILL. Slowdowns and network faults follow `scenarios.make_scenario` schedules, replayed by training step, so the same scenario can run in the simulator and in the real system.

**Tech Stack:** Python 3.13, torch 2.14 (gloo, CPU), FastAPI 0.162 + uvicorn 0.54, websockets 17, rich, httpx, Chart.js 4.4.1 from cdnjs, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-live-cluster-demo-design.md`

## Global Constraints
- Code lives in `final_year_project/demo/` on branch `feat/live-demo`. Tests go in `final_year_project/tests_demo/`; they don't collide with the existing flat `test_*.py` files.
- The agent code (`simulator.py`, `rl_agent.py`) is reused **unchanged**. The demo only builds `simulator.Observation` objects and calls `policy.decide`.
- Every emulation value comes from `demo/config.py`, with a `source` string: measured (`calibration.json`), published (paper), or assumed.
- Global batch B is never changed by the agent, only by the workload generator.
- Nodes use `torch.set_num_threads(2)` (3 nodes on an 8-core Mac).
- Gloo collective timeout is 5 s; heartbeat every 0.5 s; dead after 3 s (`simulator.CRASH_SIGNAL_AGE_S`).
- The UI and README state the real-vs-emulated table verbatim from spec §2.

## Review Focus
1. **Killing the rank-0 node mid-step:** survivors must still time out, leave the group and wait. Pinned by the integration test `test_kill_rank0_then_evict`.
2. **Two group changes in a row** (EVICT, then READMIT within seconds): there must be no stale-epoch joins. Each node ignores epoch messages older than the one it holds. Pinned by unit test `test_stale_epoch_ignored`.
3. **CLI commands with a bad node name or a bad factor:** the API returns 400 with a message, and the controller keeps running. Pinned by `test_api_rejects_bad_fault`.
4. **The policy raising an exception or returning an invalid action** (READMIT of a node that isn't evicted): it is treated as WAIT and logged. Pinned by `test_adapter_invalid_action_is_wait`.
5. **The browser connecting before any node joins:** the state snapshot still renders, with nodes shown as "starting". Pinned by `test_state_snapshot_without_nodes`.

---

### Task 1: Spike: rebuild a gloo group after SIGKILL (DONE 2026-10-28)
Result: survivors raise `RuntimeError` (gloo timeout) after 5 s; `destroy_process_group()` takes about 5 ms; re-init on `PrefixStore("e1")` with world 2 takes about 20 ms; `all_reduce` works. **In-process rebuild works, so the process-restart fallback is not needed.** The spike script is in the scratchpad and not committed.

### Task 2: Tiny GPT, data, share split

**Files:** create `demo/__init__.py`, `demo/tinygpt.py`, `demo/shares.py`; test `tests_demo/test_tinygpt.py`, `tests_demo/test_shares.py`

**Interfaces (produces):**
- `tinygpt.load_text() -> str`: downloads Tiny Shakespeare once into `demo/data/`. Offline fallback: the bundled public-domain sample `demo/data/fallback.txt`.
- `tinygpt.Vocab(text)` with `.encode(str) -> LongTensor` and `.size`
- `tinygpt.TinyGPT(vocab_size, d=128, layers=4, heads=4, ctx=64)`, an `nn.Module` taking `(B,T)` and returning logits
- `tinygpt.batch_for_step(data: LongTensor, step: int, B: int, ctx: int, seed: int = 0) -> (x, y)`: the same on every node
- `tinygpt.flat_grads(model) -> Tensor` and `tinygpt.set_flat_grads(model, flat)`
- `tinygpt.checksum(model) -> str`: sha1 of the float32 params, first 12 hex characters
- `shares.split_counts(B: int, shares: dict[str, float], members: list[str]) -> dict[str, int]`: largest remainder; sums to B; deterministic tie-break by member order

**Steps:**
- [ ] Tests:
  - `split_counts(10, {"a":.5,"b":.3,"c":.2}, ["a","b","c"]) == {"a":5,"b":3,"c":2}`
  - sum == B for 200 random cases
  - a zero share gives 0
  - `batch_for_step` is identical for the same args and differs across steps
  - accumulation equivalence: the gradient of the mean loss over 12 rows equals the sum over micro-batches of 4 with grads scaled by 4/12, within 1e-6
  - the loss of a fresh model is about `ln(vocab)` ± 0.5
- [ ] Run them and watch them fail; implement; run them and watch them pass; commit `demo: tiny GPT, deterministic batches, share split`.

### Task 3: Sourced config

**Files:** create `demo/config.py`; test `tests_demo/test_config.py`

**Produces:**
- `NodeSpec(id, label, speed_factor, cap, source)` and `DemoConfig(nodes, b_start=6, b_max=48, b_step=(2,6), b_every_s=(20,40), reboot_s=(15,40), scenario_level="realistic", step_ms_target=...)`
- `load_config(calibration_path) -> DemoConfig`: speed factors = calibration speeds / the max (Mac about 0.34, labelled "measured"; NVIDIA labelled "assumed until Windows results").
- `CITATIONS` dict: the GREYHOUND slowdown durations, the Llama 3 interruption rate, and the calibration file paths.

**Steps:**
- [ ] Tests: the factors from the real `calibration.json` give mac ≈ 0.343 (±0.01) and NVIDIA 1.0; every node has a non-empty source; a missing calibration file falls back to the defaults, with source "assumed".
- [ ] Implement; tests pass; commit.

### Task 4: Pure cluster state (no network)

**Files:** create `demo/state.py`; test `tests_demo/test_state.py`

**Produces:** `ClusterState(config)` with:
- `on_hello(node_id, pid)`
- `on_heartbeat(node_id, t)`
- `on_report(node_id, report: dict)`, where report keys are `step, epoch, loss, count, compute_s, emu_s, comm_s, wall_s, compensated`
- `on_left(node_id, epoch, reason)`
- `begin_epoch(members: list[str]) -> dict[node_id, msg]`, where msg = `{"type":"epoch","epoch","rank","world","members","shares","B"}`; rank 0 is a member that already holds state, if any
- `evict(nid)` / `readmit(nid)` / `rebalance(shares)` / `compensate(nid)` / `set_B(B)`, each returning a list of `(node_id, msg)` to send
- `alive(nid, now) -> bool`
- `snapshot(now) -> dict`, the JSON for the UI and `/api/state`
- `node_view(nid)`, with `normal_speed` (median of healthy steps 5–15), EMA `measured_speed`, `bandwidth`, `share`, `evicted`, `compensated`, `heartbeat_age`

**Steps:**
- [ ] Tests:
  - `test_state_snapshot_without_nodes`
  - `begin_epoch` gives ranks 0..n−1 and puts a node that already has state at rank 0
  - `evict` gives one `end_epoch` message to the current rank 0, and the pending membership excludes the node
  - `test_stale_epoch_ignored` (a report from an old epoch doesn't change the step)
  - speed EMA and normal speed computed from synthetic reports
  - heartbeat age crossing 3 s ⇒ `alive` False
- [ ] Implement; pass; commit.

### Task 5: Agent adapter

**Files:** create `demo/agent_adapter.py`; test `tests_demo/test_agent_adapter.py`

**Produces:**
- `build_observation(state: ClusterState, now) -> simulator.Observation`: speeds in batches/s; `comm_s` = median full-bandwidth all-reduce seconds; `max_share` = 1.0
- `AgentRunner(mode: "rl"|"rule"|"off", model_path)` with `.decide(obs) -> (Action, target|None, info)`
- `apply(action, target, state) -> list[(node_id, msg)]`

**Steps:**
- [ ] Tests:
  - the observation from synthetic state has the 3 machines, and a dead node has `alive_signal_age > 3`
  - `rule` mode returns EVICT for a dead node
  - `off` always returns WAIT
  - `test_adapter_invalid_action_is_wait`
  - `rl` mode loads `models/v2_seed1.zip` and returns a valid action for a healthy observation
- [ ] Implement; pass; commit.

### Task 6: Workload and chaos schedule

**Files:** create `demo/chaos.py`; test `tests_demo/test_chaos.py`

**Produces:**
- `Workload(config, rng)` with `.tick(now) -> new B | None` (auto) and `.bump(delta)`
- `ChaosPlayer(level, seed)` with `.due(step) -> list[FaultCmd]`, built from `scenarios.make_scenario(seed, level).events`. Crash → kill + reboot after the event's duration, converted to seconds; slowdown → throttle(1 − fraction, duration in steps); network → bandwidth(1 − fraction, duration).

**Steps:**
- [ ] Tests: B never goes above `b_max` and only grows; `ChaosPlayer(…, seed=3)` returns the same commands as the scenario's events at the same steps; durations > 0.
- [ ] Implement; pass; commit.

### Task 7: Node process

**Files:** create `demo/node.py`; test `tests_demo/test_node_local.py`

**Produces:**
- `python -m demo.node --id mac-m2 --controller 127.0.0.1:8000 [--store-port 29600]`
- `Trainer` with `.step(x, y, count, cap) -> (loss, compute_s)` (accumulation), `.apply(flat)` (clip + AdamW), `.state_tensors()` / `.load_state_tensors()` for the broadcast
- the main loop: WS connect; hello; heartbeat thread; wait for epoch; init group; rank 0 broadcasts the state; per step: config broadcast → slice → compute → emulation sleeps → all_reduce → apply → report; `end_epoch` flag → leave cleanly; exception → `group_failed` → wait

**Steps:**
- [ ] Test (world 1, real gloo group via TCPStore on a free port): 30 steps lower the loss; the emulation sleep scales the wall time by 1/speed_factor (±30%); throttle 0.5 roughly doubles compute+emu.
- [ ] Implement; pass; commit.

### Task 8: Controller app + localhost integration

**Files:** create `demo/controller.py`; test `tests_demo/test_integration.py`

**Produces:**
- the FastAPI `app`, made by `create_app(config, agent_mode, chaos, spawn_nodes: bool)`
- WebSockets `/ws/node/{id}` and `/ws/ui`
- REST from spec §8
- a background loop (10 Hz): heartbeats, workload, chaos, the agent tick (after a new step or every 1 s while stalled), reboot of killed nodes, UI broadcast at 4 Hz
- the event log `demo/runs/<ts>.jsonl`
- `NodeLauncher` (`subprocess.Popen`, kill, relaunch) for single-machine mode

**Steps:**
- [ ] Integration tests, marked `slow` and using real processes:
  - start the controller with 3 spawned nodes and agent off; after 20 steps all checksums are equal and the loss at the last step is below step 1
  - `test_kill_rank0_then_evict`: kill the rank-0 node; the step doesn't advance for 6 s; POST an EVICT through the agent `rule` mode → steps advance within 10 s, with 2 members
  - reboot + READMIT (rule) → 3 members and equal checksums
  - `test_api_rejects_bad_fault`
- [ ] Implement; pass; commit.

### Task 9: CLI

**Files:** create `demo/cli.py`, `demo/__main__.py`; test `tests_demo/test_cli.py`

**Produces:**
- `python -m demo up|down|status|watch|load|crash|slow|net|heal|chaos|agent|node`
- `up` starts the controller (uvicorn, in a subprocess) and opens the browser; it writes `demo/runs/pids.json`
- `node` runs a node for 3-laptop mode

**Steps:**
- [ ] Tests: argument parsing produces the right REST calls (httpx `MockTransport`); `status` renders a Rich table from a snapshot fixture.
- [ ] Implement; pass; commit.

### Task 10: Frontend

**Files:** create `demo/static/index.html`, `demo/static/app.js`, `demo/static/style.css`

**Steps:**
- [ ] Build the page per spec §8: node cards, header, 3 charts with fault markers, decisions timeline, event log, controls, and the "About this demo" panel.
- [ ] Start `python -m demo up`, check in the browser (Chrome tools if available, otherwise a headless screenshot), fix the layout; commit.

### Task 11: End-to-end run, README, results
- [ ] Run a 5-minute session: realistic chaos with seed 3, with `rl`, `rule` and `off`. Record steps done, stall seconds and the decisions into `demo/runs/` and a summary table.
- [ ] Write `demo/README.md`: how to run, the real-vs-emulated table, sources, the 3-laptop instructions, known limits.
- [ ] Full test suite: `pytest tests_demo` plus the existing `pytest -q`. Commit.
