# Live Cluster Demo: Design

**Date:** 2026-10-28 · **For:** the progress review (~2026-10-30) and later thesis demos
**Branch:** `feat/live-demo` (on top of `feat/realistic-sim-rl`)

## 1. Goal

A small but real distributed-training system that runs on one Mac. It has:
- three node processes that train a tiny language model together with data parallelism;
- a backend that keeps raising the workload, injects faults and lets our self-healing agent fix them live;
- a terminal CLI and a browser page, both connected to the backend.

The agent being shown is the one already trained in the simulator. It runs unchanged.

**Success looks like this:**
1. `python -m demo up` starts the backend, 3 nodes and the browser page. The loss goes down live.
2. The workload grows in random jumps, and faults happen from chaos mode or from CLI/UI commands.
3. With the RL agent on, a killed node is removed, survivors keep training, and the node is brought back after it "reboots", all without a human. With the agent off, the same crash stalls training.
4. Every node's weights are provably identical: a checksum is compared after every group change.

## 2. Real vs emulated

This table is stated in the UI ("About this demo") and in the report.

| Real | Emulated (all nodes share one Mac) |
|---|---|
| 3 OS processes; sockets between them | Hardware speed: each node has a speed factor; "mac-m2" sleeps so it is slower than the "NVIDIA" nodes. The ratio comes from `calibration.json`, else 0.6 |
| Real training: a char-level tiny GPT (~0.8M params) on Tiny Shakespeare, AdamW, loss really drops | Slowdown fault (thermal throttling): the speed factor is multiplied by 0.2–0.6 |
| Real gradient averaging: `torch.distributed` gloo `all_reduce` every step | Slow network: the node sleeps `bytes / emulated_bandwidth` before sending |
| **Crash = SIGKILL of the node process**; detected by heartbeat loss and collective timeouts | Memory limit: each node fits at most `cap` sequences at once; above that it uses gradient accumulation, which is real but slower |
| **Recovery is real**: survivors rebuild the gloo group; a returning node receives the weights and optimizer state over the network | COMPENSATE ("lighter mode"): ×1.25 speed, the same as the simulator |
| Reboot: the backend relaunches the killed process after a random 15–40 s | |

## 3. Architecture

```
 CLI (python -m demo ...) ──REST──┐        Browser ──WebSocket /ws/ui──┐
                                  ▼                                     ▼
                    CONTROLLER (FastAPI + uvicorn, one process)
                    · TCPStore master for gloo rendezvous (survives node deaths)
                    · membership / epochs / shares / global batch B
                    · workload generator · fault injector · chaos mode
                    · agent loop: telemetry → Observation → policy → action
                    · event log → runs/<time>.jsonl
                                  │ WebSocket /ws/node/{id}: heartbeats, step reports ↑, commands ↓
              ┌───────────────────┼───────────────────┐
           node mac-m2        node win-gpu-0       node win-gpu-1
              └──── gloo all_reduce of gradients (one group per epoch) ────┘
```

### Files (`final_year_project/demo/`)
| File | Responsibility |
|---|---|
| `tinygpt.py` | Tiny GPT model, dataset download/cache, deterministic batch for (step, B) |
| `shares.py` | Split B into whole sequences per node by shares (largest remainder, respecting caps) |
| `node.py` | Node process: control connection, heartbeat thread, epoch join, train loop, fault emulation |
| `controller.py` | FastAPI app: state, epochs, workload, faults, reboot, agent loop, UI broadcast, event log |
| `agent_adapter.py` | Telemetry → `simulator.Observation`; runs a `BasePolicy`; turns actions into controller operations |
| `cli.py`, `__main__.py` | `python -m demo <command>`; `up` spawns processes; other commands call the REST API |
| `static/index.html`, `app.js`, `style.css` | Frontend; no build step; Chart.js pinned from cdnjs |
| `tests/` | Unit and localhost integration tests |

## 4. How training works

- **Epoch:** a period with fixed membership. The controller assigns ranks (ordered member list), the world size and the store prefix `e<epoch>`. Nodes call `init_process_group("gloo", store=PrefixStore("e<k>", TCPStore(controller_host, port)), rank, world_size, timeout=5 s)`.
- **Epoch start:** rank 0 broadcasts the model weights, the optimizer state and the step counter. The controller always puts a member that was already training at rank 0; on the first epoch any member can be rank 0. Every member then checksums its weights and reports them, and the controller checks they are equal.
- **Each step k:**
  1. Rank 0 broadcasts the config `[config_version, B, counts per rank]`. Config changes from the controller (shares, B) reach rank 0 only, so all ranks switch at the same step.
  2. Each node takes its slice of the global batch for step k: deterministic, `(seed, k)` → B sequences, sliced by counts.
  3. Forward/backward on its slice, with the loss as the mean over its own slice. Gradients are scaled by `count_i / B`.
  4. **Emulation:** sleep `compute_time × (1/speed − 1)`, then sleep `grad_bytes / emulated_bandwidth` (zero at full bandwidth).
  5. `all_reduce(SUM)` on one flat gradient buffer.
  6. Gradient-norm clip, then AdamW step. The maths is identical on every node, so the copies stay identical.
  7. Report to the controller, without blocking: step, loss, count, compute s, emulation s, comm s, wall s.
- **A count of 0** (B is small): the node still joins the `all_reduce` with zeros.
- **Over the memory limit:** if `count_i > cap_i`, the node splits its slice into micro-batches of at most `cap_i` and accumulates gradients. The result is the same, it just takes longer, so a growing workload makes imbalance hurt more.

## 5. Faults, detection, recovery

| Fault | Injection | Detection |
|---|---|---|
| crash | controller sends SIGKILL to the pid | heartbeat age > 3 s (heartbeats come from their own thread every 0.5 s); survivors' collective times out |
| slow | command `throttle(factor, duration)` | measured speed falls (step reports) |
| net | command `bandwidth(factor, duration)` | comm time rises |

- **No automatic healing.** Like the simulator, a dead member stalls the group until the policy chooses EVICT. With the agent off, training stays stalled, which is the point of the comparison.
- **EVICT(i)** starts a new epoch without i: the survivors drop the failed group and re-init. An evicted node that is alive stays idle but keeps sending heartbeats.
- **REBALANCE** sends new shares from measured speeds (`simulator.proportional_shares`, with caps) to rank 0.
- **COMPENSATE(i)** sends `compensate` to node i (×1.25), then rebalances.
- **READMIT(i)** starts a new epoch with i; rank 0 sends i the state.
- **Reboot:** 15–40 s after a crash, the controller relaunches the process. It connects and sends heartbeats; the policy sees it as readmittable.
- **Collective failure:** a node that gets an exception from a collective discards that step (no optimizer update happened), reports `group_failed`, then waits for the next epoch. The broadcast from rank 0 at the next epoch start makes any partial divergence impossible.

## 6. Workload generator

- B starts at 6 sequences of 64 tokens.
- Every 20–40 s, B grows by a random 2–6 when auto is on, up to a maximum of 48.
- Caps (sequences per micro-batch): mac-m2 8, NVIDIA nodes 16 each, set in the config. As B grows, nodes need more micro-batches, so a badly split batch costs more and more.
- `demo load +N` sets it by hand.

## 7. Agent loop

- **When it decides:** after each finished training step, or every 1 s while training is stalled (the simulator's stall tick).
- **Observation per node:**
  - `normal_speed`: batches/s from the healthy warm-up, the median of steps 5–15;
  - `measured_speed`: EMA of `count / (compute + emulation s) / B`;
  - `alive_signal_age`;
  - `share`, `evicted`, `compensated`; `max_share = 1.0`, because caps are soft and handled by accumulation;
  - `bandwidth` = baseline comm s / current comm s (capped at 1);
  - `price` and `code_version` are constants.
- **Global:** `comm_s` = median measured all-reduce s at full bandwidth.
- **Policies:**
  - `rl`: `rl_agent.load_policy`, model chosen by flag; default `models/v2_seed1.zip`;
  - `rule`: `simulator.ReactivePolicy`;
  - `off`: always WAIT.
- **Switching:** policies can be swapped live. Each decision is logged with its inputs.
- **Safety net:** a policy exception counts as WAIT, and it's logged.

## 8. Interfaces

- **REST:**
  - `GET /api/state`
  - `POST /api/load {delta | auto: bool}`
  - `POST /api/fault {node, kind: crash|slow|net|heal, factor?, duration?}`
  - `POST /api/chaos {on}`
  - `POST /api/agent {mode}`
  - `POST /api/shutdown`
- **WS `/ws/ui`:** a state snapshot 4×/s (nodes, epoch, B, step, loss and throughput series, agent mode) plus events as they happen.
- **CLI:**
  - `up [--agent rl|rule|off] [--chaos] [--no-browser]`, `down`
  - `status`, `watch` (Rich live table)
  - `load +N|auto|manual`
  - `crash <node>`, `slow <node> <factor> [--for s]`, `net <node> <factor> [--for s]`, `heal <node>`
  - `chaos on|off`, `agent rl|rule|off`
- **Frontend (localhost:8000):**
  - 3 node cards: status, emulated hardware label, speed bar, share, count/cap, compensated flag;
  - a header with epoch, step, B, agent mode and chaos;
  - charts: loss, samples/s (goodput), and B over time, with fault markers;
  - an agent decisions timeline and an event log;
  - controls: the same as the CLI;
  - an "About this demo" panel with the real-vs-emulated table.

## 9. Testing

- **Unit:** share split (sums to B, respects caps, deterministic); deterministic batch; observation builder from synthetic telemetry; action → operation mapping; workload generator bounds.
- **Integration, localhost and real processes:**
  - 3 nodes run 20 steps: identical checksums, and the loss at step 20 is below step 1.
  - Kill a node: survivors stall; `EVICT` → training resumes within 10 s.
  - Relaunch, then `READMIT`: 3 members, identical checksums.
  - Run with `--agent rule` and chaos for 60 s: no crash of the controller, and the event log is valid JSONL.
- **Accumulation equivalence:** count 12 with cap 4 gives the same gradient as count 12 with cap 16, up to 1e-6.
- **Spike first:** rebuild a gloo group after SIGKILL of one member. If `PrefixStore` re-init after a failed collective doesn't work in-process, survivors restart their own process and reload state from rank 0 (the fallback). The fallback design adds about 2 s per recovery.

## 10. Out of scope

- An MLX node (the slot is `node.py`'s `Trainer` interface; it comes from the data-parallel bridge work).
- Running across the real laptops (the same code, a different host flag).
- Pipeline parallelism, DiLoCo, checkpoint-to-disk (in DP, recovery copies the state from survivors), and controller fault tolerance (if the controller dies, the demo ends).
