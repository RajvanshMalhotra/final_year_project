# Live demo: three training nodes and a self-healing agent

Three separate programs (nodes) train a small language model together. Every step they average their updates over the network with PyTorch's gloo all-reduce. A backend (the controller) grows the workload, injects faults and lets our trained agent fix the cluster live. You watch it in a web page or the terminal.

## Run it (one Mac)

```bash
cd final_year_project
.venv/bin/pip install -r requirements.txt fastapi "uvicorn[standard]" websockets rich httpx torch
.venv/bin/python -m demo up --agent rl --chaos --level hard
```

`up` starts the controller and three node processes, then opens http://127.0.0.1:8000.

| Command | What it does |
|---|---|
| `python -m demo status` | One table of the cluster |
| `python -m demo watch` | The same table, live (Ctrl+C stops) |
| `python -m demo crash win-gpu-1` | Kills that node's process (a real crash) |
| `python -m demo slow mac-m2 0.4 --for 60` | That node runs at 40% speed for 60 s |
| `python -m demo net win-gpu-0 0.2` | That node's link runs at 20% until healed |
| `python -m demo heal win-gpu-1` | Clears faults; restarts the process if it is down |
| `python -m demo load +8` / `-4` / `auto` / `manual` | Changes the workload (sequences per step) |
| `python -m demo chaos on --level hard --seed 3` / `off` | Replays a simulator fault scenario |
| `python -m demo agent rl` / `rule` / `off` | Who fixes faults: RL agent, simple rule, or nobody |
| `python -m demo down` | Stops everything |

The web page has the same controls: a Crash / Slow down / Slow link / Heal button on each machine.

## What happens when a node crashes

1. The node's process is killed. Its heartbeats (every 0.5 s) stop.
2. The other nodes' all-reduce times out after 5 s. They leave the group and wait.
3. After 15 to 40 s the controller restarts the killed process, like a laptop rebooting.
4. What happens next depends on who fixes faults:
   - **RL agent / simple rule:** once the heartbeat is more than 3 s old, it removes the node. The survivors form a new group (a new "epoch"), one of them sends the model and optimizer state, every node reports a checksum, and training continues. When the node is back and healthy, it is brought back the same way.
   - **Nobody:** training stays stalled until the node returns. A permanent crash stalls it for good.
5. Every 25 steps all nodes report a checksum of their weights. The page and the log show whether they are identical.

## Real vs emulated

| Real | Emulated (all three nodes share one Mac) |
|---|---|
| 3 OS processes, network sockets between them | Speed differences: the "Mac" node pauses so it runs at 0.34x an "NVIDIA" node (`calibration.json`: Mac measured; NVIDIA assumed until the Windows tests run) |
| Training a 0.83M-parameter GPT on Tiny Shakespeare; the loss is real | Slowdowns: the node pauses longer each step (durations from `scenarios.py`, shaped on GREYHOUND, USENIX ATC 2025) |
| Gradient averaging with gloo every step | Slow link: the node waits before sending |
| Crash = SIGKILL; detection from heartbeats and timeouts | Pace: 0.05 s per sequence and 0.05 s of link time on a healthy "NVIDIA" node, matching the agent's training scenarios |
| Recovery: new group, state sent over the network, checksums | Lighter mode: x1.25 speed, as in the simulator |
| The agent: the trained model, unchanged, fed live measurements | |

Every emulated value and its source is in `demo/config.py` (`CITATIONS`), and the page's "About this demo" section shows them.

## On the three real laptops

On the Mac:

```bash
python -m demo up --no-spawn --agent rl
```

On each Windows laptop (PyTorch installed, firewall open for Python):

```bash
python -m demo node --id win-gpu-0 --controller <Mac IP>:8000 --no-emulate
```

With `--no-emulate` nothing is emulated: speed, Wi-Fi and faults are whatever really happens. A crash from the page or CLI tells the node to exit. Real faults work too: close a lid, unplug the charger, start a heavy program. The Mac also runs its own node: `python -m demo node --id mac-m2 --controller 127.0.0.1:8000 --no-emulate`. (This node uses PyTorch on CPU; the MLX node comes with the cross-framework work.)

## Compare the agents on the same faults

```bash
python -m demo.compare --level hard --seed 3 --seconds 300 --modes rl rule off
```

This runs the same scenario three times, with a fresh cluster each time. It writes `demo/runs/compare-<time>.json` and prints a table of the steps done, sequences trained, seconds stalled, decisions taken and final loss.

## Files

| File | What it does |
|---|---|
| `tinygpt.py` | The model, the text, the same batches on every node, gradient accumulation, checksums |
| `shares.py` | Splits each batch into whole sequences per node |
| `config.py` | Every emulated value, with its source |
| `state.py` | The controller's logic: epochs, who is in the group, stalls, speeds (no network code; unit-tested) |
| `agent_adapter.py` | Live measurements in, the simulator's `Observation` out; runs the agent; carries out its action |
| `chaos.py` | Workload growth; replays a simulator scenario at the same training steps |
| `node.py` | One node: training, the shared optimizer, emulation, the epoch loop, heartbeats |
| `controller.py` | The backend: FastAPI, node connections, the agent loop, REST and WebSocket, the event log |
| `cli.py` | The `python -m demo ...` commands |
| `static/` | The web page |
| `compare.py` | Same faults, different agents |

Tests: `pytest tests_demo -q` (53 tests, including end-to-end runs with real processes).

## Things to know

- **The agent waits 15 steps at the start** while it learns each node's normal speed. Before that every speed reads 0.
- **Lighter mode only runs on a node that is really slow** (below 85% of its normal speed), and a re-split only happens when the shares are off by more than 2%. Both rules match how the simulator behaved when the agent was trained.
- **The controller is a single point of failure.** If it stops, the demo stops.
