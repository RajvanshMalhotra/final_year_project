"""
agent_brain.py
==============
Self-Healing Mixed-GPU Training - Task C: Agent Brain
------------------------------------------------------
Days 1-7 implementation:

  Day 1-2 : MachineDetector  +  evaluate_detector()
  Day 3   : MPCFallback (shortlist + planner shared with the simulator)
  Day 4   : DecisionModel    +  train_behavioral_cloning()  on simulator Oracle data
  Day 5-6 : HybridAgentPolicy  (shortlist mask + under-70%-sure rule)
  Day 7   : Evaluation report (__main__); tests live in test_agent_brain.py

Not here yet: reinforcement learning. ClusterEnv (simulator.py) is the RL
environment; the decision model will be fine-tuned with MaskablePPO on it.

External deps
-------------
  numpy        - arrays, statistics
  scikit-learn - MLPClassifier (lightweight neural network) + calibration

Usage
-----
  python agent_brain.py          # runs evaluation report + test suite
  pytest test_agent_brain.py -v  # tests only
"""

from __future__ import annotations

import io
import math
import random
import sys
import time

# ---- Force UTF-8 stdout so Unicode characters print on Windows terminals ----
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        pass  # stdout has no buffer in some environments (e.g. IDLE) - skip
import unittest
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---- scikit-learn (required) -------------------------------------------------
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    raise ImportError("scikit-learn is required.  pip install scikit-learn")

try:  # scikit-learn >= 1.6; cv="prefit" was removed in 1.8
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - older scikit-learn
    FrozenEstimator = None

from simulator import (
    CRASH_SIGNAL_AGE_S, SLOW_RATIO_THRESHOLD,
    Action, BasePolicy, DoNothingPolicy, MachineObs, Observation, OraclePolicy,
    PolicyOutput, RandomFaultGenerator, RebalanceOnlyPolicy, SimulatedCluster,
    TimeDelayedPolicy, _unpack, is_readmittable, plan_action, plan_action_full,
    proportional_shares, seconds_per_batch,
    run_episode, speed_ratio, step_time,
)


# =============================================================================
# HELPERS
# =============================================================================

NORMAL_SPEEDS = {"mac-m2": 1.2, "win-gpu-0": 3.5, "win-gpu-1": 3.5}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =============================================================================
# DAY 1-2 - ANOMALY DETECTOR
# =============================================================================

@dataclass
class FaultDetection:
    """Result returned by MachineDetector.update()."""
    machine_id:  str
    fault_kind:  str    # "none" | "fail-stop" | "fail-slow"
    step:        int
    speed_ratio: float  # smoothed speed / normal  (0..1+)
    ema_speed:   float
    z_score:     float  # how many noise-sigmas below normal the smoothed speed is


class MachineDetector:
    """
    Per-machine streaming detector, judged against the machine's OWN normal speed.

    Fault classification
    --------------------
    fail-stop : measured_speed <= STOP_SPEED_THRESHOLD  OR
                alive_signal_age > CRASH_SIGNAL_AGE
    fail-slow : EMA-smoothed speed < SLOW_RATIO_THRESHOLD * normal for
                CONFIRM_STEPS consecutive updates (after warmup). It stays
                flagged for as long as the machine is slow.
    none      : normal operation

    The baseline is the machine's normal speed, not a moving average, so the
    detector cannot "get used to" a slowdown and go quiet while it lasts.
    """
    EMA_ALPHA:            float = 0.5    # smoothing of the noisy speed signal
    SLOW_RATIO_THRESHOLD: float = SLOW_RATIO_THRESHOLD
    CONFIRM_STEPS:        int   = 2      # consecutive slow readings before flagging
    STOP_SPEED_THRESHOLD: float = 0.05   # steps/s; below this -> stopped
    CRASH_SIGNAL_AGE:     float = CRASH_SIGNAL_AGE_S
    WARMUP_STEPS:         int   = 5      # ignore first N steps
    NOISE_FLOOR:          float = 0.01   # minimum noise std, as a fraction of normal

    def __init__(self, machine_id: str, normal_speed: float):
        self.machine_id   = machine_id
        self.normal_speed = normal_speed
        self._ema_speed:  float = normal_speed
        self._noise_var:  float = (self.NOISE_FLOOR * normal_speed) ** 2
        self._slow_run:   int   = 0
        self._step_count: int   = 0

    def update(self, step: int, measured_speed: float,
               alive_signal_age: float) -> FaultDetection:
        self._step_count += 1

        # Hard fail-stop check
        if (measured_speed <= self.STOP_SPEED_THRESHOLD
                or alive_signal_age > self.CRASH_SIGNAL_AGE):
            self._slow_run = 0
            return FaultDetection(self.machine_id, "fail-stop", step, 0.0,
                                  self._ema_speed, float("inf"))

        self._ema_speed = (self.EMA_ALPHA * measured_speed
                           + (1 - self.EMA_ALPHA) * self._ema_speed)
        ratio = self._ema_speed / self.normal_speed if self.normal_speed > 0 else 1.0
        slow  = ratio < self.SLOW_RATIO_THRESHOLD

        # Learn the noise level only from healthy readings
        if not slow:
            resid = measured_speed - self.normal_speed
            self._noise_var = max(0.9 * self._noise_var + 0.1 * resid * resid,
                                  (self.NOISE_FLOOR * self.normal_speed) ** 2)
        z_score = (self.normal_speed - self._ema_speed) / math.sqrt(self._noise_var)

        self._slow_run = self._slow_run + 1 if slow else 0
        flagged = (self._step_count > self.WARMUP_STEPS
                   and self._slow_run >= self.CONFIRM_STEPS)

        return FaultDetection(
            machine_id  = self.machine_id,
            fault_kind  = "fail-slow" if flagged else "none",
            step        = step,
            speed_ratio = ratio,
            ema_speed   = self._ema_speed,
            z_score     = z_score,
        )


class ClusterDetector:
    """One MachineDetector per machine; scans a whole Observation at once."""

    def __init__(
        self,
        machine_normals: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        normals = machine_normals or kwargs.get("normal_speeds") or {}
        self._detectors: Dict[str, MachineDetector] = {
            mid: MachineDetector(mid, ns)
            for mid, ns in normals.items()
        }

    def scan(self, obs: Observation) -> List[FaultDetection]:
        results: List[FaultDetection] = []
        for mid, mobs in obs.machines.items():
            if mid not in self._detectors:
                self._detectors[mid] = MachineDetector(mid, mobs.normal_speed)
            det = self._detectors[mid].update(
                step             = obs.step,
                measured_speed   = mobs.measured_speed,
                alive_signal_age = mobs.alive_signal_age,
            )
            if mobs.evicted:
                # Out of the ring: not a fault we have to handle (MPC checks READMIT).
                det = FaultDetection(mid, "none", obs.step, det.speed_ratio,
                                     det.ema_speed, det.z_score)
            results.append(det)
        return results


def _naive_detect(obs: Observation) -> List[Tuple[str, str]]:
    """Naive baseline: flag machine whose speed < 50% of cluster average."""
    speeds = {mid: m.measured_speed for mid, m in obs.machines.items()}
    values = [v for v in speeds.values() if v > 0]
    avg    = sum(values) / len(values) if values else 1.0
    results = []
    for mid, spd in speeds.items():
        if spd <= 0.05:
            results.append((mid, "fail-stop"))
        elif spd < 0.50 * avg:
            results.append((mid, "fail-slow"))
        else:
            results.append((mid, "none"))
    return results


# ---- Detector evaluation result ---------------------------------------------

@dataclass
class DetectorEvalResult:
    detector_name:      str
    precision:          float
    recall:             float
    f1:                 float
    avg_time_to_notice: float
    tp: int; fp: int; fn: int

    def __str__(self) -> str:
        return (
            f"  {self.detector_name:<28s}  "
            f"P={self.precision:.3f}  R={self.recall:.3f}  F1={self.f1:.3f}  "
            f"TTN={self.avg_time_to_notice:.1f} steps  "
            f"(TP={self.tp} FP={self.fp} FN={self.fn})"
        )


def evaluate_detector(
    n_machines:    int   = 3,
    total_steps:   int   = 200,
    fault_step:    int   = 60,
    slowdown_frac: float = 0.50,
    seed:          int   = 0,
) -> Tuple[DetectorEvalResult, DetectorEvalResult]:
    """
    Generate synthetic noisy traces, slow machine 0 down from fault_step on, and
    score the EMA detector vs the naive cross-machine baseline PER STEP:
      TP : fault machine flagged during its fault window
      FN : fault machine not flagged during its fault window
      FP : any machine flagged when it is not faulty
    TTN is the number of steps from the fault to the first flag.
    Returns (ema_result, naive_result).
    """
    np_rng = np.random.RandomState(seed)
    NORMAL_SPEED = 3.5
    NOISE_STD    = 0.10

    machine_ids   = [f"gpu-{i}" for i in range(n_machines)]
    fault_machine = machine_ids[0]
    detectors = {mid: MachineDetector(mid, NORMAL_SPEED) for mid in machine_ids}

    counts = {"ema": [0, 0, 0], "naive": [0, 0, 0]}          # tp, fp, fn
    first:  Dict[str, Optional[int]] = {"ema": None, "naive": None}

    def score(name: str, mid: str, flagged: bool, step: int) -> None:
        faulty = mid == fault_machine and step >= fault_step
        if faulty and flagged:
            counts[name][0] += 1
            if first[name] is None:
                first[name] = step
        elif faulty:
            counts[name][2] += 1
        elif flagged:
            counts[name][1] += 1

    for step in range(total_steps):
        machines_obs: Dict[str, MachineObs] = {}
        for mid in machine_ids:
            noise = float(np_rng.normal(0, NOISE_STD * NORMAL_SPEED))
            if mid == fault_machine and step >= fault_step:
                spd = NORMAL_SPEED * (1.0 - slowdown_frac) + noise
            else:
                spd = NORMAL_SPEED + noise
            machines_obs[mid] = MachineObs(
                alive_signal_age = 0.1,
                normal_speed     = NORMAL_SPEED,
                measured_speed   = max(0.01, spd),
                share            = 1.0 / n_machines,
                code_version     = "v1.0",
                price            = 0.50,
                is_active        = True,
            )
        obs = Observation(time=float(step) * 0.3, step=step, machines=machines_obs)

        for mid in machine_ids:
            r = detectors[mid].update(step, machines_obs[mid].measured_speed, 0.1)
            score("ema", mid, r.fault_kind != "none", step)
        for mid, kind in _naive_detect(obs):
            score("naive", mid, kind != "none", step)

    def _result(name: str, label: str) -> DetectorEvalResult:
        tp, fp, fn = counts[name]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ttn = float(first[name] - fault_step) if first[name] is not None else float("inf")
        return DetectorEvalResult(label, p, r, f1, ttn, tp, fp, fn)

    return _result("ema", "EMA Detector"), _result("naive", "Naive Cross-Machine")


# =============================================================================
# DAY 3 - DIAGNOSIS & MPC FALLBACK
# =============================================================================

@dataclass
class DiagnosedFault:
    machine_id:   str
    fault_kind:   str    # "fail-stop" | "fail-slow" | "none"
    severity:     float  # 0..1
    duration_est: int    # steps the decision must pay off over


class MPCFallback:
    """
    Model Predictive Control fallback.

    Shortlists at most 3 actions for the diagnosed fault, predicts each one's
    goodput over the horizon with simulator.plan_action (the same model the
    Oracle uses, tested against the real cluster), and returns the best.
    WAIT wins ties.

    Shortlists
    ----------
    fail-stop  -> [EVICT, WAIT]
    fail-slow  -> [WAIT, COMPENSATE, EVICT]
    none       -> [WAIT, REBALANCE, READMIT]   (REBALANCE helps after a recovery,
                                               READMIT brings back a healed machine)
    """
    HORIZON: int = 100   # steps an action must pay off over when no estimate is given

    def __init__(self, horizon: Optional[int] = None, **kwargs):
        if horizon is not None:
            self.HORIZON = horizon
        elif "H" in kwargs:
            self.HORIZON = kwargs["H"]

    def decide(
        self,
        fault: DiagnosedFault,
        obs:   Observation,
    ) -> Tuple[Action, float, Dict[str, float]]:
        """
        Returns (best_action, best_projection, all_projections).
        all_projections: {action_name: projected goodput (steps/hour)}
        """
        horizon = fault.duration_est if fault.duration_est > 0 else self.HORIZON
        projections: Dict[str, float] = {}
        best, best_val = Action.WAIT, -1.0
        for action, target in self.shortlist(fault, obs):
            val = self._project(action, target, obs, horizon)
            projections[action.name] = max(val, projections.get(action.name, -1.0))
            if val > best_val + 1e-9:
                best, best_val = action, val
        return best, best_val, projections

    def shortlist(self, fault: DiagnosedFault, obs: Observation) -> List[Tuple[Action, Optional[str]]]:
        if fault.fault_kind == "fail-stop":
            return [(Action.EVICT, fault.machine_id), (Action.WAIT, None)]
        if fault.fault_kind == "fail-slow":
            return [(Action.WAIT, None), (Action.COMPENSATE, fault.machine_id),
                    (Action.EVICT, fault.machine_id)]
        return ([(Action.WAIT, None), (Action.REBALANCE, None)]
                + [(Action.READMIT, mid) for mid, m in obs.machines.items() if is_readmittable(m)])

    @staticmethod
    def _project(action: Action, target: Optional[str], obs: Observation, horizon: int) -> float:
        """Goodput (steps/hour) over `horizon` steps if we take `action` now."""
        overhead, st, prog = plan_action_full(obs, action, target)
        spb = seconds_per_batch(st, prog)
        if not math.isfinite(spb):
            return 0.0
        return horizon / (overhead + horizon * spb) * 3600.0

    @staticmethod
    def diagnose(detections: List[FaultDetection],
                 obs: Observation,
                 duration_est: int = 120) -> DiagnosedFault:
        """Convert FaultDetections into the single worst DiagnosedFault."""
        stops = [d for d in detections if d.fault_kind == "fail-stop"]
        slows = [d for d in detections if d.fault_kind == "fail-slow"]
        if stops:
            return DiagnosedFault(stops[0].machine_id, "fail-stop", 1.0, duration_est)
        if slows:
            worst = min(slows, key=lambda d: d.speed_ratio)
            return DiagnosedFault(worst.machine_id, "fail-slow",
                                  _clamp(1.0 - worst.speed_ratio, 0.0, 1.0), duration_est)
        return DiagnosedFault("none", "none", 0.0, 0)


# =============================================================================
# DAY 4 - FEATURIZER
# =============================================================================

def featurize(obs: Observation) -> np.ndarray:
    """
    Observation -> fixed-length float32 feature vector.

    Per-machine (sorted by id): speed_ratio, signal_age/10, share, price,
                                is_active, evicted, compensated
    Global: min / max active speed ratio, n_inactive,
            imbalance (step time now / step time with ideal shares - 1),
            any machine presumed dead, any machine ready to readmit
    """
    per_machine: List[float] = []
    ratios: List[float] = []
    n_inactive = 0
    for mid in sorted(obs.machines):
        m = obs.machines[mid]
        ratio = speed_ratio(m)
        per_machine.extend([
            ratio,
            _clamp(m.alive_signal_age, 0, 10) / 10.0,
            m.share,
            m.price,
            float(m.is_active),
            float(m.evicted),
            float(m.compensated),
        ])
        if m.is_active:
            ratios.append(ratio)
        else:
            n_inactive += 1

    speeds = {mid: m.measured_speed for mid, m in obs.machines.items() if m.is_active}
    shares = {mid: m.share for mid, m in obs.machines.items() if m.is_active}
    now, ideal = step_time(speeds, shares), step_time(speeds, proportional_shares(speeds))
    imbalance = _clamp(now / ideal - 1.0, 0.0, 5.0) if math.isfinite(now) and ideal > 0 else 5.0

    any_dead = any(not m.evicted and (m.measured_speed < 0.01
                                      or m.alive_signal_age > CRASH_SIGNAL_AGE_S)
                   for m in obs.machines.values())
    any_readmit = any(is_readmittable(m) for m in obs.machines.values())

    global_feats = [
        min(ratios) if ratios else 0.0,
        max(ratios) if ratios else 0.0,
        float(n_inactive),
        imbalance,
        float(any_dead),
        float(any_readmit),
    ]
    return np.array(per_machine + global_feats, dtype=np.float32)


# =============================================================================
# DAY 4 - DECISION MODEL
# =============================================================================

def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE: bucket predictions by confidence; in each bucket compare average
    confidence with actual accuracy; average the gaps, weighted by bucket size.
    0 = confidence means exactly what it says.
    """
    conf    = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bucket  = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bucket == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


class DecisionModel:
    """
    Lightweight neural network (MLP 64x64 ReLU) trained via behavioral
    cloning to mimic the Oracle policy.

    The rows are split in order (Oracle data is grouped by episode, so this
    keeps whole episodes apart): 60% train, 20% calibrate, 20% measure ECE.
    Calibration (Platt scaling) is fitted on data the network never saw, so the
    confidence used by the under-70%-sure rule means what it says.
    Rare actions are oversampled in the training split: the Oracle WAITs on
    almost every step, and without this the network just predicts WAIT.
    """

    MIN_CLASS_FRACTION: float = 0.10   # oversample each action to >= 10% of WAIT

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (64, 64), seed: int = 42):
        self._base = MLPClassifier(
            hidden_layer_sizes  = hidden_layer_sizes,
            activation          = "relu",
            solver              = "adam",
            max_iter            = 500,
            random_state        = seed,
            learning_rate_init  = 1e-3,
            early_stopping      = True,
            validation_fraction = 0.10,
            n_iter_no_change    = 15,
        )
        self._seed = seed
        self._calibrated: Optional[CalibratedClassifierCV] = None
        self._label_encoder = LabelEncoder()
        self._is_trained = False
        self.classes_: List[str] = []
        self.calibration_ece: float = float("nan")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionModel":
        """Train on (N,F) feature matrix and (N,) string action labels."""
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = list(self._label_encoder.classes_)
        n = len(X)
        a, b = int(n * 0.6), int(n * 0.8)

        X_tr, y_tr = self._oversample(X[:a], y_enc[:a])
        self._base.fit(X_tr, y_tr)

        if FrozenEstimator is not None:
            self._calibrated = CalibratedClassifierCV(FrozenEstimator(self._base), method="sigmoid")
        else:  # pragma: no cover - older scikit-learn
            self._calibrated = CalibratedClassifierCV(self._base, method="sigmoid", cv="prefit")
        self._calibrated.fit(X[a:b], y_enc[a:b])
        self._is_trained = True

        proba   = self._calibrated.predict_proba(X[b:])
        correct = self._calibrated.classes_[np.argmax(proba, axis=1)] == y_enc[b:]
        self.calibration_ece = expected_calibration_error(proba.max(axis=1), correct)
        return self

    def _oversample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self._seed)
        counts = np.bincount(y)
        target = int(counts.max() * self.MIN_CLASS_FRACTION)
        parts_X, parts_y = [X], [y]
        for cls, c in enumerate(counts):
            if 0 < c < target:
                idx = rng.choice(np.flatnonzero(y == cls), size=target - c, replace=True)
                parts_X.append(X[idx])
                parts_y.append(y[idx])
        return np.vstack(parts_X), np.concatenate(parts_y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, C) calibrated probability matrix over self.classes_."""
        if not self._is_trained or self._calibrated is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        proba = self._calibrated.predict_proba(X)
        full  = np.zeros((len(X), len(self.classes_)))
        full[:, self._calibrated.classes_] = proba   # classes absent from the calibration split get 0
        return full

    def predict(self, X: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Returns (action_name_list, confidences_array)."""
        proba    = self.predict_proba(X)
        best_idx = np.argmax(proba, axis=1)
        confs    = proba[np.arange(len(proba)), best_idx]
        names    = [self.classes_[i] for i in best_idx]
        return names, confs

    def predict_single(self, x: np.ndarray) -> Tuple[str, float]:
        """Convenience: single (1-D) feature vector -> (action_name, confidence)."""
        names, confs = self.predict(x.reshape(1, -1))
        return names[0], float(confs[0])

    @property
    def is_trained(self) -> bool:
        return self._is_trained


# =============================================================================
# DAY 4 - BEHAVIORAL CLONING TRAINER
# =============================================================================

def train_behavioral_cloning(
    oracle_observations: List[Observation],
    oracle_actions:      List[Action],
    hidden_layer_sizes:  Tuple[int, ...] = (64, 64),
    verbose:             bool = True,
) -> DecisionModel:
    """
    Featurize oracle data and train a DecisionModel to copy the Oracle policy.

    Parameters
    ----------
    oracle_observations : list of Observation objects (keep episodes together)
    oracle_actions      : matching list of Action enums (oracle labels)
    hidden_layer_sizes  : MLP topology
    verbose             : print training stats

    Returns
    -------
    Trained DecisionModel
    """
    if len(oracle_observations) != len(oracle_actions):
        raise ValueError("Length mismatch between observations and actions.")

    if verbose:
        print(f"  [BC] Featurizing {len(oracle_observations)} samples ...")

    X = np.vstack([featurize(o) for o in oracle_observations])
    y = np.array([a.name for a in oracle_actions])

    if verbose:
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    {u:<14s}: {c:>6d} ({100*c/len(y):.1f}%)")

    model = DecisionModel(hidden_layer_sizes=hidden_layer_sizes)
    t0    = time.perf_counter()
    model.fit(X, y)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if verbose:
        print(f"  [BC] Done in {elapsed_ms:.1f} ms. Held-out ECE: {model.calibration_ece:.3f}")

    return model


# =============================================================================
# DAY 4 - ORACLE DATA FROM THE SIMULATOR
# =============================================================================

class _RecordingPolicy(BasePolicy):
    """
    Acts with a `behavior` policy and labels every state it visits with what
    the Oracle would do there.

    Following the Oracle alone never visits the states the agent really meets
    (a crash visible only after the heartbeat timeout, a slowdown nobody has
    handled, skewed shares after a recovery), because the Oracle has already
    fixed them. Realistic behaviour policies, and later the agent itself
    (DAgger), reach those states.
    """
    name = "OracleRecorder"

    def __init__(self, oracle: OraclePolicy, behavior: BasePolicy):
        self._oracle   = oracle
        self._behavior = behavior
        self.observations: List[Observation] = []
        self.labels:       List[Action]      = []

    def bind(self, cluster, schedule, total_steps) -> None:
        self._oracle.bind(cluster, schedule, total_steps)

    def decide(self, obs: Observation) -> PolicyOutput:
        self.observations.append(obs)
        self.labels.append(_unpack(self._oracle.decide(obs))[0])
        return self._behavior.decide(obs)


class _OracleMix(BasePolicy):
    """Follows the Oracle with probability `follow_prob`, otherwise WAITs."""
    name = "OracleMix"

    def __init__(self, oracle: OraclePolicy, follow_prob: float, seed: int):
        self._oracle, self._follow, self._rng = oracle, follow_prob, random.Random(seed)

    def decide(self, obs: Observation) -> PolicyOutput:
        return self._oracle.decide(obs) if self._rng.random() < self._follow else Action.WAIT


def _default_behavior(seed: int, oracle: OraclePolicy) -> BasePolicy:
    """Rotate through behaviours that react the way a real agent can (after detection)."""
    return [
        lambda: _OracleMix(oracle, 0.7, seed),
        lambda: RebalanceOnlyPolicy(),
        lambda: TimeDelayedPolicy(10),
        lambda: TimeDelayedPolicy(60),
    ][seed % 4]()


def generate_oracle_data(
    seeds:       Iterable[int],
    total_steps: int = 1000,
    behavior:    Optional[Callable[[int, OraclePolicy], BasePolicy]] = None,
    difficulty:  str = "easy",
) -> Tuple[List[Observation], List[Action]]:
    """
    (Observation, Oracle action) pairs from simulator episodes, one episode per
    seed, kept in seed order. `behavior(seed, oracle)` picks who acts.
    `difficulty` picks the scenario level (scenarios.py); several can be mixed
    by passing "realistic,hard,chaos" (cycled per seed).
    """
    from scenarios import make_scenario
    behavior = behavior or _default_behavior
    levels   = difficulty.split(",")
    observations: List[Observation] = []
    actions:      List[Action]      = []
    for i, seed in enumerate(seeds):
        sc     = make_scenario(seed, levels[i % len(levels)], total_steps)
        oracle = OraclePolicy(sc.schedule)
        rec    = _RecordingPolicy(oracle, behavior(seed, oracle))
        run_episode(sc.make_cluster(), rec, sc.schedule, total_steps)
        observations.extend(rec.observations)
        actions.extend(rec.labels)
    return observations, actions


def train_agent_model(
    seeds:         Iterable[int],
    total_steps:   int = 1000,
    dagger_rounds: int = 1,
    verbose:       bool = False,
    difficulty:    str = "easy",
) -> DecisionModel:
    """
    Behavioral cloning + DAgger: train on Oracle labels, then let the trained
    agent drive, label the states IT visits with the Oracle, add them, retrain.
    """
    seeds = list(seeds)
    obs, acts = generate_oracle_data(seeds, total_steps, difficulty=difficulty)
    model = train_behavioral_cloning(obs, acts, verbose=verbose)
    for _ in range(dagger_rounds):
        learner = lambda seed, oracle, m=model: HybridAgentPolicy(
            m, MPCFallback(), ClusterDetector(NORMAL_SPEEDS))
        more_obs, more_acts = generate_oracle_data(seeds, total_steps, behavior=learner,
                                                   difficulty=difficulty)
        obs, acts = more_obs + obs, more_acts + acts   # older data last -> stays in train split
        model = train_behavioral_cloning(obs, acts, verbose=verbose)
    return model


# =============================================================================
# DAY 4 - MOCK ORACLE DATA GENERATOR (standalone, no simulator needed)
# =============================================================================

def generate_mock_oracle_data(
    n_samples: int = 1000,
    seed:      int = 42,
) -> Tuple[List[Observation], List[Action]]:
    """
    Synthetic (Observation, Action) pairs from a hand-written rule. Only for fast
    unit tests of the model plumbing; train real models with generate_oracle_data().

    Label logic:
      - crash     -> EVICT
      - slow_major-> COMPENSATE   (speed_ratio 40-55%)
      - slow_minor-> REBALANCE    (speed_ratio 65-80%)
      - none      -> WAIT
    """
    rng = np.random.RandomState(seed)
    MACHINE_IDS    = ["mac-m2", "win-gpu-0", "win-gpu-1"]
    NORMAL_SPEEDS  = {"mac-m2": 1.2, "win-gpu-0": 3.5, "win-gpu-1": 3.5}
    PRICES         = {"mac-m2": 0.0, "win-gpu-0": 0.5, "win-gpu-1": 0.5}

    observations: List[Observation] = []
    actions:      List[Action]      = []

    for i in range(n_samples):
        fault_kind    = rng.choice(
            ["none", "crash", "slow_minor", "slow_major"],
            p=[0.60, 0.10, 0.15, 0.15],
        )
        fault_machine = rng.choice(MACHINE_IDS)
        machines: Dict[str, MachineObs] = {}

        for mid in MACHINE_IDS:
            ns = NORMAL_SPEEDS[mid]
            if mid == fault_machine:
                if fault_kind == "crash":
                    spd, age = 0.0, 5.0
                elif fault_kind == "slow_major":
                    spd, age = ns * rng.uniform(0.40, 0.55), 0.1
                elif fault_kind == "slow_minor":
                    spd, age = ns * rng.uniform(0.65, 0.80), 0.1
                else:
                    spd, age = ns * rng.uniform(0.90, 1.10), 0.1
            else:
                spd, age = ns * rng.uniform(0.90, 1.10), rng.uniform(0.0, 0.4)

            spd = max(0.0, spd)
            n_active = sum(
                1 for m2 in MACHINE_IDS
                if not (m2 == fault_machine and fault_kind == "crash")
            )
            machines[mid] = MachineObs(
                alive_signal_age = age,
                normal_speed     = ns,
                measured_speed   = spd,
                share            = 1.0 / max(n_active, 1),
                code_version     = "v1.0",
                price            = PRICES[mid],
                is_active        = spd > 0.01,
            )

        obs = Observation(time=float(i) * 0.3, step=i, machines=machines)

        if fault_kind == "crash":
            action = Action.EVICT
        elif fault_kind == "slow_major":
            action = Action.COMPENSATE
        elif fault_kind == "slow_minor":
            action = Action.REBALANCE
        else:
            action = Action.WAIT

        observations.append(obs)
        actions.append(action)

    return observations, actions


# =============================================================================
# DAY 5-6 - HYBRID AGENT POLICY
# =============================================================================

class HybridAgentPolicy(BasePolicy):
    """
    The final self-healing agent.

    Decision flow
    -------------
    1. Detector scans the observation (every step, so its state stays current).
    2. Diagnose -> shortlist of <= 3 allowed actions (the MPC's shortlist).
    3. DecisionModel predicts (action, confidence).
    4. confidence >= 0.70 and action in the shortlist -> candidate.
    5. Shield: if the MPC projects the candidate to be more than SHIELD_MARGIN
       worse than its own best choice, the MPC's choice is used instead. A model
       can be confidently wrong in states it never saw in training.
       Otherwise -> MPCFallback picks from the shortlist.
    Without a detector there is no diagnosis, so the mask and shield are skipped.
    Total latency target: < 1.0 ms.

    Attributes
    ----------
    mpc_call_count  : times MPC decided.
    nn_call_count   : times the NN prediction was used.
    decision_times  : per-step latencies in ms.
    """

    name = "HybridAgentPolicy"
    CONFIDENCE_THRESHOLD: float = 0.70
    SHIELD_MARGIN:        float = 0.05

    def __init__(
        self,
        decision_model:   DecisionModel,
        mpc_fallback:     MPCFallback,
        cluster_detector: Optional[ClusterDetector] = None,
    ):
        self._model    = decision_model
        self._mpc      = mpc_fallback
        self._detector = cluster_detector

        self.mpc_call_count: int = 0
        self.nn_call_count:  int = 0
        self.shield_count:   int = 0   # NN actions overruled by the shield
        self.decision_times: List[float] = []

    def decide(self, obs: Observation) -> PolicyOutput:
        """Make a decision; records latency internally."""
        t0     = time.perf_counter()
        action = self._decide_internal(obs)
        self.decision_times.append((time.perf_counter() - t0) * 1_000)
        return action

    def _decide_internal(self, obs: Observation) -> PolicyOutput:
        # Steps 1-2: detect, diagnose, shortlist
        detections = self._detector.scan(obs) if self._detector is not None else []
        fault      = MPCFallback.diagnose(detections, obs)
        allowed    = ({a for a, _ in self._mpc.shortlist(fault, obs)}
                      if self._detector is not None else set(Action))

        # Step 3: NN inference
        if self._model.is_trained:
            action_name, confidence = self._model.predict_single(featurize(obs))
        else:
            action_name, confidence = "WAIT", 0.0

        # Step 4: under-70%-sure rule + shortlist mask
        nn_action: Optional[Action] = None
        if confidence >= self.CONFIDENCE_THRESHOLD and action_name in Action.__members__:
            if Action[action_name] in allowed:
                nn_action = Action[action_name]

        if nn_action is not None and self._detector is None:
            self.nn_call_count += 1
            return self._with_target(nn_action, fault)

        # Step 5: MPC (as fallback, or as the shield's yardstick)
        mpc_action, best, projections = self._mpc.decide(fault, obs)
        if nn_action is not None:
            if projections.get(nn_action.name, 0.0) >= (1 - self.SHIELD_MARGIN) * best:
                self.nn_call_count += 1
                return self._with_target(nn_action, fault)
            self.shield_count += 1
        self.mpc_call_count += 1
        return self._with_target(mpc_action, fault)

    @staticmethod
    def _with_target(action: Action, fault: DiagnosedFault) -> PolicyOutput:
        if action in (Action.EVICT, Action.COMPENSATE) and fault.machine_id != "none":
            return (action, fault.machine_id)
        return action

    def summary(self) -> str:
        total  = self.nn_call_count + self.mpc_call_count
        avg_ms = (sum(self.decision_times) / len(self.decision_times)
                  if self.decision_times else 0.0)
        nn_pct  = 100 * self.nn_call_count  / total if total else 0
        mpc_pct = 100 * self.mpc_call_count / total if total else 0
        return (
            f"  HybridAgentPolicy  total={total}  "
            f"NN={self.nn_call_count} ({nn_pct:.0f}%)  "
            f"MPC={self.mpc_call_count} ({mpc_pct:.0f}%, shield {self.shield_count})  "
            f"avg_latency={avg_ms:.4f} ms"
        )


# =============================================================================
# EVALUATION REPORT  (printed by __main__)
# =============================================================================

def _banner(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")

def _section(title: str) -> None:
    print(f"\n{'-'*72}\n  {title}\n{'-'*72}")



def run_evaluation_report() -> None:
    _banner("TASK C - AGENT BRAIN  |  Full Evaluation Report")

    # ------------------------------------------------------------------
    # 1. Detector Evaluation
    # ------------------------------------------------------------------
    _section("1 / 4   ANOMALY DETECTOR EVALUATION  (per-step, 5 seeds)")
    print("  200-step synthetic traces, 10% noise, 50% slowdown at step 60 ...")

    all_ema, all_naive = [], []
    for seed in range(5):
        ema_r, naive_r = evaluate_detector(
            n_machines=3, total_steps=200,
            fault_step=60, slowdown_frac=0.50, seed=seed,
        )
        all_ema.append(ema_r)
        all_naive.append(naive_r)

    def _avg(lst, attr):
        return sum(getattr(x, attr) for x in lst) / len(lst)

    print(f"\n  {'Detector':<28}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>6}  {'TTN (steps)':>12}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*12}")
    for name, lst in [("EMA Detector", all_ema), ("Naive Cross-Machine", all_naive)]:
        print(
            f"  {name:<28}  "
            f"{_avg(lst,'precision'):>10.3f}  "
            f"{_avg(lst,'recall'):>8.3f}  "
            f"{_avg(lst,'f1'):>6.3f}  "
            f"{_avg(lst,'avg_time_to_notice'):>12.1f}"
        )

    # ------------------------------------------------------------------
    # 2. Behavioral Cloning on simulator Oracle data
    # ------------------------------------------------------------------
    _section("2 / 4   DECISION MODEL  (Behavioral Cloning of the simulator Oracle)")
    print("  Recording Oracle labels on 30 training episodes (seeds 100-129), + 1 DAgger round ...")

    model = train_agent_model(range(100, 130), total_steps=1000, dagger_rounds=1, verbose=True)

    obs_test, act_test = generate_oracle_data(range(0, 5), total_steps=1000)
    X_test = np.vstack([featurize(o) for o in obs_test])
    y_test = np.array([a.name for a in act_test])
    preds, confs = model.predict(X_test)
    preds = np.array(preds)
    print(f"\n  Test episodes (seeds 0-4): accuracy {np.mean(preds == y_test)*100:.1f}%, "
          f"ECE {expected_calibration_error(confs, preds == y_test):.3f}")
    print(f"\n  Per-class accuracy on test set:")
    print(f"  {'Action':<14}  {'Support':>8}  {'Correct':>8}  {'Acc':>8}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*8}")
    for cls in sorted(set(y_test)):
        mask    = y_test == cls
        support = int(mask.sum())
        correct = int((preds[mask] == cls).sum())
        print(f"  {cls:<14}  {support:>8d}  {correct:>8d}  {correct/support:>8.1%}")

    # ------------------------------------------------------------------
    # 3. The agent inside the simulator vs the baselines
    # ------------------------------------------------------------------
    _section("3 / 4   AGENT IN THE SIMULATOR  (seeds 0-4, never used for training)")

    def make_agent() -> HybridAgentPolicy:
        return HybridAgentPolicy(model, MPCFallback(), ClusterDetector(NORMAL_SPEEDS))

    contenders = [
        ("DoNothing",           lambda gen: DoNothingPolicy()),
        ("Act at once",         lambda gen: RebalanceOnlyPolicy()),
        ("Wait 60, then act",   lambda gen: TimeDelayedPolicy(60)),
        ("Hybrid agent",        lambda gen: make_agent()),
        ("Oracle (upper bound)", lambda gen: OraclePolicy(gen)),
    ]
    print(f"\n  {'Policy':<22}  {'Goodput/hr':>11}  {'Goodput/$':>10}  {'Wasted s':>9}  "
          f"{'NN/MPC':>10}  {'ms/decision':>11}")
    print(f"  {'─'*22}  {'─'*11}  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*11}")
    for name, factory in contenders:
        ghr, gusd, waste, ms = [], [], [], []
        nn = mpc = 0
        for seed in range(5):
            gen = RandomFaultGenerator(seed=seed, total_steps=1000)
            pol = factory(gen)
            r = run_episode(SimulatedCluster(seed=seed), pol, gen.as_schedule(), 1000)
            ghr.append(r.goodput_per_hr); gusd.append(r.goodput_per_usd)
            waste.append(r.total_wasted_time_s); ms.append(r.avg_decision_ms)
            if isinstance(pol, HybridAgentPolicy):
                nn += pol.nn_call_count; mpc += pol.mpc_call_count
        split = f"{nn}/{mpc}" if nn + mpc else "-"
        print(f"  {name:<22}  {np.mean(ghr):>11.0f}  {np.mean(gusd):>10.0f}  "
              f"{np.mean(waste):>9.1f}  {split:>10}  {np.mean(ms):>11.4f}")

    # ------------------------------------------------------------------
    # 4. MPC Sanity Check
    # ------------------------------------------------------------------
    _section("4 / 4   MPC FALLBACK SANITY CHECK")

    c = SimulatedCluster(seed=0)
    c.inject_slowdown("win-gpu-0", 0.50, 100)
    fault4 = DiagnosedFault("win-gpu-0", "fail-slow", severity=0.50, duration_est=100)
    best4, _, proj4 = MPCFallback().decide(fault4, c.get_observation())

    print(f"\n  Scenario: win-gpu-0 at 50% speed for 100 steps (fail-slow)")
    print(f"  MPC projections (Goodput / hr):")
    for aname, val in sorted(proj4.items(), key=lambda x: -x[1]):
        tag = "  <- CHOSEN" if aname == best4.name else ""
        print(f"    {aname:<14s}: {val:>12.2f}{tag}")
    print(f"\n  Decision: {best4.name}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Step 1: Full evaluation report
    run_evaluation_report()

    # Step 2: Unit test suite
    _banner("UNIT TEST SUITE  (Day 7)")
    print()

    suite  = unittest.defaultTestLoader.loadTestsFromName("test_agent_brain")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    _banner("SUMMARY")
    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"  Tests run : {total}")
    print(f"  Passed    : {passed}  {'OK ' * min(passed, 15)}")
    print(f"  Failed    : {failed}  {'!! ' * min(failed, 15)}")
    print(f"\n  Status    : {'ALL TESTS PASSED' if failed == 0 else str(failed) + ' TEST(S) FAILED'}")
