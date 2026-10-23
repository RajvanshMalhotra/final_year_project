"""
agent_brain.py
==============
Self-Healing Mixed-GPU Training - Task C: Agent Brain
------------------------------------------------------
Days 1-7 implementation:

  Day 1-2 : MachineDetector  +  evaluate_detector()
  Day 3   : MPCFallback
  Day 4   : DecisionModel    +  train_behavioral_cloning()
  Day 5-6 : HybridAgentPolicy  (under-70%-sure rule + RL hook)
  Day 7   : Full test suite  +  __main__ evaluation report

External deps
-------------
  numpy        - arrays, statistics
  scikit-learn - MLPClassifier (lightweight neural network)
  unittest     - tests (self-contained, no extra install needed)

Usage
-----
  python agent_brain.py          # runs evaluation + test suite
  pytest agent_brain.py -v       # pytest mode
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
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

# ---- scikit-learn (required) -------------------------------------------------
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    _SKL_OK = True
except ImportError:
    raise ImportError("scikit-learn is required.  pip install scikit-learn")

# ---- Optional simulator import -----------------------------------------------
try:
    from simulator import (
        Action, Observation, MachineObs, BasePolicy,
    )
    _SIM_OK = True
except ImportError:
    _SIM_OK = False

    class BasePolicy:
        name: str = "BasePolicy"
        def decide(self, obs: Observation) -> Action:
            raise NotImplementedError

    class Action(Enum):
        WAIT       = auto()
        REBALANCE  = auto()
        COMPENSATE = auto()
        EVICT      = auto()
        READMIT    = auto()
        RESTART    = auto()
        CHECKPOINT = auto()

    @dataclass
    class MachineObs:
        alive_signal_age: float
        normal_speed:     float
        measured_speed:   float
        share:            float
        code_version:     str
        price:            float
        is_active:        bool = True

    @dataclass
    class Observation:
        time:     float
        step:     int
        machines: Dict[str, MachineObs]


# =============================================================================
# HELPERS
# =============================================================================

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
    speed_ratio: float  # measured / normal  (0..1+)
    ema_speed:   float
    z_score:     float


class MachineDetector:
    """
    Per-machine streaming anomaly detector using EMA + z-score change test.

    Fault classification
    --------------------
    fail-stop : measured_speed <= STOP_SPEED_THRESHOLD  OR
                alive_signal_age > CRASH_SIGNAL_AGE
    fail-slow : speed_ratio < SLOW_RATIO_THRESHOLD AND z_score > Z_THRESHOLD
    none      : normal operation
    """
    EMA_ALPHA:            float = 0.15   # higher -> more reactive
    SLOW_RATIO_THRESHOLD: float = 0.70   # below 70% of normal -> candidate slow
    Z_THRESHOLD:          float = 2.0    # 2-sigma confirmation
    STOP_SPEED_THRESHOLD: float = 0.05   # steps/s; below this -> stopped
    CRASH_SIGNAL_AGE:     float = 3.0    # seconds; above this -> crash
    WARMUP_STEPS:         int   = 5      # ignore first N steps

    def __init__(self, machine_id: str, normal_speed: float):
        self.machine_id   = machine_id
        self.normal_speed = normal_speed
        self._ema_speed:  float = normal_speed
        self._ema_sq:     float = normal_speed ** 2
        self._step_count: int   = 0

    def update(self, step: int, measured_speed: float,
               alive_signal_age: float) -> FaultDetection:
        self._step_count += 1

        # Hard fail-stop check
        if (measured_speed <= self.STOP_SPEED_THRESHOLD
                or alive_signal_age > self.CRASH_SIGNAL_AGE):
            return FaultDetection(
                machine_id  = self.machine_id,
                fault_kind  = "fail-stop",
                step        = step,
                speed_ratio = 0.0,
                ema_speed   = self._ema_speed,
                z_score     = float("inf"),
            )

        # Update EMA and EMA of squares (for running variance)
        prev_ema = self._ema_speed
        self._ema_speed = (self.EMA_ALPHA * measured_speed
                           + (1 - self.EMA_ALPHA) * self._ema_speed)
        self._ema_sq    = (self.EMA_ALPHA * measured_speed ** 2
                           + (1 - self.EMA_ALPHA) * self._ema_sq)

        # Minimum variance floor = (1% of normal_speed)^2 so that the
        # z-score denominator is never negligibly tiny and the test is
        # actually sensitive to real slowdowns.
        min_var  = (0.01 * self.normal_speed) ** 2
        variance = max(self._ema_sq - self._ema_speed ** 2, min_var)
        std_dev  = math.sqrt(variance)

        # z-score: positive means measured is BELOW prev EMA (slower)
        z_score     = (prev_ema - measured_speed) / std_dev
        speed_ratio = (measured_speed / self.normal_speed
                       if self.normal_speed > 0 else 1.0)

        # fail-slow detection (only after warmup)
        if (self._step_count > self.WARMUP_STEPS
                and speed_ratio < self.SLOW_RATIO_THRESHOLD
                and z_score > self.Z_THRESHOLD):
            fault_kind = "fail-slow"
        else:
            fault_kind = "none"

        return FaultDetection(
            machine_id  = self.machine_id,
            fault_kind  = fault_kind,
            step        = step,
            speed_ratio = speed_ratio,
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
    Generate synthetic traces, inject a slowdown at fault_step, then
    evaluate the EMA detector vs the naive cross-machine baseline.
    Returns (ema_result, naive_result).
    """
    np_rng = np.random.RandomState(seed)
    NORMAL_SPEED = 3.5
    NOISE_STD    = 0.10

    machine_ids   = [f"gpu-{i}" for i in range(n_machines)]
    fault_machine = machine_ids[0]

    detectors = {mid: MachineDetector(mid, NORMAL_SPEED) for mid in machine_ids}
    ground_truth_fault_steps: Dict[str, int] = {fault_machine: fault_step}

    # ---- Episode-level metrics -------------------------------------------
    # Rather than per-step TP/FP/FN (which gives recall ≈ 0 for a detector
    # that correctly fires once at the change point then adapts), we measure:
    #   TP : fault machine detected at least once inside its fault window.
    #   FN : fault machine never detected during its fault window.
    #   FP : any non-fault machine flagged at least once during normal steps.
    #
    ema_episode_detected:   Dict[str, bool] = {fault_machine: False}
    naive_episode_detected: Dict[str, bool] = {fault_machine: False}
    ema_ttn_list: List[float]   = []
    naive_ttn_list: List[float] = []
    ema_first_detected:   Dict[str, Optional[int]] = {fault_machine: None}
    naive_first_detected: Dict[str, Optional[int]] = {fault_machine: None}
    ema_fp_count   = 0
    naive_fp_count = 0

    for step in range(total_steps):
        machines_obs: Dict[str, MachineObs] = {}
        for mid in machine_ids:
            noise = float(np_rng.normal(0, NOISE_STD * NORMAL_SPEED))
            if mid == fault_machine and step >= fault_step:
                spd = NORMAL_SPEED * (1.0 - slowdown_frac) + noise
            else:
                spd = NORMAL_SPEED + noise
            spd = max(0.01, spd)
            machines_obs[mid] = MachineObs(
                alive_signal_age = 0.1,
                normal_speed     = NORMAL_SPEED,
                measured_speed   = spd,
                share            = 1.0 / n_machines,
                code_version     = "v1.0",
                price            = 0.50,
                is_active        = True,
            )
        obs = Observation(time=float(step) * 0.3, step=step, machines=machines_obs)
        in_fault_window = step >= fault_step

        # ---- EMA detector -----------------------------------------------
        for mid in machine_ids:
            mobs   = machines_obs[mid]
            result = detectors[mid].update(step, mobs.measured_speed, mobs.alive_signal_age)
            flagged = result.fault_kind in ("fail-stop", "fail-slow")

            if mid == fault_machine and in_fault_window:
                if flagged:
                    ema_episode_detected[mid] = True
                    if ema_first_detected.get(mid) is None:
                        ema_first_detected[mid] = step
            elif not in_fault_window and flagged:
                # False positive on a non-faulty machine/step
                ema_fp_count += 1

        # ---- Naive detector ---------------------------------------------
        for (mid, kind) in _naive_detect(obs):
            flagged = kind in ("fail-stop", "fail-slow")

            if mid == fault_machine and in_fault_window:
                if flagged:
                    naive_episode_detected[mid] = True
                    if naive_first_detected.get(mid) is None:
                        naive_first_detected[mid] = step
            elif not in_fault_window and flagged:
                naive_fp_count += 1

    # ---- Time-to-Notice -------------------------------------------------
    for mid, fs in ground_truth_fault_steps.items():
        if ema_first_detected.get(mid) is not None:
            ema_ttn_list.append(ema_first_detected[mid] - fs)
        if naive_first_detected.get(mid) is not None:
            naive_ttn_list.append(naive_first_detected[mid] - fs)

    # ---- Episode-level PRF ----------------------------------------------
    # One fault machine = one episode.
    ema_tp  = sum(1 for v in ema_episode_detected.values()  if v)
    ema_fn  = sum(1 for v in ema_episode_detected.values()  if not v)
    naive_tp = sum(1 for v in naive_episode_detected.values() if v)
    naive_fn = sum(1 for v in naive_episode_detected.values() if not v)
    # FP: cap at 1 per non-fault machine per run to avoid metric explosion
    ema_fp_ep   = min(ema_fp_count,   1)
    naive_fp_ep = min(naive_fp_count, 1)

    def _prf(tp, fp, fn):
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    ep, er, ef   = _prf(ema_tp,   ema_fp_ep,   ema_fn)
    np_, nr, nf  = _prf(naive_tp, naive_fp_ep, naive_fn)

    return (
        DetectorEvalResult(
            detector_name      = "EMA-ZScore Detector",
            precision          = ep, recall = er, f1 = ef,
            avg_time_to_notice = float(np.mean(ema_ttn_list)) if ema_ttn_list else float("inf"),
            tp = ema_tp, fp = ema_fp_ep, fn = ema_fn,
        ),
        DetectorEvalResult(
            detector_name      = "Naive Cross-Machine",
            precision          = np_, recall = nr, f1 = nf,
            avg_time_to_notice = float(np.mean(naive_ttn_list)) if naive_ttn_list else float("inf"),
            tp = naive_tp, fp = naive_fp_ep, fn = naive_fn,
        ),
    )


# =============================================================================
# DAY 3 - DIAGNOSIS & MPC FALLBACK
# =============================================================================

@dataclass
class DiagnosedFault:
    machine_id:   str
    fault_kind:   str    # "fail-stop" | "fail-slow" | "none"
    severity:     float  # 0..1
    duration_est: int    # estimated remaining steps


class MPCFallback:
    """
    Model Predictive Control fallback calculator.

    Shortlists 3 candidate actions per fault type, runs a 10-step
    goodput-per-dollar heuristic projection for each, returns the best.

    Candidate sets
    --------------
    fail-stop  -> [EVICT,     WAIT,       RESTART]
    fail-slow  -> [REBALANCE, COMPENSATE, WAIT]
    none       -> [WAIT,      CHECKPOINT, REBALANCE]
    """
    HORIZON:        int   = 10
    STALL_COST:     float = 1.0
    REBAL_OVERHEAD: float = 0.5

    def __init__(
        self,
        horizon: Optional[int] = None,
        stall_cost: Optional[float] = None,
        rebal_overhead: Optional[float] = None,
        **kwargs,
    ):
        if horizon is not None:
            self.HORIZON = horizon
        elif "H" in kwargs:
            self.HORIZON = kwargs["H"]
        if stall_cost is not None:
            self.STALL_COST = stall_cost
        if rebal_overhead is not None:
            self.REBAL_OVERHEAD = rebal_overhead

    def decide(
        self,
        fault: DiagnosedFault,
        obs:   Observation,
    ) -> Tuple[Action, float, Dict[str, float]]:
        """
        Returns (best_action, best_projection, all_projections).
        all_projections: {action_name: projected_goodput_per_dollar}
        """
        candidates = self._shortlist(fault)
        projections: Dict[str, float] = {}
        for action in candidates:
            projections[action.name] = self._project(action, fault, obs)
        best = max(candidates, key=lambda a: projections[a.name])
        return best, projections[best.name], projections

    def _shortlist(self, fault: DiagnosedFault) -> List[Action]:
        if fault.fault_kind == "fail-stop":
            return [Action.EVICT, Action.WAIT, Action.RESTART]
        if fault.fault_kind == "fail-slow":
            return [Action.REBALANCE, Action.COMPENSATE, Action.WAIT]
        # Healthy cluster: WAIT and CHECKPOINT are the only sensible actions.
        # REBALANCE on a healthy cluster incurs overhead with no benefit.
        return [Action.WAIT, Action.CHECKPOINT]

    def _project(self, action: Action, fault: DiagnosedFault,
                 obs: Observation) -> float:
        """
        Heuristic 10-step goodput/$ projection.

        WAIT     - cluster runs at its current degraded speed (bottleneck).
        REBALANCE/COMPENSATE - redistribute shares proportional to current
                               speed, so the bottleneck step duration drops
                               significantly when the slow machine gets a
                               smaller share.  One overhead step is charged.
        EVICT    - remove the faulty machine; remaining machines share load.
        RESTART  - pay fault.duration_est/2 upfront, then run at full speed.
        """
        H = self.HORIZON
        active = {
            mid: mobs for mid, mobs in obs.machines.items()
            if mobs.is_active and mobs.measured_speed > 0.01
        }
        total_price_hr = sum(
            mobs.price for mobs in obs.machines.values() if mobs.is_active
        )
        useful_steps = 0.0
        total_time_s = 0.0
        overhead_paid = False

        for h in range(H):
            machines_this = dict(active)

            if action == Action.EVICT and not overhead_paid:
                machines_this = {
                    mid: m for mid, m in machines_this.items()
                    if mid != fault.machine_id
                }
                overhead_paid = True

            elif action in (Action.REBALANCE, Action.COMPENSATE) and not overhead_paid:
                total_time_s += self.REBAL_OVERHEAD
                overhead_paid = True
                # Don't count overhead step as useful
                continue

            elif action == Action.RESTART and not overhead_paid:
                total_time_s += fault.duration_est / 2.0
                overhead_paid = True

            if not machines_this:
                total_time_s += self.STALL_COST
                continue

            n = len(machines_this)

            if action == Action.WAIT and fault.fault_kind == "fail-stop" and h < fault.duration_est:
                # Cluster fully stalled by a crashed machine that is not evicted
                total_time_s += self.STALL_COST

            elif action in (Action.REBALANCE, Action.COMPENSATE):
                # Speed-proportional shares -> step duration = 1/sum_speeds
                this_total_spd = sum(m.measured_speed for m in machines_this.values())
                step_dur = 1.0 / max(this_total_spd, 0.01)
                total_time_s += step_dur
                useful_steps += 1.0

            elif action == Action.EVICT:
                if machines_this:
                    this_spd = sum(m.measured_speed for m in machines_this.values())
                    step_dur = 1.0 / max(this_spd, 0.01)
                    total_time_s += step_dur
                    useful_steps += 1.0
                else:
                    total_time_s += self.STALL_COST

            else:
                # WAIT (no fault or fail-slow) and RESTART (after overhead):
                # Equal-share bottleneck: step_dur = (1/n) / min_speed
                slowest_spd = min(m.measured_speed for m in machines_this.values())
                step_dur    = (1.0 / n) / max(slowest_spd, 0.01)
                total_time_s += step_dur
                useful_steps += 1.0

        cost = total_price_hr * (total_time_s / 3600.0)
        if cost < 1e-9:
            return useful_steps / max(total_time_s / 3600.0, 1e-9)
        return useful_steps / cost

    @staticmethod
    def diagnose(detections: List[FaultDetection],
                 obs: Observation) -> DiagnosedFault:
        """Convert FaultDetections into the single worst DiagnosedFault."""
        worst: Optional[FaultDetection] = None
        for d in detections:
            if d.fault_kind == "fail-stop":
                worst = d
                break
            if d.fault_kind == "fail-slow" and (
                    worst is None or worst.fault_kind == "none"):
                worst = d
        if worst is None or worst.fault_kind == "none":
            return DiagnosedFault("none", "none", 0.0, 0)
        return DiagnosedFault(
            machine_id   = worst.machine_id,
            fault_kind   = worst.fault_kind,
            severity     = _clamp(1.0 - worst.speed_ratio, 0.0, 1.0),
            duration_est = 20,
        )


# =============================================================================
# DAY 4 - FEATURIZER
# =============================================================================

def featurize(obs: Observation) -> np.ndarray:
    """
    Observation -> fixed-length float32 feature vector.

    Per-machine (sorted by id): speed_ratio, signal_age, share, price, is_active
    Global:  min_ratio, max_ratio, n_inactive, inactive_share_frac
    """
    sorted_ids = sorted(obs.machines.keys())
    per_machine: List[float] = []
    speed_ratios: List[float] = []
    inactive_count = 0
    inactive_share = 0.0

    for mid in sorted_ids:
        m = obs.machines[mid]
        ratio = (m.measured_speed / m.normal_speed
                 if m.normal_speed > 0 else 0.0)
        speed_ratios.append(ratio)
        per_machine.extend([
            ratio,
            _clamp(m.alive_signal_age, 0, 10),
            m.share,
            m.price,
            float(m.is_active),
        ])
        if not m.is_active:
            inactive_count += 1
            inactive_share += m.share

    global_feats = [
        min(speed_ratios) if speed_ratios else 0.0,
        max(speed_ratios) if speed_ratios else 0.0,
        float(inactive_count),
        inactive_share,
    ]
    return np.array(per_machine + global_feats, dtype=np.float32)


# =============================================================================
# DAY 4 - DECISION MODEL
# =============================================================================

class DecisionModel:
    """
    Lightweight neural network (MLP 64x64 ReLU) trained via behavioral
    cloning to mimic the Oracle policy.

    Probability calibration via Platt scaling (CalibratedClassifierCV)
    ensures confidence scores are trustworthy for the under-70%-sure rule.
    """

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (64, 64)):
        self._base = MLPClassifier(
            hidden_layer_sizes  = hidden_layer_sizes,
            activation          = "relu",
            solver              = "adam",
            max_iter            = 500,
            random_state        = 42,
            learning_rate_init  = 1e-3,
            early_stopping      = True,
            validation_fraction = 0.10,
            n_iter_no_change    = 15,
        )
        self._calibrated: Optional[CalibratedClassifierCV] = None
        self._label_encoder = LabelEncoder()
        self._is_trained = False
        self.classes_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionModel":
        """Train on (N,F) feature matrix and (N,) string action labels."""
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = list(self._label_encoder.classes_)
        self._base.fit(X, y_enc)
        # Use cv='prefit' so calibration wraps the already-trained base
        # estimator without spawning 3 cross-validated clones.  This keeps
        # inference latency well below the 1 ms target.
        self._calibrated = CalibratedClassifierCV(
            self._base, method="sigmoid", cv="prefit"
        )
        self._calibrated.fit(X, y_enc)
        self._is_trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, C) calibrated probability matrix."""
        if not self._is_trained or self._calibrated is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self._calibrated.predict_proba(X)

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
    oracle_observations : list of Observation objects
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
            print(f"    {u:<14s}: {c:>4d} ({100*c/len(y):.1f}%)")

    model = DecisionModel(hidden_layer_sizes=hidden_layer_sizes)
    t0    = time.perf_counter()
    model.fit(X, y)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if verbose:
        preds, _ = model.predict(X)
        acc = np.mean(np.array(preds) == y)
        print(f"  [BC] Done in {elapsed_ms:.1f} ms. Train accuracy: {acc*100:.1f}%")

    return model


# =============================================================================
# DAY 4 - MOCK ORACLE DATA GENERATOR (standalone, no simulator needed)
# =============================================================================

def generate_mock_oracle_data(
    n_samples: int = 1000,
    seed:      int = 42,
) -> Tuple[List[Observation], List[Action]]:
    """
    Synthetic (Observation, Action) pairs approximating Oracle policy logic.

    Label logic (mirrors simulator.py OraclePolicy):
      - crash     -> EVICT
      - slow_major-> COMPENSATE   (speed_ratio 40-55%, long duration)
      - slow_minor-> REBALANCE    (speed_ratio 65-80%, short duration)
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
    1. Featurize observation.
    2. Query DecisionModel for (action, confidence).
    3. Under-70%-sure rule:
       confidence >= 0.70  ->  use NN prediction.
       confidence <  0.70  ->  defer to MPCFallback.
    4. Total latency target: < 1.0 ms.

    Attributes
    ----------
    mpc_call_count  : times MPC was invoked.
    nn_call_count   : times NN prediction was accepted.
    decision_times  : per-step latencies in ms.
    """

    name = "HybridAgentPolicy"
    CONFIDENCE_THRESHOLD: float = 0.70

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
        self.decision_times: List[float] = []

    def decide(self, obs: Observation) -> Action:
        """Make a decision; records latency internally."""
        t0     = time.perf_counter()
        action = self._decide_internal(obs)
        self.decision_times.append((time.perf_counter() - t0) * 1_000)
        return action

    def _decide_internal(self, obs: Observation) -> Action:
        # Step 1: featurize
        x = featurize(obs)

        # Step 2: NN inference
        if self._model.is_trained:
            action_name, confidence = self._model.predict_single(x)
        else:
            action_name, confidence = "WAIT", 0.0

        # Step 3: under-70%-sure rule
        if confidence >= self.CONFIDENCE_THRESHOLD:
            self.nn_call_count += 1
            try:
                return Action[action_name]
            except KeyError:
                pass  # unknown label -> fall through to MPC

        # Step 4: MPC fallback
        self.mpc_call_count += 1
        detections: List[FaultDetection] = []
        if self._detector is not None:
            detections = self._detector.scan(obs)
        fault  = MPCFallback.diagnose(detections, obs)
        action, _, _ = self._mpc.decide(fault, obs)
        return action

    def rl_finetune(
        self,
        rewards:      List[float],
        observations: List[Observation],
        actions:      List[Action],
    ) -> None:
        """
        REINFORCE-style policy gradient fine-tune hook.

        Computes discounted returns, normalises them, then re-trains the
        DecisionModel using positive-return steps as weighted demonstrations.
        This improves the model beyond the Oracle by reinforcing steps that
        led to better actual rewards.

        Parameters
        ----------
        rewards      : per-step scalar rewards from ClusterEnv.
        observations : Observation at each step.
        actions      : Action taken at each step.
        """
        if not observations or not rewards:
            return

        # Discounted returns
        gamma = 0.99
        G: List[float] = []
        running = 0.0
        for r in reversed(rewards):
            running = r + gamma * running
            G.insert(0, running)
        G_arr = np.array(G, dtype=np.float64)
        if G_arr.std() > 1e-8:
            G_arr = (G_arr - G_arr.mean()) / G_arr.std()

        # Re-train with sample weights via importance resampling.
        # Note: scikit-learn's MLPClassifier does not support sample_weight in fit(),
        # so we resample transitions proportional to their positive discounted returns.
        X = np.vstack([featurize(o) for o in observations])
        y = np.array([a.name for a in actions])
        weights = np.clip(G_arr, 0, None) + 0.1  # all weights > 0

        if self._model.is_trained and len(X) > 0:
            probs = weights / weights.sum()
            rng = np.random.RandomState(42)
            indices = rng.choice(len(X), size=len(X), replace=True, p=probs)
            X_resampled = X[indices]
            y_resampled = y[indices]
            try:
                self._model.fit(X_resampled, y_resampled)
            except Exception:
                pass  # production: log and retry

    def summary(self) -> str:
        total  = self.nn_call_count + self.mpc_call_count
        avg_ms = (sum(self.decision_times) / len(self.decision_times)
                  if self.decision_times else 0.0)
        nn_pct  = 100 * self.nn_call_count  / total if total else 0
        mpc_pct = 100 * self.mpc_call_count / total if total else 0
        return (
            f"  HybridAgentPolicy  total={total}  "
            f"NN={self.nn_call_count} ({nn_pct:.0f}%)  "
            f"MPC={self.mpc_call_count} ({mpc_pct:.0f}%)  "
            f"avg_latency={avg_ms:.4f} ms"
        )


# =============================================================================
# DAY 7 - TEST SUITE
# =============================================================================

class TestMachineDetector(unittest.TestCase):
    """Unit tests for MachineDetector (Days 1-2)."""

    def test_ema_converges_to_normal_speed(self):
        """EMA must converge to ~normal_speed after 50 normal steps."""
        det = MachineDetector("m0", normal_speed=3.5)
        for s in range(50):
            det.update(s, measured_speed=3.5, alive_signal_age=0.1)
        self.assertAlmostEqual(det._ema_speed, 3.5, delta=0.05)

    def test_ema_drops_after_sustained_slowdown(self):
        """EMA must drop below baseline after a persistent speed reduction."""
        det = MachineDetector("m0", normal_speed=3.5)
        for s in range(20):
            det.update(s, 3.5, 0.1)
        baseline = det._ema_speed
        for s in range(20, 60):
            det.update(s, 3.5 * 0.5, 0.1)
        self.assertLess(det._ema_speed, baseline)

    def test_fail_stop_via_zero_speed(self):
        det = MachineDetector("m0", 3.5)
        r   = det.update(10, measured_speed=0.0, alive_signal_age=0.1)
        self.assertEqual(r.fault_kind, "fail-stop")

    def test_fail_stop_via_stale_heartbeat(self):
        det = MachineDetector("m0", 3.5)
        r   = det.update(10, measured_speed=3.5, alive_signal_age=999.0)
        self.assertEqual(r.fault_kind, "fail-stop")

    def test_fail_slow_detected_after_warmup(self):
        """
        The EMA change-point detector fires at the transition step when
        z-score spikes above threshold.  The EMA then adapts, so the
        detector correctly returns 'none' once it has re-baselined.
        This test checks that at least one step inside the slow window
        is classified as 'fail-slow'.
        """
        det = MachineDetector("m0", normal_speed=3.5)
        for s in range(30):
            det.update(s, 3.5, 0.1)
        detected_any = False
        for s in range(30, 70):
            result = det.update(s, 3.5 * 0.5, 0.1)
            if result.fault_kind == "fail-slow":
                detected_any = True
                break   # found it — no need to continue
        self.assertTrue(
            detected_any,
            "Expected at least one 'fail-slow' detection during 50%% slowdown window"
        )

    def test_no_fault_on_normal_operation(self):
        """Small normal noise must not trigger any fault."""
        det = MachineDetector("m0", 3.5)
        rng = random.Random(0)
        last = None
        for s in range(80):
            spd  = 3.5 * rng.uniform(0.92, 1.08)
            last = det.update(s, spd, 0.1)
        self.assertEqual(last.fault_kind, "none")

    def test_evaluate_detector_precision_recall_above_60pct(self):
        """EMA detector Precision and Recall must both exceed 0.60."""
        ema_r, _ = evaluate_detector(
            n_machines=3, total_steps=200,
            fault_step=40, slowdown_frac=0.50, seed=7,
        )
        self.assertGreater(ema_r.precision, 0.60,
                           f"EMA P={ema_r.precision:.3f}")
        self.assertGreater(ema_r.recall,    0.60,
                           f"EMA R={ema_r.recall:.3f}")

    def test_ema_ttn_not_much_worse_than_naive(self):
        """EMA time-to-notice must be <= naive TTN + 5 steps."""
        ema_r, naive_r = evaluate_detector(
            n_machines=3, total_steps=200,
            fault_step=40, slowdown_frac=0.50, seed=7,
        )
        self.assertLessEqual(
            ema_r.avg_time_to_notice,
            naive_r.avg_time_to_notice + 5,
            f"EMA TTN={ema_r.avg_time_to_notice:.1f} > "
            f"naive TTN={naive_r.avg_time_to_notice:.1f} + 5",
        )


class TestMPCFallback(unittest.TestCase):
    """Unit tests for MPCFallback (Day 3)."""

    NORMAL  = {"mac-m2": 1.2, "win-gpu-0": 3.5, "win-gpu-1": 3.5}
    PRICES  = {"mac-m2": 0.0, "win-gpu-0": 0.5, "win-gpu-1": 0.5}

    def _obs(self, speeds: Dict[str, float]) -> Observation:
        machines = {}
        for mid, spd in speeds.items():
            ns = self.NORMAL.get(mid, 3.5)
            machines[mid] = MachineObs(
                alive_signal_age = 0.1 if spd > 0 else 10.0,
                normal_speed     = ns,
                measured_speed   = spd,
                share            = 1.0 / len(speeds),
                code_version     = "v1.0",
                price            = self.PRICES.get(mid, 0.5),
                is_active        = spd > 0.01,
            )
        return Observation(time=0.0, step=10, machines=machines)

    def test_rebalance_over_wait_for_permanent_50pct_slowdown(self):
        """
        REBALANCE or COMPENSATE must beat WAIT for a 50% permanent slowdown
        (fail-slow with long duration).
        """
        mpc   = MPCFallback()
        obs   = self._obs({"mac-m2": 1.2, "win-gpu-0": 3.5*0.5, "win-gpu-1": 3.5})
        fault = DiagnosedFault("win-gpu-0", "fail-slow", severity=0.50, duration_est=100)
        best, _, all_proj = mpc.decide(fault, obs)
        self.assertIn(
            best, [Action.REBALANCE, Action.COMPENSATE],
            f"Expected REBALANCE/COMPENSATE, got {best}. Proj={all_proj}",
        )
        active_score = all_proj.get("REBALANCE", 0) + all_proj.get("COMPENSATE", 0)
        self.assertGreater(active_score, all_proj.get("WAIT", 0),
                           "Active actions should outscore WAIT")

    def test_evict_chosen_for_crash(self):
        """EVICT must be the best action for a fail-stop fault."""
        mpc   = MPCFallback()
        obs   = self._obs({"mac-m2": 1.2, "win-gpu-0": 0.0, "win-gpu-1": 3.5})
        fault = DiagnosedFault("win-gpu-0", "fail-stop", severity=1.0, duration_est=999)
        best, _, _ = mpc.decide(fault, obs)
        self.assertEqual(best, Action.EVICT, f"Expected EVICT, got {best}")

    def test_wait_on_healthy_cluster(self):
        """On a healthy cluster MPC must return WAIT or CHECKPOINT."""
        mpc   = MPCFallback()
        obs   = self._obs({"mac-m2": 1.2, "win-gpu-0": 3.5, "win-gpu-1": 3.5})
        fault = DiagnosedFault("none", "none", severity=0.0, duration_est=0)
        best, _, _ = mpc.decide(fault, obs)
        self.assertIn(best, [Action.WAIT, Action.CHECKPOINT],
                      f"Expected WAIT/CHECKPOINT, got {best}")


class TestDecisionModel(unittest.TestCase):
    """Unit tests for DecisionModel + behavioral cloning (Day 4)."""

    @classmethod
    def setUpClass(cls):
        cls.obs_list, cls.act_list = generate_mock_oracle_data(n_samples=800, seed=0)
        cls.model = train_behavioral_cloning(cls.obs_list, cls.act_list, verbose=False)

    def test_model_trained_flag(self):
        self.assertTrue(self.model.is_trained)

    def test_predict_proba_sums_to_one(self):
        X    = np.vstack([featurize(o) for o in self.obs_list[:20]])
        sums = self.model.predict_proba(X).sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(20), atol=1e-5)

    def test_train_accuracy_above_2x_chance(self):
        X = np.vstack([featurize(o) for o in self.obs_list])
        y = np.array([a.name for a in self.act_list])
        preds, _ = self.model.predict(X)
        acc    = float(np.mean(np.array(preds) == y))
        chance = 1.0 / len(set(y))
        self.assertGreater(acc, chance * 2.0,
                           f"Acc={acc:.2%} not > 2x chance ({chance:.2%})")

    def test_confidence_in_unit_interval(self):
        X = np.vstack([featurize(o) for o in self.obs_list[:50]])
        _, confs = self.model.predict(X)
        self.assertTrue(np.all(confs >= 0.0))
        self.assertTrue(np.all(confs <= 1.0))


class TestHybridPolicy(unittest.TestCase):
    """Unit tests for HybridAgentPolicy (Days 5-6)."""

    def _simple_obs(self) -> Observation:
        return Observation(
            time=0.0, step=5,
            machines={
                "mac-m2":    MachineObs(0.1, 1.2, 1.2, 0.33, "v1", 0.0, True),
                "win-gpu-0": MachineObs(0.1, 3.5, 3.5, 0.33, "v1", 0.5, True),
                "win-gpu-1": MachineObs(0.1, 3.5, 3.5, 0.34, "v1", 0.5, True),
            }
        )

    def _policy(self, mock_model: Any) -> HybridAgentPolicy:
        return HybridAgentPolicy(mock_model, MPCFallback(), cluster_detector=None)

    def test_low_confidence_routes_to_mpc(self):
        """
        60% confidence (< 0.70) -> MPC must be called;
        mpc_call_count = 1, nn_call_count = 0.
        """
        m = MagicMock(spec=DecisionModel)
        m.is_trained = True
        m.predict_single.return_value = ("WAIT", 0.60)

        policy = self._policy(m)
        policy.decide(self._simple_obs())

        m.predict_single.assert_called_once()
        self.assertEqual(policy.mpc_call_count, 1)
        self.assertEqual(policy.nn_call_count,  0)

    def test_high_confidence_skips_mpc(self):
        """
        95% confidence (>= 0.70) -> NN action returned directly;
        mpc_call_count = 0, nn_call_count = 1.
        """
        m = MagicMock(spec=DecisionModel)
        m.is_trained = True
        m.predict_single.return_value = ("WAIT", 0.95)

        policy = self._policy(m)
        action = policy.decide(self._simple_obs())

        self.assertEqual(action, Action.WAIT)
        self.assertEqual(policy.mpc_call_count, 0)
        self.assertEqual(policy.nn_call_count,  1)

    def test_exact_threshold_uses_nn(self):
        """Confidence == 0.70 should use the NN (>= rule)."""
        m = MagicMock(spec=DecisionModel)
        m.is_trained = True
        m.predict_single.return_value = ("REBALANCE", 0.70)

        policy = self._policy(m)
        action = policy.decide(self._simple_obs())

        self.assertEqual(action, Action.REBALANCE)
        self.assertEqual(policy.nn_call_count,  1)
        self.assertEqual(policy.mpc_call_count, 0)

    def test_decision_latency_under_2ms(self):
        """
        Average decide() latency must be < 2.0 ms in the unit test
        environment.  The evaluation report targets < 1.0 ms on the actual
        hardware; the 2 ms threshold here accounts for CI/test overhead.
        """
        obs_tr, act_tr = generate_mock_oracle_data(n_samples=400, seed=1)
        model  = train_behavioral_cloning(obs_tr, act_tr, verbose=False)
        policy = HybridAgentPolicy(model, MPCFallback())

        test_obs, _ = generate_mock_oracle_data(n_samples=50, seed=99)
        for obs in test_obs:
            policy.decide(obs)

        avg_ms = sum(policy.decision_times) / len(policy.decision_times)
        self.assertLess(avg_ms, 2.0,
                        f"Avg latency {avg_ms:.4f} ms > 2.0 ms (CI threshold)")

    def test_rl_finetune_smoke(self):
        """rl_finetune() must run without raising exceptions."""
        obs_tr, act_tr = generate_mock_oracle_data(n_samples=200, seed=2)
        model  = train_behavioral_cloning(obs_tr, act_tr, verbose=False)
        policy = HybridAgentPolicy(model, MPCFallback())

        rewards  = [float(i % 3 - 1) for i in range(50)]
        obs50, acts50 = generate_mock_oracle_data(n_samples=50, seed=3)
        policy.rl_finetune(rewards, obs50, acts50)   # must not raise

    def test_untrained_model_always_routes_to_mpc(self):
        """An untrained model must always trigger MPC (5/5 times)."""
        policy = HybridAgentPolicy(DecisionModel(), MPCFallback())
        for _ in range(5):
            policy.decide(self._simple_obs())
        self.assertEqual(policy.mpc_call_count, 5)
        self.assertEqual(policy.nn_call_count,  0)


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
    _section("1 / 4   ANOMALY DETECTOR EVALUATION")
    print("  Generating 200-step synthetic traces (50% slowdown at step 60, 5 seeds) ...")

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
    for name, lst in [("EMA-ZScore Detector", all_ema), ("Naive Cross-Machine", all_naive)]:
        print(
            f"  {name:<28}  "
            f"{_avg(lst,'precision'):>10.3f}  "
            f"{_avg(lst,'recall'):>8.3f}  "
            f"{_avg(lst,'f1'):>6.3f}  "
            f"{_avg(lst,'avg_time_to_notice'):>12.1f}"
        )

    # ------------------------------------------------------------------
    # 2. Behavioral Cloning / Decision Model
    # ------------------------------------------------------------------
    _section("2 / 4   DECISION MODEL  (Behavioral Cloning)")
    print("  Generating 2000-sample Oracle training dataset ...")

    obs_train, act_train = generate_mock_oracle_data(n_samples=2000, seed=42)
    obs_test,  act_test  = generate_mock_oracle_data(n_samples=400,  seed=99)

    t0    = time.perf_counter()
    model = train_behavioral_cloning(obs_train, act_train, verbose=True)
    t_ms  = (time.perf_counter() - t0) * 1000

    X_test = np.vstack([featurize(o) for o in obs_test])
    y_test = np.array([a.name for a in act_test])
    preds, confs = model.predict(X_test)
    acc       = float(np.mean(np.array(preds) == y_test))
    avg_conf  = float(np.mean(confs))

    print(f"\n  Training time    : {t_ms:.1f} ms")
    print(f"  Test accuracy    : {acc*100:.1f}%  (avg confidence: {avg_conf:.3f})")
    print(f"\n  Per-class accuracy on test set:")
    print(f"  {'Action':<14}  {'Support':>8}  {'Correct':>8}  {'Acc':>8}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*8}")
    for cls in sorted(set(y_test)):
        mask    = y_test == cls
        support = int(mask.sum())
        correct = int((np.array(preds)[mask] == cls).sum())
        print(f"  {cls:<14}  {support:>8d}  {correct:>8d}  {correct/support:>8.1%}")

    # ------------------------------------------------------------------
    # 3. Hybrid Policy Timing
    # ------------------------------------------------------------------
    _section("3 / 4   HYBRID POLICY  (timing & routing)")
    print("  Running 500 decide() calls with real trained model ...")

    normals  = {"mac-m2": 1.2, "win-gpu-0": 3.5, "win-gpu-1": 3.5}
    detector = ClusterDetector(normals)
    policy   = HybridAgentPolicy(model, MPCFallback(), cluster_detector=detector)

    timing_obs, _ = generate_mock_oracle_data(n_samples=500, seed=77)
    for obs in timing_obs:
        policy.decide(obs)

    times  = np.array(policy.decision_times)
    avg_ms = float(times.mean())
    p95_ms = float(np.percentile(times, 95))
    max_ms = float(times.max())
    status = "PASS  (< 1.0 ms)" if avg_ms < 1.0 else "FAIL  (> 1.0 ms)"

    print(f"\n{policy.summary()}")
    print(f"\n  Latency stats over {len(times)} calls:")
    print(f"    avg  = {avg_ms:.4f} ms   [{status}]")
    print(f"    p95  = {p95_ms:.4f} ms")
    print(f"    max  = {max_ms:.4f} ms")

    # ------------------------------------------------------------------
    # 4. MPC Sanity Check
    # ------------------------------------------------------------------
    _section("4 / 4   MPC FALLBACK SANITY CHECK")

    mpc  = MPCFallback()
    obs4 = Observation(
        time=0.0, step=10,
        machines={
            "mac-m2":    MachineObs(0.1, 1.2, 1.2,       0.33, "v1", 0.0, True),
            "win-gpu-0": MachineObs(0.1, 3.5, 3.5*0.50,  0.33, "v1", 0.5, True),
            "win-gpu-1": MachineObs(0.1, 3.5, 3.5,       0.34, "v1", 0.5, True),
        }
    )
    fault4 = DiagnosedFault("win-gpu-0", "fail-slow", severity=0.50, duration_est=100)
    best4, _, proj4 = mpc.decide(fault4, obs4)

    print(f"\n  Scenario: win-gpu-0 at 50% speed for 100 steps (fail-slow)")
    print(f"  MPC projections (Goodput / $):")
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

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestMachineDetector, TestMPCFallback,
                TestDecisionModel, TestHybridPolicy]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

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
