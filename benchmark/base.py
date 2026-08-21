"""
The benchmark harness: one protocol every model satisfies, and a resumable
runner that scores them all with identical code.

The protocol is deliberately narrow. A model must be able to produce a
CUMULATIVE INCIDENCE FUNCTION in the exact shape `rp_core.metrics.evaluate()`
consumes; anything that cannot is not eligible as a baseline, because absolute
risk is the endpoint this project is about. Everything else - how it is
fitted, what library it wraps, whether it is classical or neural - is the
model's own business.

The deep-learning baselines (DeepHit, Neural Fine-Gray, DSM) land in a later
task and slot in here without changing this file.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class SurvivalBaseline(Protocol):
    """Fit on training rows, predict CIFs for any rows."""

    name: str

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray,
            **kw) -> None: ...

    def predict_cif(self, X: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """(n, T, K=2) cumulative incidence, in the layout `evaluate()` takes."""
        ...


def derive_seed(model: str, fold, imputation, base: int = 42) -> int:
    """Per-fit seed from (model, fold, imputation).

    Deterministic and independent across cells, so one model's random state
    cannot depend on how many fits ran before it - which would make results
    depend on execution order.
    """
    key = f"{base}|{model}|{fold}|{imputation}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % (2 ** 31 - 1)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

RECORD_COLUMNS = [
    "model", "fold", "imputation", "eval_set", "cause", "horizon",
    "auc", "cindex", "brier", "ici", "e50", "e90", "emax",
    "citl_predicted", "citl_observed", "citl_difference",
    "coherence_max", "n_rows", "n_events_cause", "fit_seconds",
    "predict_seconds", "calibration_method", "seed", "converged",
    "null_brier", "beats_null",
]


def records_from_evaluation(result: dict, model: str, fold, imputation,
                            eval_set: str, fit_seconds: float,
                            predict_seconds: float, seed: int,
                            causes=(1, 2), converged=None, null_result=None) -> list[dict]:
    """Flatten one `evaluate()` result into LONG format - one row per
    (model, fold, imputation, cause, horizon).

    Long format because every downstream question (pool over imputations,
    compare models within a horizon, subgroup tables) is then a groupby rather
    than a reshape.
    """
    horizons = result["meta"]["t_eval"]
    coherence = result.get("checks", {}).get("coherence", {}).get(
        "max_abs_deviation", np.nan)
    rows = []
    for k in causes:
        auc = result["discrimination"]["auc"].get(k, [np.nan] * len(horizons))
        cidx = result["discrimination"].get("cindex", {}).get(
            k, [np.nan] * len(horizons))
        brier = result["calibration"]["brier"].get(k, [np.nan] * len(horizons))
        null_brier = (null_result["calibration"]["brier"].get(
                          k, [np.nan] * len(horizons))
                      if null_result else [np.nan] * len(horizons))
        austin = result["calibration"].get("austin", {}).get(k)
        citl = result["calibration"].get("citl", {}).get(k)
        for j, t in enumerate(horizons):
            a = austin[j] if austin else {}
            c = citl[j] if citl else {}
            b_model, b_null = _at(brier, j), _at(null_brier, j)
            beats = (bool(b_model < b_null)
                     if np.isfinite(b_model) and np.isfinite(b_null) else None)
            rows.append({
                "model": model, "fold": fold, "imputation": imputation,
                "eval_set": eval_set, "cause": int(k), "horizon": float(t),
                "auc": _at(auc, j), "cindex": _at(cidx, j),
                "brier": _at(brier, j),
                "ici": a.get("ICI", np.nan), "e50": a.get("E50", np.nan),
                "e90": a.get("E90", np.nan), "emax": a.get("Emax", np.nan),
                "citl_predicted": c.get("predicted_mean", np.nan),
                "citl_observed": c.get("observed_aj", np.nan),
                "citl_difference": c.get("difference", np.nan),
                "coherence_max": coherence,
                "n_rows": result["meta"]["n"],
                "n_events_cause": result["meta"]["n_events"].get(k, np.nan),
                "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "calibration_method": a.get("method", ""),
                "seed": seed,
                # Recorded per cause: a baseline whose optimiser did not
                # converge is not a fair comparator, and 25 cells run
                # unattended is exactly where that goes unnoticed.
                "converged": (True if converged is None
                              else bool(converged.get(k, True))),
                "null_brier": b_null,
                "beats_null": beats,
            })
    return rows


def _at(seq, j):
    try:
        return float(seq[j])
    except (TypeError, IndexError):
        return float("nan")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkCell:
    """One unit of work: fit a model on one training set, score it on one or
    more evaluation sets."""
    model_name: str
    fold: object
    imputation: object
    train: dict                      # {"X":, "time":, "event":}
    evaluate_on: dict                # eval_set name -> same dict shape
    factory: object = None           # callable(seed) -> SurvivalBaseline
    fit_kwargs: dict = field(default_factory=dict)


def _survival_for(model, X, t_eval, cif) -> np.ndarray:
    """Overall survival to score against, from the model where it has one.

    This matters more than it looks. If S were always taken as 1 - F1 - F2 then
    `F1 + F2 + S - 1` would be identically zero and the coherence column - a
    paper result, not merely a test - would measure nothing.

    A model that carries its own survival (Royston-Parmar, cause-specific Cox)
    supplies it, and the residual is that model's real integration error. A
    model that does not (Fine-Gray, whose two subdistribution fits know nothing
    of each other; RSF) has no survival to give, so the clipped complement is
    used and the residual becomes exactly the excess mass max(0, F1 + F2 - 1) -
    the quantity worth reporting for those models.
    """
    fn = getattr(model, "predict_survival", None)
    if callable(fn):
        return np.asarray(fn(X, t_eval), dtype=np.float64)
    return np.clip(1.0 - cif.sum(axis=-1), 0.0, 1.0)



def _null_cif(time_train, event_train, t_eval, n_rows, causes=(1, 2)):
    """Marginal Aalen-Johansen from TRAINING rows, broadcast to every
    evaluation row — the floor any usable model must clear.

    A model that cannot beat 'predict the population average for everyone'
    has not learned anything, and a silent training failure looks exactly
    like this while raising nothing.
    """
    from metrics.calibration import aalen_johansen
    aj = np.stack(
        [aalen_johansen(time_train, event_train, t_eval,
                        cause=k, variance=False)[0]
         for k in causes], axis=-1)                    # (T, K)
    return np.broadcast_to(aj[None], (n_rows, len(t_eval), len(causes))).copy()



def _completed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        done = pd.read_parquet(path, columns=["model", "fold", "imputation",
                                              "eval_set"])
    except Exception:
        return set()
    return set(map(tuple, done.drop_duplicates().to_numpy().tolist()))


def run_benchmark(cells, t_eval, output_path, evaluate_fn=None,
                  resume: bool = True, verbose: bool = True,
                  evaluate_kwargs: dict | None = None) -> pd.DataFrame:
    """Fit and score every cell, appending long-format records to `output_path`.

    RESUMABLE: completed (model, fold, imputation, eval_set) combinations are
    skipped, so an interrupted run continues rather than restarting. Because
    each cell's seed comes from `derive_seed`, a resumed run reproduces exactly
    what an uninterrupted one would have produced.

    `evaluate_fn` defaults to `rp_core.metrics.evaluate`; it is injectable so
    tests can run a cheap version.
    """
    output_path = Path(output_path)
    if evaluate_fn is None:
        from metrics import evaluate as evaluate_fn
    evaluate_kwargs = dict(evaluate_kwargs or {})
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))

    done = _completed_keys(output_path) if resume else set()
    existing = (pd.read_parquet(output_path)
                if resume and output_path.exists() else None)
    new_rows = []
    null_cache = {}

    for cell in cells:
        keys = [(cell.model_name, cell.fold, cell.imputation, name)
                for name in cell.evaluate_on]
        if all(k in done for k in keys):
            if verbose:
                print(f"  skip (done): {cell.model_name} fold={cell.fold} "
                      f"m={cell.imputation}")
            continue

        seed = derive_seed(cell.model_name, cell.fold, cell.imputation)
        model = cell.factory(seed)
        t0 = time.time()
        model.fit(cell.train["X"], cell.train["time"], cell.train["event"],
                  **cell.fit_kwargs)
        fit_seconds = time.time() - t0

        for name, data in cell.evaluate_on.items():
            if (cell.model_name, cell.fold, cell.imputation, name) in done:
                continue
            t0 = time.time()
            cif = model.predict_cif(data["X"], t_eval)
            predict_seconds = time.time() - t0
            surv = _survival_for(model, data["X"], t_eval, cif)
            result = evaluate_fn(cif, data["time"], data["event"], t_eval,
                                 surv=surv, **evaluate_kwargs)
            
            nkey = (cell.fold, cell.imputation, name)
            if nkey not in null_cache:
                ncif = _null_cif(cell.train["time"], cell.train["event"],
                                 t_eval, len(data["time"]))
                null_cache[nkey] = evaluate_fn(
                    ncif, data["time"], data["event"], t_eval,
                    surv=1.0 - ncif.sum(axis=-1),
                    **{**evaluate_kwargs, "bootstrap": False,
                       "do_cindex": False, "calibration": False})
                    #**{**evaluate_kwargs, "n_boot": 0})
            null_result = null_cache[nkey]

            failed = [k for k in (1, 2)
                      for j, t in enumerate(t_eval)
                      if _at(result["calibration"]["brier"].get(k, []), j)
                      >= _at(null_result["calibration"]["brier"].get(k, []), j)]
            if failed:
                print(f"  !! {cell.model_name} fold={cell.fold} "
                      f"m={cell.imputation} {name}: DOES NOT BEAT NULL for "
                      f"(cause, horizon) {failed} - treat this cell as unusable")
                
            converged = getattr(model, "converged", None)
            if isinstance(converged, dict) and not all(converged.values()):
                bad = [k for k, ok in converged.items() if not ok]
                print(f"  !! {cell.model_name} fold={cell.fold} "
                      f"m={cell.imputation}: NOT CONVERGED for cause(s) {bad} "
                      f"- treat this cell as unusable")
            new_rows += records_from_evaluation(
                result, cell.model_name, cell.fold, cell.imputation, name,
                fit_seconds, predict_seconds, seed,
                converged=converged if isinstance(converged, dict) else None,
                null_result=null_result)
            if verbose:
                print(f"  {cell.model_name:12s} fold={cell.fold} "
                      f"m={cell.imputation} {name:12s} "
                      f"fit {fit_seconds:6.1f}s predict {predict_seconds:5.1f}s "
                      f"coherence {result.get('checks', {}).get('coherence', {}).get('max_abs_deviation', float('nan')):.2e}")

        if new_rows:
            frame = pd.DataFrame(new_rows, columns=RECORD_COLUMNS)
            out = (pd.concat([existing, frame], ignore_index=True)
                   if existing is not None else frame)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, index=False)

    frame = pd.DataFrame(new_rows, columns=RECORD_COLUMNS)
    if existing is not None:
        frame = pd.concat([existing, frame], ignore_index=True)
    if len(frame):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
    return frame


def coherence_table(results: pd.DataFrame) -> pd.DataFrame:
    """max |F1 + F2 + S - 1| per model.

    A paper result, not merely a test. It is structural for the Royston-Parmar
    formulation, depends on the integration for Cox, and is not guaranteed at
    all for Fine-Gray, whose two subdistribution models are fitted
    independently.
    """
    return (results.groupby("model")["coherence_max"]
            .agg(["max", "mean", "count"])
            .rename(columns={"max": "worst", "mean": "average",
                             "count": "n_records"})
            .sort_values("worst"))
