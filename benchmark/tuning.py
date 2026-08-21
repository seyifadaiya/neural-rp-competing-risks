"""
One hyperparameter search for every model, classical and neural alike.

A baseline tuned less carefully than the proposed model is not a baseline, so
the point of this module is that the SEARCH PROCEDURE is shared code rather
than shared intention: same sampler, same budget, same folds, same selection
rule, and every search space declared in one place where the budgets can be
compared at a glance.

Two rules are enforced rather than documented:

  * Tuning runs on the DEVELOPMENT window only, imputation m = 1. The temporal
    and exploratory windows are held out for evaluation, and touching them
    during a search would quietly invalidate every number computed on them
    afterwards. `assert_tunable` raises rather than warns.
  * Selection uses held-out LOG-LIKELIHOOD where the model has one and
    integrated Brier score where it does not. Which criterion was used is
    recorded per model - comparing a likelihood-selected model against a
    Brier-selected one is fine, but only if the reader knows.

Every trial is logged (config, per-fold scores, wall time) to parquet. That
log is a supplementary table, not a debugging aid: it is what lets a reviewer
see that the comparators were given a real search.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Windows that must never be seen during a search.
FORBIDDEN_WINDOWS = ("temporal", "exploratory")

TUNING_IMPUTATION = 1

TRIAL_COLUMNS = [
    "model", "trial", "criterion", "score", "score_std", "n_folds",
    "fit_seconds", "seed", "config", "failed", "error",
]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def assert_tunable(window=None, imputation=None) -> None:
    """Refuse to search on held-out windows or the wrong imputation.

    This is the one place where a mistake is both easy to make and impossible
    to detect afterwards from the results table, so it raises.
    """
    if window is not None:
        w = np.asarray(window)
        present = set(np.unique(w).tolist())
        bad = present.intersection(FORBIDDEN_WINDOWS)
        if bad:
            raise ValueError(
                f"tuning data contains held-out window(s) {sorted(bad)}. "
                f"Hyperparameter search runs on the development window only; "
                f"anything selected against temporal or exploratory rows makes "
                f"the evaluation on those rows meaningless.")
    if imputation is not None and int(imputation) != TUNING_IMPUTATION:
        raise ValueError(
            f"tuning uses imputation m={TUNING_IMPUTATION} only, got "
            f"m={imputation}. Searching across imputations would spend the "
            f"budget on imputation noise rather than on hyperparameters.")


# ---------------------------------------------------------------------------
# Search spaces - all in one place, so budgets are visibly comparable
# ---------------------------------------------------------------------------

@dataclass
class SearchSpace:
    """A model's grid plus the criterion it is selected on."""
    name: str
    grid: dict
    criterion: str = "ibs"            # "loglik" (higher better) | "ibs" (lower)
    notes: str = ""

    def size(self) -> int:
        n = 1
        for v in self.grid.values():
            n *= len(v)
        return n

    def sample(self, rng) -> dict:
        return {k: _pick(rng, v) for k, v in self.grid.items()}

    def enumerate(self) -> list:
        keys = list(self.grid)
        return [dict(zip(keys, combo))
                for combo in itertools.product(*(self.grid[k] for k in keys))]


def _pick(rng, values):
    v = values[int(rng.integers(len(values)))]
    return list(v) if isinstance(v, tuple) else v


# Widths are drawn from the same ladder for every neural model, so no model
# gets a quietly larger capacity budget than another.
_WIDTHS = [(128, 64), (256, 128), (512, 256)]
_HEADS = [(32,), (64,), (128,)]
_DROPOUT = [0.0, 0.1, 0.2, 0.3]
_LR = [3e-4, 1e-3, 3e-3]
_BATCH = [128, 256, 512]

SEARCH_SPACES = {
    "deephit": SearchSpace(
        "deephit",
        {"hidden": _WIDTHS, "head": _HEADS, "dropout": _DROPOUT, "lr": _LR,
         "batch_size": _BATCH,
         # The discretisation is a modelling choice that materially changes
         # the answer, so it is tuned and reported, not fixed silently.
         "num_durations": [20, 50, 100],
         "alpha": [0.0, 0.2, 0.5], "sigma": [0.1, 0.25, 0.5]},
        criterion="ibs",
        notes="DeepHit's training loss mixes likelihood with a ranking term, "
              "so it is not a likelihood and cannot be used for selection."),
    "neural_fine_gray": SearchSpace(
        "neural_fine_gray",
        {"hidden": _WIDTHS, "head": _HEADS, "dropout": _DROPOUT, "lr": _LR,
         "batch_size": _BATCH, "act": ["ReLU", "Tanh"]},
        criterion="loglik",
        notes="exact likelihood by automatic differentiation"),
    "dsm": SearchSpace(
        "dsm",
        {"hidden": _WIDTHS, "dropout": _DROPOUT, "lr": _LR,
         "batch_size": _BATCH,
         "k": [1, 3, 4, 6, 8],
         "distribution": ["Weibull", "LogNormal"]},
        criterion="loglik",
        notes="mixture components k tuned; parametric-mixture contrast"),
    "variant_a": SearchSpace(
        "variant_a",
        {"hidden_dims": _WIDTHS, "latent_dim": [32, 64, 128],
         "dropout": _DROPOUT, "encoder_lr": _LR, "spline_lr": [3e-3, 1e-2, 3e-2]},
        criterion="loglik",
        notes="the proposed model, tuned by the same procedure as its "
              "comparators"),
    "variant_b": SearchSpace(
        "variant_b",
        {"hidden_dims": _WIDTHS, "latent_dim": [32, 64, 128],
         "dropout": _DROPOUT, "encoder_lr": _LR, "spline_lr": [3e-3, 1e-2, 3e-2],
         "head_lr": _LR},
        criterion="loglik",
        notes="the proposed model, tuned by the same procedure as its "
              "comparators"),
}


def search_space_table() -> pd.DataFrame:
    """The supplementary table: what each model was allowed to search over."""
    rows = []
    for name, sp in SEARCH_SPACES.items():
        rows.append({"model": name, "criterion": sp.criterion,
                     "n_hyperparameters": len(sp.grid),
                     "grid_size": sp.size(),
                     "parameters": ", ".join(sorted(sp.grid)),
                     "notes": sp.notes})
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_fold(model, X_val, time_val, event_val, t_eval, criterion: str,
               n_grid: int = 50):
    """One fold's score, oriented so that HIGHER IS ALWAYS BETTER.

    Integrated Brier is negated rather than left as-is, because a selection
    rule that silently flips direction per model is how the wrong
    configuration gets chosen without anyone noticing.
    """
    if criterion == "loglik":
        fn = getattr(model, "compute_nll", None)
        if fn is None:
            raise ValueError(f"{getattr(model, 'name', model)} declares the "
                             f"'loglik' criterion but exposes no compute_nll")
        return -float(fn(X_val, time_val, event_val))

    from metrics import integrated_brier_score
    # IBS integrates over [0, t_max] on its own dense grid, so it takes a
    # prediction FUNCTION rather than predictions at `t_eval` - the metrics
    # package never holds a model.
    ibs = integrated_brier_score(lambda g: model.predict_cif(X_val, g),
                                 time_val, event_val,
                                 t_max=float(np.max(t_eval)), n_grid=n_grid)
    return -float(np.mean(list(ibs.values())))


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

@dataclass
class TuningResult:
    model: str
    criterion: str
    best_config: dict
    best_score: float
    budget: int
    n_evaluated: int
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        return {"model": self.model, "criterion": self.criterion,
                "best_config": self.best_config, "best_score": self.best_score,
                "budget": self.budget, "n_evaluated": self.n_evaluated}


def random_search(model_name: str, factory, folds, t_eval, budget: int = 30,
                  space: SearchSpace | None = None, seed: int = 42,
                  output_path=None, window=None, imputation=TUNING_IMPUTATION,
                  verbose: bool = True) -> TuningResult:
    """Random search over `space`, scored by cross-validation.

    `factory(config, seed) -> model` builds a model from one sampled config.
    `folds` is a sequence of (train, validate) dicts, each `{"X", "time",
    "event"}` - fold 0-4 of the development window at imputation m = 1.

    A trial that raises is RECORDED AS FAILED and the search continues. Deep
    models diverge on bad learning rates, and losing a whole search to one
    such config would be worse than losing one trial - but a silently dropped
    trial would make the budget a lie, so failures stay in the log.
    """
    space = space or SEARCH_SPACES[model_name]
    assert_tunable(window, imputation)

    rng = np.random.default_rng(seed)
    # Never evaluate the same config twice: with small grids, sampling with
    # replacement would silently spend the budget on duplicates.
    seen, configs = set(), []
    exhaustive = space.size() <= budget
    pool = space.enumerate() if exhaustive else None
    while len(configs) < (space.size() if exhaustive else budget):
        cfg = pool[len(configs)] if exhaustive else space.sample(rng)
        key = repr(sorted((k, tuple(v) if isinstance(v, list) else v)
                          for k, v in cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)

    rows, best, best_cfg = [], -np.inf, None
    for i, cfg in enumerate(configs):
        t0 = time.time()
        scores, failed, err = [], False, ""
        try:
            for f, fold in enumerate(folds):
                model = factory(cfg, seed + 1000 * i + f)
                model.fit(fold["train"]["X"], fold["train"]["time"],
                          fold["train"]["event"])
                scores.append(score_fold(
                    model, fold["validate"]["X"], fold["validate"]["time"],
                    fold["validate"]["event"], t_eval, space.criterion))
        except Exception as exc:                       # noqa: BLE001
            failed, err = True, f"{type(exc).__name__}: {exc}"[:300]
        elapsed = time.time() - t0

        mean = float(np.mean(scores)) if scores and not failed else float("nan")
        rows.append({
            "model": model_name, "trial": i, "criterion": space.criterion,
            "score": mean,
            "score_std": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "n_folds": len(scores), "fit_seconds": elapsed,
            "seed": seed + 1000 * i, "config": repr(cfg),
            "failed": failed, "error": err,
        })
        if not failed and np.isfinite(mean) and mean > best:
            best, best_cfg = mean, cfg
        if verbose:
            tag = "FAILED" if failed else f"{mean:+.5f}"
            print(f"  [{model_name}] trial {i + 1}/{len(configs)} "
                  f"{tag:>12s}  {elapsed:6.1f}s  {cfg}")

    trials = pd.DataFrame(rows, columns=TRIAL_COLUMNS)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            trials = pd.concat([pd.read_parquet(output_path), trials],
                               ignore_index=True)
        trials.to_parquet(output_path, index=False)

    if best_cfg is None:
        raise RuntimeError(
            f"every one of the {len(configs)} {model_name} trials failed; "
            f"see the trial log's `error` column")

    return TuningResult(model=model_name, criterion=space.criterion,
                        best_config=best_cfg, best_score=best,
                        budget=budget, n_evaluated=len(configs), trials=trials)


def make_cv_folds(X, time, event, fold_id, folds=(0, 1, 2, 3, 4)):
    """Development-window CV folds in the shape `random_search` wants.

    Uses the PRE-ASSIGNED `cv_fold` column rather than re-splitting, so the
    tuning folds are the same folds everything else uses - grouped by
    recipient and stratified, as `make_splits` built them.
    """
    fold_id = np.asarray(fold_id)
    out = []
    for f in folds:
        va = fold_id == f
        tr = ~va
        if not va.any() or not tr.any():
            continue
        out.append({
            "fold": int(f),
            "train": {"X": np.asarray(X)[tr], "time": np.asarray(time)[tr],
                      "event": np.asarray(event)[tr]},
            "validate": {"X": np.asarray(X)[va], "time": np.asarray(time)[va],
                         "event": np.asarray(event)[va]},
        })
    return out


def tuning_summary(results) -> pd.DataFrame:
    """One row per model: what was searched, how, and what won."""
    rows = []
    for r in results:
        rows.append({"model": r.model, "criterion": r.criterion,
                     "budget": r.budget, "n_evaluated": r.n_evaluated,
                     "grid_size": SEARCH_SPACES[r.model].size()
                     if r.model in SEARCH_SPACES else np.nan,
                     "best_score": r.best_score,
                     "best_config": repr(r.best_config)})
    return pd.DataFrame(rows)
