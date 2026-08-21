"""
Loading feature-builder output for the benchmark and tuning notebooks.

Lives here rather than in either notebook so the two cannot drift apart, and
so the loading rules - which are where a silent, invalidating error would hide
- are testable without running a notebook.

Two guards matter more than the rest:

  * NO OUTCOME COLUMN MAY REACH A DESIGN MATRIX. `GTIME_YEARS` or `event_type`
    leaking into X would produce a model that looks excellent and means
    nothing, and nothing downstream would notice.
  * THE TEMPORAL AND EXPLORATORY WINDOWS ARE NOT LOADED. They are touched once,
    at the very end of the project, after every modelling decision is locked.
    Loading them here - even to "just look" - spends them.

Both raise rather than warn.
"""

from __future__ import annotations

import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Columns the feature builder passes through alongside the design matrix. If
# any of these appears among the feature names, the build is wrong.
FORBIDDEN_FEATURE_COLUMNS = frozenset({
    "GTIME_YEARS", "event_type", "PT_CODE", "window", "cv_fold", "TX_DATE",
    "pretrain_split",
})

# Never loaded by these notebooks.
HELD_OUT_SETS = frozenset({"temporal", "exploratory"})

TIME_COLUMN = "GTIME_YEARS"
EVENT_COLUMN = "event_type"
CAUSES = (1, 2)


# ---------------------------------------------------------------------------
# Logging - these runs take hours, so a stalled run must look different from a
# slow one.
# ---------------------------------------------------------------------------

def log(message: str, **fields) -> None:
    """Timestamped, flushed progress line."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extra = "  ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[{stamp}] {message}{('  ' + extra) if extra else ''}", flush=True)


def human(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def assert_not_held_out(*names: str) -> None:
    """Refuse to touch the temporal or exploratory windows.

    These are the project's last untouched data. Every look at them - fitting,
    scoring, or plotting - costs some of the independence that makes the final
    evaluation worth reporting, so this raises rather than warns.
    """
    bad = sorted({n for n in names if n in HELD_OUT_SETS})
    if bad:
        raise ValueError(
            f"refusing to load held-out window(s) {bad}. These are evaluated "
            f"once, at the end of the project, after every modelling decision "
            f"is locked. If that moment has genuinely arrived, load them in a "
            f"dedicated final-evaluation notebook, not this one.")


def assert_no_outcome_columns(columns) -> None:
    """No outcome or administrative column may appear among the features."""
    leaked = sorted(set(map(str, columns)) & FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(
            f"outcome/administrative column(s) {leaked} found in the design "
            f"matrix. A model fitted on these would score superbly and mean "
            f"nothing, and no downstream check would catch it. Rebuild the "
            f"features before going further.")


# ---------------------------------------------------------------------------
# Paths and loading
# ---------------------------------------------------------------------------

def fold_sets(fold: int) -> tuple:
    """The (train, test) applied-set names for one development fold."""
    return f"dev_fold{fold}_train", f"dev_fold{fold}_test"


def _stem(features_dir: Path, applied_to: str, imputation: int) -> Path:
    return Path(features_dir) / f"{applied_to}_m{int(imputation)}"


def load_set(features_dir, fit_set: str, applied_to: str, imputation: int = 1,
             representation: str = "onehot") -> dict:
    """One applied set as `{"X", "time", "event", "columns", ...}`.

    `fit_set` names the set whose `FeatureSpec` produced this one; it is used
    to locate the spec and cross-check the column list, which is what would
    catch a train/test pair built from different specs.

    The feature builder writes arrays as .npz and the outcome columns as a
    separate .parquet, so this reads both and rejoins them:

        {applied_to}_m{m}_embedding.npz   continuous, categorical_codes,
                                          columns, cardinalities
        {applied_to}_m{m}_onehot.npz      design, columns
        {applied_to}_m{m}_outcomes.parquet

    For `representation="embedding"` the returned X is
    `[continuous | categorical codes]`, which is the layout the deep baselines
    take; `n_continuous` and `cardinalities` come back alongside it so they can
    be passed on without being re-derived.
    """
    if representation not in ("onehot", "embedding"):
        raise ValueError(f"unknown representation {representation!r}")
    assert_not_held_out(fit_set, applied_to)

    features_dir = Path(features_dir)
    stem = _stem(features_dir, applied_to, imputation)
    arrays_path = Path(f"{stem}_{representation}.npz")
    outcomes_path = Path(f"{stem}_outcomes.parquet")
    for p in (arrays_path, outcomes_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Expected the feature builder's output for "
                f"'{applied_to}' at imputation m={imputation}; check "
                f"FEATURES_DIR and that the build has run.")

    d = np.load(arrays_path, allow_pickle=True)
    columns = [str(c) for c in d["columns"].tolist()]
    assert_no_outcome_columns(columns)

    if representation == "onehot":
        X = np.asarray(d["design"], dtype=np.float64)
        cardinalities, n_continuous = [], X.shape[1]
    else:
        cont = np.asarray(d["continuous"], dtype=np.float64)
        codes = np.asarray(d["categorical_codes"], dtype=np.float64)
        X = np.hstack([cont, codes]) if codes.size else cont
        cardinalities = [int(c) for c in d["cardinalities"].tolist()]
        n_continuous = cont.shape[1]

    y = pd.read_parquet(outcomes_path)
    for col in (TIME_COLUMN, EVENT_COLUMN):
        if col not in y.columns:
            raise KeyError(f"{outcomes_path} has no {col!r} column")
    time = y[TIME_COLUMN].to_numpy(dtype=np.float64)
    event = y[EVENT_COLUMN].to_numpy()
    event = np.asarray(event).astype(np.int64)

    if len(time) != len(X):
        raise ValueError(
            f"{applied_to} m{imputation}: {len(X)} feature rows but "
            f"{len(time)} outcome rows - the arrays and outcomes are not the "
            f"same set of patients")
    if not np.all(np.isfinite(time)) or time.min() <= 0:
        raise ValueError(f"{applied_to}: follow-up times must be positive and "
                         f"finite (min {np.nanmin(time)})")
    unexpected = set(np.unique(event).tolist()) - {0, *CAUSES}
    if unexpected:
        raise ValueError(f"{applied_to}: unexpected event codes {unexpected}; "
                         f"expected 0 (censored), 1, 2")

    out = {"X": X, "time": time, "event": event, "columns": columns,
           "cardinalities": cardinalities, "n_continuous": n_continuous,
           "representation": representation, "applied_to": applied_to,
           "fit_set": fit_set, "imputation": int(imputation),
           "n_rows": len(time)}

    spec_path = features_dir / f"spec_{fit_set}_m{int(imputation)}.json"
    if spec_path.exists():
        out["spec_path"] = str(spec_path)
        with open(spec_path) as fh:
            spec = json.load(fh)
        expected = _expected_width(spec, representation)
        # A train/test pair built from different specs is the failure this
        # catches: same column count, different meaning per column.
        if expected is not None and expected != X.shape[1]:
            raise ValueError(
                f"{applied_to} has {X.shape[1]} {representation} columns but "
                f"spec_{fit_set}_m{imputation} implies {expected}. The applied "
                f"set was not built from this spec.")
    return out


def _expected_width(spec: dict, representation: str):
    """Design width implied by a serialised `FeatureSpec`.

    `n_onehot_features` and `n_embedding_features` are PROPERTIES on the
    dataclass and are not written to the JSON, so they have to be recomputed
    from the fields that are. Reading the absent keys instead would leave this
    check silently dead - passing not because the widths agree but because it
    never compared anything.
    """
    cont = spec.get("continuous_order")
    cats = spec.get("categorical_order")
    if cont is None or cats is None:
        return None
    if representation == "embedding":
        return len(cont) + len(cats)
    levels = spec.get("categorical_levels") or {}
    reference = spec.get("categorical_reference") or {}
    width = len(cont)
    for c in cats:
        lv = levels.get(c)
        if lv is None:
            return None
        # one column per NON-reference level
        width += sum(1 for v in lv if v != reference.get(c))
    return width


def load_fold(features_dir, fold: int, imputation: int = 1,
              representation: str = "onehot") -> tuple:
    """`(train, test)` for one development fold, both from the train fit."""
    fit, test = fold_sets(fold)
    return (load_set(features_dir, fit, fit, imputation, representation),
            load_set(features_dir, fit, test, imputation, representation))


def as_cell_data(loaded: dict) -> dict:
    """Strip a loaded set down to what `BenchmarkCell` takes."""
    return {"X": loaded["X"], "time": loaded["time"], "event": loaded["event"]}


def describe(loaded: dict) -> str:
    e = loaded["event"]
    counts = "  ".join(f"cause{k} {int((e == k).sum()):,}" for k in CAUSES)
    return (f"{loaded['applied_to']} m{loaded['imputation']} "
            f"[{loaded['representation']}]  {loaded['n_rows']:,} rows x "
            f"{loaded['X'].shape[1]} cols   {counts}  "
            f"censored {int((e == 0).sum()):,}")


def available_sets(features_dir) -> pd.DataFrame:
    """What the feature builder actually produced, for the config cell to
    print before anything long-running starts."""
    rows = []
    for p in sorted(Path(features_dir).glob("*_outcomes.parquet")):
        stem = p.name[: -len("_outcomes.parquet")]
        name, _, m = stem.rpartition("_m")
        rows.append({"applied_to": name, "imputation": int(m) if m else -1,
                     "held_out": name in HELD_OUT_SETS,
                     "onehot": Path(str(p)[:-len("_outcomes.parquet")]
                                    + "_onehot.npz").exists(),
                     "embedding": Path(str(p)[:-len("_outcomes.parquet")]
                                       + "_embedding.npz").exists()})
    return pd.DataFrame(rows).sort_values(["applied_to", "imputation"])


# ---------------------------------------------------------------------------
# RSF at production settings
# ---------------------------------------------------------------------------

# `RSFBaseline` exposes ntree/nodesize/ntime, which is all the test suite
# needed. The production run also wants nsplit, perf.type, block.size and
# rf.cores, and `rp_core` is not to be modified - so the extra options are
# substituted into the package's own R template here.
#
# The template is swapped in around the parent's `_run` rather than
# reimplementing it. Copying those thirty lines of CSV marshalling would risk
# them drifting out of step with the verified path, which is a worse failure
# than the swap being slightly unusual.
_EXTRA_RSF_OPTIONS = """
options(rf.cores = __RFCORES__)
"""


def _tuned_template(nsplit: int, perf_type: str, block_size, rf_cores) -> str:
    from benchmark.classical import RSF_R_TEMPLATE

    template = RSF_R_TEMPLATE
    block = "NULL" if block_size is None else str(int(block_size))
    cores = ("parallel::detectCores()" if rf_cores is None
             else str(int(rf_cores)))
    # Extra arguments onto the rfsrc() call the package already builds.
    old = ('fit <- rfsrc(f, data = train, ntree = __NTREE__, '
           'nodesize = __NODESIZE__,\n             ntime = __NTIME__,')
    if old not in template:
        raise RuntimeError(
            "the rp_core RSF template changed shape; update _tuned_template "
            "in notebooks/io_helpers.py rather than guessing")
    new = ('fit <- rfsrc(f, data = train, ntree = __NTREE__, '
           'nodesize = __NODESIZE__,\n             ntime = __NTIME__, '
           f'nsplit = {int(nsplit)}, perf.type = "{perf_type}",\n'
           f'             block.size = {block},')
    template = template.replace(old, new, 1)
    return (_EXTRA_RSF_OPTIONS.replace("__RFCORES__", cores)
            + template)


def production_rsf(columns=None, seed: int = 42, ntree: int = 500,
                   nodesize: int = 30, ntime: int = 100, nsplit: int = 10,
                   perf_type: str = "none", block_size=None, rf_cores=None,
                   timeout: int = 24 * 3600):
    """`RSFBaseline` with the production R options the benchmark run wants.

    `perf.type = "none"` and `block.size = NULL` switch off the OOB error
    bookkeeping, which is pure overhead here - the benchmark scores on held-out
    folds through `metrics.evaluate`, so nothing reads rfsrc's own estimate.
    `nsplit = 10` uses random split points rather than an exhaustive scan,
    which is where most of the saving comes from on continuous covariates.
    """
    from benchmark.classical import RSFBaseline

    template = _tuned_template(nsplit, perf_type, block_size, rf_cores)

    class _ProductionRSF(RSFBaseline):
        """RSFBaseline running the production R template."""

        def _run(self, X, t_eval):
            import benchmark.classical as classical
            original = classical.RSF_R_TEMPLATE
            classical.RSF_R_TEMPLATE = template
            try:
                return super()._run(X, t_eval)
            finally:
                classical.RSF_R_TEMPLATE = original

    model = _ProductionRSF(name="rsf", columns=columns, seed=seed, ntree=ntree,
                           nodesize=nodesize, ntime=ntime, timeout=timeout)
    model.r_options = {"ntree": ntree, "nodesize": nodesize, "ntime": ntime,
                       "nsplit": nsplit, "perf.type": perf_type,
                       "block.size": block_size,
                       "rf.cores": rf_cores or "detectCores()"}
    return model


def rsf_factory(columns=None, **kw):
    """`factory(seed) -> model`, the shape `run_benchmark` expects."""
    return lambda seed: production_rsf(columns=columns, seed=seed, **kw)


# ---------------------------------------------------------------------------
# Post-run checks
# ---------------------------------------------------------------------------

def beats_null(results, null_brier=None):
    """Per (model, fold, cause, horizon): is the Brier score better than the
    marginal predictor's?

    `null_brier` maps (cause, horizon) -> Brier of the Aalen-Johansen marginal.
    Without it, beats_null is left as NaN — "unknown" must not read as "passed".
    """
    out = results.copy()
    
    if "beats_null" in out.columns and out["beats_null"].notna().any():
        return out
    
    if null_brier:
        ref = out.apply(
            lambda r: null_brier.get((int(r["cause"]), float(r["horizon"])),
                                     np.nan), axis=1)
        out["null_brier"] = ref
        out["beats_null"] = out["brier"] < ref
    else:
        out["null_brier"] = np.nan
        out["beats_null"] = np.nan
    return out


def failure_report(results: pd.DataFrame) -> pd.DataFrame:
    """Cells worth looking at before trusting anything: non-convergence, or a
    model that failed to beat the null."""
    checked = beats_null(results)
    
    conv  = checked["converged"].fillna(True).astype(bool).to_numpy()
    beats = checked["beats_null"].fillna(True).astype(bool).to_numpy()
    bad = checked[(~conv) | (~beats)]
    
    #bad = checked[(~checked["converged"].fillna(True))
                  #| (~checked["beats_null"].fillna(True))]
    cols = ["model", "fold", "imputation", "eval_set", "cause", "horizon",
            "brier", "null_brier", "converged", "beats_null"]
    return bad[[c for c in cols if c in bad.columns]]


def add_rp_core_to_path(repo_root=None) -> Path:
    """Put `rp_core` on `sys.path` without assuming a working directory."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    pkg = root / "rp_core"
    if not pkg.exists():
        raise FileNotFoundError(f"rp_core not found under {root}")
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    return pkg


@contextlib.contextmanager
def timed(label: str):
    """Time a block and log both ends, so a stall is visible while it happens
    rather than only in hindsight."""
    import time as _time
    log(f"START  {label}")
    t0 = _time.time()
    try:
        yield
    finally:
        log(f"DONE   {label}", elapsed=human(_time.time() - t0))


# ---------------------------------------------------------------------------
# Variants A and B behind the SurvivalBaseline protocol
# ---------------------------------------------------------------------------

class VariantBaseline:
    """`VariantA` / `VariantB` in the shape `run_benchmark` and the tuning
    harness consume.

    The variants are `nn.Module`s taking `(log_t, x_cont, x_cat)`, not
    `SurvivalBaseline`s taking `(X, time, event)`, so something has to bridge
    them. It lives here rather than in `rp_core` because `rp_core` is not to be
    modified, and it is built entirely from that package's public API - the
    same `RCSBasis`, `time_knots`, `fit_scaler`, `train_model` and
    `cif.predict_cif` the verified tests use, so no estimation logic is
    reimplemented.

    `X` is the EMBEDDING representation, `[continuous | integer codes]`, split
    back apart here into the two tensors the encoder expects.
    """

    def __init__(self, variant: str = "variant_b", n_continuous=None,
                 cardinalities=(), columns=None, seed: int = 42,
                 hidden_dims=(256, 128), latent_dim: int = 64,
                 dropout: float = 0.2, encoder_lr: float = 1e-3,
                 spline_lr: float = 1e-2, head_lr: float = 1e-3,
                 df_time: int = 4, max_epochs: int = 200, batch_size: int = 512,
                 patience: int = 15, val_fraction: float = 0.15, **unused):
        if variant not in ("variant_a", "variant_b"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.name = variant
        self.n_continuous = n_continuous
        self.cardinalities = tuple(int(c) for c in cardinalities)
        self.columns = list(columns) if columns is not None else None
        self.seed = int(seed)
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.encoder_lr = float(encoder_lr)
        self.spline_lr = float(spline_lr)
        self.head_lr = float(head_lr)
        self.df_time = int(df_time)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.model = None

    # -- input -------------------------------------------------------------
    def _split(self, X):
        import torch
        X = np.asarray(X, dtype=np.float64)
        n_cont = (self.n_continuous if self.n_continuous is not None
                  else X.shape[1] - len(self.cardinalities))
        x_cont = torch.tensor(X[:, :n_cont], dtype=torch.float32)
        codes = X[:, n_cont:]
        x_cat = (torch.tensor(codes.astype(np.int64), dtype=torch.long)
                 if codes.shape[1] else None)
        return x_cont, x_cat

    def _batch(self, X, time, event):
        import torch
        x_cont, x_cat = self._split(X)
        b = {"log_t": torch.log(torch.tensor(np.maximum(
                 np.asarray(time, dtype=np.float64), 1e-8), dtype=torch.float32)),
             "x_cont": x_cont,
             "event_type": torch.tensor(np.asarray(event).astype(np.int64))}
        if x_cat is not None:
            b["x_cat"] = x_cat
        return b

    # -- protocol ----------------------------------------------------------
    def fit(self, X, time, event, **kw):
        import torch
        from benchmark.classical import time_knots
        from encoder import EncoderConfig, fit_scaler
        from neural_rp import (VariantA, VariantAConfig, VariantB,
                               VariantBConfig)
        from rcs_basis import RCSBasis
        from train import TrainConfig, set_seed, train_model

        set_seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event).astype(np.int64)

        # Internal validation split for early stopping, seeded so a refit
        # reproduces itself.
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(time))
        n_val = max(1, int(round(self.val_fraction * len(time))))
        va, tr = perm[:n_val], perm[n_val:]

        knots = {k: time_knots(time[tr], event[tr], k, self.df_time - 1)
                 for k in (1, 2)}
        basis = {k: RCSBasis(knots[k]) for k in (1, 2)}

        enc = EncoderConfig(
            n_continuous=(self.n_continuous if self.n_continuous is not None
                          else X.shape[1] - len(self.cardinalities)),
            categorical_cardinalities=self.cardinalities,
            hidden_dims=self.hidden_dims, latent_dim=self.latent_dim,
            dropout=self.dropout)

        b_tr = self._batch(X[tr], time[tr], event[tr])
        b_va = self._batch(X[va], time[va], event[va])
        mean, std = fit_scaler(b_tr["x_cont"])

        if self.variant == "variant_a":
            cfg = VariantAConfig(encoder=enc, spline_lr=self.spline_lr,
                                 encoder_lr=self.encoder_lr)
            self.model = VariantA(basis, cfg, mean, std)
        else:
            cfg = VariantBConfig(encoder=enc, spline_lr=self.spline_lr,
                                 encoder_lr=self.encoder_lr,
                                 head_lr=self.head_lr)
            self.model = VariantB(basis, cfg, mean, std)

        tcfg = TrainConfig(max_epochs=self.max_epochs,
                           batch_size=self.batch_size, patience=self.patience,
                           scheduler="cosine", seed=self.seed, verbose=False)
        train_model(self.model, b_tr, b_va, tcfg)
        self.model.eval()
        return self

    def _require_fit(self):
        if self.model is None:
            raise RuntimeError("fit() must be called before predicting")

    def predict_cif(self, X, t_eval, n_nodes=None, batch_size: int = 2048):
        import cif as cif_module
        self._require_fit()
        x_cont, x_cat = self._split(X)
        kw = {} if n_nodes is None else {"n_nodes": n_nodes}
        cif, _ = cif_module.predict_cif(
            self.model, x_cont, np.atleast_1d(np.asarray(t_eval,
                                                         dtype=np.float64)),
            batch_size=batch_size, x_cat=x_cat, **kw)
        return np.asarray(cif, dtype=np.float64)

    def predict_survival(self, X, t_eval, batch_size: int = 2048):
        """Event-free survival from the model's own cumulative hazards.

        Supplying this keeps the coherence column a real measurement: S here is
        exp(-sum_k H_k) computed from the fit, so the residual is quadrature
        error rather than the identically-zero complement.
        """
        import cif as cif_module
        self._require_fit()
        x_cont, x_cat = self._split(X)
        s = cif_module.predict_survival(
            self.model, x_cont, np.atleast_1d(np.asarray(t_eval,
                                                         dtype=np.float64)),
            batch_size=batch_size, x_cat=x_cat)
        return np.asarray(s, dtype=np.float64)

    def compute_nll(self, X, time, event) -> float:
        """Mean held-out negative log-likelihood - the tuning criterion.

        Named to match the deep baselines so `score_fold` can treat all five
        models identically.
        """
        import torch
        self._require_fit()
        b = self._batch(X, time, event)
        with torch.no_grad():
            return float(self.model.loss(b["log_t"], b["x_cont"],
                                         b["event_type"], b.get("x_cat")))


def variant_factory(variant: str, n_continuous, cardinalities, columns=None,
                    **fixed):
    """`factory(config, seed) -> VariantBaseline`, for the tuning harness."""
    def make(config, seed):
        return VariantBaseline(variant=variant, n_continuous=n_continuous,
                               cardinalities=cardinalities, columns=columns,
                               seed=seed, **{**fixed, **dict(config)})
    return make
