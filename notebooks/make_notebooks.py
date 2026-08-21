"""Generate the two runner notebooks.

Written as a script rather than hand-edited JSON so the cells stay valid and
can be regenerated after a change without hand-patching notebook internals.
Run `uv run python notebooks/make_notebooks.py` to rebuild.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


# ===========================================================================
# Shared configuration cell
# ===========================================================================

CONFIG = r'''
# =====================================================================
# CONFIGURATION - the only cell to edit
# =====================================================================
from pathlib import Path

REPO_ROOT    = Path(r"")          # repository root; leave "" to auto-detect
FEATURES_DIR = Path(r"")          # feature-builder output
RESULTS_DIR  = Path(r"")          # benchmark parquet + logs

T_EVAL = [1.0, 3.0, 5.0]          # years
FOLDS  = [0, 1, 2, 3, 4]

# =====================================================================
# Nothing below this line needs editing.
# =====================================================================
import sys, os, json, time
import numpy as np
import pandas as pd

_here = Path.cwd()
if not str(REPO_ROOT):
    # Notebook may be run from notebooks/ or from the repository root.
    REPO_ROOT = _here.parent if (_here / "io_helpers.py").exists() else _here
REPO_ROOT = Path(REPO_ROOT).resolve()

sys.path.insert(0, str(REPO_ROOT / "notebooks"))
import io_helpers as ioh
ioh.add_rp_core_to_path(REPO_ROOT)

FEATURES_DIR = Path(FEATURES_DIR)
RESULTS_DIR = Path(RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

assert FEATURES_DIR.exists(), f"FEATURES_DIR does not exist: {FEATURES_DIR}"
ioh.log("configuration", repo=REPO_ROOT, features=FEATURES_DIR,
        results=RESULTS_DIR)

# What the feature builder actually produced. The held_out column is a
# reminder, not a permission: the loader refuses those sets outright.
_avail = ioh.available_sets(FEATURES_DIR)
print(_avail.to_string(index=False))
'''


# ===========================================================================
# Notebook 1 - RSF benchmark
# ===========================================================================

NB1 = [
    md(r"""
# 01 - RSF benchmark

Random survival forest across the five development folds, at **imputation
m = 1 only**.

**Why one imputation.** RSF is the runtime bottleneck of the whole project -
roughly 15 hours for the full matrix even at reduced settings, against
minutes for every other classical baseline. Between-imputation variance is
negligible at this project's missingness rates (0.26% cold ischemia, ~4.5%
biopsy grades within biopsied rows), so the full m = 1..5 sweep would buy
almost nothing for about five times the cost. **This is a stated deviation in
the paper, not a hidden one** - report it in the methods alongside the
missingness rates that justify it.

**The temporal and exploratory windows are never loaded here.** They are
scored once, at the end of the project, after every modelling decision is
locked. `io_helpers.load_set` raises if asked for them.

Runs unattended for hours: every step is timestamped, and the run is
resumable, so an interruption continues rather than restarting.
"""),
    code(CONFIG),
    md(r"""
## 1. Single-fit timing probe

**Run this before launching anything.** One fold, timed, with the projected
total printed. A cell that says "projected 40 hours" is worth far more before
the run than after it.

Settings, and why:

| Option | Value | Reason |
|---|---|---|
| `ntree` | 500 | |
| `nodesize` | 30 | larger than the default 15; shallower trees, much cheaper |
| `ntime` | 100 | **not a tuning knob.** Left unset, `rfsrc` evaluates CIFs at every distinct event time - at n = 37,000 that is a multi-gigabyte array per cause |
| `nsplit` | 10 | random split points instead of an exhaustive scan; most of the saving on continuous covariates |
| `perf.type` | `"none"` | switches off OOB error bookkeeping, which nothing here reads - scoring is on held-out folds via `metrics.evaluate` |
| `block.size` | `NULL` | same reason |
| `rf.cores` | all cores | |
"""),
    code(r'''
from benchmark import BenchmarkCell, coherence_table, run_benchmark

RSF_OPTIONS = dict(ntree=500, nodesize=30, ntime=100, nsplit=10,
                   perf_type="none", block_size=None, rf_cores=None)

# --- FALLBACK, if the probe below exceeds ~30 minutes ------------------
# Drop to ntree=300, but ONLY after confirming the OOB error has plateaued -
# fewer trees is a weaker comparator, and "it was slow" is not a method.
# To check, fit once with perf_type="oob" and inspect the error curve, then
# set the tree count where it flattens.
#     RSF_OPTIONS.update(ntree=300)
# Whatever is used, REPORT IT: an RSF at 300 trees is not an RSF at 1000.
# -----------------------------------------------------------------------

IMPUTATION = 1

train0, test0 = ioh.load_fold(FEATURES_DIR, FOLDS[0], IMPUTATION, "onehot")
print(ioh.describe(train0))
print(ioh.describe(test0))

with ioh.timed(f"RSF probe fit on fold {FOLDS[0]}"):
    t0 = time.time()
    probe = ioh.production_rsf(columns=train0["columns"], seed=0, **RSF_OPTIONS)
    probe.fit(train0["X"], train0["time"], train0["event"])
    cif = probe.predict_cif(test0["X"], np.asarray(T_EVAL))
    probe_seconds = time.time() - t0

n_cells = len(FOLDS) + 1          # five CV folds plus the final dev_full fit
print(f"\nR options: {probe.r_options}")
print(f"one fit + predict : {ioh.human(probe_seconds)}")
print(f"PROJECTED TOTAL   : {ioh.human(probe_seconds * n_cells)} "
      f"for {n_cells} cells")
print(f"CIF sanity        : shape {cif.shape}, range "
      f"[{cif.min():.4f}, {cif.max():.4f}], F1+F2 max {cif.sum(-1).max():.4f}")

if probe_seconds > 30 * 60:
    print("\n!! over 30 minutes per fit - consider the ntree=300 fallback "
          "above, after checking the OOB curve has plateaued.")
'''),
    md(r"""
## 2. Build the cells

Five cross-validation cells plus one final fit on the full development window.

**The final cell evaluates on `dev_full` only.** Adding `temporal` or
`exploratory` to `evaluate_on` here would spend the project's last untouched
data on an intermediate result. They are scored once, in a separate final
notebook, after every modelling decision is locked - including this one.
"""),
    code(r'''
cells, loaded = [], {}

for fold in FOLDS:
    tr, te = ioh.load_fold(FEATURES_DIR, fold, IMPUTATION, "onehot")
    loaded[fold] = (tr, te)
    cells.append(BenchmarkCell(
        model_name="rsf", fold=fold, imputation=IMPUTATION,
        train=ioh.as_cell_data(tr),
        evaluate_on={"test": ioh.as_cell_data(te)},
        factory=ioh.rsf_factory(columns=tr["columns"], **RSF_OPTIONS)))
    ioh.log(f"fold {fold}", train=f"{tr['n_rows']:,}", test=f"{te['n_rows']:,}")

# Final fit on the whole development window, scored in-sample only.
# evaluate_on deliberately contains NOTHING but dev_full - see the note above.
#
# fold = -1, not "dev_full": run_benchmark writes `fold` as a parquet column,
# and mixing ints with a string makes it an object column that pyarrow refuses
# ("Could not convert 'dev_full' with type str: tried to convert to int64").
# It would also break any later join against the other models' results, whose
# folds are integers. -1 means "fitted on the whole development window, not a
# CV fold"; the eval_set column carries the readable name.
full = ioh.load_set(FEATURES_DIR, "dev_full", "dev_full", IMPUTATION, "onehot")
DEV_FULL_FOLD = -1
cells.append(BenchmarkCell(
    model_name="rsf", fold=DEV_FULL_FOLD, imputation=IMPUTATION,
    train=ioh.as_cell_data(full),
    evaluate_on={"dev_full": ioh.as_cell_data(full)},
    factory=ioh.rsf_factory(columns=full["columns"], **RSF_OPTIONS)))

for c in cells:
    assert set(c.evaluate_on) <= {"test", "dev_full"}, \
        f"unexpected evaluation set in cell {c.fold}: {list(c.evaluate_on)}"
ioh.assert_not_held_out(*[k for c in cells for k in c.evaluate_on])

assert all(isinstance(c.fold, (int, np.integer)) for c in cells), \
    "fold must stay integer-typed or the results parquet cannot be written"
print(f"\n{len(cells)} cells: {[c.fold for c in cells]} "
      f"(-1 = full development window)")
'''),
    md(r"""
## 3. Run

`resume=True`, so an interrupted run continues from the parquet rather than
starting over. Seeds come from `derive_seed(model, fold, imputation)`, which is
a hash rather than a counter - a resumed run therefore reproduces exactly what
an uninterrupted one would have produced, and restarting is free of
consequences.

Expect this cell to take hours. The timestamps are how a stalled run is
distinguished from a slow one.
"""),
    code(r'''
RESULTS_PATH = RESULTS_DIR / "rsf_results.parquet"

with ioh.timed(f"RSF benchmark -> {RESULTS_PATH.name}"):
    results = run_benchmark(cells, np.asarray(T_EVAL), RESULTS_PATH,
                            resume=True, verbose=True)

ioh.log("written", path=RESULTS_PATH, rows=len(results))
results.head()
'''),
    md(r"""
## 4. Post-run checks

Read these before trusting anything downstream.

On the coherence column: RSF has no joint survival function of its own -
`rfsrc` returns none for the competing-risks family - so `predict_survival`
reconstructs one from the forest's own cumulative hazards by the product-limit
recursion. The residual is therefore a real property of the bagged estimator
(averaging CIFs and averaging cumulative hazards do not commute), and it runs
around 8e-02. It is **not** comparable to the Royston-Parmar residual, which is
quadrature error in a check that could have failed.
"""),
    code(r'''
print("COHERENCE  max |F1 + F2 + S - 1|")
print(coherence_table(results).to_string())

print("\nCELLS NEEDING ATTENTION (non-convergence, or no better than the null)")
bad = ioh.failure_report(results)
print("  none" if bad.empty else bad.to_string(index=False))

print("\nPER-FOLD TIMING   (fold -1 = full development window)")
timing = (results.groupby(["fold", "eval_set"])[["fit_seconds",
                                                 "predict_seconds"]]
          .first().round(1))
print(timing.to_string())
print(f"\ntotal fit time: "
      f"{ioh.human(results.groupby('fold')['fit_seconds'].first().sum())}")

print("\nHELD-OUT BRIER by cause and horizon")
print(results[results.eval_set == "test"]
      .pivot_table(index="horizon", columns="cause", values="brier",
                   aggfunc="mean").round(5).to_string())
'''),
    md(r"""
## 5. Summary

**Produced:** `rsf_results.parquet` in `RESULTS_DIR` - long format, one row per
(model, fold, imputation, eval_set, cause, horizon), with AUC, C-index, Brier,
Austin calibration (ICI/E50/E90/Emax), calibration-in-the-large, the coherence
residual, timings, seeds, and a per-cause `converged` flag.

**Covers:** RSF only, five development folds plus a full-development fit, at
imputation m = 1. Temporal and exploratory untouched.

**Consumed by:** the model-comparison stage, which pools this with the other
classical baselines' results and the neural variants'. Concatenate on the same
schema - `RECORD_COLUMNS` in `benchmark/base.py` - and group by model.

**To report in the methods:** the RSF settings printed by the probe cell
(`probe.r_options`), and the single-imputation deviation with the missingness
rates that justify it.
"""),
    code(r'''
print(f"rsf_results.parquet : {RESULTS_PATH}")
print(f"rows                : {len(results):,}")
print(f"models              : {sorted(results.model.unique())}")
print(f"folds               : {sorted(map(str, results.fold.unique()))}")
print(f"imputations         : {sorted(results.imputation.unique())}")
print(f"eval sets           : {sorted(results.eval_set.unique())}")
print(f"RSF settings        : {RSF_OPTIONS}")
'''),
]


# ===========================================================================
# Notebook 2 - tuning
# ===========================================================================

NB2 = [
    md(r"""
# 02 - Hyperparameter search

Random search for all five models, **fold 0 only, at full n, 30
configurations each**.

**Why one fold rather than a subsample.** Both save time; they are not
equivalent. Dropping from five folds to one adds selection *noise*, which is
symmetric across models and shows up as a less precise ranking. Subsampling
rows introduces *bias*: regularisation optima shift with sample size, and they
shift unequally across architectures, so the model whose optimum moves most is
penalised. Tuning Neural Fine-Gray on less data than its competitors would
risk beating the closest published comparator because it was tuned worse -
precisely the failure the fairness rules exist to prevent.

The cost of that choice is borne by cell 7, which checks the top NFG
configurations transport to a second fold.

**Never touches the temporal or exploratory windows.**
"""),
    code(CONFIG),
    md(r"""
## 1. Search spaces, side by side

All five declared in one place, so the budgets are visibly comparable rather
than comparable by assertion. Every grid is far larger than the 30-trial
budget, so no model is quietly searched exhaustively while another is sampled.

**Selection criterion** is held-out log-likelihood where the model provides
one, and integrated Brier score otherwise. DeepHit is the exception: its
training loss mixes likelihood with a ranking term and is not a likelihood, so
it is selected on IBS. Which criterion each model used is recorded in the trial
log and printed at selection time - comparing a likelihood-selected model with
a Brier-selected one is fine, but only if the reader knows.
"""),
    code(r'''
from benchmark import SEARCH_SPACES, make_cv_folds, random_search, \
    search_space_table
from benchmark.deep import (DeepConfig, DSMBaseline, DeepHitBaseline,
                            NeuralFineGrayBaseline, available_deep_baselines)

BUDGET = 30
MAX_EPOCHS = 200          # variants A/B; early stopping usually ends sooner
TUNING_FOLD = 0
TUNING_IMPUTATION = 1
MODELS = ["variant_a", "variant_b", "deephit", "neural_fine_gray", "dsm"]

table = search_space_table()
print(table[["model", "criterion", "n_hyperparameters", "grid_size"]]
      .to_string(index=False))
print()
for name in MODELS:
    sp = SEARCH_SPACES[name]
    print(f"{name:18s} budget {BUDGET:3d} of {sp.size():6d} configurations "
          f"({100 * BUDGET / sp.size():.1f}% of the grid)  "
          f"criterion={sp.criterion}")
    print(f"{'':18s} {', '.join(sorted(sp.grid))}")

# Deep baselines must actually be importable before hours are spent.
for name, (ok, why) in available_deep_baselines().items():
    print(f"\n{name:18s} {'available' if ok else 'UNAVAILABLE: ' + why}")
'''),
    md(r"""
## 2. Load fold 0, and build the CV folds the harness wants

Two representations: the neural and deep models consume `embedding`
(`[continuous | integer codes]`), the classical ones `onehot`. Both come from
the same fitted `FeatureSpec` - identical information, never re-encoded.
"""),
    code(r'''
tune_train, tune_val = ioh.load_fold(FEATURES_DIR, TUNING_FOLD,
                                     TUNING_IMPUTATION, "embedding")
print(ioh.describe(tune_train))
print(ioh.describe(tune_val))

N_CONTINUOUS = tune_train["n_continuous"]
CARDINALITIES = tune_train["cardinalities"]
COLUMNS = tune_train["columns"]
print(f"\nembedding: {N_CONTINUOUS} continuous + {len(CARDINALITIES)} "
      f"categorical (cardinalities {CARDINALITIES})")

# The harness takes a list of {"train":, "validate":} dicts. One entry here,
# because this is a single-fold search by design.
folds = [{"fold": TUNING_FOLD,
          "train": ioh.as_cell_data(tune_train),
          "validate": ioh.as_cell_data(tune_val)}]

# Explicit guard: the search must never see a held-out window.
ioh.assert_not_held_out(tune_train["applied_to"], tune_val["applied_to"])
'''),
    md(r"""
## 3. Timing probe

One fit per model at full n, before committing to 30 configurations each.
Neural Fine-Gray is the expensive one by a wide margin - roughly 1,210 s per
fit at 37,000 rows against 25 s for DeepHit and 10 s for DSM - so its 30
configurations dominate the total.

Its cost is not linear in rows and not a fixed schedule: the number of
iterations before early stopping fires grows with n. That is why the probe
measures rather than extrapolates.
"""),
    code(r'''
def make_model(name, config, seed):
    """One model from a sampled config - the factory the harness calls."""
    if name == "deephit":
        return DeepHitBaseline(columns=COLUMNS, n_continuous=N_CONTINUOUS,
                               cardinalities=CARDINALITIES, seed=seed,
                               config=DeepConfig(**config))
    if name == "neural_fine_gray":
        return NeuralFineGrayBaseline(columns=COLUMNS,
                                      n_continuous=N_CONTINUOUS,
                                      cardinalities=CARDINALITIES, seed=seed,
                                      config=DeepConfig(**config))
    if name == "dsm":
        return DSMBaseline(columns=COLUMNS, n_continuous=N_CONTINUOUS,
                           cardinalities=CARDINALITIES, seed=seed,
                           config=DeepConfig(**config))
    if name in ("variant_a", "variant_b"):
        # The proposed models, through the same protocol and the same search
        # procedure as their comparators. VariantBaseline wraps rp_core's own
        # VariantA/VariantB, RCSBasis, train_model and cif.predict_cif - no
        # estimation logic is reimplemented here.
        return ioh.VariantBaseline(variant=name, n_continuous=N_CONTINUOUS,
                                   cardinalities=CARDINALITIES,
                                   columns=COLUMNS, seed=seed,
                                   max_epochs=MAX_EPOCHS, **config)
    raise KeyError(f"no factory for {name!r}")


probe_seconds = {}
for name in MODELS:
    cfg = SEARCH_SPACES[name].sample(np.random.default_rng(0))
    try:
        with ioh.timed(f"probe {name}"):
            t0 = time.time()
            m = make_model(name, cfg, seed=0)
            m.fit(tune_train["X"], tune_train["time"], tune_train["event"])
            probe_seconds[name] = time.time() - t0
    except Exception as exc:
        print(f"{name:18s} PROBE FAILED: {type(exc).__name__}: {exc}")
        continue
    total = probe_seconds[name] * BUDGET
    print(f"{name:18s} {ioh.human(probe_seconds[name]):>8s} per fit  ->  "
          f"{ioh.human(total):>8s} for {BUDGET} configurations")

if probe_seconds:
    print(f"\nPROJECTED TOTAL: "
          f"{ioh.human(sum(probe_seconds.values()) * BUDGET)}")
'''),
    md(r"""
## 4. Run the search, one model at a time

Trials are written to parquet **after each model finishes**, so an interruption
during Neural Fine-Gray - the long one - does not lose the cheaper models that
already completed. A failed trial is recorded rather than dropped: a silently
skipped configuration would make the reported budget a lie.
"""),
    code(r'''
TRIALS_PATH = RESULTS_DIR / "tuning_trials.parquet"
results = {}

for name in MODELS:
    ioh.log(f"searching {name}", budget=BUDGET,
            criterion=SEARCH_SPACES[name].criterion)
    try:
        with ioh.timed(f"random search: {name}"):
            results[name] = random_search(
                name,
                factory=lambda cfg, seed, _n=name: make_model(_n, cfg, seed),
                folds=folds, t_eval=np.asarray(T_EVAL), budget=BUDGET,
                seed=42, output_path=TRIALS_PATH,
                window=np.array(["development"]),
                imputation=TUNING_IMPUTATION, verbose=True)
        r = results[name]
        ioh.log(f"{name} done", best=f"{r.best_score:+.5f}",
                evaluated=r.n_evaluated)
    except Exception as exc:
        ioh.log(f"{name} FAILED: {type(exc).__name__}: {exc}")

ioh.log("trials written", path=TRIALS_PATH)
'''),
    md(r"""
## 5. Selection

The top five per model, not just the winner. If the best two configurations
are separated by less than the spread across nearby configurations, the
selection is within noise and the winner is arbitrary - worth seeing before
committing it to the full matrix.
"""),
    code(r'''
trials = pd.read_parquet(TRIALS_PATH)

for name, r in results.items():
    print(f"\n{'=' * 70}\n{name}   criterion={r.criterion}   "
          f"evaluated {r.n_evaluated}/{r.budget}")
    print(f"BEST  score {r.best_score:+.6f}")
    print(f"      {r.best_config}")
    top = (trials[(trials.model == name) & (~trials.failed)]
           .nlargest(5, "score")[["score", "score_std", "fit_seconds",
                                  "config"]])
    print("\ntop 5:")
    print(top.to_string(index=False))
    if len(top) > 1:
        gap = top.score.iloc[0] - top.score.iloc[1]
        spread = top.score.iloc[0] - top.score.iloc[-1]
        flag = "  <-- KNIFE EDGE" if spread > 0 and gap < 0.2 * spread else ""
        print(f"\ngap to runner-up {gap:.6f}, spread across top 5 "
              f"{spread:.6f}{flag}")

n_failed = int(trials.failed.sum())
if n_failed:
    print(f"\n{n_failed} failed trial(s):")
    print(trials[trials.failed][["model", "trial", "error"]].to_string(index=False))
'''),
    md(r"""
## 6. Write the selected configurations

`selected_configs.json` is the artifact the full development run consumes. It
records the criterion and the budget alongside each configuration, so the run
that uses it can state how the configuration was chosen without going back to
the trial log.
"""),
    code(r'''
SELECTED_PATH = RESULTS_DIR / "selected_configs.json"

payload = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "tuning_fold": TUNING_FOLD,
    "tuning_imputation": TUNING_IMPUTATION,
    "budget": BUDGET,
    "t_eval": list(T_EVAL),
    "note": ("Single-fold search at full n. One fold adds selection noise "
             "(symmetric across models); subsampling rows would add bias "
             "(asymmetric), so the fold was cut rather than the rows."),
    "models": {name: {"config": r.best_config, "criterion": r.criterion,
                      "score": float(r.best_score),
                      "n_evaluated": int(r.n_evaluated)}
               for name, r in results.items()},
}
SELECTED_PATH.write_text(json.dumps(payload, indent=2, default=str))
ioh.log("selected configs written", path=SELECTED_PATH,
        models=len(payload["models"]))
print(json.dumps(payload, indent=2, default=str))
'''),
    md(r"""
## 7. Neural Fine-Gray transport check

Single-fold selection is noisier than five-fold. This re-fits the top three NFG
configurations on **fold 1** and checks the ranking holds.

**If the ranking reorders, the selection is within noise.** That is not a
failure of the run - it is the information the check exists to produce, and it
matters because NFG is the closest published comparator to this project's
model. Committing a configuration chosen by noise would mean any margin over
it is partly an artifact of the search. If it reorders, either average the top
few configurations' scores across both folds and take that winner, or spend
the extra folds on NFG alone and say so.
"""),
    code(r'''
TRANSPORT_FOLD = 1

if "neural_fine_gray" not in results:
    print("Neural Fine-Gray was not searched; nothing to transport-check.")
else:
    import ast
    from benchmark.tuning import score_fold

    nfg_trials = (trials[(trials.model == "neural_fine_gray")
                         & (~trials.failed)].nlargest(3, "score"))
    tr1, va1 = ioh.load_fold(FEATURES_DIR, TRANSPORT_FOLD, TUNING_IMPUTATION,
                             "embedding")
    ioh.assert_not_held_out(tr1["applied_to"], va1["applied_to"])
    print(ioh.describe(tr1))

    rows = []
    for rank, (_, row) in enumerate(nfg_trials.iterrows(), start=1):
        cfg = ast.literal_eval(row["config"])
        with ioh.timed(f"NFG rank {rank} on fold {TRANSPORT_FOLD}"):
            m = make_model("neural_fine_gray", cfg, seed=1000 + rank)
            m.fit(tr1["X"], tr1["time"], tr1["event"])
            s = score_fold(m, va1["X"], va1["time"], va1["event"],
                           np.asarray(T_EVAL),
                           SEARCH_SPACES["neural_fine_gray"].criterion)
        rows.append({"rank_fold0": rank, "score_fold0": row["score"],
                     "score_fold1": s, "config": row["config"]})

    check = pd.DataFrame(rows)
    check["rank_fold1"] = check.score_fold1.rank(ascending=False).astype(int)
    print("\n" + check[["rank_fold0", "rank_fold1", "score_fold0",
                        "score_fold1"]].to_string(index=False))

    held = bool((check.rank_fold0.values == check.rank_fold1.values).all())
    winner_held = bool(check.rank_fold1.iloc[0] == 1)
    print(f"\nfull ranking held : {held}")
    print(f"winner held       : {winner_held}")
    if not winner_held:
        print("\n!! The fold-0 winner is not the fold-1 winner. The selection "
              "is within noise. Do NOT quietly keep the fold-0 choice - "
              "either average across both folds and take that winner, or run "
              "the remaining folds for NFG and report the wider search.")
'''),
    md(r"""
## 8. Summary

**Produced:**

- `tuning_trials.parquet` - every trial: model, config, per-fold score, wall
  time, seed, and the failure reason where one applies. This is a supplementary
  table for the paper, not a debugging aid: it is what shows a reviewer the
  comparators were given a real search.
- `selected_configs.json` - the chosen configuration per model, with its
  criterion, score, and budget.

**Covers:** fold 0, imputation m = 1, 30 configurations per model, at full n.
Temporal and exploratory untouched.

**Consumed by:** the full development run, which fixes these configurations and
fits them across all folds and imputations.

**To report in the methods:** the budget and grid size per model (cell 1), the
criterion per model, and the transport-check result from cell 7.
"""),
    code(r'''
print(f"tuning_trials.parquet : {TRIALS_PATH}")
print(f"selected_configs.json : {SELECTED_PATH}")
print(f"models searched       : {sorted(results)}")
print(f"trials logged         : {len(trials):,} "
      f"({int(trials.failed.sum())} failed)")
print(f"total search time     : "
      f"{ioh.human(trials.fit_seconds.sum())}")
for name, r in results.items():
    print(f"  {name:18s} {r.criterion:7s} {r.best_score:+.6f}")
'''),
]


def main():
    for name, cells in (("01_rsf_benchmark.ipynb", NB1),
                        ("02_tuning.ipynb", NB2)):
        path = HERE / name
        path.write_text(json.dumps(notebook(cells), indent=1))
        print(f"wrote {path}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
