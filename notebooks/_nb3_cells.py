"""Cells for `03_development_matrix.ipynb`.

Kept in its own module rather than inline in `make_notebooks.py`: the cell
bodies are themselves triple-quoted strings, and nesting those inside another
triple-quoted block silently truncates at the first inner delimiter.

`build(md, code)` takes the two cell constructors so this module does not
import from `make_notebooks` and create a cycle.
"""

CONFIG = r'''
# =====================================================================
# CONFIGURATION - the only cell to edit
# =====================================================================
from pathlib import Path
import os

REPO_ROOT    = Path(r"")          # repository root; leave "" to auto-detect
FEATURES_DIR = Path(r"")          # feature-builder output
RESULTS_DIR  = Path(r"")          # where 01 and 02 wrote their results
NFG_PATH     = Path(r"")          # clone of Jeanselme/NeuralFineGray

T_EVAL      = [1.0, 3.0, 5.0]     # years
FOLDS       = [0, 1, 2, 3, 4]
IMPUTATIONS = [1, 2, 3, 4, 5]

# =====================================================================
# Nothing below this line needs editing.
# =====================================================================
import sys, json, time
import numpy as np
import pandas as pd

_here = Path.cwd()
if not str(REPO_ROOT):
    REPO_ROOT = _here.parent if (_here / "io_helpers.py").exists() else _here
REPO_ROOT = Path(REPO_ROOT).resolve()

# NFG_PATH must be set BEFORE the deep wrappers are asked what is available:
# the baseline reports itself unavailable when it cannot find the clone, and an
# unavailable model is silently OMITTED from the results rather than raising.
# Losing the closest published comparator that way would not surface until the
# results table was read, hours later.
assert str(NFG_PATH), (
    "NFG_PATH is empty. Neural Fine-Gray is the closest published comparator "
    "to this project's model; without it this run quietly omits it.\n"
    "  git clone --recursive https://github.com/Jeanselme/NeuralFineGray.git")
NFG_PATH = Path(NFG_PATH).resolve()
assert (NFG_PATH / "nfg").is_dir(), f"no nfg/ package under {NFG_PATH}"
os.environ["NFG_PATH"] = str(NFG_PATH)

sys.path.insert(0, str(REPO_ROOT / "notebooks"))
import io_helpers as ioh
ioh.add_rp_core_to_path(REPO_ROOT)

FEATURES_DIR = Path(FEATURES_DIR)
RESULTS_DIR = Path(RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
assert FEATURES_DIR.exists(), f"FEATURES_DIR does not exist: {FEATURES_DIR}"

from benchmark.deep import available_deep_baselines
_avail = available_deep_baselines(str(NFG_PATH))
for _name, (_ok, _why) in _avail.items():
    print(f"  {_name:18s} {'available' if _ok else 'UNAVAILABLE: ' + _why}")
assert all(ok for ok, _ in _avail.values()), (
    "a deep baseline is unavailable; fix that before spending hours on a run "
    "that would quietly omit it")

ioh.log("configuration", repo=REPO_ROOT, features=FEATURES_DIR,
        results=RESULTS_DIR)
RESULTS_PATH = RESULTS_DIR / "development_results.parquet"
RSF_PATH = RESULTS_DIR / "rsf_results.parquet"
SELECTED_PATH = RESULTS_DIR / "selected_configs.json"
'''


CELL_CONFIGS = r'''
assert SELECTED_PATH.exists(), (
    f"{SELECTED_PATH} not found - run 02_tuning.ipynb first")
selected = json.loads(SELECTED_PATH.read_text())

print(f"tuned on fold {selected.get('tuning_fold')} at "
      f"m={selected.get('tuning_imputation')}, budget "
      f"{selected.get('budget')} configurations each")
print(f"created {selected.get('created')}\n")
for name, entry in selected["models"].items():
    print(f"{name:18s} criterion={entry['criterion']:7s} "
          f"score={entry['score']:+.6f}  (not comparable across rows)")
    print(f"{'':18s} {entry['config']}")

NEURAL = ["variant_a", "variant_b", "deephit", "neural_fine_gray", "dsm"]
CLASSICAL = ["royston_parmar", "cox", "fine_gray", "rp_kdri_epts"]
MODELS = NEURAL + CLASSICAL

missing = [m for m in NEURAL if m not in selected["models"]]
assert not missing, (
    f"no tuned configuration for {missing}. Running those on defaults would "
    f"make them weaker comparators than the ones that were tuned.")
print(f"\n{len(MODELS)} models x {len(FOLDS)} folds x {len(IMPUTATIONS)} "
      f"imputations = {len(MODELS) * len(FOLDS) * len(IMPUTATIONS)} CV cells "
      f"+ {len(MODELS) * len(IMPUTATIONS)} dev_full cells")
'''


CELL_FACTORIES = r'''
from benchmark import (BenchmarkCell, CoxBaseline, FineGrayBaseline,
                       MinimalReferenceBaseline, RoystonParmarBaseline,
                       coherence_table, run_benchmark)
from benchmark.deep import (DeepConfig, DSMBaseline, DeepHitBaseline,
                            NeuralFineGrayBaseline)

DEEP_CLASSES = {"deephit": DeepHitBaseline,
                "neural_fine_gray": NeuralFineGrayBaseline,
                "dsm": DSMBaseline}
CLASSICAL_CLASSES = {"royston_parmar": RoystonParmarBaseline,
                     "cox": CoxBaseline,
                     "fine_gray": FineGrayBaseline,
                     "rp_kdri_epts": MinimalReferenceBaseline}

REPRESENTATION = {**{m: "embedding" for m in NEURAL},
                  **{m: "onehot" for m in CLASSICAL}}


def make_factory(model, loaded):
    """`factory(seed) -> baseline`, configured for one training set."""
    cfg = dict(selected["models"].get(model, {}).get("config", {}))

    if model in ("variant_a", "variant_b"):
        return lambda seed: ioh.VariantBaseline(
            variant=model, n_continuous=loaded["n_continuous"],
            cardinalities=loaded["cardinalities"], columns=loaded["columns"],
            seed=seed, **cfg)

    if model in DEEP_CLASSES:
        cls = DEEP_CLASSES[model]
        extra = {"path": str(NFG_PATH)} if model == "neural_fine_gray" else {}
        return lambda seed: cls(
            columns=loaded["columns"], n_continuous=loaded["n_continuous"],
            cardinalities=loaded["cardinalities"], seed=seed,
            config=DeepConfig(**cfg), **extra)

    cls = CLASSICAL_CLASSES[model]
    if model == "rp_kdri_epts":
        # The minimal reference keeps only the two composites and selects its
        # own columns, so it takes no spline list.
        return lambda seed: cls(columns=loaded["columns"], seed=seed)
    spline_cols = ioh.spline_columns_for(loaded)
    return lambda seed: cls(columns=loaded["columns"], seed=seed,
                            spline_columns=spline_cols)


# One sanity load: both representations must describe the same patients.
_fit = ioh.fold_sets(FOLDS[0])[0]
_e = ioh.load_set(FEATURES_DIR, _fit, _fit, IMPUTATIONS[0], "embedding")
_o = ioh.load_set(FEATURES_DIR, _fit, _fit, IMPUTATIONS[0], "onehot")
assert np.array_equal(_e["time"], _o["time"]), \
    "embedding and one-hot disagree on follow-up time - not the same patients"
assert np.array_equal(_e["event"], _o["event"]), \
    "embedding and one-hot disagree on event type - not the same patients"
print(ioh.describe(_e))
print(ioh.describe(_o))
_spl = ioh.spline_columns_for(_o)
print(f"\ncontinuous columns given splines ({len(_spl)}): {_spl[:6]} ...")
del _e, _o
'''


CELL_PROBE = r'''
probe_seconds = {}
for model in MODELS:
    rep = REPRESENTATION[model]
    tr, te = ioh.load_fold(FEATURES_DIR, FOLDS[0], IMPUTATIONS[0], rep)
    try:
        t0 = time.time()
        m = make_factory(model, tr)(0)
        m.fit(tr["X"], tr["time"], tr["event"])
        cif = m.predict_cif(te["X"], np.asarray(T_EVAL))
        probe_seconds[model] = time.time() - t0
        print(f"  {model:18s} [{rep:9s}] {ioh.human(probe_seconds[model]):>8s}"
              f"   F1+F2 max {cif.sum(-1).max():.4f}")
    except Exception as exc:
        print(f"  {model:18s} PROBE FAILED: {type(exc).__name__}: {exc}")
    finally:
        del tr, te

n_cells_per_model = len(FOLDS) * len(IMPUTATIONS) + len(IMPUTATIONS)
total = sum(probe_seconds.values()) * n_cells_per_model
print(f"\n{n_cells_per_model} cells per model x {len(MODELS)} models = "
      f"{n_cells_per_model * len(MODELS)} cells")
print(f"PROJECTED TOTAL: {ioh.human(total)}")
if len(probe_seconds) < len(MODELS):
    print("\n!! not every model probed successfully - fix that before running")
'''


CELL_RUN = r'''
DEV_FULL_FOLD = -1
model_seconds = {}

for model in MODELS:
    rep = REPRESENTATION[model]
    t_model = time.time()
    ioh.log(f"START {model}", representation=rep, cells=n_cells_per_model)

    for m_imp in IMPUTATIONS:
        cells = []
        for fold in FOLDS:
            tr, te = ioh.load_fold(FEATURES_DIR, fold, m_imp, rep)
            cells.append(BenchmarkCell(
                model_name=model, fold=int(fold), imputation=int(m_imp),
                train=ioh.as_cell_data(tr),
                evaluate_on={"test": ioh.as_cell_data(te)},
                factory=make_factory(model, tr)))

        full = ioh.load_set(FEATURES_DIR, "dev_full", "dev_full", m_imp, rep)
        cells.append(BenchmarkCell(
            model_name=model, fold=DEV_FULL_FOLD, imputation=int(m_imp),
            train=ioh.as_cell_data(full),
            evaluate_on={"dev_full": ioh.as_cell_data(full)},
            factory=make_factory(model, full)))

        # Guards, on every cell, every time.
        for c in cells:
            expected = {"dev_full"} if c.fold == DEV_FULL_FOLD else {"test"}
            assert set(c.evaluate_on) == expected, (
                f"{model} fold={c.fold}: evaluate_on is "
                f"{sorted(c.evaluate_on)}, expected {sorted(expected)}")
            assert isinstance(c.fold, (int, np.integer)), \
                "fold must stay integer-typed or the parquet write fails"
            ioh.assert_not_held_out(*c.evaluate_on)

        run_benchmark(cells, np.asarray(T_EVAL), RESULTS_PATH,
                      resume=True, verbose=True)
        del cells, full

    model_seconds[model] = time.time() - t_model
    ioh.log(f"DONE  {model}", elapsed=ioh.human(model_seconds[model]))

ioh.log("matrix complete", total=ioh.human(sum(model_seconds.values())))
results = pd.read_parquet(RESULTS_PATH)
print(f"\n{len(results):,} records in {RESULTS_PATH.name}")
'''


CELL_COHERENCE = r'''
rsf = pd.read_parquet(RSF_PATH) if RSF_PATH.exists() else None
if rsf is None:
    print(f"!! {RSF_PATH.name} not found - RSF will be missing from the table")
    combined = results
else:
    combined = pd.concat([results, rsf], ignore_index=True)
    print(f"concatenated {len(rsf):,} RSF records (m=1 only, from notebook 01)")

print("\nCOHERENCE  max |F1 + F2 + S - 1|")
print(coherence_table(combined).to_string())
print("\nRead this by MECHANISM, not as a ranking: zero by construction "
      "(deephit, neural_fine_gray)")
print("and small-by-measurement (royston_parmar, cox, dsm) are different "
      "quantities - see the README.")
'''


CELL_UNUSABLE = r'''
print("UNUSABLE CELLS  (these are not merely weak - do not pool them)\n")

bad_conv = combined[~combined["converged"].fillna(True)]
if bad_conv.empty:
    print("  convergence : all cells converged")
else:
    print(f"  !! {len(bad_conv)} record(s) from non-converged fits:")
    print(bad_conv[["model", "fold", "imputation", "cause", "horizon"]]
          .drop_duplicates().to_string(index=False))

# The null is the Aalen-Johansen marginal fitted on each fold's TRAINING rows,
# per fold rather than pooled - the folds differ in event mix, and one shared
# threshold would flatter some and penalise others.
null_table = ioh.null_brier_table(FEATURES_DIR, FOLDS, IMPUTATIONS[0], T_EVAL)
checked = ioh.flag_against_null(combined, null_table)
lost = checked[checked["beats_null"] == False]        # noqa: E712
if lost.empty:
    print("\n  vs null     : every scored cell beats the marginal")
else:
    print(f"\n  !! {len(lost)} record(s) NO BETTER THAN THE MARGINAL:")
    print(lost[["model", "fold", "imputation", "cause", "horizon",
                "brier", "null_brier"]].to_string(index=False))
print("\n  (dev_full rows are scored in-sample and have no matching null "
      "fold, so they are left unflagged rather than compared to a number "
      "that does not apply to them.)")
'''


CELL_CALIBRATION = r'''
print("CALIBRATION METHOD per model")
print("(an elevated Fine-Gray fallback rate contaminates ICI in exactly the")
print(" way that survives review - the number is still produced, just weaker)\n")
methods = (combined.groupby(["model", "calibration_method"]).size()
           .unstack(fill_value=0))
print(methods.to_string())
if methods.shape[1] > 1:
    share = methods.div(methods.sum(axis=1), axis=0)
    print("\nshare per model:")
    print(share.round(3).to_string())
'''


CELL_TIMING = r'''
print("WALL TIME per model\n")
# fit + predict, because RSFBaseline defers ALL its work to a single R
# subprocess at predict time: its fit_seconds reads 0.0, and a table built on
# that column alone would say the bottleneck of the whole project was free.
per_cell = (combined.groupby(["model", "fold", "imputation", "eval_set"])
            [["fit_seconds", "predict_seconds"]].first().reset_index())
per_cell["total_seconds"] = per_cell["fit_seconds"] + per_cell["predict_seconds"]
timing = (per_cell.groupby("model")["total_seconds"]
          .agg(["sum", "mean", "count"])
          .rename(columns={"sum": "total_s", "mean": "per_cell_s",
                           "count": "cells"})
          .sort_values("total_s", ascending=False))
timing["total"] = timing["total_s"].map(ioh.human)
print(timing[["cells", "per_cell_s", "total"]].round(1).to_string())
print(f"\ngrand total (fit + predict): "
      f"{ioh.human(per_cell['total_seconds'].sum())}")
print("RSF's fit_seconds is 0.0 by construction - it builds the forest at "
      "predict time - so the column above is fit + predict for every model.")
'''


CELL_HEADLINE = r'''
METRICS = ["ici", "e50", "e90", "brier", "auc", "coherence_max"]
cv = combined[combined["eval_set"] == "test"]

summary = (cv.groupby(["model", "cause", "horizon"])[METRICS]
           .agg(["mean", "std"]))
summary.columns = [f"{a}_{b}" for a, b in summary.columns]


def _pm(mu, sd):
    if not np.isfinite(mu):
        return "n/a"
    return f"{mu:.4f} +/- {sd:.4f}" if np.isfinite(sd) else f"{mu:.4f}"


for cause in sorted(cv["cause"].unique()):
    for horizon in sorted(cv["horizon"].unique()):
        block = summary.xs((cause, horizon), level=("cause", "horizon"))
        print(f"\n=== cause {int(cause)}, {horizon:.0f} years   "
              f"(mean +/- SD over {cv['fold'].nunique()} folds x "
              f"{cv['imputation'].nunique()} imputations)")
        rows = []
        for model in block.index:
            row = {"model": model}
            for metric in METRICS:
                row[metric] = _pm(block.loc[model, f"{metric}_mean"],
                                  block.loc[model, f"{metric}_std"])
            rows.append(row)
        table = pd.DataFrame(rows)
        # Sorted by ICI: calibration is the endpoint, so it orders the table.
        order = block["ici_mean"].sort_values().index
        table = table.set_index("model").loc[order].reset_index()
        print(table.to_string(index=False))

summary.to_csv(RESULTS_DIR / "development_summary.csv")
ioh.log("summary written", path=RESULTS_DIR / "development_summary.csv")
'''


CELL_SUMMARY = r'''
print(f"development_results.parquet : {RESULTS_PATH}")
print(f"development_summary.csv     : {RESULTS_DIR / 'development_summary.csv'}")
print(f"rsf_results.parquet         : {RSF_PATH} "
      f"({'present' if RSF_PATH.exists() else 'MISSING'})")
print(f"\nrecords     : {len(results):,}"
      f" (+{0 if rsf is None else len(rsf):,} RSF, concatenated at analysis)")
print(f"models      : {sorted(results.model.unique())}")
print(f"folds       : {sorted(results.fold.unique())}  (-1 = dev_full)")
print(f"imputations : {sorted(results.imputation.unique())}")
print(f"eval sets   : {sorted(results.eval_set.unique())}")
print(f"wall time   : {ioh.human(sum(model_seconds.values()))}")
'''


def build(md, code):
    """The notebook's cells, in order."""
    return [
        md(r"""
# 03 - Full development matrix

Every model except RSF, across the whole development window.

**5 folds x 5 imputations x 9 models = 225 fits**, each scored on its held-out
fold, plus **45 `dev_full` fits** (one per model per imputation) scored on
`dev_full` only. 270 cells, a few hours.

| | Models | Representation | Configuration |
|---|---|---|---|
| Neural | `variant_a`, `variant_b`, `deephit`, `neural_fine_gray`, `dsm` | `embedding` | `selected_configs.json` |
| Classical | `royston_parmar`, `cox`, `fine_gray`, `rp_kdri_epts` | `onehot` | no tuning needed |

Both representations come from the same fitted `FeatureSpec` - identical
information, never re-encoded. That is what makes the comparison fair.

**RSF is not re-run.** It is already banked in `rsf_results.parquet` from
notebook 01 and is concatenated at analysis time only.

**Configurations are read, not chosen.** A model retuned after seeing
development results is no longer honestly tuned, so nothing here adjusts a
hyperparameter.

**Temporal and exploratory are never loaded** - including by this notebook,
whose results may well prompt another modelling decision. They are scored once,
at the very end, after every such decision is locked.
"""),
        code(CONFIG),
        md(r"""
## 1. The selected configurations

Printed so the run is self-documenting: the results parquet records a seed and
a model name, not the architecture behind them.

**The tuning scores are not comparable across models.** Each is recorded under
its own criterion - held-out log-likelihood for the variants, Neural Fine-Gray
and DSM; integrated Brier for DeepHit, whose training loss mixes likelihood
with a ranking term and is therefore not a likelihood. Beyond that, a Weibull
mixture and a spline cumulative hazard do not share an estimand, so even two
"log-likelihood" figures sit on different scales. They are provenance, not a
comparison. **The comparison is the CIF-based metrics this notebook produces**,
which every model is scored on identically.
"""),
        code(CELL_CONFIGS),
        md(r"""
## 2. Factories

One place where a model name becomes a fitted object. The classical baselines
are handed `spline_columns` explicitly, read from the `FeatureSpec`, rather
than letting the >=20-distinct-values heuristic decide - that gate is a safety
net, and relying on it would silently expand a many-levelled ordinal.
"""),
        code(CELL_FACTORIES),
        md(r"""
## 3. Timing probe

One fit per model at full n on fold 0, m = 1, before committing to 270 cells.
Expected from the tuning run: variants and DSM 10-25 s, DeepHit ~26 s, Neural
Fine-Gray ~50 s. If the projection lands far above three hours, find out why
rather than letting it run overnight.
"""),
        code(CELL_PROBE),
        md(r"""
## 4-5. Build and run, one model at a time

Cells are built and run **per (model, imputation) group** rather than all 270
up front. `BenchmarkCell` holds its arrays, so building the whole matrix first
would keep every fold of every imputation in memory at once - gigabytes of
design matrices, nearly all idle. Grouping bounds it to one imputation's worth,
at the cost of re-reading the feature files per model: seconds, against hours
of fitting.

Ordering by model also makes progress legible. With ~270 cells over hours, a
per-model elapsed line is what distinguishes a stalled run from a slow one.

`resume=True`, so an interruption continues from the parquet. Seeds come from
`derive_seed(model, fold, imputation)` - a hash, not a counter - so a resumed
run reproduces exactly what an uninterrupted one would have produced.

`dev_full` cells use **`fold = -1`, an integer**. `run_benchmark` writes `fold`
as a parquet column; a string there makes it an object column that pyarrow
refuses outright, and would break any later join against the other models'
integer folds.
"""),
        code(CELL_RUN),
        md(r"""
## 6. Post-run checks

Read these before anything downstream. RSF is concatenated here - only here -
from notebook 01's output.
"""),
        code(CELL_COHERENCE),
        code(CELL_UNUSABLE),
        code(CELL_CALIBRATION),
        code(CELL_TIMING),
        md(r"""
## 7. Headline table

**Calibration first.** The endpoint is calibrated absolute risk, not
discrimination: a model can rank patients well and still be wrong about how
much risk each one carries, which is the failure this whole exercise exists to
detect. AUC is reported, but it is not the result. Rows are ordered by ICI.

Mean +/- SD across folds and imputations, per model x cause x horizon. The SD
mixes fold and imputation variance, so read it as spread rather than as a
standard error.
"""),
        code(CELL_HEADLINE),
        md(r"""
## 8. Summary

**Produced**

- `development_results.parquet` - long format, one row per (model, fold,
  imputation, eval_set, cause, horizon), with AUC, C-index, Brier, Austin
  calibration (ICI/E50/E90/Emax), calibration-in-the-large, the coherence
  residual, timings, seeds, and a per-cause `converged` flag. `fold = -1`
  marks the full-development fits.
- `development_summary.csv` - the headline table, mean and SD across folds and
  imputations.

**Covers** nine models across 5 folds x 5 imputations, plus a full-development
fit per model per imputation. RSF is *not* in this file; it lives in
`rsf_results.parquet` at m = 1 and is concatenated at analysis time.

**Temporal and exploratory remain untouched.**

**Consumed by** the final evaluation, which locks the model choice made here
and scores it once on the held-out windows. That notebook is the only place
those windows are loaded, and it should not be written until every decision
this table prompts has been made.

**To report in the methods:** the selected configurations (cell 1), the
calibration-method shares (cell 6), RSF's single-imputation deviation, and
whatever this table changes about the model choice.
"""),
        code(CELL_SUMMARY),
    ]
