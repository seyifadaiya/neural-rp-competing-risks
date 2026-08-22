# Runner notebooks

Two long-running, unattended jobs against local data. Both restartable, both
timestamped, and neither touches the temporal or exploratory windows.

| Notebook | What it does | Rough cost |
|---|---|---|
| `01_rsf_benchmark.ipynb` | RSF across the 5 development folds + a full-development fit, at m = 1 | hours; the probe cell projects it before launch |
| `02_tuning.ipynb` | 30-configuration random search for all five models on fold 0 | dominated by Neural Fine-Gray |
| `03_development_matrix.ipynb` | The other 9 models across 5 folds x 5 imputations + `dev_full` — 270 cells | 2–3 hours |

Run them in order: 03 consumes `selected_configs.json` from 02 and concatenates
`rsf_results.parquet` from 01 at analysis time. It does **not** re-run RSF.

`io_helpers.py` holds everything they share: the loader, the two guards, the
RSF production options, the Variant A/B adapter, the spline-column lookup, and
the null-model Brier table. `make_notebooks.py` regenerates the notebooks —
edit that, not the `.ipynb` JSON. Notebook 03's cells live in `_nb3_cells.py`,
because its cell bodies are themselves triple-quoted strings and nesting those
inside another triple-quoted block truncates silently.

## Before running

Edit the **single configuration cell** at the top of each notebook:

```python
REPO_ROOT    = Path(r"")   # leave "" to auto-detect
FEATURES_DIR = Path(r"")   # feature-builder output
RESULTS_DIR  = Path(r"")   # where results are written
```

Nothing else in either notebook contains a path.

Neural Fine-Gray additionally needs its repository, which is not on PyPI:

```bash
git clone --recursive https://github.com/Jeanselme/NeuralFineGray.git
export NFG_PATH=/path/to/NeuralFineGray
```

Without it that one model reports itself unavailable and is skipped — it is
never silently replaced by something else.

## The two guards

Both raise rather than warn, because both failures are invisible afterwards.

1. **No outcome column may reach a design matrix.** `GTIME_YEARS` or
   `event_type` among the features would produce a model that scores superbly
   and means nothing.
2. **The temporal and exploratory windows are never loaded.** They are scored
   once, at the end of the project, after every modelling decision is locked.
   `load_set` refuses them outright, in either argument position.

## Testing

```bash
uv run pytest notebooks/test_io_helpers.py -q
```

19 tests, synthetic data only. They cover the guards, the spec width
cross-check, and the RSF option plumbing — not the notebooks themselves, which
need real data and hours.

## Deviations to report in the paper

- **RSF at imputation m = 1 only.** It is the runtime bottleneck; between-
  imputation variance is negligible at this project's missingness rates. State
  it in the methods with those rates.
- **Tuning on fold 0 only, at full n.** Cutting folds adds selection *noise*
  (symmetric across models); subsampling rows would add *bias* (asymmetric, as
  regularisation optima shift with sample size). The fold was cut rather than
  the rows so no model is tuned on less data than its competitors. Cell 7 of
  the tuning notebook checks the top Neural Fine-Gray configurations transport
  to a second fold; report that result.
- **Whatever RSF settings the probe cell prints.** An RSF at 300 trees is not
  an RSF at 1000.
- **DeepHit's `num_durations` came from a search run before a discretisation
  bug was fixed.** pycox's Kaplan-Meier sums event codes, so competing-risks
  codes 0/1/2 counted every cause-2 event twice and the quantile bins were
  placed too early. The bug is fixed, but `selected_configs.json` predates the
  fix, so DeepHit's bin count is a fixed fair configuration rather than a tuned
  optimum. Either re-run the DeepHit arm of notebook 02 (~1 hour) or say so.
