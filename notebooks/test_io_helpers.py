"""Smoke test for the notebook helpers, on synthetic data only.

The notebooks themselves are not executed here - they run for hours against
real data. What IS testable without them is the part where a silent,
invalidating mistake would live: the loader's guards, the spec cross-check,
and the RSF option plumbing. This builds a miniature copy of the feature
builder's on-disk layout and exercises those.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rp_core"))

import io_helpers as ioh  # noqa: E402

CONT = ["AGE", "AGE_DON", "KDRI_RAO"]
CATS = {"GENDER": ["F", "M"], "DIAB": ["none", "type1", "type2"]}
REFERENCE = {"GENDER": "F", "DIAB": "none"}


def write_set(root: Path, name: str, m: int = 1, n: int = 120, seed: int = 0,
              leak: bool = False):
    """One applied set in exactly the layout `features/build.ipynb` writes."""
    rng = np.random.default_rng(seed)
    cont = rng.normal(size=(n, len(CONT)))
    codes = np.column_stack([rng.integers(0, len(v), n) for v in CATS.values()])
    onehot_cols = list(CONT)
    for c, levels in CATS.items():
        onehot_cols += [f"{c}={lv}" for lv in levels if lv != REFERENCE[c]]
    design = np.hstack([cont] + [
        np.eye(len(levels))[codes[:, j]][:, [i for i, lv in enumerate(levels)
                                             if lv != REFERENCE[c]]]
        for j, (c, levels) in enumerate(CATS.items())])

    emb_cols = list(CONT) + list(CATS)
    if leak:                       # the failure the guard exists to catch
        emb_cols = emb_cols + ["GTIME_YEARS"]
        cont = np.hstack([cont, rng.normal(size=(n, 1))])

    stem = root / f"{name}_m{m}"
    np.savez_compressed(f"{stem}_embedding.npz", continuous=cont,
                        categorical_codes=codes,
                        columns=np.array(emb_cols, dtype=object),
                        cardinalities=np.array([len(v) for v in CATS.values()]))
    np.savez_compressed(f"{stem}_onehot.npz", design=design,
                        columns=np.array(onehot_cols, dtype=object))
    t = rng.uniform(0.05, 9.0, n)
    e = rng.integers(0, 3, n)
    pd.DataFrame({"GTIME_YEARS": t, "event_type": e,
                  "PT_CODE": np.arange(n), "window": "development",
                  "cv_fold": 0}).to_parquet(f"{stem}_outcomes.parquet",
                                            index=False)


def write_spec(root: Path, fit_name: str, m: int = 1):
    payload = {
        "categorical_levels": {k: list(v) for k, v in CATS.items()},
        "categorical_reference": dict(REFERENCE),
        "continuous_scalers": {},
        "continuous_order": list(CONT),
        "categorical_order": list(CATS),
        "fit_name": f"{fit_name}_m{m}",
        "n_fit_rows": 120,
        "notes": [],
    }
    (root / f"spec_{fit_name}_m{m}.json").write_text(json.dumps(payload))


@pytest.fixture
def features(tmp_path):
    for fold in (0, 1):
        fit, test = ioh.fold_sets(fold)
        write_spec(tmp_path, fit)
        write_set(tmp_path, fit, seed=fold * 10)
        write_set(tmp_path, test, n=40, seed=fold * 10 + 1)
    write_spec(tmp_path, "dev_full")
    write_set(tmp_path, "dev_full", n=160, seed=99)
    # Held-out windows exist on disk, as they will in reality.
    write_set(tmp_path, "temporal", n=50, seed=7)
    write_set(tmp_path, "exploratory", n=30, seed=8)
    return tmp_path


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_held_out_windows_cannot_be_loaded(features):
    """The guard must fire on both windows, in either argument position."""
    for name in ("temporal", "exploratory"):
        with pytest.raises(ValueError, match="held-out"):
            ioh.load_set(features, "dev_full", name)
        with pytest.raises(ValueError, match="held-out"):
            ioh.load_set(features, name, "dev_full")
    # ... and the files are genuinely there, so the guard is what stopped it,
    # not a missing file.
    assert (features / "temporal_m1_onehot.npz").exists()


def test_outcome_column_in_features_is_rejected(features, tmp_path):
    """An outcome column among the features must raise, not warn."""
    write_spec(tmp_path, "leaky")
    write_set(tmp_path, "leaky", leak=True)
    with pytest.raises(ValueError, match="GTIME_YEARS"):
        ioh.load_set(tmp_path, "leaky", "leaky", representation="embedding")


@pytest.mark.parametrize("bad", sorted(ioh.FORBIDDEN_FEATURE_COLUMNS))
def test_every_forbidden_column_is_caught(bad):
    with pytest.raises(ValueError, match="design"):
        ioh.assert_no_outcome_columns(["AGE", bad, "KDRI_RAO"])
    ioh.assert_no_outcome_columns(["AGE", "KDRI_RAO"])


def test_spec_cross_check_is_live(features, tmp_path):
    """The width check must actually compare something.

    n_onehot_features is a property and is absent from the serialised spec, so
    reading that key would leave this check permanently passing. This asserts
    it fails when the widths genuinely disagree.
    """
    spec = json.loads((features / "spec_dev_fold0_train_m1.json").read_text())
    assert ioh._expected_width(spec, "embedding") == len(CONT) + len(CATS)
    # one non-reference column per categorical level
    assert ioh._expected_width(spec, "onehot") == len(CONT) + 1 + 2

    spec["continuous_order"] = CONT + ["AN_EXTRA_COLUMN"]
    (tmp_path / "spec_dev_fold0_train_m1.json").write_text(json.dumps(spec))
    for f in features.glob("dev_fold0_*"):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    with pytest.raises(ValueError, match="not built from this spec"):
        ioh.load_set(tmp_path, "dev_fold0_train", "dev_fold0_train")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_onehot_and_embedding(features):
    tr, te = ioh.load_fold(features, 0, representation="onehot")
    assert tr["X"].shape == (120, len(CONT) + 3)
    assert te["X"].shape == (40, len(CONT) + 3)
    assert tr["time"].shape == (120,) and tr["event"].shape == (120,)
    assert set(np.unique(tr["event"])) <= {0, 1, 2}
    assert tr["time"].min() > 0

    etr, ete = ioh.load_fold(features, 0, representation="embedding")
    # embedding X is [continuous | codes], which is what the deep wrappers take
    assert etr["X"].shape == (120, len(CONT) + len(CATS))
    assert etr["n_continuous"] == len(CONT)
    assert etr["cardinalities"] == [len(v) for v in CATS.values()]
    # same patients, same outcomes, whichever representation
    assert np.array_equal(etr["time"], tr["time"])
    assert np.array_equal(etr["event"], tr["event"])


def test_missing_file_says_what_is_missing(features):
    with pytest.raises(FileNotFoundError, match="FEATURES_DIR"):
        ioh.load_set(features, "dev_fold0_train", "dev_fold9_train")


def test_as_cell_data_and_describe(features):
    tr, _ = ioh.load_fold(features, 0)
    cell = ioh.as_cell_data(tr)
    assert set(cell) == {"X", "time", "event"}
    assert "dev_fold0_train" in ioh.describe(tr)


def test_available_sets_flags_held_out(features):
    table = ioh.available_sets(features)
    assert set(table.applied_to) >= {"dev_fold0_train", "dev_full", "temporal"}
    assert table.set_index("applied_to").loc["temporal", "held_out"]
    assert not table.set_index("applied_to").loc["dev_full", "held_out"]
    assert table["onehot"].all() and table["embedding"].all()


# ---------------------------------------------------------------------------
# RSF production options
# ---------------------------------------------------------------------------

def test_production_rsf_injects_the_requested_options():
    """The extra R options must reach the template, and the package's own
    template must be restored afterwards."""
    import benchmark.classical as classical

    before = classical.RSF_R_TEMPLATE
    tpl = ioh._tuned_template(nsplit=10, perf_type="none", block_size=None,
                              rf_cores=None)
    assert "nsplit = 10" in tpl
    assert 'perf.type = "none"' in tpl
    assert "block.size = NULL" in tpl
    assert "options(rf.cores = parallel::detectCores())" in tpl
    # the package's placeholders must survive, or the parent's _run breaks
    for token in ("__TRAIN__", "__TEST__", "__OUT__", "__TIMES__", "__SEED__",
                  "__NTREE__", "__NODESIZE__", "__NTIME__"):
        assert token in tpl, token
    assert classical.RSF_R_TEMPLATE is before, "global left modified"

    m = ioh.production_rsf(columns=["a", "b"], seed=3, ntree=500, nodesize=30)
    assert m.name == "rsf" and m.seed == 3
    assert m.ntree == 500 and m.nodesize == 30 and m.ntime == 100
    assert m.r_options["nsplit"] == 10
    assert hasattr(m, "fit") and hasattr(m, "predict_cif")


def test_rsf_factory_shape():
    f = ioh.rsf_factory(columns=["a"], ntree=300)
    m = f(11)
    assert m.seed == 11 and m.ntree == 300


def test_template_change_is_detected():
    """If rp_core's template is restructured, this must fail loudly rather
    than silently producing an rfsrc call without the extra options."""
    import benchmark.classical as classical

    original = classical.RSF_R_TEMPLATE
    classical.RSF_R_TEMPLATE = "totally different template"
    try:
        with pytest.raises(RuntimeError, match="template changed shape"):
            ioh._tuned_template(10, "none", None, None)
    finally:
        classical.RSF_R_TEMPLATE = original


# ---------------------------------------------------------------------------
# Post-run checks
# ---------------------------------------------------------------------------

"""def test_failure_report_surfaces_bad_cells():
    df = pd.DataFrame({
        "model": ["rsf"] * 4, "fold": [0, 1, 2, 3], "imputation": [1] * 4,
        "eval_set": ["test"] * 4, "cause": [1] * 4, "horizon": [5.0] * 4,
        "brier": [0.10, 0.12, 0.40, 0.11],
        "converged": [True, False, True, True],
    })
    bad = ioh.failure_report(df)
    assert set(bad.fold) == {1, 2}, "should flag non-convergence AND a bad Brier"

    null = {(1, 5.0): 0.20}
    checked = ioh.beats_null(df, null)
    assert bool(checked.loc[checked.fold == 2, "beats_null"].item()) is False
    assert bool(checked.loc[checked.fold == 0, "beats_null"].item()) is True
"""

def test_failure_report_surfaces_bad_cells():
    # As the harness writes it: base.py computes beats_null against the
    # Aalen-Johansen marginal, so failure_report must USE that column.
    df = pd.DataFrame({
        "model": ["rsf"] * 4, "fold": [0, 1, 2, 3], "imputation": [1] * 4,
        "eval_set": ["test"] * 4, "cause": [1] * 4, "horizon": [5.0] * 4,
        "brier": [0.10, 0.12, 0.40, 0.11],
        "null_brier": [0.20] * 4,
        "beats_null": [True, True, False, True],
        "converged": [True, False, True, True],
    })
    assert set(ioh.failure_report(df).fold) == {1, 2}

    # Without the harness column, an explicit reference reproduces it.
    raw = df.drop(columns=["null_brier", "beats_null"])
    checked = ioh.beats_null(raw, {(1, 5.0): 0.20})
    assert bool(checked.loc[checked.fold == 2, "beats_null"].item()) is False
    assert bool(checked.loc[checked.fold == 0, "beats_null"].item()) is True

    # With neither, "unknown" must not read as "passed".
    assert set(ioh.failure_report(raw).fold) == {1}

def test_logging_helpers():
    assert ioh.human(30).endswith("s")
    assert ioh.human(600).endswith("m")
    assert ioh.human(7200).endswith("h")
    with ioh.timed("a labelled block"):
        pass
