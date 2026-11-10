"""Code checks on synthetic data only - no STAR data, no results."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("step2", HERE / "02_cohort_cutoffs.py")
step2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step2)


def fake(changes, n_per_day=40, seed=0):
    """changes: list of (start_date, kdri_cutoff); KDPI>=86 iff KDRI >= cutoff."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2015-01-01", "2016-12-31")
    d = pd.DataFrame({"tx_date": np.repeat(days, n_per_day)})
    d["kdri"] = np.round(rng.lognormal(0.1, 0.3, len(d)), 4)
    cut = pd.Series(np.nan, index=d.index)
    for start, c in changes:
        cut[d.tx_date >= pd.Timestamp(start)] = c
    d["high"] = d.kdri >= cut
    d["kdpi"] = np.where(d.high, 90, 80)
    return d


def test_best_threshold_separable():
    k = np.array([1.0, 1.1, 1.2, 1.5, 1.6]); h = np.array([0, 0, 0, 1, 1], bool)
    c, wrong, la, hb = step2.best_threshold(k, h)
    assert (c, wrong) == (1.5, 0)


def test_best_threshold_counts_wrong_side():
    k = np.array([1.0, 1.1, 1.7, 1.5, 1.6]); h = np.array([0, 1, 0, 1, 1], bool)
    c, wrong, la, hb = step2.best_threshold(k, h)
    assert (c, wrong, la, hb) == (1.1, 1, 1, 0)


def test_segmentation_finds_mid_year_change():
    d = fake([("2015-01-01", 1.60), ("2016-04-20", 1.70)])
    g = d[d.tx_date.dt.year == 2016]
    segs = step2.segment(g, min_side=30)
    assert len(segs) == 2
    splits = []
    segs = step2.segment(g, min_side=30, splits=splits)
    w = splits[0]
    assert w["first_new_table_date_earliest"] <= pd.Timestamp("2016-04-20").date() <= w["first_new_table_date_latest"]
    assert all(step2.summarise(s)["n_wrong_side"] == 0 for s in segs)


def test_no_split_when_table_constant():
    d = fake([("2015-01-01", 1.60)])
    assert len(step2.segment(d, min_side=30)) == 1
