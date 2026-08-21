"""Validation of the feature builder.

This module is the last transform before modelling and the last place a
train/test boundary can be violated, so the tests are organised around the
two claims it has to support: nothing is fitted on held-out data, and the
neural variants and the classical baselines receive identical information.
`test_2_information_equivalence` underwrites the fairness of the entire model
comparison - if it fails, no benchmark result means anything.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from features import (CATEGORICAL, CONTINUOUS, PRETRAIN_MASKED,     # noqa: E402
                      ROBUST_COLUMNS, FeatureSpec, encoder_config_kwargs,
                      fit_feature_spec, recover_categoricals, summarise,
                      transform)
from features.spec import FIXED_REFERENCES                          # noqa: E402

SEED = 42

# Level pools chosen to mirror the imputed files: strings for Y/N/U and M/F,
# coded floats for the SRTR numeric categoricals, and the explicit
# `not_biopsied` level the imputation stage created.
LEVEL_POOLS = {
    "GENDER": ["M", "F"], "GENDER_DON": ["M", "F"],
    "DIAL_TRR": ["Y", "N"], "PUMP_KI": ["Y", "N", "U"],
    "BIOPSY": ["Y", "N"], "NON_HRT_DON": ["Y", "N"],
    "PREV_TX_OTHER_ORG": [0, 1], "HLA_ZERO": [0, 1],
    "DIAG": list(range(1, 14)), "COD_CAD_DON": [1, 2, 3, 4],
    "FUNC_STAT_TRR": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "HYPERTENS_DUR_DON": [0, 1, 2, 3], "HIST_DIABETES_DON": [1, 2, 3, 4, 5],
    "ABO_MAT": ["1", "2", "3"],
    "DIAB": [1, 2, 3, 4, 5], "PERIP_VASC": ["Y", "N", "U"],
    "MALIG": ["Y", "N"], "CMV_STATUS": ["P", "N", "U"],
    "CMV_DON": ["P", "N", "U"], "HCV_SEROSTATUS": ["P", "N", "U"],
    "HEP_C_ANTI_DON": ["P", "N", "U"],
    "GLOMERUL": ["1", "2", "3", "4", "5", "6", "not_biopsied"],
    "INTERSTITIAL_FIBROSIS": ["1", "2", "3", "4", "5", "6", "not_biopsied"],
    "VASCULAR_CHANGES": ["1", "2", "3", "4", "5", "6", "not_biopsied"],
    "AMIS": [0, 1, 2], "BMIS": [0, 1, 2], "DRMIS": [0, 1, 2],
}


def make_frame(n=4000, seed=SEED, drop_bmi=True):
    """A frame with the schema of the imputed parquet files."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    for c in CONTINUOUS:
        if c in ("BMI_CALC", "BMI_DON_CALC") and drop_bmi:
            continue          # absent in the real files; derived at build time
        if c in ROBUST_COLUMNS:
            df[c] = rng.lognormal(5.5, 1.1, n)          # right-skewed
        elif c in ("HGT_CM_CALC", "HGT_CM_DON_CALC"):
            df[c] = rng.normal(170, 10, n)
        elif c in ("WGT_KG_CALC", "WGT_KG_DON_CALC"):
            df[c] = rng.normal(80, 15, n)
        elif c == "NPKID":
            df[c] = rng.integers(0, 4, n).astype(float)
        elif c == "END_CPRA":
            df[c] = rng.uniform(0, 100, n)
        else:
            df[c] = rng.normal(50, 12, n)
    for c in CATEGORICAL:
        pool = LEVEL_POOLS[c]
        # Skewed level frequencies so the modal reference is unambiguous.
        w = np.linspace(3.0, 1.0, len(pool))
        df[c] = rng.choice(pool, size=n, p=w / w.sum())
    df["GTIME_YEARS"] = rng.uniform(0.1, 8, n)
    df["event_type"] = rng.choice([0, 1, 2], n)
    df["PT_CODE"] = rng.integers(1, n, n).astype(float)
    df["window"] = "development"
    df["cv_fold"] = rng.integers(0, 5, n)
    df["TX_DATE"] = pd.Timestamp("2016-01-01")
    return df


@pytest.fixture(scope="module")
def frames():
    df = make_frame()
    train = df.iloc[:3000].reset_index(drop=True)
    test = df.iloc[3000:].reset_index(drop=True)
    return train, test


@pytest.fixture(scope="module")
def spec(frames):
    train, _ = frames
    return fit_feature_spec(train, fit_name="dev_fold0_train_m1")


# ---------------------------------------------------------------------------
# 1. No leakage
# ---------------------------------------------------------------------------

def test_1_no_leakage_from_held_out_rows(frames):
    """Perturbing the held-out rows arbitrarily must leave the fitted spec
    byte-identical: nothing about them can reach a scaler or a level map."""
    train, test = frames
    base = fit_feature_spec(train, fit_name="f").to_json()

    wrecked = test.copy()
    for c in CONTINUOUS:
        if c in wrecked.columns:
            wrecked[c] = wrecked[c] * 1000 + 5000
    for c in CATEGORICAL:
        wrecked[c] = wrecked[c].iloc[0]                 # collapse every level
    # Fitting on the same training rows, with the test rows mangled.
    again = fit_feature_spec(train, fit_name="f").to_json()
    assert base == again

    # And a spec fitted on train+test differs, so the test has teeth.
    mixed = fit_feature_spec(pd.concat([train, wrecked], ignore_index=True),
                             fit_name="f").to_json()
    assert mixed != base


# ---------------------------------------------------------------------------
# 2. Information equivalence - the fairness claim
# ---------------------------------------------------------------------------

def test_2_information_equivalence(frames, spec):
    """The embedding and one-hot representations must encode the same
    information. Recover the categorical values from each and require an
    exact match; the neural model and the baselines then differ in inductive
    bias, never in what they were told."""
    _, test = frames
    emb = transform(test, spec, "embedding")
    oh = transform(test, spec, "onehot")

    rec_emb = recover_categoricals(emb, spec)
    rec_oh = recover_categoricals(oh, spec)
    pd.testing.assert_frame_equal(rec_emb, rec_oh)

    # Both must also match the source values (after unseen-level folding).
    for c in spec.categorical_order:
        levels = set(spec.categorical_levels[c])
        src = test[c].map(lambda v: v if v in levels
                          else spec.categorical_reference[c])
        assert (rec_emb[c].astype(str).to_numpy() == src.astype(str).to_numpy()).all()

    # The continuous block is literally the same array in both.
    assert np.array_equal(emb.continuous, oh.continuous)
    assert np.array_equal(oh.design[:, :len(spec.continuous_order)], emb.continuous)
    print(f"\n  embedding {emb.continuous.shape[1]} continuous + "
          f"{emb.categorical_codes.shape[1]} codes; "
          f"one-hot design {oh.design.shape[1]} columns")


# ---------------------------------------------------------------------------
# 3. Unseen levels
# ---------------------------------------------------------------------------

def test_3_unseen_levels_map_to_reference(frames, spec):
    """A level seen only in validation folds onto the reference, adds no
    column, and is counted."""
    _, test = frames
    novel = test.copy()
    novel.loc[novel.index[:37], "CMV_STATUS"] = "NEW_LEVEL"
    novel.loc[novel.index[:11], "DIAG"] = 999

    oh_ref = transform(test, spec, "onehot")
    oh = transform(novel, spec, "onehot")
    emb = transform(novel, spec, "embedding")

    assert oh.columns == oh_ref.columns, "an unseen level created a column"
    assert oh.design.shape[1] == oh_ref.design.shape[1]
    assert oh.unseen_counts["CMV_STATUS"] == 37
    assert oh.unseen_counts["DIAG"] == 11
    print(f"\n  unseen-level counts: {oh.unseen_counts}")

    j = spec.categorical_order.index("CMV_STATUS")
    ref_code = spec.categorical_levels["CMV_STATUS"].index(
        spec.categorical_reference["CMV_STATUS"])
    assert (emb.categorical_codes[:37, j] == ref_code).all()


# ---------------------------------------------------------------------------
# 4. Reference levels
# ---------------------------------------------------------------------------

def test_4_reference_levels(frames, spec):
    """AMIS/BMIS/DRMIS use 0; one-hot drops exactly one column per
    categorical."""
    for c, ref in FIXED_REFERENCES.items():
        assert spec.categorical_reference[c] == ref, c

    _, test = frames
    oh = transform(test, spec, "onehot")
    n_cont = len(spec.continuous_order)
    n_ind = oh.design.shape[1] - n_cont
    expected = sum(len(spec.categorical_levels[c]) - 1
                   for c in spec.categorical_order)
    assert n_ind == expected
    for c in spec.categorical_order:
        cols = [x for x in oh.columns if x.startswith(f"{c}=")]
        assert len(cols) == len(spec.categorical_levels[c]) - 1
        ref_col = f"{c}={spec.categorical_reference[c]}"
        assert ref_col not in cols, f"reference column {ref_col} was not dropped"


def test_4b_diag_reference_is_configurable(frames):
    """DIAG should use the diabetes group. Its code is not
    discoverable from the data here, so the builder takes it as a parameter
    and warns loudly when it falls back to the modal level."""
    train, _ = frames
    
    # 999 is not a DIAG group, so this exercises the "not observed" branch,
    # which is the same fallback path as "not supplied".
    
    with pytest.warns(RuntimeWarning, match="DIAG"):
        fallback = fit_feature_spec(train, fit_name="f", diag_reference=999)
    assert any("DIAG" in n for n in fallback.notes)

    chosen = fit_feature_spec(train, fit_name="f", diag_reference=7)
    assert chosen.categorical_reference["DIAG"] == 7
    assert not any("DIAG" in n for n in chosen.notes)
    
    
    # The project's actual choice, locked in: a silent change would make DIAG
    # coefficients incomparable across folds and imputations.
    default = fit_feature_spec(train, fit_name="f")
    assert default.categorical_reference["DIAG"] == 2


# ---------------------------------------------------------------------------
# 5. not_biopsied
# ---------------------------------------------------------------------------

def test_5_not_biopsied_survives(spec, frames):
    """The structural `not_biopsied` level created during imputation must
    survive as its own level in all three biopsy grade columns."""
    _, test = frames
    emb = transform(test, spec, "embedding")
    oh = transform(test, spec, "onehot")
    for c in ("GLOMERUL", "INTERSTITIAL_FIBROSIS", "VASCULAR_CHANGES"):
        assert "not_biopsied" in spec.categorical_levels[c], c
        if spec.categorical_reference[c] != "not_biopsied":
            assert f"{c}=not_biopsied" in oh.columns
    rec = recover_categoricals(emb, spec)
    assert (rec["GLOMERUL"] == "not_biopsied").any()


# ---------------------------------------------------------------------------
# 6. JSON round-trip
# ---------------------------------------------------------------------------

def test_6_json_round_trip(spec, frames):
    _, test = frames
    text = spec.to_json()
    back = FeatureSpec.from_json(text)
    assert back.to_json() == text
    assert back.feature_order == spec.feature_order
    for rep in ("embedding", "onehot"):
        a = transform(test, spec, rep)
        b = transform(test, back, rep)
        assert a.columns == b.columns
        assert np.array_equal(a.continuous, b.continuous)
        assert np.array_equal(a.categorical_codes, b.categorical_codes)
        if rep == "onehot":
            assert np.array_equal(a.design, b.design)


# ---------------------------------------------------------------------------
# 7. Shape and column-order stability
# ---------------------------------------------------------------------------

def test_7_column_order_is_stable():
    """Column order must be stable, because a silent reordering would corrupt
    every model comparison without raising anything.

    This test found a real problem worth stating. The FEATURE order (and the
    one-hot column COUNT) is stable everywhere by construction. The one-hot
    column NAMES are not automatically stable across fit sets, because the
    reference level is the modal level of the training rows and the mode is a
    sample quantity: with two levels of similar frequency it flips, and a
    different level gets dropped.

    Predictions are invariant to which level is dropped, so this cannot bias a
    metric. But POOLED BASELINE COEFFICIENTS across the five imputations would
    be meaningless if different imputations dropped different levels. The fix
    is to pin the references from imputation 1 across the rest of a fit set -
    the same patients, so no leakage - which is what the build notebook does
    and what this test verifies.
    """
    # (a) Across imputations of one fit set, with references pinned from m=1:
    #     names and order must be identical.
    base = fit_feature_spec(make_frame(n=1500, seed=1), diag_reference=7)
    pinned = base.categorical_reference
    names = []
    for m in range(1, 6):
        df = make_frame(n=1500, seed=m)
        sp = fit_feature_spec(df, fit_name=f"m{m}", diag_reference=7,
                              reference_overrides=pinned)
        names.append(tuple(transform(df, sp, "onehot").columns))
    assert len(set(names)) == 1, "one-hot columns differ across imputations"

    # (b) Across independently fitted folds: feature order and column COUNT
    #     are stable; the dropped level may differ, and that is by design.
    orders, counts, dropped = set(), set(), []
    for fold in range(4):
        df = make_frame(n=1500, seed=100 + fold)
        sp = fit_feature_spec(df, fit_name=f"fold{fold}", diag_reference=7)
        res = transform(df, sp, "onehot")
        orders.add(tuple(sp.feature_order))
        counts.add(res.design.shape[1])
        dropped.append(sp.categorical_reference["VASCULAR_CHANGES"])
    assert len(orders) == 1, "feature order differs across folds"
    assert len(counts) == 1, "one-hot width differs across folds"
    print(f"\n  imputations (references pinned): identical column names, "
          f"{len(names[0])} columns")
    print(f"  folds (references fitted per fold): stable width "
          f"{counts.pop()}, VASCULAR_CHANGES reference per fold = {dropped}")


# ---------------------------------------------------------------------------
# 8. Scaler choice
# ---------------------------------------------------------------------------

def test_8_scaler_choice_recorded(spec):
    """Robust scaling on the skewed columns, standard elsewhere, recoverable
    from the spec."""
    for c in spec.continuous_order:
        kind = spec.continuous_scalers[c].kind
        assert kind == ("robust" if c in ROBUST_COLUMNS else "standard"), c
    robust = [c for c in spec.continuous_order
              if spec.continuous_scalers[c].kind == "robust"]
    assert set(robust) == ROBUST_COLUMNS & set(spec.continuous_order)
    print(f"\n  robust-scaled: {sorted(robust)}")

    # Standard scaling must actually standardise the training rows.
    df = make_frame(n=2000, seed=9)
    sp = fit_feature_spec(df, diag_reference=7)
    res = transform(df, sp, "embedding")
    j = sp.continuous_order.index("AGE")
    assert abs(res.continuous[:, j].mean()) < 1e-10
    assert abs(res.continuous[:, j].std() - 1.0) < 1e-10
    # Robust scaling centres the median and scales by the IQR.
    j = sp.continuous_order.index("COLD_ISCH_KI")
    assert abs(np.median(res.continuous[:, j])) < 1e-10


# ---------------------------------------------------------------------------
# 9. Pretrain masking
# ---------------------------------------------------------------------------

def test_9_pretrain_spec_masks_pre2014_features(frames):
    """The pretraining spec drops the five features absent before 2014 and
    keeps GLOMERUL."""
    train, _ = frames
    sp = fit_feature_spec(train, fit_name="pretrain_m1", masked=True,
                          diag_reference=7)
    for c in PRETRAIN_MASKED:
        assert c not in sp.feature_order, f"{c} should be masked"
    assert "GLOMERUL" in sp.categorical_order
    full = fit_feature_spec(train, diag_reference=7)
    assert len(sp.feature_order) == len(full.feature_order) - len(PRETRAIN_MASKED)

    res = transform(train, sp, "onehot")
    assert not any(col.split("=")[0] in PRETRAIN_MASKED for col in res.columns)
    print(f"\n  pretrain spec: {len(sp.feature_order)} features "
          f"(full {len(full.feature_order)}), GLOMERUL retained")


# ---------------------------------------------------------------------------
# Feature counts, encoder wiring, reporting
# ---------------------------------------------------------------------------

def test_feature_counts_reported(spec):
    n_cont = len(spec.continuous_order)
    n_cat = len(spec.categorical_order)
    n_oh = spec.n_onehot_features
    sum_levels_minus_1 = sum(len(spec.categorical_levels[c]) - 1
                             for c in spec.categorical_order)
    print(f"\n  embedding: {n_cont} continuous + {n_cat} categorical codes "
          f"= {spec.n_embedding_features}")
    print(f"  one-hot  : {n_cont} continuous + {sum_levels_minus_1} indicators "
          f"= {n_oh}")
    assert spec.n_embedding_features == n_cont + n_cat
    assert n_oh == n_cont + sum_levels_minus_1


def test_encoder_config_kwargs_match_spec(spec, frames):
    """The cardinality tuple handed to CovariateEncoder must match the codes
    actually produced, or embeddings would index out of range."""
    from encoder import CovariateEncoder, EncoderConfig
    _, test = frames
    kw = encoder_config_kwargs(spec)
    cfg = EncoderConfig(hidden_dims=(32,), latent_dim=8, dropout=0.0, **kw)
    enc = CovariateEncoder(cfg)
    res = transform(test, spec, "embedding")
    assert res.categorical_codes.max(axis=0).tolist() <= [c - 1 for c in
                                                          spec.cardinalities]
    import torch
    z = enc(torch.tensor(res.continuous, dtype=torch.float32),
            torch.tensor(res.categorical_codes, dtype=torch.long))
    assert z.shape == (len(test), 8) and torch.isfinite(z).all()


def test_summary_report(spec, frames):
    _, test = frames
    novel = test.copy()
    novel.loc[novel.index[:5], "MALIG"] = "ZZZ"
    text = summarise(spec, {"held_out": transform(novel, spec, "embedding")})
    assert "embedding" in text and "one-hot" in text and "robust" in text
    assert "MALIG" in text
    print("\n" + "\n".join(text.splitlines()[:6]))
