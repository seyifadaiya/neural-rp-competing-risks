"""
Fitted feature specification: everything learned from training data.

This is the last transform before modelling and therefore the last place a
train/test boundary can be violated. Category level maps, reference levels
and continuous scalers are all LEARNED, so `fit_feature_spec` is called on
training rows only and `transform` is applied to everything else.

One `FeatureSpec` produces BOTH representations - integer codes for the
neural variants' embeddings and one-hot for the classical baselines. That
single fitted object is the mechanism guaranteeing every model receives
identical information; `tests/test_features.py::test_2_information_equivalence`
asserts it rather than trusting it.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CONTINUOUS = [
    "AGE", "AGE_DON", "HGT_CM_CALC", "HGT_CM_DON_CALC",
    "WGT_KG_CALC", "WGT_KG_DON_CALC", #"BMI_CALC", "BMI_DON_CALC",
    "CREAT_DON", "KDRI_RAO", "EPTS_FINAL", "DIALYSIS_VINT_DAYS",
    "DAYSWAIT_CHRON_KI", "COLD_ISCH_KI", "BSA_RATIO", "NPKID",
    "GLOMERUL_COUNT",
    # END_CPRA is a percentage (0-100) and is treated as continuous. It is
    # named in the pretrain mask list but appears in neither variable list in
    # the task spec; see README for the count reconciliation.
    "END_CPRA",
]

CATEGORICAL = [
    "GENDER", "GENDER_DON", "DIAL_TRR", "PUMP_KI", "BIOPSY", "NON_HRT_DON",
    "PREV_TX_OTHER_ORG", "HLA_ZERO",
    "DIAG", "COD_CAD_DON", "FUNC_STAT_TRR", "HYPERTENS_DUR_DON",
    "HIST_DIABETES_DON", "ABO_MAT",
    "DIAB", "PERIP_VASC", "MALIG", "CMV_STATUS", "CMV_DON",
    "HCV_SEROSTATUS", "HEP_C_ANTI_DON",
    "GLOMERUL", "INTERSTITIAL_FIBROSIS", "VASCULAR_CHANGES",
    # Numeric 0/1/2 per locus, deliberately made categorical so the classical
    # baselines can express that DR mismatch matters more than A or B - which
    # a single linear term cannot.
    "AMIS", "BMIS", "DRMIS",
]

# Right-skewed: robust (median / IQR) rather than mean / SD.
ROBUST_COLUMNS = {"DIALYSIS_VINT_DAYS", "DAYSWAIT_CHRON_KI", "COLD_ISCH_KI"}

# Carried through untouched for modelling and evaluation.
PASSTHROUGH = ["GTIME_YEARS", "event_type", "PT_CODE", "window", "cv_fold",
               "TX_DATE"]

# Dropped after imputation; excluded at build time rather than reimputed.
DROPPED = ["ETHCAT", "ETHCAT_DON"]

# Absent pre-2014, so excluded from the pretraining spec. GLOMERUL is retained.
PRETRAIN_MASKED = ["END_CPRA", "EPTS_FINAL", "INTERSTITIAL_FIBROSIS",
                   "VASCULAR_CHANGES", "GLOMERUL_COUNT"]

# Reference levels that are fixed by the analysis plan rather than chosen as
# the modal level. AMIS/BMIS/DRMIS use zero mismatch as the natural baseline.
FIXED_REFERENCES: dict[str, Any] = {"AMIS": 0, "BMIS": 0, "DRMIS": 0}

# DIAG should use the hypertensive/vascular group as reference, but its code
# depends on how DIAG was grouped upstream and is not discoverable from here.
# Set this once (the build notebook prints the observed levels with counts);
# until it is set, DIAG falls back to the modal level with a warning, which is
# a valid fit but not the intended contrast.
DIAG_REFERENCE_CODE: Any = 2


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

@dataclass
class Scaler:
    """Fitted centre/scale for one continuous column, plus which kind."""
    kind: str            # "standard" (mean/SD) or "robust" (median/IQR)
    center: float
    scale: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.center) / self.scale


def _fit_scaler(values: np.ndarray, kind: str) -> Scaler:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if kind == "robust":
        centre = float(np.median(v))
        q1, q3 = np.percentile(v, [25, 75])
        scale = float(q3 - q1)
    else:
        centre = float(v.mean())
        scale = float(v.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        # Constant (or near-constant) column: centre only, never divide by ~0.
        scale = 1.0
    return Scaler(kind=kind, center=centre, scale=scale)


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """Everything learned from training data. Fitted once, applied many times."""
    categorical_levels: dict[str, list]
    categorical_reference: dict[str, Any]
    continuous_scalers: dict[str, Scaler]
    continuous_order: list[str]
    categorical_order: list[str]
    fit_name: str = ""
    n_fit_rows: int = 0
    notes: list[str] = field(default_factory=list)

    # -- derived views ------------------------------------------------------

    @property
    def feature_order(self) -> list[str]:
        """Canonical order: continuous block, then categoricals."""
        return list(self.continuous_order) + list(self.categorical_order)

    @property
    def cardinalities(self) -> list[int]:
        """Levels per categorical, in `categorical_order` - the tuple
        `CovariateEncoder` expects."""
        return [len(self.categorical_levels[c]) for c in self.categorical_order]

    def onehot_columns(self) -> list[str]:
        """Design-matrix column names: continuous, then one column per
        NON-reference level of each categorical."""
        cols = list(self.continuous_order)
        for c in self.categorical_order:
            ref = self.categorical_reference[c]
            cols += [f"{c}={lv}" for lv in self.categorical_levels[c] if lv != ref]
        return cols

    @property
    def n_embedding_features(self) -> int:
        return len(self.continuous_order) + len(self.categorical_order)

    @property
    def n_onehot_features(self) -> int:
        return len(self.onehot_columns())

    # -- serialisation ------------------------------------------------------

    def to_json(self, path=None) -> str:
        payload = {
            "categorical_levels": {k: list(v)
                                   for k, v in self.categorical_levels.items()},
            "categorical_reference": self.categorical_reference,
            "continuous_scalers": {k: asdict(v)
                                   for k, v in self.continuous_scalers.items()},
            "continuous_order": list(self.continuous_order),
            "categorical_order": list(self.categorical_order),
            "fit_name": self.fit_name,
            "n_fit_rows": int(self.n_fit_rows),
            "notes": list(self.notes),
        }
        text = json.dumps(payload, indent=2, sort_keys=False, default=_json_default)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_json(cls, text_or_path) -> "FeatureSpec":
        text = text_or_path
        if isinstance(text_or_path, (str,)) and not text_or_path.lstrip().startswith("{"):
            with open(text_or_path) as fh:
                text = fh.read()
        d = json.loads(text)
        return cls(
            categorical_levels={k: list(v)
                                for k, v in d["categorical_levels"].items()},
            categorical_reference=d["categorical_reference"],
            continuous_scalers={k: Scaler(**v)
                                for k, v in d["continuous_scalers"].items()},
            continuous_order=list(d["continuous_order"]),
            categorical_order=list(d["categorical_order"]),
            fit_name=d.get("fit_name", ""),
            n_fit_rows=int(d.get("n_fit_rows", 0)),
            notes=list(d.get("notes", [])),
        )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _native(v):
    """numpy scalar -> python scalar, so levels survive a JSON round-trip."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    return v


def _sorted_levels(values: pd.Series) -> list:
    vals = [_native(v) for v in pd.unique(values.dropna())]
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                  for v in vals)
    # Deterministic order regardless of row order - column order stability
    # across folds and imputations depends on this.
    return sorted(vals) if numeric else sorted(vals, key=str)


def derive_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI_CALC / BMI_DON_CALC from height and weight when absent.

    Deterministic arithmetic on existing columns, so it introduces no fitted
    quantity and no leakage. The imputed files carry height and weight but
    not BMI.
    """
    df = df.copy()
    for bmi, hgt, wgt in (("BMI_CALC", "HGT_CM_CALC", "WGT_KG_CALC"),
                          ("BMI_DON_CALC", "HGT_CM_DON_CALC", "WGT_KG_DON_CALC")):
        if bmi not in df.columns and hgt in df.columns and wgt in df.columns:
            h = pd.to_numeric(df[hgt], errors="coerce") / 100.0
            df[bmi] = pd.to_numeric(df[wgt], errors="coerce") / np.where(
                h > 0, h ** 2, np.nan)
    return df


def check_schema(df: pd.DataFrame, masked: bool = False) -> dict:
    """Report which expected columns are present, missing or unexpected.

    Called before fitting so a schema mismatch surfaces as a readable report
    rather than a KeyError twenty files into a build.
    """
    cont, cat = expected_columns(masked)
    present = set(df.columns)
    return {
        "missing_continuous": [c for c in cont if c not in present],
        "missing_categorical": [c for c in cat if c not in present],
        "missing_passthrough": [c for c in PASSTHROUGH if c not in present],
        "present_but_dropped": [c for c in DROPPED if c in present],
        "unexpected": sorted(present - set(cont) - set(cat)
                             - set(PASSTHROUGH) - set(DROPPED)),
    }


def expected_columns(masked: bool = False) -> tuple[list[str], list[str]]:
    cont = [c for c in CONTINUOUS if not (masked and c in PRETRAIN_MASKED)]
    cat = [c for c in CATEGORICAL if not (masked and c in PRETRAIN_MASKED)]
    return cont, cat


def fit_feature_spec(df_train: pd.DataFrame, fit_name: str = "",
                     masked: bool = False,
                     diag_reference: Any = None,
                     reference_overrides: dict | None = None,
                     strict: bool = False) -> FeatureSpec:
    """Fit level maps, reference levels and scalers on TRAINING rows only.

    masked:          apply the pretraining mask (drops the five pre-2014
                     features; GLOMERUL is retained).
    diag_reference:  code for the hypertensive/vascular DIAG group; falls back
                     to the modal level with a warning when unknown.
    reference_overrides:
                     pin reference levels instead of taking the modal level.

                     Why this exists: the modal level is a SAMPLE quantity, so
                     with two levels of similar frequency it can flip between
                     imputations of the same fit set. Predictions are invariant
                     to which level is dropped, but pooled BASELINE
                     COEFFICIENTS across the five imputations are not - pooling
                     is meaningless if imputation 1 drops one level and
                     imputation 3 drops another. The build notebook therefore
                     fits imputation 1, then passes its references to
                     imputations 2-5 of the SAME fit set. Those are the same
                     patients, so this is not leakage; it is one fitted choice
                     applied consistently.
    strict:          raise instead of warn when expected columns are missing.
    """
    df = derive_bmi(df_train)
    cont_cols, cat_cols = expected_columns(masked)
    notes: list[str] = []

    report = check_schema(df, masked=masked)
    missing = report["missing_continuous"] + report["missing_categorical"]
    if missing:
        msg = f"expected columns absent from the data and skipped: {missing}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg, RuntimeWarning)
        notes.append(msg)
    cont_cols = [c for c in cont_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    scalers, levels, refs = {}, {}, {}
    for c in cont_cols:
        kind = "robust" if c in ROBUST_COLUMNS else "standard"
        scalers[c] = _fit_scaler(pd.to_numeric(df[c], errors="coerce").to_numpy(),
                                 kind)

    diag_ref = diag_reference if diag_reference is not None else DIAG_REFERENCE_CODE
    overrides = dict(reference_overrides or {})
    for c in cat_cols:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        lv = _sorted_levels(s)
        levels[c] = lv
        if c in overrides:
            ref = _native(overrides[c])
            if ref not in lv:
                warnings.warn(f"{c}: pinned reference {ref!r} not observed in "
                              f"these training rows; using the modal level",
                              RuntimeWarning)
                notes.append(f"{c}: pinned reference {ref!r} absent, used modal")
                ref = _native(s.mode(dropna=True).iloc[0])
        elif c in FIXED_REFERENCES:
            ref = FIXED_REFERENCES[c]
            if ref not in lv:
                warnings.warn(f"{c}: fixed reference {ref!r} not observed in "
                              f"training data; using the modal level instead",
                              RuntimeWarning)
                notes.append(f"{c}: fixed reference {ref!r} absent, used modal")
                ref = _native(s.mode(dropna=True).iloc[0])
        elif c == "DIAG":
            if diag_ref is not None and diag_ref in lv:
                ref = _native(diag_ref)
            else:
                ref = _native(s.mode(dropna=True).iloc[0])
                note = ("DIAG: hypertensive/vascular reference code not "
                        "supplied (or not observed); fell back to the modal "
                        f"level {ref!r}. Set spec.DIAG_REFERENCE_CODE or pass "
                        "diag_reference= to apply the intended contrast.")
                warnings.warn(note, RuntimeWarning)
                notes.append(note)
        else:
            ref = _native(s.mode(dropna=True).iloc[0])
        refs[c] = ref

    if overrides:
        notes.append(f"reference levels pinned for {sorted(overrides)}")
    if masked:
        notes.append(f"pretraining spec: masked {sorted(set(PRETRAIN_MASKED))}")

    return FeatureSpec(
        categorical_levels=levels,
        categorical_reference=refs,
        continuous_scalers=scalers,
        continuous_order=cont_cols,
        categorical_order=cat_cols,
        fit_name=fit_name,
        n_fit_rows=int(len(df_train)),
        notes=notes,
    )
