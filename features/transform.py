"""
Applying a fitted `FeatureSpec` to produce the two model representations.

    "embedding" -> standardised continuous block + integer codes per
                   categorical, plus the cardinality tuple `CovariateEncoder`
                   expects.
    "onehot"    -> standardised continuous block + one-hot with the reference
                   level dropped, a design matrix for the classical baselines.

Both come from the SAME fitted spec. Nothing here learns anything from the
data it is applied to: unseen levels fall back to the reference level and are
counted, never added as new columns, because a level appearing only in
temporal validation must not change the design matrix the model was trained
on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .spec import FeatureSpec, PASSTHROUGH, derive_bmi


@dataclass
class TransformResult:
    """Arrays plus everything needed to audit the transform afterwards."""
    representation: str
    continuous: np.ndarray                      # (n, n_continuous) float64
    categorical_codes: np.ndarray | None        # (n, n_categorical) int64
    design: np.ndarray | None                   # (n, n_onehot) float64
    columns: list                               # names, in array order
    cardinalities: list                         # levels per categorical
    unseen_counts: dict                         # column -> count mapped to ref
    unseen_values: dict                         # column -> the unseen values
    passthrough: pd.DataFrame                   # outcome / admin columns
    n_rows: int = 0
    warnings: list = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        """The transform as a DataFrame, for inspection or CSV export."""
        if self.representation == "onehot":
            return pd.DataFrame(self.design, columns=self.columns)
        block = np.hstack([self.continuous,
                           self.categorical_codes.astype(np.float64)])
        return pd.DataFrame(block, columns=self.columns)


def _codes_for(series: pd.Series, levels: list, reference):
    """Integer code per row, with unseen values folded onto the reference."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    lookup = {lv: i for i, lv in enumerate(levels)}

    def norm(v):
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and float(v).is_integer():
            return int(v)
        return v

    vals = [norm(v) for v in series.to_numpy()]
    ref_code = lookup[reference]
    codes = np.empty(len(vals), dtype=np.int64)
    unseen = {}
    for i, v in enumerate(vals):
        c = lookup.get(v)
        if c is None:
            codes[i] = ref_code
            unseen[v] = unseen.get(v, 0) + 1
        else:
            codes[i] = c
    return codes, unseen


def transform(df: pd.DataFrame, spec: FeatureSpec,
              representation: str = "embedding") -> TransformResult:
    """Apply `spec` to `df`. `representation` in {"embedding", "onehot"}."""
    if representation not in ("embedding", "onehot"):
        raise ValueError(f"unknown representation {representation!r}")
    df = derive_bmi(df)

    missing = [c for c in spec.feature_order if c not in df.columns]
    if missing:
        raise KeyError(f"columns required by the spec are absent: {missing}")

    n = len(df)
    cont = np.empty((n, len(spec.continuous_order)), dtype=np.float64)
    for j, c in enumerate(spec.continuous_order):
        cont[:, j] = spec.continuous_scalers[c].apply(
            pd.to_numeric(df[c], errors="coerce").to_numpy())
    warns = []
    if not np.isfinite(cont).all():
        bad = [spec.continuous_order[j]
               for j in range(cont.shape[1]) if not np.isfinite(cont[:, j]).all()]
        warns.append(f"non-finite values after scaling in {bad} - the imputed "
                     f"inputs should contain none")

    codes = np.empty((n, len(spec.categorical_order)), dtype=np.int64)
    unseen_counts, unseen_values = {}, {}
    for j, c in enumerate(spec.categorical_order):
        col_codes, unseen = _codes_for(df[c], spec.categorical_levels[c],
                                       spec.categorical_reference[c])
        codes[:, j] = col_codes
        total = int(sum(unseen.values()))
        if total:
            unseen_counts[c] = total
            unseen_values[c] = sorted(unseen.items(), key=lambda kv: -kv[1])

    design, columns = None, None
    if representation == "onehot":
        blocks = [cont]
        columns = list(spec.continuous_order)
        for j, c in enumerate(spec.categorical_order):
            levels = spec.categorical_levels[c]
            ref = spec.categorical_reference[c]
            keep = [i for i, lv in enumerate(levels) if lv != ref]
            oh = np.zeros((n, len(keep)), dtype=np.float64)
            for col_i, lvl_i in enumerate(keep):
                oh[:, col_i] = (codes[:, j] == lvl_i).astype(np.float64)
            blocks.append(oh)
            columns += [f"{c}={levels[i]}" for i in keep]
        design = np.hstack(blocks)
    else:
        columns = list(spec.feature_order)

    present = [c for c in PASSTHROUGH if c in df.columns]
    return TransformResult(
        representation=representation,
        continuous=cont,
        categorical_codes=codes,
        design=design,
        columns=columns,
        cardinalities=spec.cardinalities,
        unseen_counts=unseen_counts,
        unseen_values=unseen_values,
        passthrough=df[present].reset_index(drop=True),
        n_rows=n,
        warnings=warns,
    )


def recover_categoricals(result: TransformResult, spec: FeatureSpec) -> pd.DataFrame:
    """Read the categorical values back out of whichever representation.

    Used by the information-equivalence test: if the embedding and one-hot
    forms recover the same values, the neural model and the baselines are
    seeing the same information, which is what makes the comparison fair.
    """
    out = {}
    for j, c in enumerate(spec.categorical_order):
        levels = spec.categorical_levels[c]
        ref = spec.categorical_reference[c]
        if result.representation == "embedding":
            out[c] = [levels[i] for i in result.categorical_codes[:, j]]
        else:
            keep = [i for i, lv in enumerate(levels) if lv != ref]
            names = [f"{c}={levels[i]}" for i in keep]
            idx = [result.columns.index(nm) for nm in names]
            block = result.design[:, idx] if idx else np.zeros((result.n_rows, 0))
            vals = []
            for row in block:
                hit = np.flatnonzero(row > 0.5)
                vals.append(levels[keep[hit[0]]] if hit.size else ref)
            out[c] = vals
    return pd.DataFrame(out)


def encoder_config_kwargs(spec: FeatureSpec) -> dict:
    """The pieces `rp_core.encoder.EncoderConfig` needs from a fitted spec."""
    return {"n_continuous": len(spec.continuous_order),
            "categorical_cardinalities": tuple(spec.cardinalities)}


def summarise(spec: FeatureSpec, results: dict | None = None) -> str:
    """Human-readable summary: feature counts, cardinalities, scaler kinds and
    any unseen-level counts."""
    lines = [f"FeatureSpec '{spec.fit_name}' fitted on {spec.n_fit_rows:,} rows",
             f"  embedding : {len(spec.continuous_order)} continuous "
             f"+ {len(spec.categorical_order)} categorical codes "
             f"= {spec.n_embedding_features} columns",
             f"  one-hot   : {len(spec.continuous_order)} continuous "
             f"+ {spec.n_onehot_features - len(spec.continuous_order)} indicator "
             f"(sum of levels-1) = {spec.n_onehot_features} columns",
             "  continuous scalers:"]
    for c in spec.continuous_order:
        s = spec.continuous_scalers[c]
        lines.append(f"    {c:22s} {s.kind:8s} centre={s.center:12.4f} "
                     f"scale={s.scale:12.4f}")
    lines.append("  categorical levels (reference marked *):")
    for c in spec.categorical_order:
        lv = spec.categorical_levels[c]
        ref = spec.categorical_reference[c]
        shown = ", ".join(f"{v}*" if v == ref else str(v) for v in lv)
        lines.append(f"    {c:22s} k={len(lv):2d}  [{shown}]")
    if spec.notes:
        lines.append("  notes:")
        lines += [f"    - {n}" for n in spec.notes]
    if results:
        lines.append("  unseen levels mapped to reference:")
        any_unseen = False
        for name, res in results.items():
            if res.unseen_counts:
                any_unseen = True
                for col, cnt in res.unseen_counts.items():
                    lines.append(f"    {name}: {col} -> {cnt} rows "
                                 f"{res.unseen_values[col][:5]}")
        if not any_unseen:
            lines.append("    (none)")
    return "\n".join(lines)
