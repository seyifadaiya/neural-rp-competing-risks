"""
Competing-risks evaluation metrics.

MODEL-AGNOSTIC BY CONSTRUCTION. Every function takes arrays -

    cif    (n, T, K)   predicted CIF per patient, horizon, cause
    time   (n,)        observed follow-up in years
    event  (n,)        0 censored, 1..K cause
    t_eval (T,)        horizons

- and never a model object, so the neural variants and every classical
baseline are scored by identical code. That is what makes the comparison
fair rather than merely careful.

Priority: calibration, not discrimination. The JASN ten-country study found
competing and non-competing models achieve near-identical discrimination for
graft failure, so calibration and CIF accuracy - especially within high-risk
strata - carry the argument.
"""

import numpy as np

from .calibration import (aalen_johansen, brier_score, calibration_curve,
                          calibration_in_the_large, fit_fine_gray,
                          grouped_calibration, integrated_brier_score,
                          rcs_design_np)
from .checks import coherence, monotonicity_and_bounds
from .discrimination import time_dependent_auc, time_dependent_cindex
from .ipcw import CensoringDistribution, brier_weights, make_censoring
from .subgroups import evaluate_by_subgroup, quantile_strata

__all__ = [
    "evaluate", "bootstrap_ci",
    "time_dependent_auc", "time_dependent_cindex",
    "brier_score", "integrated_brier_score", "calibration_curve",
    "grouped_calibration", "calibration_in_the_large", "aalen_johansen",
    "fit_fine_gray", "rcs_design_np",
    "coherence", "monotonicity_and_bounds",
    "CensoringDistribution", "make_censoring", "brier_weights",
    "evaluate_by_subgroup", "quantile_strata",
]


def bootstrap_ci(fn, n, n_boot=500, seed=42, alpha=0.05, subsample=None):
    """Percentile bootstrap over PATIENTS.

    `fn(idx) -> dict of arrays`. `subsample` caps the resample size, which
    matters only for the C-index: it is a 2-D dominance count and therefore
    the one metric here that is not a vectorised cumulative sum. Leave it None
    for statistically standard behaviour.
    """
    rng = np.random.default_rng(seed)
    size = subsample or n
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=size)
        draws.append(fn(idx))
    keys = draws[0].keys()
    out = {}
    for key in keys:
        stack = np.stack([np.asarray(d[key], dtype=np.float64) for d in draws])
        out[key] = {
            "lo": np.quantile(stack, alpha / 2, axis=0),
            "hi": np.quantile(stack, 1 - alpha / 2, axis=0),
            "se": stack.std(axis=0, ddof=1),
        }
    return out


def evaluate(cif, time, event, t_eval, surv=None, causes=(1, 2),
             n_boot=500, seed=42, cens_fit_on=None, bootstrap=True,
             boot_subsample=None, do_cindex=True, calibration=True,
             n_groups=10):
    """Every metric in one structured record.

    One call per model per imputation gives the benchmark harness a
    comparable row. Returns a nested dict:

        discrimination : auc, cindex (+ bootstrap CIs)
        calibration    : brier, ICI/E50/E90/Emax, curve, deciles, CITL
        checks         : coherence, monotonicity/bounds
        meta           : n, event counts, horizons, settings
    """
    cif = np.asarray(cif, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
    causes = list(causes)
    n = len(time)
    cens = make_censoring(time, event, fit_on=cens_fit_on)
    g_at_horizons = cens.warn_if_unstable(t_eval)

    result = {"meta": {
        "n": int(n),
        "t_eval": t_eval.tolist(),
        "causes": causes,
        "n_events": {int(k): int((event == k).sum()) for k in causes},
        "n_censored": int((event == 0).sum()),
        "censoring_fitted_on": "training" if cens_fit_on else "evaluation",
        "n_boot": int(n_boot) if bootstrap else 0,
        "seed": int(seed),
        "G_at_horizons": g_at_horizons.tolist(),
    }}

    auc = time_dependent_auc(cif, time, event, t_eval, cens=cens, causes=causes)
    disc = {"auc": {int(k): v.tolist() for k, v in auc.items()}}
    if do_cindex:
        cidx = time_dependent_cindex(cif, time, event, t_eval, cens=cens,
                                     causes=causes)
        disc["cindex"] = {int(k): v.tolist() for k, v in cidx.items()}
    result["discrimination"] = disc

    bs = brier_score(cif, time, event, t_eval, cens=cens, causes=causes)
    cal = {"brier": {int(k): v.tolist() for k, v in bs.items()}}
    if calibration:
        cal["austin"], cal["deciles"], cal["citl"] = {}, {}, {}
        for k in causes:
            per_t, dec_t, citl_t = [], [], []
            for j, t in enumerate(t_eval):
                curve = calibration_curve(cif[:, j, k - 1], time, event, t,
                                          cause=k, cens=cens)
                per_t.append({m: curve[m] for m in ("ICI", "E50", "E90",
                                                    "Emax", "method")}
                             | {"curve_predicted": curve["curve_predicted"].tolist(),
                                "curve_observed": curve["curve_observed"].tolist()})
                dec_t.append(grouped_calibration(cif[:, j, k - 1], time, event,
                                                 t, cause=k, n_groups=n_groups))
                citl_t.append(calibration_in_the_large(cif[:, j, k - 1], time,
                                                       event, t, cause=k))
            cal["austin"][int(k)] = per_t
            cal["deciles"][int(k)] = dec_t
            cal["citl"][int(k)] = citl_t
    result["calibration"] = cal

    checks = {"monotonicity_bounds": monotonicity_and_bounds(cif, surv)}
    if surv is not None:
        checks["coherence"] = coherence(cif, surv)
    result["checks"] = checks

    if bootstrap and n_boot > 0:
        def _one(idx):
            c, tt, ee = cif[idx], time[idx], event[idx]
            cn = make_censoring(tt, ee, fit_on=cens_fit_on)
            rec = {}
            a = time_dependent_auc(c, tt, ee, t_eval, cens=cn, causes=causes)
            b = brier_score(c, tt, ee, t_eval, cens=cn, causes=causes)
            for k in causes:
                rec[f"auc_{k}"] = a[k]
                rec[f"brier_{k}"] = b[k]
            if do_cindex:
                ci = time_dependent_cindex(c, tt, ee, t_eval, cens=cn,
                                           causes=causes)
                for k in causes:
                    rec[f"cindex_{k}"] = ci[k]
            return rec

        boot = bootstrap_ci(_one, n, n_boot=n_boot, seed=seed,
                            subsample=boot_subsample)
        result["bootstrap"] = {
            key: {s: v[s].tolist() for s in ("lo", "hi", "se")}
            for key, v in boot.items()
        }
    return result
