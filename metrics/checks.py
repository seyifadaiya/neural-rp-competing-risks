"""
Coherence and sanity checks, carried through as REPORTED METRICS rather than
only as tests. F_1 + F_2 + S = 1 is a structural property of the
Royston-Parmar competing-risks model, so its measured deviation belongs in
the results table next to the discrimination and calibration numbers.
"""

import numpy as np


def coherence(cif, surv):
    """max |sum_k F_k(t) + S(t) - 1| over the evaluation set."""
    cif = np.asarray(cif, dtype=np.float64)
    surv = np.asarray(surv, dtype=np.float64)
    dev = np.abs(cif.sum(axis=-1) + surv - 1.0)
    return {"max_abs_deviation": float(dev.max()),
            "mean_abs_deviation": float(dev.mean())}


def monotonicity_and_bounds(cif, surv=None, tol=1e-9):
    """CIFs non-decreasing in t and inside [0, 1]; survival non-increasing."""
    cif = np.asarray(cif, dtype=np.float64)
    d_cif = np.diff(cif, axis=1)
    out = {
        "min_cif_increment": float(d_cif.min()) if d_cif.size else 0.0,
        "min_cif": float(cif.min()),
        "max_cif": float(cif.max()),
        "n_monotonicity_violations": int((d_cif < -tol).sum()),
        "n_bound_violations": int(((cif < -tol) | (cif > 1 + tol)).sum()),
    }
    if surv is not None:
        surv = np.asarray(surv, dtype=np.float64)
        d_s = np.diff(surv, axis=1)
        out["max_surv_increment"] = float(d_s.max()) if d_s.size else 0.0
        out["n_survival_violations"] = int((d_s > tol).sum())
    return out
