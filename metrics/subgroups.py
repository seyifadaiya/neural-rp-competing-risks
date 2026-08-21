"""
Stratified evaluation. The project's claim is about calibration in
high-risk groups, so per-stratum metrics are a first-class output rather
than a follow-up analysis.
"""

import warnings

import numpy as np

MIN_EVENTS_WARN = 50


def evaluate_by_subgroup(evaluate_fn, cif, time, event, t_eval, strata,
                         stratum_names=None, causes=(1, 2), **opts):
    """Run `evaluate_fn` within each level of `strata`.

    Returns {level: result}, each carrying its own n and per-cause event
    counts. A stratum with fewer than ~50 events of a cause gets a warning
    recorded in its result: calibration estimates there are unstable, and a
    silently wide ICI is easy to over-read.
    """
    strata = np.asarray(strata)
    cif = np.asarray(cif, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    levels = np.unique(strata)

    results = {}
    for lev in levels:
        sel = strata == lev
        name = (stratum_names or {}).get(lev, str(lev)) if stratum_names else str(lev)
        counts = {int(k): int(((event[sel] > 0) & (event[sel] == k)).sum())
                  for k in causes}
        warns = [f"cause {k}: only {v} events (<{MIN_EVENTS_WARN}); "
                 f"calibration estimates are unstable"
                 for k, v in counts.items() if v < MIN_EVENTS_WARN]
        for w in warns:
            warnings.warn(f"stratum {name}: {w}", RuntimeWarning)
        res = evaluate_fn(cif[sel], time[sel], event[sel], t_eval,
                          causes=causes, **opts)
        res["n"] = int(sel.sum())
        res["n_events"] = counts
        res["warnings"] = warns
        results[name] = res
    return results


def quantile_strata(x, n_groups=4, labels=None):
    """Convenience: split a continuous variable into quantile strata."""
    x = np.asarray(x, dtype=np.float64)
    edges = np.quantile(x, np.linspace(0, 1, n_groups + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_groups - 1)
    if labels is None:
        labels = [f"Q{i + 1}" for i in range(n_groups)]
    return np.asarray(labels)[idx]
