"""
Shared inverse-probability-of-censoring-weight machinery.

All metrics in this package are IPCW-weighted using a Kaplan-Meier estimate
of the CENSORING distribution G(t) = P(C > t) - i.e. the KM fit with
censoring as the event and every cause pooled as "censored" for that fit.

float64 throughout; pure numpy, no model objects.
"""

import warnings

import numpy as np

MIN_G = 1e-10          # floor on G to keep weights finite


class CensoringDistribution:
    """Kaplan-Meier estimate of G(t) = P(C > t).

    Two evaluations matter and are easy to conflate:

      * `G(t)`  - right-continuous, used for subjects still at risk at the
                  horizon t;
      * `G_minus(t)` - left-continuous, G(t-), used for subjects whose own
                  event time is t. Using G(t) there would divide by a
                  quantity that already reflects censoring AT that instant.
    """

    def __init__(self, time, event):
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event)
        is_cens = (event == 0).astype(np.float64)

        order = np.argsort(time, kind="mergesort")
        t_sorted, c_sorted = time[order], is_cens[order]
        uniq, first = np.unique(t_sorted, return_index=True)
        n = len(t_sorted)
        at_risk = n - first                                   # n_i
        d_cens = np.add.reduceat(c_sorted, first)             # censorings at t_i

        factors = 1.0 - d_cens / at_risk
        self.times = uniq
        self.surv = np.cumprod(factors)                       # G(t_i)
        # G just before each unique time = value carried in from the left.
        self.surv_minus = np.concatenate([[1.0], self.surv[:-1]])
    def warn_if_unstable(self, t_eval, floor=0.05):
        """Warn only about horizons where G(t) is actually small.

        G approaching 0 at the very end of follow-up is normal (with uniform
        administrative censoring it reaches 0 by design) and says nothing
        about a horizon at 1 or 5 years, so warning on the global minimum
        would cry wolf on every dataset. What matters is G at the horizons
        being evaluated, where a small value means a few long-followed
        subjects carry enormous IPCW weight.
        """
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        g = self.G(t_eval)
        bad = t_eval[g < floor]
        if bad.size:
            warnings.warn(
                f"censoring survival G(t) < {floor} at horizons "
                f"{np.round(bad, 3).tolist()} (min {g.min():.3g}); IPCW "
                f"weights there are dominated by few subjects and the "
                f"estimates are unstable", RuntimeWarning)
        return g

    def G(self, t):
        """G(t), right-continuous."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        idx = np.searchsorted(self.times, t, side="right") - 1
        out = np.where(idx >= 0, self.surv[np.clip(idx, 0, None)], 1.0)
        return np.maximum(out, MIN_G)

    def G_minus(self, t):
        """G(t-), left-continuous."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        idx = np.searchsorted(self.times, t, side="left")
        out = np.where(idx < len(self.times),
                       self.surv_minus[np.clip(idx, 0, len(self.times) - 1)],
                       self.surv[-1])
        # Times beyond the last observed point keep the final value.
        beyond = t > self.times[-1]
        out = np.where(beyond, self.surv[-1], out)
        return np.maximum(out, MIN_G)


def make_censoring(time, event, fit_on=None):
    """Build the censoring distribution.

    `fit_on` may be a (time, event) tuple to estimate G on a different set
    (e.g. the training data). Default is the evaluation set, per the spec.
    """
    if fit_on is None:
        return CensoringDistribution(time, event)
    return CensoringDistribution(fit_on[0], fit_on[1])


def brier_weights(time, event, t, cens: CensoringDistribution):
    """Standard three-group IPCW weights for a Brier score at horizon t.

      * an event of any cause at T_i <= t  -> 1 / G(T_i-)
      * still event-free at t              -> 1 / G(t)
      * censored before t                  -> 0  (contributes nothing)
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    had_event = (time <= t) & (event > 0)
    still_at_risk = time > t
    w = np.zeros_like(time)
    w[had_event] = 1.0 / cens.G_minus(time[had_event])
    w[still_at_risk] = 1.0 / cens.G(np.full(still_at_risk.sum(), t))
    return w
