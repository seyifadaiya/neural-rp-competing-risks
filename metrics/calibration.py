"""
Brier score and calibration - the priority of this package.

The project's primary endpoint is CALIBRATED ABSOLUTE RISK, not
discrimination: the JASN ten-country study found competing and
non-competing models achieve near-identical discrimination for graft
failure, so a discrimination-led result would be weak. Everything here is
built and tested to that standard.

Observed risk is estimated by AALEN-JOHANSEN wherever competing events
exist - never Kaplan-Meier, which overestimates absolute risk in their
presence and is the exact error this project characterises.

Calibration curves follow Austin, Putter, Lee & van Buuren (2022),
"Graphical calibration curves and the integrated calibration index (ICI) for
competing risk models", Diagnostic and Prognostic Research 6:2 - a Fine-Gray
model with a restricted cubic spline of the predicted CIF as the only
covariate.
"""

import warnings

import numpy as np

from .ipcw import brier_weights, make_censoring

# --------------------------------------------------------------------------
# Restricted cubic spline (numpy; verified against rp_core.rcs_basis)
# --------------------------------------------------------------------------

def rcs_design_np(v, knots):
    """[v, B_1(v), ..., B_{m-2}(v)] - the same truncated-power RCS basis as
    `rp_core/rcs_basis.py`, kept in pure numpy so the metrics path has no
    torch dependency. `test_metrics.py` checks the two agree exactly."""
    v = np.asarray(v, dtype=np.float64).reshape(-1, 1)
    knots = np.asarray(knots, dtype=np.float64)
    k1, km, inner = knots[0], knots[-1], knots[1:-1]
    lam = (km - inner) / (km - k1)
    cube = lambda z: np.clip(z, 0, None) ** 3
    B = cube(v - inner) - lam * cube(v - k1) - (1 - lam) * cube(v - km)
    return np.hstack([v, B])


def default_knots(v, n_knots=4):
    """Harrell-style quantile knot placement, collapsing to fewer knots when
    the predictor has little spread."""
    v = np.asarray(v, dtype=np.float64)
    qs = {3: [.10, .50, .90], 4: [.05, .35, .65, .95],
          5: [.05, .275, .50, .725, .95]}[n_knots]
    knots = np.quantile(v, qs)
    knots = np.unique(knots)
    while len(knots) < 3 and n_knots > 3:
        n_knots -= 1
        knots = np.unique(np.quantile(v, qs[:n_knots]))
    return knots


# --------------------------------------------------------------------------
# Aalen-Johansen
# --------------------------------------------------------------------------

def aalen_johansen(time, event, t_eval, cause=1, variance=True):
    """Nonparametric cumulative incidence with the Aalen (1978) variance,
    the same estimator and variance R's cmprsk::cuminc reports.

        F_k(t) = sum_{t_i <= t} S(t_{i-1}) d_ki / n_i
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))

    order = np.argsort(time, kind="mergesort")
    time, event = time[order], event[order]
    uniq, first = np.unique(time, return_index=True)
    n = len(time)
    at_risk = (n - first).astype(np.float64)
    d_all = np.add.reduceat((event > 0).astype(np.float64), first)
    d_k = np.add.reduceat((event == cause).astype(np.float64), first)

    surv_prev = np.concatenate([[1.0], np.cumprod(1.0 - d_all / at_risk)[:-1]])
    inc = surv_prev * d_k / at_risk
    cif = np.cumsum(inc)

    idx = np.searchsorted(uniq, t_eval, side="right") - 1
    est = np.where(idx >= 0, cif[np.clip(idx, 0, None)], 0.0)
    if not variance:
        return est, None

    # Aalen (1978) variance. Written with CUMULATIVE SUMS rather than the
    # textbook double loop: expanding [F(t) - F(t_i)]^2 makes every term a
    # prefix sum, turning O(M^2) into O(M). At n = 50,000 there are ~35,000
    # distinct event times, and the naive form cost ~57 s for a single
    # cause-horizon - it dominated the whole evaluation budget.
    a = d_all / (at_risk * np.maximum(at_risk - d_all, 1e-12))
    b = surv_prev ** 2 * ((at_risk - d_k) / at_risk) * d_k / at_risk ** 2
    c = surv_prev * d_k / at_risk ** 2
    A, B, C = np.cumsum(a), np.cumsum(b), np.cumsum(c)
    AF, AFF, CF = np.cumsum(a * cif), np.cumsum(a * cif ** 2), np.cumsum(c * cif)
    var_at = (cif ** 2 * A - 2 * cif * AF + AFF + B - 2 * cif * C + 2 * CF)
    var = np.where(idx >= 0, var_at[np.clip(idx, 0, None)], 0.0)
    return est, np.maximum(var, 0.0)


# --------------------------------------------------------------------------
# Fine-Gray (subdistribution hazard) - the engine behind the Austin curve
# --------------------------------------------------------------------------

class FineGrayFit:
    """Fine-Gray proportional subdistribution hazards model.

    Fitted by Newton-Raphson on the IPCW-weighted partial likelihood. The
    risk-set sums are computed with CUMULATIVE SUMS rather than an explicit
    (events x subjects) matrix, which is what makes this usable at 50k
    patients: the weight of a subject who had a competing event at T_j is
    G(t_m)/G(T_j), so

        S0(t_m) = sum_{T_j >= t_m} e^eta_j  +  G(t_m) * sum_{T_j < t_m, comp} e^eta_j / G(T_j)

    and both terms are running sums over sorted time.
    """

    def __init__(self, converged, beta, cum_base, base_times, n_iter,
                 message="", centre=None, scale=None):
        self.converged = converged
        self.beta = beta
        self.cum_base = cum_base
        self.base_times = base_times
        self.n_iter = n_iter
        self.message = message
        self.centre = centre
        self.scale = scale

    @property
    def beta_original(self):
        """Coefficients on the ORIGINAL covariate scale (fitting standardises)."""
        return self.beta / self.scale if self.scale is not None else self.beta

    def baseline(self, t):
        idx = np.searchsorted(self.base_times, np.atleast_1d(t), side="right") - 1
        return np.where(idx >= 0, self.cum_base[np.clip(idx, 0, None)], 0.0)

    def predict(self, X, t):
        """F(t | X) = 1 - exp(-H0(t) exp(X beta)), undoing the standardisation."""
        X = np.asarray(X, dtype=np.float64)
        if self.centre is not None:
            X = (X - self.centre) / self.scale
        eta = np.clip(X @ self.beta, -500, 500)
        return 1.0 - np.exp(-self.baseline(t) * np.exp(eta))


def fit_fine_gray(X, time, event, cause=1, cens=None, max_iter=100, tol=1e-9,
                  ridge=1e-8):
    """Newton-Raphson with backtracking on the weighted partial likelihood.

    Two numerical points matter, both learned the hard way elsewhere in this
    project. The covariate columns are CENTRED AND SCALED before fitting: a
    truncated-power spline basis evaluated on predicted risks in [0, 1] has
    cubic columns orders of magnitude smaller than the linear one, and the
    near-collinearity makes the Hessian solve unreliable even though Newton is
    affine-invariant in exact arithmetic. And each step is accepted only if it
    increases the log partial likelihood, with halving otherwise - an
    undamped Newton step on that geometry overshoots and the fit was silently
    falling back to the cruder smoother.
    """
    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    n, p = X.shape
    if cens is None:
        cens = make_censoring(time, event)

    centre = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale <= 0] = 1.0
    Xz = (X - centre) / scale

    order = np.argsort(time, kind="mergesort")
    T, E, Xs = time[order], event[order], Xz[order]
    g_at = cens.G(T)

    ev_times = np.unique(T[E == cause])
    if ev_times.size < p + 2:
        return FineGrayFit(False, np.zeros(p), np.zeros(0), np.zeros(0), 0,
                           "too few events of the cause of interest",
                           centre, scale)

    at_risk_start = np.searchsorted(T, ev_times, side="left")
    comp_idx = np.flatnonzero((E > 0) & (E != cause))
    comp_T, comp_g = T[comp_idx], g_at[comp_idx]
    comp_before = np.searchsorted(comp_T, ev_times, side="left")
    g_evt = cens.G(ev_times)

    # Per-event-time case sums, grouped by searchsorted rather than by
    # scanning the whole array once per event time: the naive form is a
    # Python loop of O(n) scans, i.e. ~6e8 comparisons at n = 50,000, and it
    # dominated the entire evaluation budget.
    case_mask = E == cause
    case_grp = np.searchsorted(ev_times, T[case_mask])
    d_events = np.bincount(case_grp, minlength=ev_times.size).astype(np.float64)
    x_sum_events = np.zeros((ev_times.size, p))
    np.add.at(x_sum_events, case_grp, Xs[case_mask])

    def _rev(arr):
        c = np.concatenate([np.cumsum(arr[::-1], axis=0)[::-1],
                            np.zeros((1,) + arr.shape[1:])])
        return c[at_risk_start]

    def _fwd(arr):
        c = np.concatenate([np.zeros((1,) + arr.shape[1:]),
                            np.cumsum(arr, axis=0)])
        return c[comp_before]

    Xc = Xs[comp_idx]

    def risk_sums(beta, need_hess=True):
        w = np.exp(np.clip(Xs @ beta, -500, 500))
        wc = w[comp_idx] / comp_g
        s0 = _rev(w[:, None])[:, 0] + g_evt * _fwd(wc[:, None])[:, 0]
        s1 = _rev(w[:, None] * Xs) + g_evt[:, None] * _fwd(wc[:, None] * Xc)
        s2 = None
        if need_hess:
            wxx = (w[:, None, None] * Xs[:, :, None] * Xs[:, None, :]).reshape(n, p * p)
            wxxc = (wc[:, None, None] * Xc[:, :, None] * Xc[:, None, :]).reshape(-1, p * p)
            s2 = (_rev(wxx) + g_evt[:, None] * _fwd(wxxc)).reshape(-1, p, p)
        return np.maximum(s0, 1e-300), s1, s2

    def log_pl(beta):
        s0, _, _ = risk_sums(beta, need_hess=False)
        lin = float((x_sum_events @ beta).sum())
        return lin - float((d_events * np.log(s0)).sum())

    beta = np.zeros(p)
    cur = log_pl(beta)
    converged, msg = False, ""
    for it in range(max_iter):
        s0, s1, s2 = risk_sums(beta)
        mu = s1 / s0[:, None]
        grad = x_sum_events.sum(axis=0) - (d_events[:, None] * mu).sum(axis=0)
        info = ((d_events[:, None, None]
                 * (s2 / s0[:, None, None] - mu[:, :, None] * mu[:, None, :]))
                .sum(axis=0))
        info = info + ridge * np.eye(p)
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info, grad, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            msg = "non-finite Newton step"
            break
        # Backtracking: accept only an increase in the log partial likelihood.
        factor, improved = 1.0, False
        for _ in range(40):
            cand = beta + factor * step
            val = log_pl(cand)
            if np.isfinite(val) and val >= cur:
                beta, prev, cur, improved = cand, cur, val, True
                break
            factor *= 0.5
        if not improved:
            converged = np.max(np.abs(grad)) < 1e-6
            msg = "" if converged else "line search failed to improve"
            break
        if np.max(np.abs(factor * step)) < tol or abs(cur - prev) < tol:
            converged = True
            break
    else:
        msg = "did not converge in max_iter"

    s0, _, _ = risk_sums(beta, need_hess=False)
    cum_base = np.cumsum(d_events / s0)
    return FineGrayFit(converged, beta, cum_base, ev_times, it + 1, msg,
                       centre, scale)


# --------------------------------------------------------------------------
# Brier score
# --------------------------------------------------------------------------

def brier_score(cif, time, event, t_eval, cens=None, causes=None):
    """IPCW Brier score per cause per horizon."""
    cif = np.asarray(cif, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
    causes = causes or list(range(1, cif.shape[2] + 1))
    if cens is None:
        cens = make_censoring(time, event)
    n = len(time)

    out = {}
    for k in causes:
        vals = np.empty(len(t_eval))
        for j, t in enumerate(t_eval):
            w = brier_weights(time, event, t, cens)
            y = ((time <= t) & (event == k)).astype(np.float64)
            vals[j] = float(np.sum(w * (y - cif[:, j, k - 1]) ** 2) / n)
        out[k] = vals
    return out


def integrated_brier_score(cif_fn, time, event, t_max, n_grid=100, cens=None,
                           causes=(1, 2)):
    """IBS over [0, t_max] by trapezoid on a dense grid.

    `cif_fn(t_grid) -> (n, len(t_grid), K)` so the caller supplies predictions
    on the integration grid (the metrics package never holds a model).
    """
    grid = np.linspace(t_max / n_grid, t_max, n_grid)
    cif_grid = np.asarray(cif_fn(grid), dtype=np.float64)
    bs = brier_score(cif_grid, time, event, grid, cens=cens, causes=list(causes))
    return {k: float(np.trapezoid(v, grid) / t_max) for k, v in bs.items()}


# --------------------------------------------------------------------------
# Austin calibration
# --------------------------------------------------------------------------

def calibration_curve(cif_k, time, event, t, cause=1, cens=None, n_knots=4,
                      n_curve=100):
    """Austin et al. (2022) smoothed calibration curve for a competing risk.

    Regresses the observed subdistribution risk on a restricted cubic spline
    of the predicted CIF via a Fine-Gray model, then reads the smoothed
    observed risk off that fit at each subject's own prediction.

    Returns ICI (mean |observed - predicted|), E50, E90, Emax, the curve for
    plotting, and which smoother was used.
    """
    p = np.asarray(cif_k, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    if cens is None:
        cens = make_censoring(time, event)

    knots = default_knots(p, n_knots)
    method, obs = "fine_gray_rcs", None
    if len(knots) >= 3 and np.unique(p).size > len(knots):
        X = rcs_design_np(p, knots)
        fit = fit_fine_gray(X, time, event, cause=cause, cens=cens)
        if fit.converged:
            obs = fit.predict(X, t)
        else:
            method = f"fallback_local_aj ({fit.message})"
    else:
        method = "fallback_local_aj (insufficient spread in predictions)"

    if obs is None:
        obs = _local_aj_smoother(p, time, event, t, cause)

    obs = np.clip(obs, 0.0, 1.0)
    diff = np.abs(obs - p)
    grid = np.linspace(np.quantile(p, 0.01), np.quantile(p, 0.99), n_curve)
    if method.startswith("fine_gray"):
        curve = np.clip(fit.predict(rcs_design_np(grid, knots), t), 0, 1)
    else:
        curve = np.interp(grid, np.sort(p), obs[np.argsort(p)])
    return {
        "ICI": float(np.mean(diff)),
        "E50": float(np.median(diff)),
        "E90": float(np.quantile(diff, 0.90)),
        "Emax": float(np.max(diff)),
        "curve_predicted": grid,
        "curve_observed": curve,
        "method": method,
    }


def _local_aj_smoother(p, time, event, t, cause, n_windows=25, frac=0.25):
    """Fallback smoother: Aalen-Johansen within overlapping windows of the
    predicted risk, interpolated back to each subject.

    Hazard-based and fully nonparametric, so it cannot fail to converge; it is
    less smooth than the Fine-Gray fit, which is why it is only the fallback.
    """
    order = np.argsort(p)
    n = len(p)
    half = max(int(frac * n / 2), 25)
    centres = np.linspace(half, n - half - 1, n_windows).astype(int)
    xs, ys = [], []
    for c in centres:
        sel = order[max(0, c - half):min(n, c + half)]
        est, _ = aalen_johansen(time[sel], event[sel], [t], cause=cause,
                                variance=False)
        xs.append(p[sel].mean())
        ys.append(float(est[0]))
    xs, ys = np.asarray(xs), np.asarray(ys)
    keep = np.argsort(xs)
    return np.interp(p, xs[keep], ys[keep])


def grouped_calibration(cif_k, time, event, t, cause=1, n_groups=10):
    """Deciles of predicted CIF; observed risk by AALEN-JOHANSEN with a CI.

    Kaplan-Meier would overestimate observed risk here - it treats competing
    events as censored, i.e. as if those patients could still fail - which is
    precisely the bias this project exists to characterise.
    """
    p = np.asarray(cif_k, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    edges = np.quantile(p, np.linspace(0, 1, n_groups + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    grp = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_groups - 1)

    rows = []
    for g in range(n_groups):
        sel = grp == g
        if sel.sum() == 0:
            continue
        est, var = aalen_johansen(time[sel], event[sel], [t], cause=cause)
        se = float(np.sqrt(var[0])) if var is not None else np.nan
        rows.append({
            "group": g + 1,
            "n": int(sel.sum()),
            "n_events_cause": int(((time[sel] <= t) & (event[sel] == cause)).sum()),
            # The group-level prediction is the MEAN OF INDIVIDUAL predictions,
            # never the prediction at the mean covariate (Jensen).
            "predicted_mean": float(p[sel].mean()),
            "observed_aj": float(est[0]),
            "observed_lo": float(max(est[0] - 1.96 * se, 0.0)),
            "observed_hi": float(min(est[0] + 1.96 * se, 1.0)),
        })
    return rows


def calibration_in_the_large(cif_k, time, event, t, cause=1):
    """Mean predicted CIF vs the overall Aalen-Johansen estimate."""
    p = np.asarray(cif_k, dtype=np.float64)
    est, var = aalen_johansen(time, event, [t], cause=cause)
    se = float(np.sqrt(var[0]))
    return {"predicted_mean": float(p.mean()),
            "observed_aj": float(est[0]),
            "observed_se": se,
            "difference": float(p.mean() - est[0])}
