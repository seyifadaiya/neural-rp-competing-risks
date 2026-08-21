"""
Classical baselines, given every advantage they can legitimately use.

The project's claim is that neural representation learning improves
CALIBRATED ABSOLUTE RISK beyond the best classical models. That is only
defensible if the comparators are strong, so three rules hold throughout:

1. Continuous covariates enter through restricted cubic splines, never
   linearly (see `covariate_splines.py`).
2. The validated composites KDRI_RAO and EPTS_FINAL are supplied alongside
   their components - pre-packaged nonlinearity a linear model could not
   otherwise discover.
3. Classical models consume the `onehot` representation from the same fitted
   `FeatureSpec` the neural models' `embedding` comes from. Identical
   information; never re-encoded.

Coherence differs by construction across these models, and that contrast is
itself a result:

  * Royston-Parmar : F1 + F2 + S = 1 is STRUCTURAL (~1e-07, quadrature only).
  * Cox            : holds only if the CIF integration is done correctly.
  * Fine-Gray      : NOT guaranteed - the two subdistribution models are
                     fitted independently, so nothing ties them together.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch

import cif as cif_module
from metrics.calibration import fit_fine_gray
from rcs_basis import RCSBasis
from rp_classical import ClassicalRP, fit_classical_rp

from .covariate_splines import (CovariateSplineSpec, apply_covariate_splines,
                                fit_covariate_splines)

CAUSES = (1, 2)


def time_knots(time, event, cause, n_internal=3):
    """Time-spline knots at quantiles of uncensored log event times for one
    cause - the same placement the neural variants use, so the baseline and
    the neural model differ in the covariate map alone."""
    lt = np.log(np.asarray(time)[np.asarray(event) == cause])
    if lt.size < n_internal + 2:
        lt = np.log(np.asarray(time))
    return torch.tensor(np.quantile(lt, np.linspace(0, 1, n_internal + 2)),
                        dtype=torch.float64)


class _RPForCIF(torch.nn.Module):
    """Adapter exposing a fitted `ClassicalRP` in the shape `rp_core.cif`
    expects (`forward`, `basis`, `dtype_`), so the verified CIF quadrature is
    reused rather than reimplemented."""

    def __init__(self, model: ClassicalRP):
        super().__init__()
        self.model = model
        self.basis = model.basis
        self.dtype_ = torch.float64

    def forward(self, log_t, x_cont, x_cat=None):
        return self.model.log_H_and_deriv(
            torch.as_tensor(log_t, dtype=torch.float64),
            torch.as_tensor(x_cont, dtype=torch.float64))


# ---------------------------------------------------------------------------
# 1 & 5. Cause-specific Royston-Parmar
# ---------------------------------------------------------------------------

class RoystonParmarBaseline:
    """Cause-specific Royston-Parmar on the RCS-expanded design matrix.

    The primary comparator, and the one that needs no new estimator:
    `fit_classical_rp` is already verified against rstpm2 to 1.4e-07, and CIFs
    come from the verified `rp_core.cif` quadrature. Coherence is therefore
    structural here - any deviation is quadrature error alone.
    """

    def __init__(self, name="royston_parmar", columns=None, spline_columns=None,
                 df_time=4, seed=42, expand=True):
        self.name = name
        self.columns = list(columns) if columns is not None else None
        self.spline_columns = spline_columns
        self.df_time = df_time
        self.seed = seed
        self.expand = expand
        self.spline_spec: CovariateSplineSpec | None = None
        self.model = None
        self._adapter = None

    def fit(self, X, time, event, **kw):
        X = np.asarray(X, dtype=np.float64)
        cols = self.columns or [f"x{i}" for i in range(X.shape[1])]
        if self.expand:
            self.spline_spec = fit_covariate_splines(
                X, cols, only=self.spline_columns)
            Xd = apply_covariate_splines(X, self.spline_spec)
        else:
            Xd = X
        knots = {k: time_knots(time, event, k, self.df_time - 1) for k in CAUSES}
        self.model = ClassicalRP(
            knots, p=Xd.shape[1],
            basis={k: RCSBasis(knots[k]) for k in CAUSES})
        fit_classical_rp(self.model,
                         torch.tensor(np.asarray(time, dtype=np.float64)),
                         torch.tensor(np.asarray(event)),
                         torch.tensor(Xd))
        self._adapter = _RPForCIF(self.model)
        return self

    def _design(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (apply_covariate_splines(X, self.spline_spec)
                if self.spline_spec is not None else X)

    def predict_cif(self, X, t_eval, n_nodes=None, batch_size=2048):
        Xd = torch.tensor(self._design(X))
        kwargs = {} if n_nodes is None else {"n_nodes": n_nodes}
        cif, _ = cif_module.predict_cif(self._adapter, Xd,
                                        np.asarray(t_eval, dtype=np.float64),
                                        batch_size=batch_size, **kwargs)
        return np.asarray(cif, dtype=np.float64)

    def predict_survival(self, X, t_eval, batch_size=2048):
        """Overall (all-cause) survival exp(-sum_k H_k), from the model itself.

        Supplying this rather than letting the harness infer S = 1 - F1 - F2 is
        what makes the coherence column a real measurement: S here is computed
        from the fitted cumulative hazards, so F1 + F2 + S - 1 is the quadrature
        error and nothing else.
        """
        Xd = torch.tensor(self._design(X))
        surv = cif_module.predict_survival(
            self._adapter, Xd, np.asarray(t_eval, dtype=np.float64),
            batch_size=batch_size)
        return np.asarray(surv, dtype=np.float64)

    def spec_json(self) -> str:
        return self.spline_spec.to_json() if self.spline_spec else "{}"


class MinimalReferenceBaseline(RoystonParmarBaseline):
    """Cause-specific RP on KDRI_RAO and EPTS_FINAL only, RCS-expanded.

    The floor every richer model must clear. It anchors the question
    clinicians actually ask: what does everything beyond the two standard
    indices buy?
    """

    def __init__(self, name="rp_kdri_epts", columns=None,
                 keep=("KDRI_RAO", "EPTS_FINAL"), df_time=4, seed=42):
        super().__init__(name=name, columns=columns, df_time=df_time, seed=seed)
        self.keep = tuple(keep)

    def fit(self, X, time, event, **kw):
        X = np.asarray(X, dtype=np.float64)
        cols = self.columns or [f"x{i}" for i in range(X.shape[1])]
        idx = [i for i, c in enumerate(cols) if c in self.keep]
        if not idx:
            raise KeyError(f"none of {self.keep} present in the design matrix; "
                           f"pass columns= so the reference model can find them")
        self._keep_idx = idx
        sub_cols = [cols[i] for i in idx]
        self.columns_full = cols
        super_self = super()
        self.columns = sub_cols
        out = super_self.fit(X[:, idx], time, event, **kw)
        self.columns = cols
        return out

    def predict_cif(self, X, t_eval, **kw):
        X = np.asarray(X, dtype=np.float64)[:, self._keep_idx]
        return super().predict_cif(X, t_eval, **kw)

    def predict_survival(self, X, t_eval, **kw):
        X = np.asarray(X, dtype=np.float64)[:, self._keep_idx]
        return super().predict_survival(X, t_eval, **kw)


# ---------------------------------------------------------------------------
# 2. Cause-specific Cox
# ---------------------------------------------------------------------------

class CoxBaseline:
    """Two cause-specific Cox models (the competing event censored), Breslow
    baseline hazards, CIF by discrete integration over observed event times:

        F_k(t|x) = sum_{t_j <= t} S(t_{j-1}|x) * dH_k(t_j|x)

    Unlike the Royston-Parmar formulation, coherence here is NOT structural:
    F1 + F2 + S = 1 holds only because the same event-time grid and the same
    running survival are used for both causes. That contrast is worth
    reporting rather than hiding.
    """

    def __init__(self, name="cox", columns=None, spline_columns=None, seed=42,
                 penalizer=1e-6, expand=True, batch_size=512):
        self.name = name
        self.columns = list(columns) if columns is not None else None
        self.spline_columns = spline_columns
        self.seed = seed
        self.penalizer = penalizer
        self.expand = expand
        self.spline_spec = None
        self.beta = {}
        self.base_times = {}
        self.base_cumhaz = {}
        self.batch_size = int(batch_size)
        self._cache = None

    def fit(self, X, time, event, **kw):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event)
        cols = self.columns or [f"x{i}" for i in range(X.shape[1])]
        if self.expand:
            self.spline_spec = fit_covariate_splines(X, cols,
                                                     only=self.spline_columns)
            Xd = apply_covariate_splines(X, self.spline_spec)
        else:
            Xd = X
        self._centre = Xd.mean(axis=0)
        self._scale = np.where(Xd.std(axis=0) > 0, Xd.std(axis=0), 1.0)
        Z = (Xd - self._centre) / self._scale

        for k in CAUSES:
            b = _fit_cox(Z, time, (event == k).astype(int),
                         penalizer=self.penalizer)
            self.beta[k] = b
            t_k, H_k = _breslow_baseline(Z, time, (event == k).astype(int), b)
            self.base_times[k] = t_k
            self.base_cumhaz[k] = H_k
        # One shared, sorted grid of all event times, so both causes integrate
        # against the same running survival - the reason coherence holds.
        self.grid = np.unique(np.concatenate([self.base_times[k] for k in CAUSES]))
        self._cache = None
        return self

    def _design(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xd = (apply_covariate_splines(X, self.spline_spec)
              if self.spline_spec is not None else X)
        return (Xd - self._centre) / self._scale

    def _baseline_increments(self, grid):
        """dH0_k on `grid`: (G,) per cause. Independent of the subject, so it
        is computed once per grid rather than once per batch."""
        out = {}
        for k in CAUSES:
            H0 = np.interp(grid, self.base_times[k], self.base_cumhaz[k],
                           left=0.0)
            out[k] = np.diff(np.concatenate([[0.0], H0]))
        return out

    def _increments(self, Z, grid=None, dH0=None):
        """dH_k per subject on the grid: (n, G) per cause."""
        grid = self.grid if grid is None else grid
        dH0 = self._baseline_increments(grid) if dH0 is None else dH0
        return {k: dH0[k][None, :] * np.exp(Z @ self.beta[k])[:, None]
                for k in CAUSES}

    def _integrate(self, Z, grid=None, dH0=None):
        """(F_1, F_2, S) on the shared event-time grid.

        Survival is the PRODUCT-LIMIT form, S(t_j) = prod_{i<=j}(1 - dH_i),
        not exp(-sum dH). That choice is what makes coherence exact rather
        than approximate, and it is worth spelling out because it is the whole
        content of "the integration done correctly":

            S(t_{j-1}) - S(t_j) = S(t_{j-1}) dH(t_j)

        so summing S(t_{j-1}) dH_k over both causes telescopes to 1 - S(t).
        With exp(-sum dH) the identity only holds to O(dH^2), which on this
        data showed up as a coherence error of ~1e-03 - three orders of
        magnitude worse than the Royston-Parmar model's quadrature error, and
        entirely an artefact of the integration rather than of Cox itself.
        """
        inc = self._increments(Z, grid, dH0)
        total = np.clip(sum(inc[k] for k in CAUSES), 0.0, 1.0 - 1e-12)
        keep = 1.0 - total
        surv = np.cumprod(keep, axis=1)                      # S(t_j)
        surv_prev = surv / np.maximum(keep, 1e-300)          # S(t_{j-1})
        # Rescale the per-cause increments if clipping bit, so the two causes
        # still partition the total hazard exactly.
        raw_total = sum(inc[k] for k in CAUSES)
        scale = np.where(raw_total > 0, total / np.maximum(raw_total, 1e-300), 0.0)
        cif = {k: np.cumsum(surv_prev * inc[k] * scale, axis=1) for k in CAUSES}
        return cif, surv

    def predict_cif(self, X, t_eval):
        return self._predict(X, t_eval)[0]

    def predict_survival(self, X, t_eval):
        return self._predict(X, t_eval)[1]

    def _predict(self, X, t_eval):
        """CIFs and survival in one pass, memoised.

        BATCHED, and the grid is truncated at max(t_eval), because the
        integration is over an (n_batch, G) array with G = number of distinct
        event times. On SRTR-scale data G runs into six figures, so the
        unbatched form would try to allocate an n x G matrix per cause and die.
        Only the columns at the evaluation horizons are kept, so peak memory is
        `batch_size * G` regardless of how many rows are scored.
        """
        Xarr = np.asarray(X, dtype=np.float64)
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        key = (Xarr.tobytes(), t_eval.tobytes())
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]

        # Nothing beyond the last horizon can affect the answer.
        grid = self.grid[self.grid <= t_eval.max()]
        if grid.size == 0:
            grid = self.grid[:1]
        dH0 = self._baseline_increments(grid)
        idx = np.searchsorted(grid, t_eval, side="right") - 1
        keep = idx >= 0

        Z = self._design(Xarr)
        n = Z.shape[0]
        cif = np.zeros((n, len(t_eval), 2))
        surv = np.ones((n, len(t_eval)))
        step = max(1, min(n, self.batch_size))
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            cif_grid, surv_grid = self._integrate(Z[lo:hi], grid, dH0)
            for j, i in enumerate(idx):
                if not keep[j]:
                    continue
                surv[lo:hi, j] = surv_grid[:, i]
                for k in CAUSES:
                    cif[lo:hi, j, k - 1] = cif_grid[k][:, i]

        out = (np.clip(cif, 0.0, 1.0), surv)
        self._cache = (key, out)
        return out


def _fit_cox(Z, time, status, penalizer=1e-6, max_iter=100, tol=1e-10):
    """Breslow-tie Cox partial likelihood by Newton-Raphson.

    Risk-set sums are reverse cumulative sums over time-sorted rows, which is
    what keeps this O(n log n) rather than O(n * n_events) - the same trick the
    Fine-Gray fit uses.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n, p = Z.shape
    order = np.argsort(time, kind="mergesort")
    Zs, ts, ds = Z[order], np.asarray(time)[order], np.asarray(status)[order]

    ev_times = np.unique(ts[ds == 1])
    if ev_times.size == 0:
        return np.zeros(p)
    start = np.searchsorted(ts, ev_times, side="left")
    ev_grp = np.searchsorted(ev_times, ts[ds == 1])
    d_ev = np.bincount(ev_grp, minlength=ev_times.size).astype(np.float64)
    x_sum = np.zeros((ev_times.size, p))
    np.add.at(x_sum, ev_grp, Zs[ds == 1])

    def rev(arr):
        c = np.concatenate([np.cumsum(arr[::-1], axis=0)[::-1],
                            np.zeros((1,) + arr.shape[1:])])
        return c[start]

    def grad_and_info(beta):
        """Score and observed information, accumulated ONE COVARIATE AT A TIME.

        The textbook form builds S2 as an (n, p, p) array of weighted outer
        products and reverse-cumsums it. That is what the first version did,
        and it is fine on toy data and fatal on real data: at n = 100k and
        p = 150 the temporary alone is ~18 GB, and the (G, p, p) result is
        another 9 GB.

        Only the sum over event times is ever needed, so row `a` of the
        information matrix is accumulated from a single (n, p) reverse cumsum
        and discarded. Same arithmetic, same result, memory O(n * p) instead
        of O(n * p^2).
        """
        w = np.exp(np.clip(Zs @ beta, -500, 500))
        s0 = np.maximum(rev(w[:, None])[:, 0], 1e-300)
        s1 = rev(w[:, None] * Zs)
        mu = s1 / s0[:, None]
        grad = x_sum.sum(axis=0) - (d_ev[:, None] * mu).sum(axis=0)
        dz = d_ev / s0
        info = np.empty((p, p))
        for a in range(p):
            s2_a = rev(w[:, None] * Zs[:, [a]] * Zs)          # (G, p)
            info[a] = ((dz[:, None] * s2_a).sum(axis=0)
                       - (d_ev[:, None] * mu[:, [a]] * mu).sum(axis=0))
        info += penalizer * np.eye(p)
        return grad, info

    def log_pl(beta):
        w = np.exp(np.clip(Zs @ beta, -500, 500))
        s0 = np.maximum(rev(w[:, None])[:, 0], 1e-300)
        return float((x_sum @ beta).sum() - (d_ev * np.log(s0)).sum())

    beta = np.zeros(p)
    cur = log_pl(beta)
    for _ in range(max_iter):
        grad, info = grad_and_info(beta)
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info, grad, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        # Accept only an increase in the log partial likelihood.
        factor, improved = 1.0, False
        for _ in range(40):
            cand = beta + factor * step
            val = log_pl(cand)
            if np.isfinite(val) and val >= cur:
                beta, prev, cur, improved = cand, cur, val, True
                break
            factor *= 0.5
        if not improved or abs(cur - prev) < tol:
            break
    return beta


def _breslow_baseline(Z, time, status, beta):
    """Breslow cumulative baseline hazard at the observed event times."""
    order = np.argsort(time, kind="mergesort")
    Zs, ts, ds = Z[order], np.asarray(time)[order], np.asarray(status)[order]
    ev_times = np.unique(ts[ds == 1])
    if ev_times.size == 0:
        return np.array([0.0]), np.array([0.0])
    start = np.searchsorted(ts, ev_times, side="left")
    w = np.exp(np.clip(Zs @ beta, -500, 500))
    rc = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
    s0 = np.maximum(rc[start], 1e-300)
    d = np.bincount(np.searchsorted(ev_times, ts[ds == 1]),
                    minlength=ev_times.size).astype(np.float64)
    return ev_times, np.cumsum(d / s0)


# ---------------------------------------------------------------------------
# 3. Fine-Gray
# ---------------------------------------------------------------------------

class FineGrayBaseline:
    """One subdistribution-hazard model per cause, F_k = 1 - exp(-H_k^sub).

    Reuses the Fine-Gray estimator validated against `cmprsk::crr` to 3.6e-11
    in the metrics module, including its convergence fixes (centred and scaled
    columns; Newton steps accepted only on likelihood improvement).

    F1 + F2 + S = 1 is NOT guaranteed here: the two models are fitted
    independently and nothing couples them. The observed deviation is reported
    rather than corrected, because it is a substantive point in favour of the
    Royston-Parmar formulation.
    """

    def __init__(self, name="fine_gray", columns=None, spline_columns=None,
                 seed=42, expand=True):
        self.name = name
        self.columns = list(columns) if columns is not None else None
        self.spline_columns = spline_columns
        self.seed = seed
        self.expand = expand
        self.spline_spec = None
        self.fits = {}

    def fit(self, X, time, event, **kw):
        X = np.asarray(X, dtype=np.float64)
        cols = self.columns or [f"x{i}" for i in range(X.shape[1])]
        if self.expand:
            self.spline_spec = fit_covariate_splines(X, cols,
                                                     only=self.spline_columns)
            Xd = apply_covariate_splines(X, self.spline_spec)
        else:
            Xd = X
        for k in CAUSES:
            self.fits[k] = fit_fine_gray(Xd, np.asarray(time, dtype=np.float64),
                                         np.asarray(event), cause=k)
        return self

    @property
    def converged(self) -> dict:
        return {k: bool(f.converged) for k, f in self.fits.items()}

    def predict_cif(self, X, t_eval):
        X = np.asarray(X, dtype=np.float64)
        Xd = (apply_covariate_splines(X, self.spline_spec)
              if self.spline_spec is not None else X)
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        cif = np.zeros((Xd.shape[0], len(t_eval), 2))
        for k in CAUSES:
            for j, t in enumerate(t_eval):
                cif[:, j, k - 1] = self.fits[k].predict(Xd, float(t))
        return np.clip(cif, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Random survival forest (R randomForestSRC via subprocess)
# ---------------------------------------------------------------------------

RSF_R_TEMPLATE = r'''
suppressMessages(library(randomForestSRC))
train <- read.csv("__TRAIN__"); test <- read.csv("__TEST__")
te <- as.numeric(strsplit("__TIMES__", ",")[[1]])
set.seed(__SEED__)
xn <- setdiff(names(train), c("time","event"))
f <- as.formula(paste("Surv(time, event) ~", paste(xn, collapse=" + ")))
fit <- rfsrc(f, data = train, ntree = __NTREE__, nodesize = __NODESIZE__,
             ntime = __NTIME__,
             splitrule = "logrankCR", samptype = "swor", seed = -__SEED__)
p <- predict(fit, newdata = test)
# cif is (n, n_times, n_causes); interpolate each cause onto the eval grid.
out <- NULL
for (k in 1:2) {
  m <- t(apply(p$cif[, , k], 1, function(r)
        approx(p$time.interest, r, te, rule = 2)$y))
  out <- cbind(out, m)
}
# Event-free survival. The competing-risks family returns no $survival, so it
# is rebuilt from the cause-specific cumulative hazards by the SAME
# product-limit recursion rfsrc uses for the CIFs:
#     S(t_j) = prod_{i <= j} (1 - dH_1(t_i) - dH_2(t_i))
# Carried through so the harness scores coherence against the forest's own
# survival rather than inferring it from the CIFs, which would make the
# deviation identically zero and measure nothing.
d1 <- cbind(p$chf[, 1, 1], t(apply(p$chf[, , 1], 1, diff)))
d2 <- cbind(p$chf[, 1, 2], t(apply(p$chf[, , 2], 1, diff)))
keep <- pmax(1 - pmax(d1, 0) - pmax(d2, 0), 0)
S <- t(apply(keep, 1, cumprod))
sv <- t(apply(S, 1, function(r) approx(p$time.interest, r, te, rule = 2)$y))
out <- cbind(out, sv)
write.csv(out, "__OUT__", row.names = FALSE)
'''


class RSFBaseline:
    """Competing-risks random survival forest (`splitrule = "logrankCR"`),
    driven through R by subprocess with CSV exchange.

    Subprocess rather than rpy2: it keeps the R session isolated per fit, so a
    segfault in the forest cannot take the whole benchmark down, and it needs
    no rpy2 build. THIS IS THE RUNTIME BOTTLENECK - benchmark one fit before
    launching the full matrix.

    Note on coherence: rfsrc returns no `$survival` for the competing-risks
    family, so `predict_survival` rebuilds it from the forest's own
    cause-specific cumulative hazards by the product-limit recursion. The
    result does not reproduce the forest's reported CIFs exactly, because the
    ensemble averages CIFs and cumulative hazards separately and averaging does
    not commute with the Aalen-Johansen recursion. The residual is a real
    property of the bagged estimator, not an artefact of this wrapper.
    """

    def __init__(self, name="rsf", columns=None, seed=42, ntree=300,
                 nodesize=15, ntime=250, rscript="Rscript", timeout=7200):
        self.name = name
        self.columns = list(columns) if columns is not None else None
        self.seed = seed
        self.ntree = ntree
        self.nodesize = nodesize
        # `ntime` is NOT a tuning knob, it is what makes the model runnable.
        # Left unset, rfsrc evaluates the CIFs at every distinct event time:
        # at n = 50,000 that is a (50000, ~23000, 2) array for `cif` and
        # another for `chf` - about 36 GB, which does not fail fast but
        # thrashes. One run was still going after 90 minutes. On a 250-point
        # grid the same arrays are ~200 MB, and since the CIFs are interpolated
        # onto `t_eval` regardless, the grid costs nothing that matters.
        self.ntime = int(ntime)
        self.rscript = rscript
        self.timeout = timeout
        self._train = None
        self._cache = None

    @staticmethod
    def available(rscript="Rscript") -> bool:
        try:
            r = subprocess.run(
                [rscript, "-e",
                 'cat(requireNamespace("randomForestSRC", quietly=TRUE))'],
                capture_output=True, text=True, timeout=120)
            return "TRUE" in r.stdout
        except Exception:
            return False

    def fit(self, X, time, event, **kw):
        # The forest is refitted at predict time in the same R session, since
        # rfsrc gives no portable serialisation across processes; the training
        # rows are held here.
        self._train = (np.asarray(X, dtype=np.float64),
                       np.asarray(time, dtype=np.float64),
                       np.asarray(event))
        return self

    def predict_cif(self, X, t_eval):
        return self._predict(X, t_eval)[0]

    def predict_survival(self, X, t_eval):
        """Event-free survival from the same forest.

        rfsrc reports it alongside the CIFs, so it costs nothing extra and lets
        the coherence column measure the forest's own internal consistency.
        """
        return self._predict(X, t_eval)[1]

    def _predict(self, X, t_eval):
        """One R call serves both outputs; the last result is memoised because
        the harness asks for the CIFs and the survival in succession."""
        key = (np.asarray(X, dtype=np.float64).tobytes(),
               np.atleast_1d(np.asarray(t_eval, dtype=np.float64)).tobytes())
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        out = self._run(X, t_eval)
        self._cache = (key, out)
        return out

    def _run(self, X, t_eval):
        if self._train is None:
            raise RuntimeError("fit() must be called before predict_cif()")
        Xtr, ttr, etr = self._train
        X = np.asarray(X, dtype=np.float64)
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        cols = self.columns or [f"x{i}" for i in range(Xtr.shape[1])]
        safe = [f"v{i}" for i in range(len(cols))]

        tmp = tempfile.mkdtemp(prefix="rsf_")
        ftrain = os.path.join(tmp, "train.csv").replace("\\", "/")
        ftest = os.path.join(tmp, "test.csv").replace("\\", "/")
        fout = os.path.join(tmp, "cif.csv").replace("\\", "/")
        header = ",".join(["time", "event"] + safe)
        np.savetxt(ftrain, np.column_stack([ttr, etr, Xtr]), delimiter=",",
                   header=header, comments="")
        np.savetxt(ftest, np.column_stack([np.ones(len(X)), np.zeros(len(X)), X]),
                   delimiter=",", header=header, comments="")
        code = (RSF_R_TEMPLATE.replace("__TRAIN__", ftrain)
                .replace("__TEST__", ftest).replace("__OUT__", fout)
                .replace("__TIMES__", ",".join(map(str, t_eval)))
                .replace("__SEED__", str(int(self.seed)))
                .replace("__NTREE__", str(int(self.ntree)))
                .replace("__NODESIZE__", str(int(self.nodesize)))
                .replace("__NTIME__", str(int(self.ntime))))
        r = subprocess.run([self.rscript, "-e", code], capture_output=True,
                           text=True, timeout=self.timeout)
        if r.returncode != 0 or not os.path.exists(fout):
            raise RuntimeError(f"randomForestSRC failed:\n{r.stderr[-2000:]}")
        flat = np.genfromtxt(fout, delimiter=",", skip_header=1)
        flat = np.atleast_2d(flat)
        T = len(t_eval)
        cif = np.stack([flat[:, :T], flat[:, T:2 * T]], axis=-1)
        surv = flat[:, 2 * T:3 * T]
        return np.clip(cif, 0.0, 1.0), np.clip(surv, 0.0, 1.0)


def make_default_baselines(columns, spline_columns=None, seed=42,
                           include_rsf=True, rsf_kwargs=None):
    """The five classical baselines, each as a `factory(seed)` for the runner."""
    rsf_kwargs = dict(rsf_kwargs or {})
    factories = {
        "royston_parmar": lambda s: RoystonParmarBaseline(
            columns=columns, spline_columns=spline_columns, seed=s),
        "cox": lambda s: CoxBaseline(columns=columns,
                                     spline_columns=spline_columns, seed=s),
        "fine_gray": lambda s: FineGrayBaseline(
            columns=columns, spline_columns=spline_columns, seed=s),
        "rp_kdri_epts": lambda s: MinimalReferenceBaseline(columns=columns,
                                                           seed=s),
    }
    if include_rsf:
        factories["rsf"] = lambda s: RSFBaseline(columns=columns, seed=s,
                                                 **rsf_kwargs)
    return factories
