"""Validation of the competing-risks evaluation metrics.

The organising idea of this suite is the project's own thesis: a model can
discriminate perfectly and still be badly miscalibrated, so calibration is
tested at least as hard as discrimination. `test_2_miscalibration_is_detected`
states that argument in code.

Where an independent implementation exists it is used: the Fine-Gray fit is
checked against R's `cmprsk::crr`, the Aalen-Johansen estimator against
`cmprsk::cuminc`, and AUC/Brier against `riskRegression::Score`.
"""

import subprocess
import sys
import tempfile
import time as _time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from metrics import (aalen_johansen, brier_score, calibration_curve,       # noqa: E402
                     calibration_in_the_large, evaluate, evaluate_by_subgroup,
                     fit_fine_gray, grouped_calibration, quantile_strata,
                     rcs_design_np, time_dependent_auc, time_dependent_cindex)
from metrics.ipcw import make_censoring                                    # noqa: E402

SEED = 42
SHAPE = {1: 1.3, 2: 0.9}
SCALE = {1: 0.042, 2: 0.085}
BETA = {1: np.array([0.5, -0.3, 0.4, -0.2]),
        2: np.array([-0.2, 0.4, -0.3, 0.5])}
T_EVAL = np.array([1.0, 3.0, 5.0])


def simulate(n=5000, seed=SEED, c_max=9.0, spread=1.0):
    """Weibull cause-specific hazards, so the TRUE CIF is computable."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.standard_normal(n) * spread,
                         rng.standard_normal(n) * spread,
                         (rng.random(n) < 0.5) * 1.0,
                         (rng.random(n) < 0.3) * 1.0])
    latent = {}
    for k in (1, 2):
        e = rng.exponential(size=n)
        latent[k] = (e / (SCALE[k] * np.exp(x @ BETA[k]))) ** (1 / SHAPE[k])
    t_ev = np.minimum(latent[1], latent[2])
    cause = np.where(latent[1] <= latent[2], 1, 2)
    if c_max is None:
        return t_ev, cause, x                      # no censoring at all
    c = rng.uniform(0, c_max, size=n)
    time = np.minimum(t_ev, c)
    return time, np.where(t_ev <= c, cause, 0), x


def true_cif(x, t_grid, n_int=3000, chunk=4000):
    """F_k(t|x) = int_0^t h_k S du by fine quadrature - the exact target.

    Chunked over patients: the (n x n_int) intermediates are ~1 GB per cause
    at n = 40,000, which is enough to bring the machine to its knees.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((len(x), len(t_grid), 2))
    lp_all = {k: np.exp(x @ BETA[k]) for k in (1, 2)}
    for start in range(0, len(x), chunk):
        sl = slice(start, start + chunk)
        lp = {k: lp_all[k][sl] for k in (1, 2)}
        for j, t in enumerate(t_grid):
            u = np.linspace(1e-9, t, n_int)
            up = {k: u[None, :] ** SHAPE[k] for k in (1, 2)}
            H = sum(SCALE[k] * lp[k][:, None] * up[k] for k in (1, 2))
            S = np.exp(-H, out=H)                      # reuse the buffer
            for k in (1, 2):
                h = (SHAPE[k] * SCALE[k] * lp[k][:, None]
                     * u[None, :] ** (SHAPE[k] - 1))
                out[sl, j, k - 1] = np.trapezoid(h * S, u, axis=1)
    return out


def _rscript(code, arrays=None):
    """Run an R snippet, optionally writing a CSV first. None if R fails."""
    path = None
    if arrays is not None:
        fd, path = tempfile.mkstemp(suffix=".csv")
        import os
        os.close(fd)
        np.savetxt(path, arrays[0], delimiter=",", header=arrays[1], comments="")
        code = code.replace("__CSV__", path.replace("\\", "/"))
    """try:
        r = subprocess.run(["Rscript", "-e", code], capture_output=True,
                           text=True, timeout=600)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    
    
    except Exception as exc:
        print(f"  [_rscript] {type(exc).__name__}: {exc}")
        return None
    """
    #except Exception:
        #return None
        
    try:
        r = subprocess.run(["Rscript", "-e", code], capture_output=True,
                           text=True, timeout=600)
        if r.returncode != 0:
            print(f"  [_rscript] R failed:\n{r.stderr[-500:]}")
            return None
        out = r.stdout.strip()
        if not out:
            print("  [_rscript] R produced no output")
            return None
        return out
    except Exception as exc:
        print(f"  [_rscript] {type(exc).__name__}: {exc}")
        return None    


@pytest.fixture(scope="module")
def data():
    time, event, x = simulate()
    return time, event, x, true_cif(x, T_EVAL)


# ---------------------------------------------------------------------------
# Building blocks validated against independent implementations
# ---------------------------------------------------------------------------

def test_rcs_basis_matches_verified_implementation():
    """The numpy RCS used by the calibration smoother must equal the torch
    basis already verified against rstpm2."""
    import torch
    from rcs_basis import rcs_design
    v = np.linspace(0.01, 0.9, 200)
    knots = np.array([0.05, 0.2, 0.5, 0.8])
    mine = rcs_design_np(v, knots)
    theirs = rcs_design(torch.tensor(v), torch.tensor(knots)).numpy()
    assert np.abs(mine - theirs).max() < 1e-12


def test_fine_gray_matches_cmprsk():
    """The Fine-Gray engine behind the Austin calibration curve, against
    R's cmprsk::crr."""
    rng = np.random.default_rng(5)
    n = 3000
    x = np.column_stack([rng.standard_normal(n), (rng.random(n) < 0.5) * 1.0,
                         rng.standard_normal(n)])
    bt = {1: np.array([0.6, -0.4, 0.25]), 2: np.array([-0.3, 0.5, 0.1])}
    latent = {}
    for k, (sh, sc) in {1: (1.2, 0.05), 2: (0.9, 0.06)}.items():
        latent[k] = (rng.exponential(size=n) / (sc * np.exp(x @ bt[k]))) ** (1 / sh)
    t_ev = np.minimum(latent[1], latent[2])
    cause = np.where(latent[1] <= latent[2], 1, 2)
    c = rng.uniform(0, 8, size=n)
    time, event = np.minimum(t_ev, c), np.where(t_ev <= c, cause, 0)

    fit = fit_fine_gray(x, time, event, cause=1)
    assert fit.converged, fit.message
    out = _rscript('''
suppressMessages(library(cmprsk)); d <- read.csv("__CSV__")
f <- crr(d$time, d$event, cov1 = as.matrix(d[, c("x1","x2","x3")]),
         failcode = 1, cencode = 0)
cat(paste(round(f$coef, 10), collapse = ","))
''', (np.column_stack([time, event, x]), "time,event,x1,x2,x3"))
    if out is None:
        pytest.skip("R/cmprsk unavailable")
    r_beta = np.array([float(z) for z in out.split(",")])
    diff = np.abs(fit.beta_original - r_beta).max()
    print(f"\n  Fine-Gray vs cmprsk::crr: max |diff| = {diff:.3e}")
    print(f"    python {np.round(fit.beta_original, 6)}")
    print(f"    R      {np.round(r_beta, 6)}")
    assert diff < 1e-5


def test_aalen_johansen_matches_cmprsk(data):
    time, event, _, _ = data
    est, var = aalen_johansen(time, event, T_EVAL, cause=1)
    out = _rscript('''
suppressMessages(library(cmprsk)); d <- read.csv("__CSV__")
ci <- cuminc(d$time, d$event, cencode = 0); te <- c(1,3,5)
e <- approx(ci[["1 1"]]$time, ci[["1 1"]]$est, te, rule=2)$y
v <- approx(ci[["1 1"]]$time, ci[["1 1"]]$var, te, rule=2)$y
cat(paste(round(c(e, v), 12), collapse = ","))
''', (np.column_stack([time, event]), "time,event"))
    if out is None:
        pytest.skip("R/cmprsk unavailable")
    vals = np.array([float(z) for z in out.split(",")])
    r_est, r_var = vals[:3], vals[3:6]
    diff = np.abs(est - r_est).max()
    d_var = np.abs(np.sqrt(var) - np.sqrt(r_var)).max()
    print(f"\n  Aalen-Johansen vs cmprsk::cuminc: est {diff:.3e}, "
          f"SE {d_var:.3e}")
    assert diff < 1e-3
    assert d_var < 1e-4, "variance disagrees with cmprsk"


# ---------------------------------------------------------------------------
# 1. Perfect-prediction limit
# ---------------------------------------------------------------------------

def test_1_perfect_prediction_limit(data):
    """Feed the TRUE CIFs: calibration should be near-perfect and the Brier
    score at its irreducible minimum (no perturbation can beat it)."""
    time, event, x, cif = data
    auc = time_dependent_auc(cif, time, event, T_EVAL)
    cidx = time_dependent_cindex(cif, time, event, T_EVAL)
    bs = brier_score(cif, time, event, T_EVAL)

    print("\n  true-CIF predictions:")
    for k in (1, 2):
        print(f"    cause{k}: AUC {np.round(auc[k],4)}  C {np.round(cidx[k],4)}"
              f"  Brier {np.round(bs[k],5)}")
        assert np.all(auc[k] > 0.6) and np.all(auc[k] <= 1.0)
        assert np.all(cidx[k] > 0.6) and np.all(cidx[k] <= 1.0)

    for k in (1, 2):
        cal = calibration_curve(cif[:, 1, k - 1], time, event, 3.0, cause=k)
        print(f"    cause{k} @3y: ICI {cal['ICI']:.5f}  E50 {cal['E50']:.5f} "
              f"E90 {cal['E90']:.5f}  [{cal['method']}]")
        assert cal["method"].startswith("fine_gray")
        # Not exactly 0: the smoother estimates observed risk from finite
        # data. test_1b confirms it shrinks with n, i.e. it is noise not bias.
        assert cal["ICI"] < 0.02

    # Irreducible minimum: perturbing the true CIF can only make Brier worse.
    for mult in (0.8, 1.25):
        worse = brier_score(np.clip(cif * mult, 0, 1), time, event, T_EVAL)
        for k in (1, 2):
            assert np.all(worse[k] >= bs[k] - 1e-12), (
                f"perturbed prediction beat the truth for cause {k}")


def test_1b_ici_is_noise_not_bias():
    """ICI under perfect prediction must shrink with n - the estimator is
    consistent, so the residual at n=5,000 is smoother noise."""
    icis = {}
    for n in (5_000, 20_000):
        time, event, x = simulate(n=n, seed=1)
        cif = true_cif(x, np.array([3.0]))
        icis[n] = calibration_curve(cif[:, 0, 0], time, event, 3.0,
                                    cause=1)["ICI"]
    print(f"\n  ICI(perfect): n=5,000 -> {icis[5000]:.5f}, "
          f"n=20,000 -> {icis[20000]:.5f}")
    assert icis[20_000] < icis[5_000] * 0.85


# ---------------------------------------------------------------------------
# 2. Miscalibration is detected while discrimination is not
# ---------------------------------------------------------------------------

def test_2_miscalibration_is_detected(data):
    """The paper's core methodological argument, in code.

    Multiplying the true CIF by a constant is a monotone transformation, so
    AUC and the C-index are unchanged - a model can rank patients perfectly
    and still be badly wrong about absolute risk. Only the calibration
    metrics notice, which is why they carry this project.
    """
    time, event, x, cif = data
    base_auc = time_dependent_auc(cif, time, event, T_EVAL)
    base_c = time_dependent_cindex(cif, time, event, T_EVAL)
    base_cal = {k: calibration_curve(cif[:, 1, k - 1], time, event, 3.0, cause=k)
                for k in (1, 2)}

    for mult in (1.5, 0.67):
        bad = np.clip(cif * mult, 0.0, 0.999)
        auc = time_dependent_auc(bad, time, event, T_EVAL)
        cidx = time_dependent_cindex(bad, time, event, T_EVAL)
        print(f"\n  multiplier {mult}:")
        for k in (1, 2):
            cal = calibration_curve(bad[:, 1, k - 1], time, event, 3.0, cause=k)
            d_auc = np.abs(auc[k] - base_auc[k]).max()
            d_c = np.abs(cidx[k] - base_c[k]).max()
            print(f"    cause{k}: dAUC {d_auc:.2e}  dC {d_c:.2e}  "
                  f"ICI {base_cal[k]['ICI']:.5f} -> {cal['ICI']:.5f}  "
                  f"E90 {base_cal[k]['E90']:.5f} -> {cal['E90']:.5f}")
            # Discrimination is invariant (up to the few values clipped at
            # the ceiling, which create ties).
            assert d_auc < 5e-3 and d_c < 5e-3
            # Calibration is not.
            assert cal["ICI"] > base_cal[k]["ICI"] * 3
            assert cal["E50"] > base_cal[k]["E50"] * 2
            assert cal["E90"] > base_cal[k]["E90"] * 2


# ---------------------------------------------------------------------------
# 3. IPCW correctness
# ---------------------------------------------------------------------------

def test_3_ipcw_recovers_uncensored_values():
    """Under independent administrative censoring, IPCW-weighted Brier and
    AUC must recover the values computed on the same data uncensored."""
    n = 40_000
    t_unc, c_unc, x = simulate(n=n, seed=7, c_max=None)
    cif = true_cif(x, T_EVAL)
    ev_unc = c_unc.copy()

    rng = np.random.default_rng(7)
    c = rng.uniform(0, 9.0, size=n)
    time = np.minimum(t_unc, c)
    event = np.where(t_unc <= c, c_unc, 0)

    bs_u = brier_score(cif, t_unc, ev_unc, T_EVAL)
    bs_c = brier_score(cif, time, event, T_EVAL)
    auc_u = time_dependent_auc(cif, t_unc, ev_unc, T_EVAL)
    auc_c = time_dependent_auc(cif, time, event, T_EVAL)
    print("\n  uncensored vs IPCW-censored (n=40,000):")
    for k in (1, 2):
        print(f"    cause{k} Brier {np.round(bs_u[k],5)} vs "
              f"{np.round(bs_c[k],5)}  (max diff {np.abs(bs_u[k]-bs_c[k]).max():.4f})")
        print(f"    cause{k} AUC   {np.round(auc_u[k],4)} vs "
              f"{np.round(auc_c[k],4)}  (max diff {np.abs(auc_u[k]-auc_c[k]).max():.4f})")
        assert np.abs(bs_u[k] - bs_c[k]).max() < 0.01
        assert np.abs(auc_u[k] - auc_c[k]).max() < 0.02


# ---------------------------------------------------------------------------
# 4. Competing-event subjects belong in the controls
# ---------------------------------------------------------------------------

def _naive_auc_excluding_competing(cif, time, event, t, cause, j):
    """The WRONG definition: competing events dropped rather than kept as
    controls. Used only to show the size of the error."""
    cens = make_censoring(time, event)
    is_case = (time <= t) & (event == cause)
    is_ctrl = time > t                              # competing events dropped
    w = np.zeros_like(time)
    w[is_case] = 1.0 / cens.G_minus(time[is_case])
    w[is_ctrl] = 1.0 / cens.G(np.full(is_ctrl.sum(), t))
    m = cif[:, j, cause - 1]
    order = np.argsort(m, kind="mergesort")
    m_s, case_w, ctrl_w = m[order], (w * is_case)[order], (w * is_ctrl)[order]
    cum = np.concatenate([[0.0], np.cumsum(ctrl_w)])
    uniq, first = np.unique(m_s, return_index=True)
    grp = np.searchsorted(uniq, m_s)
    conc = cum[first][grp] + 0.5 * np.add.reduceat(ctrl_w, first)[grp]
    return float((case_w * conc).sum() / (case_w.sum() * ctrl_w.sum()))


def test_4_competing_events_are_controls(data):
    """With substantial competing risk the two definitions differ materially,
    so this is not a cosmetic distinction."""
    time, event, x, cif = data
    correct = time_dependent_auc(cif, time, event, T_EVAL, causes=[1])[1]
    print("\n  AUC cause 1 - competing events as controls vs dropped:")
    worst = 0.0
    for j, t in enumerate(T_EVAL):
        naive = _naive_auc_excluding_competing(cif, time, event, t, 1, j)
        worst = max(worst, abs(correct[j] - naive))
        print(f"    t={t:.0f}: correct {correct[j]:.4f}  naive {naive:.4f}  "
              f"diff {correct[j]-naive:+.4f}")
    assert worst > 0.005, (
        "the two definitions barely differ here, so this test would not "
        "detect the error it exists to catch")


# ---------------------------------------------------------------------------
# 5. Aalen-Johansen vs Kaplan-Meier in grouped calibration
# ---------------------------------------------------------------------------

def test_5_grouped_calibration_uses_aalen_johansen(data):
    """KM treats competing events as censored - as if those patients could
    still fail - and therefore OVERSTATES absolute risk. The module must use
    Aalen-Johansen; the gap is reported because it is a useful number."""
    time, event, x, cif = data
    rows = grouped_calibration(cif[:, 1, 0], time, event, 3.0, cause=1)

    def km_cif(tt, ee, t):
        """1 - KM survival treating only cause 1 as the event."""
        order = np.argsort(tt, kind="mergesort")
        tt, ee = tt[order], ee[order]
        uniq, first = np.unique(tt, return_index=True)
        n_at = len(tt) - first
        d = np.add.reduceat((ee == 1).astype(float), first)
        surv = np.cumprod(1 - d / n_at)
        idx = np.searchsorted(uniq, t, side="right") - 1
        return 1.0 - (surv[idx] if idx >= 0 else 1.0)

    edges = np.quantile(cif[:, 1, 0], np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    grp = np.clip(np.searchsorted(edges, cif[:, 1, 0], side="right") - 1, 0, 9)
    print("\n  decile  n     predicted   observed(AJ)   observed(KM)   KM-AJ")
    gaps = []
    for r in rows:
        sel = grp == r["group"] - 1
        km = km_cif(time[sel], event[sel], 3.0)
        gaps.append(km - r["observed_aj"])
        print(f"    {r['group']:2d}   {r['n']:5d}   {r['predicted_mean']:.4f}"
              f"      {r['observed_aj']:.4f}         {km:.4f}      "
              f"{km - r['observed_aj']:+.4f}")
    gaps = np.array(gaps)
    print(f"  mean KM-minus-AJ bias across deciles: {gaps.mean():+.4f} "
          f"(max {gaps.max():+.4f})")
    assert np.all(gaps > -1e-9), "KM should never fall below AJ"
    assert gaps.mean() > 0.005, "expected a material upward KM bias"
    # And the module's own output must track AJ, not KM.
    pred = np.array([r["predicted_mean"] for r in rows])
    obs = np.array([r["observed_aj"] for r in rows])
    assert np.abs(pred - obs).mean() < np.abs(pred - (obs + gaps)).mean()


# ---------------------------------------------------------------------------
# 6. Group-level predictions are averages of individual CIFs
# ---------------------------------------------------------------------------

def test_6_marginal_cif_is_average_of_individual_predictions():
    """mean_i F_k(t|x_i) != F_k(t|x_bar) by Jensen, and the gap widens with
    covariate spread. The module must use the average of predictions."""
    time, event, x = simulate(n=8000, seed=3, spread=2.0)
    cif = true_cif(x, np.array([3.0]))
    avg_of_pred = float(cif[:, 0, 0].mean())
    pred_at_mean = float(true_cif(x.mean(axis=0, keepdims=True),
                                  np.array([3.0]))[0, 0, 0])
    print(f"\n  mean_i F_1(3y|x_i) = {avg_of_pred:.5f}   "
          f"F_1(3y|x_bar) = {pred_at_mean:.5f}   "
          f"gap = {avg_of_pred - pred_at_mean:+.5f}")
    assert abs(avg_of_pred - pred_at_mean) > 0.01, (
        "covariate spread too small for this test to be meaningful")

    citl = calibration_in_the_large(cif[:, 0, 0], time, event, 3.0, cause=1)
    assert abs(citl["predicted_mean"] - avg_of_pred) < 1e-12
    assert abs(citl["predicted_mean"] - pred_at_mean) > 0.01
    rows = grouped_calibration(cif[:, 0, 0], time, event, 3.0, cause=1)
    for r in rows:
        assert 0.0 <= r["predicted_mean"] <= 1.0


# ---------------------------------------------------------------------------
# 7. Reference agreement with riskRegression::Score
# ---------------------------------------------------------------------------

def test_7_matches_riskregression(data):
    """Cross-check AUC and Brier against R's riskRegression::Score, the
    standard implementation of the Blanche IPCW estimators."""
    time, event, x, cif = data
    arr = np.column_stack([time, event, cif[:, :, 0]])
    out = _rscript('''
suppressMessages({library(riskRegression); library(survival); library(prodlim)})
d <- read.csv("__CSV__"); te <- c(1,3,5)
risk <- as.matrix(d[, c("p1","p3","p5")])
s <- Score(list(m = risk), formula = Hist(time, event) ~ 1, data = d,
           times = te, cause = 1, metrics = c("auc","brier"),
           se.fit = FALSE, conf.int = FALSE, null.model = FALSE)
a <- s$AUC$score$AUC; b <- s$Brier$score$Brier
cat(paste(c(round(a, 10), round(b, 10)), collapse = ","))
''', (arr, "time,event,p1,p3,p5"))
    if out is None:
        pytest.skip("R/riskRegression unavailable")
    vals = np.array([float(z) for z in out.split(",")])
    r_auc, r_brier = vals[:3], vals[3:6]
    mine_auc = time_dependent_auc(cif, time, event, T_EVAL, causes=[1])[1]
    mine_bs = brier_score(cif, time, event, T_EVAL, causes=[1])[1]
    d_auc = np.abs(mine_auc - r_auc).max()
    d_bs = np.abs(mine_bs - r_brier).max()
    print(f"\n  vs riskRegression::Score")
    print(f"    AUC   mine {np.round(mine_auc,6)}  R {np.round(r_auc,6)}"
          f"  max diff {d_auc:.3e}")
    print(f"    Brier mine {np.round(mine_bs,6)}  R {np.round(r_brier,6)}"
          f"  max diff {d_bs:.3e}")
    assert d_auc < 1e-3
    assert d_bs < 1e-3


# ---------------------------------------------------------------------------
# 8. Bootstrap
# ---------------------------------------------------------------------------

def test_8_bootstrap_ci_sanity(data):
    """Coarse coverage check: intervals should contain the full-sample point
    estimate essentially always, and their width should shrink with n."""
    time, event, x, cif = data
    res = evaluate(cif, time, event, T_EVAL, n_boot=200, seed=SEED,
                   do_cindex=False, calibration=False)
    boot = res["bootstrap"]
    point = np.array(res["discrimination"]["auc"][1])
    lo = np.array(boot["auc_1"]["lo"])
    hi = np.array(boot["auc_1"]["hi"])
    print(f"\n  AUC cause 1 point {np.round(point,4)}")
    print(f"    95% CI lo {np.round(lo,4)} hi {np.round(hi,4)}  "
          f"width {np.round(hi-lo,4)}")
    assert np.all(lo <= point + 1e-9) and np.all(point <= hi + 1e-9)
    assert np.all(hi - lo < 0.15)

    # Coverage of the true AUC over repeated simulation (coarse, 60 reps).
    hits = 0
    reps = 60
    for r in range(reps):
        tt, ee, xx = simulate(n=1500, seed=1000 + r)
        cc = true_cif(xx, np.array([3.0]))
        rr = evaluate(cc, tt, ee, np.array([3.0]), n_boot=60, seed=r,
                      do_cindex=False, calibration=False)
        lo_r = rr["bootstrap"]["auc_1"]["lo"][0]
        hi_r = rr["bootstrap"]["auc_1"]["hi"][0]
        if lo_r <= point[1] <= hi_r:
            hits += 1
    cover = hits / reps
    print(f"    coverage of the large-sample AUC over {reps} reps: {cover:.2f}")
    assert 0.80 <= cover <= 1.0


# ---------------------------------------------------------------------------
# Entry point, subgroups, timing
# ---------------------------------------------------------------------------

def test_evaluate_entry_point(data):
    """One call per model per imputation returns a complete comparable record."""
    time, event, x, cif = data
    surv = 1.0 - cif.sum(axis=-1)
    res = evaluate(cif, time, event, T_EVAL, surv=surv, n_boot=50, seed=SEED)
    for key in ("meta", "discrimination", "calibration", "checks", "bootstrap"):
        assert key in res
    assert set(res["calibration"]) >= {"brier", "austin", "deciles", "citl"}
    assert res["checks"]["coherence"]["max_abs_deviation"] < 1e-9
    a = res["calibration"]["austin"][1][1]
    print(f"\n  evaluate(): cause1 @3y ICI {a['ICI']:.5f} E50 {a['E50']:.5f} "
          f"E90 {a['E90']:.5f} Emax {a['Emax']:.5f} [{a['method']}]")
    print(f"    CITL: {res['calibration']['citl'][1][1]}")
    assert len(res["calibration"]["deciles"][1][1]) == 10


def test_subgroup_evaluation(data):
    time, event, x, cif = data
    strata = quantile_strata(x[:, 0], n_groups=3, labels=["low", "mid", "high"])
    out = evaluate_by_subgroup(evaluate, cif, time, event, T_EVAL, strata,
                               n_boot=0, bootstrap=False, do_cindex=False)
    print("\n  subgroup   n     events(1/2)   ICI@3y")
    for name, res in out.items():
        ici = res["calibration"]["austin"][1][1]["ICI"]
        print(f"    {name:6s}  {res['n']:5d}   "
              f"{res['n_events'][1]:4d}/{res['n_events'][2]:4d}     {ici:.5f}")
        assert res["n"] > 0
    assert set(out) == {"low", "mid", "high"}


def test_timing_budget():
    """Report the benchmark budget: 50,000 patients x 3 horizons."""
    time, event, x = simulate(n=50_000, seed=11)
    cif = true_cif(x, T_EVAL, n_int=600)
    t0 = _time.time()
    evaluate(cif, time, event, T_EVAL, bootstrap=False, do_cindex=True)
    t_point = _time.time() - t0

    t0 = _time.time()
    evaluate(cif, time, event, T_EVAL, n_boot=500, seed=SEED,
             do_cindex=False, calibration=False)
    t_boot = _time.time() - t0

    t0 = _time.time()
    evaluate(cif[:5000], time[:5000], event[:5000], T_EVAL, n_boot=20,
             seed=SEED, do_cindex=True, calibration=False)
    t_c20 = _time.time() - t0
    print(f"\n  TIMING (50,000 patients x 3 horizons)")
    print(f"    point estimates, all metrics incl. C-index : {t_point:6.1f} s")
    print(f"    500 bootstrap replicates, AUC + Brier      : {t_boot:6.1f} s")
    print(f"    C-index bootstrap cost, extrapolated       : "
          f"{t_c20 / 20 * 500 * 10:6.0f} s at n=50,000 "
          f"(scales ~n log n; see README)")
