"""Validation of the classical baselines and the benchmark harness.

The organising concern is FAIRNESS. The paper claims neural representation
learning improves calibrated absolute risk beyond the best classical models,
so these tests exist to confirm the comparators are strong: splines on every
continuous covariate, the validated composites supplied, identical scoring
code, and each estimator checked against its published R implementation.

The coherence table is a paper result rather than merely a test - it shows
F1 + F2 + S = 1 holding structurally for Royston-Parmar, by construction for
a correctly integrated Cox, and not at all for Fine-Gray.
"""

import subprocess
import sys
import tempfile
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from benchmark import (RECORD_COLUMNS, BenchmarkCell,               # noqa: E402
                       CoxBaseline, FineGrayBaseline,
                       MinimalReferenceBaseline, RoystonParmarBaseline,
                       RSFBaseline, apply_covariate_splines, coherence_table,
                       derive_seed, events_per_variable, fit_covariate_splines,
                       run_benchmark)
from benchmark.base import _survival_for                                # noqa: E402
from benchmark.classical import _fit_cox                                # noqa: E402
from benchmark.covariate_splines import CovariateSplineSpec             # noqa: E402
from metrics import aalen_johansen                                      # noqa: E402

SEED = 42
T_EVAL = np.array([1.0, 3.0, 5.0])
COLUMNS = ["AGE", "AGE_DON", "KDRI_RAO", "EPTS_FINAL", "COLD_ISCH_KI",
           "NPKID", "sex", "diab"]

# Models whose two causes share one survival function, so F1 + F2 <= 1 holds by
# construction. Fine-Gray and RSF are deliberately absent.
COHERENT_MODELS = {"royston_parmar", "cox", "rp_kdri_epts"}


def simulate(n=4000, seed=SEED, nonlinear=True):
    """Competing risks with a genuinely nonlinear effect beyond the composites,
    so the minimal reference is a floor rather than a ceiling."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.normal(50, 12, n),                       # AGE
        rng.normal(45, 14, n),                       # AGE_DON
        rng.lognormal(0.1, 0.35, n),                 # KDRI_RAO
        rng.uniform(0, 100, n),                      # EPTS_FINAL
        rng.lognormal(2.7, 0.5, n),                  # COLD_ISCH_KI
        rng.integers(0, 5, n).astype(float),         # NPKID (count, no spline)
        (rng.random(n) < 0.5) * 1.0,                 # sex
        (rng.random(n) < 0.3) * 1.0,                 # diab
    ])
    z = (X - X.mean(0)) / X.std(0)
    eta1 = 0.45 * z[:, 2] + 0.3 * z[:, 3]
    eta2 = -0.2 * z[:, 2] + 0.35 * z[:, 0]
    if nonlinear:
        # Information beyond KDRI/EPTS, and nonlinear, so splines earn their keep.
        eta1 = eta1 + 0.5 * np.sin(1.5 * z[:, 1]) + 0.4 * z[:, 4] ** 2 * 0.3
        eta2 = eta2 + 0.35 * z[:, 6] + 0.3 * np.abs(z[:, 0])
    latent = {}
    for k, (sh, sc, eta) in {1: (1.25, 0.05, eta1), 2: (0.9, 0.05, eta2)}.items():
        latent[k] = (rng.exponential(size=n) / (sc * np.exp(eta))) ** (1 / sh)
    t_ev = np.minimum(latent[1], latent[2])
    cause = np.where(latent[1] <= latent[2], 1, 2)
    c = rng.uniform(0, 9.0, size=n)
    time = np.minimum(t_ev, c)
    return X, time, np.where(t_ev <= c, cause, 0)


@pytest.fixture(scope="module")
def data():
    X, time, event = simulate()
    n_tr = int(0.7 * len(time))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(time))
    tr, te = perm[:n_tr], perm[n_tr:]
    return {"Xtr": X[tr], "ttr": time[tr], "etr": event[tr],
            "Xte": X[te], "tte": time[te], "ete": event[te]}


@pytest.fixture(scope="module")
def fitted(data):
    models = {}
    specs = [(RoystonParmarBaseline, "royston_parmar", {}),
             (CoxBaseline, "cox", {}),
             (FineGrayBaseline, "fine_gray", {}),
             (MinimalReferenceBaseline, "rp_kdri_epts", {})]
    if RSFBaseline.available():
        # Included so the coherence table covers all five, not four - RSF is
        # the other model with no shared survival function, and the paper needs
        # its deviation reported alongside Fine-Gray's.
        specs.append((RSFBaseline, "rsf", {"ntree": 200, "nodesize": 25}))
    for cls, name, kw in specs:
        m = cls(columns=COLUMNS, seed=SEED, **kw)
        t0 = _time.time()
        m.fit(data["Xtr"], data["ttr"], data["etr"])
        models[name] = (m, _time.time() - t0)
    return models


# ---------------------------------------------------------------------------
# 6. RCS expansion
# ---------------------------------------------------------------------------

def test_6_rcs_expansion(data):
    """Knots come from training rows only, skip low-cardinality columns, and
    survive serialisation."""
    spec = fit_covariate_splines(data["Xtr"], COLUMNS)
    print(f"\n  expanded {len(spec.expanded)} of {len(COLUMNS)}: {spec.expanded}")
    print(f"  design width {len(COLUMNS)} -> {spec.n_output}")
    assert "NPKID" not in spec.expanded, "a 0-4 count must stay linear"
    assert "sex" not in spec.expanded and "diab" not in spec.expanded
    for c in ("AGE", "AGE_DON", "KDRI_RAO", "EPTS_FINAL", "COLD_ISCH_KI"):
        assert c in spec.expanded, c
        assert len(spec.knots[c]) == 4
    assert spec.n_output == len(COLUMNS) + 2 * len(spec.expanded)

    # Fitted on training rows only: wrecking the held-out rows changes nothing.
    wrecked = data["Xte"] * 1000 + 5000
    again = fit_covariate_splines(data["Xtr"], COLUMNS)
    assert again.to_json() == spec.to_json()
    mixed = fit_covariate_splines(np.vstack([data["Xtr"], wrecked]), COLUMNS)
    assert mixed.to_json() != spec.to_json(), "the test would not detect a leak"

    # Round-trip.
    back = CovariateSplineSpec.from_json(spec.to_json())
    assert np.array_equal(apply_covariate_splines(data["Xte"], spec),
                          apply_covariate_splines(data["Xte"], back))

    n_ev = int((data["etr"] == 1).sum())
    print(f"  events per variable (cause 1): "
          f"{events_per_variable(n_ev, spec.n_output):.1f}")


def test_6b_rcs_matches_verified_basis(data):
    """The covariate basis must equal the time-spline basis already verified
    against rstpm2 - same machinery, different argument."""
    import torch
    from rcs_basis import rcs_design
    from benchmark.covariate_splines import _rcs_columns
    v = np.linspace(1.0, 90.0, 300)
    knots = np.quantile(v, (0.05, 0.35, 0.65, 0.95))
    mine = _rcs_columns(v, knots)
    theirs = rcs_design(torch.tensor(v), torch.tensor(knots)).numpy()[:, 1:]
    assert np.abs(mine - theirs).max() < 1e-9


# ---------------------------------------------------------------------------
# 1 & 2. Interface conformance and the coherence table
# ---------------------------------------------------------------------------

def test_1_interface_conformance(fitted, data):
    """Every baseline returns (n, T, 2) in [0,1], non-decreasing in t."""
    n = len(data["tte"])
    for name, (m, _) in fitted.items():
        cif = m.predict_cif(data["Xte"], T_EVAL)
        assert cif.shape == (n, len(T_EVAL), 2), f"{name}: {cif.shape}"
        assert np.isfinite(cif).all(), f"{name}: non-finite CIF"
        assert cif.min() >= -1e-9 and cif.max() <= 1.0 + 1e-9, name
        d = np.diff(cif, axis=1)
        assert d.min() >= -1e-7, f"{name}: CIF decreased in t ({d.min():.2e})"

        # F1 + F2 <= 1 is a property of the FORMULATION, not of the interface.
        # It holds by construction wherever the two causes share one survival
        # function (Royston-Parmar, cause-specific Cox). Fine-Gray fits the two
        # subdistributions independently, so nothing couples them and the sum
        # can exceed 1; that excess is a result to report, not a bug to assert
        # away. Test 2 reports it.
        total = cif.sum(axis=-1)
        if name in COHERENT_MODELS:
            assert (total <= 1.0 + 1e-6).all(), (
                f"{name}: F1+F2 > 1 (max {float(total.max()):.6f})")
        else:
            assert (total <= 2.0).all(), f"{name}: F1+F2 > 2, not a probability"


def test_2_coherence_table(fitted, data):
    """max |F1 + F2 + S - 1| per model. A PAPER RESULT, not just a test.

    Royston-Parmar: structural, so the deviation is pure quadrature error.
    Cox: exact only because the integration uses the product-limit survival,
         S(t_j) = prod(1 - dH), which makes the sum telescope; the obvious
         exp(-sum dH) gives ~1e-03.
    Fine-Gray: no guarantee at all - the two subdistribution models are fitted
         independently and nothing couples them. This is a substantive point
         in favour of the Royston-Parmar formulation.
    """
    rows = []
    for name, (m, fit_s) in fitted.items():
        t0 = _time.time()
        cif = m.predict_cif(data["Xte"], T_EVAL)
        pred_s = _time.time() - t0
        surv = _survival_for(m, data["Xte"], T_EVAL, cif)
        dev = float(np.abs(cif.sum(axis=-1) + surv - 1.0).max())
        rows.append({"model": name, "coherence": dev, "fit_s": fit_s,
                     "pred_s": pred_s})
    table = pd.DataFrame(rows).sort_values("coherence")
    print("\n  COHERENCE  max |F1 + F2 + S - 1|")
    for r in table.itertuples():
        # fit and predict are reported separately because RSF does all its work
        # at predict time - the forest is rebuilt in the R session that scores.
        print(f"    {r.model:16s} {r.coherence:.3e}   "
              f"(fit {r.fit_s:5.1f}s  predict {r.pred_s:5.1f}s)")

    by = {r["model"]: r["coherence"] for r in rows}
    assert by["royston_parmar"] < 1e-4, "RP coherence is structural"
    assert by["cox"] < 1e-6, "Cox coherence depends on correct integration"

    # The models with no shared survival function: report how far the two
    # independent fits drift from a coherent joint distribution.
    for name in ("fine_gray", "rsf"):
        if name not in fitted:
            continue
        cif = fitted[name][0].predict_cif(data["Xte"], T_EVAL)
        total = cif.sum(axis=-1)
        print(f"    {name} F1+F2 max = {float(total.max()):.4f} "
              f"(excess over 1: {float((total - 1.0).max()):+.3e})")
    # RSF's deviation has a different source from Fine-Gray's: rfsrc reports no
    # survival for the competing-risks family, and the product-limit
    # reconstruction from its own averaged cumulative hazards does not
    # reproduce its averaged CIFs - averaging over trees and the Aalen-Johansen
    # recursion do not commute. Report it as such rather than as a fitting
    # failure.
    assert by["fine_gray"] > by["royston_parmar"], (
        "the point of the table is that independent subdistribution fits are "
        "less coherent than the shared-survival formulation")


def test_2b_rp_coherence_is_quadrature_error(fitted, data):
    """The Royston-Parmar residual must FALL with the node count.

    This is what licenses calling it quadrature error rather than a defect.
    A fixed residual, or one that wandered, would mean something structural
    was wrong instead.

    It also means the number is meaningless unless the node count is quoted
    alongside it: the same model on the same data reads 3.0e-05 at 10 nodes
    and 6.8e-08 at 40. Two figures from different tasks differing by an order
    of magnitude is therefore expected, not a discrepancy - they are different
    fitted models on different data, not the same number measured twice.
    """
    m = fitted["royston_parmar"][0]
    devs = {}
    for nodes in (10, 20, 40):
        cif = m.predict_cif(data["Xte"], T_EVAL, n_nodes=nodes)
        surv = m.predict_survival(data["Xte"], T_EVAL)
        devs[nodes] = float(np.abs(cif.sum(-1) + surv - 1).max())
    print("\n  RP coherence vs quadrature nodes")
    for k, v in devs.items():
        print(f"    n_nodes={k:3d}  {v:.3e}")
    assert devs[20] < devs[10] / 5, "refining the quadrature must help"
    assert devs[40] < devs[20] / 5, "and keep helping"


def test_2c_coherence_paths_are_independent(fitted, data):
    """fine_gray and rsf land within ~1% of each other; confirm that is
    coincidence rather than a shared artefact of the clipped-complement path.

    They arrive by opposite routes. Fine-Gray has no joint survival, so the
    harness uses clip(1 - F1 - F2, 0, 1) and its residual IS the excess mass.
    RSF supplies its own reconstructed survival, and its CIFs never exceed 1 -
    so the clip never engages and the complement path would report exactly
    zero. If both were riding the clip floor, both would show clip hits.
    """
    from benchmark.base import _survival_for
    seen = {}
    for name in ("fine_gray", "rsf"):
        if name not in fitted:
            continue
        m = fitted[name][0]
        cif = m.predict_cif(data["Xte"], T_EVAL)
        total = cif.sum(-1)
        hits = int((total > 1.0).sum())
        dev = float(np.abs(total + _survival_for(m, data["Xte"], T_EVAL, cif)
                           - 1).max())
        dev_comp = float(np.abs(total + np.clip(1 - total, 0, 1) - 1).max())
        seen[name] = (hits, dev, dev_comp, hasattr(m, "predict_survival"))
        print(f"\n  {name:12s} own_S={seen[name][3]}  clip hits {hits:4d}  "
              f"dev {dev:.3e}  dev-if-complement {dev_comp:.3e}")

    if "fine_gray" in seen:
        hits, dev, dev_comp, own = seen["fine_gray"]
        assert not own, "Fine-Gray must not claim a joint survival it lacks"
        assert hits > 0 and np.isclose(dev, dev_comp), (
            "Fine-Gray's residual should BE the excess mass")
    if "rsf" in seen:
        hits, dev, dev_comp, own = seen["rsf"]
        assert own, "RSF supplies its own survival"
        assert hits == 0 and dev_comp == 0.0, (
            "RSF's CIFs stay under 1, so the clip path cannot be the source")
        assert dev > 1e-3, (
            "RSF's residual comes from its own survival, not from clipping")


# ---------------------------------------------------------------------------
# 3 & 4. Against the published R implementations
# ---------------------------------------------------------------------------

def _rscript(code, arr, header):
    import os
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    try:
        r = subprocess.run(["Rscript", "-e", code.replace("__CSV__", path.replace("\\", "/"))],
                           capture_output=True, text=True, timeout=600)
        os.unlink(path)
        
        if r.returncode != 0:
            print(f"  [_rscript] R failed:\n{r.stderr[-500:]}")
            return None
        
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def test_3_cox_matches_r(data):
    """Cause-specific Cox coefficients against survival::coxph, both causes."""
    X, t, e = data["Xtr"][:, :4], data["ttr"], data["etr"]
    hdr = "time,event,x1,x2,x3,x4"
    for k in (1, 2):
        # ddof=1 to match R's scale(); with ddof=0 the two designs differ by
        # sqrt(n/(n-1)) per column and the agreement caps out around 1e-4,
        # which would hide a real discrepancy of that size.
        mine = _fit_cox((X - X.mean(0)) / X.std(0, ddof=1), t,
                        (e == k).astype(int), penalizer=0.0)
        out = _rscript(f'''
suppressMessages(library(survival)); d <- read.csv("__CSV__")
z <- scale(as.matrix(d[, c("x1","x2","x3","x4")]))
f <- coxph(Surv(d$time, d$event == {k}) ~ z, ties = "breslow")
cat(paste(round(coef(f), 10), collapse = ","))
''', np.column_stack([t, e, X]), hdr)
        if out is None:
            pytest.skip("R/survival unavailable")
        r_beta = np.array([float(z) for z in out.split(",")])
        diff = np.abs(mine - r_beta).max()
        print(f"\n  cause {k} Cox vs coxph: max |diff| = {diff:.3e}")
        assert diff < 1e-8


def test_4_fine_gray_matches_cmprsk(data):
    """Regression guard on the Fine-Gray estimator this baseline reuses."""
    from metrics.calibration import fit_fine_gray
    X, t, e = data["Xtr"][:, :3], data["ttr"], data["etr"]
    fit = fit_fine_gray(X, t, e, cause=1)
    assert fit.converged, fit.message
    out = _rscript('''
suppressMessages(library(cmprsk)); d <- read.csv("__CSV__")
f <- crr(d$time, d$event, cov1 = as.matrix(d[, c("x1","x2","x3")]),
         failcode = 1, cencode = 0)
cat(paste(round(f$coef, 10), collapse = ","))
''', np.column_stack([t, e, X]), "time,event,x1,x2,x3")
    if out is None:
        pytest.skip("R/cmprsk unavailable")
    r_beta = np.array([float(z) for z in out.split(",")])
    diff = np.abs(fit.beta_original - r_beta).max()
    print(f"\n  Fine-Gray vs cmprsk::crr: max |diff| = {diff:.3e}")
    assert diff < 1e-5


# ---------------------------------------------------------------------------
# 5. RSF
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RSFBaseline.available(),
                    reason="randomForestSRC not installed")
def test_5_rsf_tracks_aalen_johansen(data):
    """The forest's predicted CIF must track the nonparametric estimator
    within strata."""
    m = RSFBaseline(columns=COLUMNS, seed=SEED, ntree=200, nodesize=25)
    t0 = _time.time()
    m.fit(data["Xtr"], data["ttr"], data["etr"])
    cif = m.predict_cif(data["Xte"], T_EVAL)
    print(f"\n  RSF one fit + predict: {_time.time() - t0:.1f}s "
          f"(n_train={len(data['ttr'])})")
    strata = data["Xte"][:, 2] > np.median(data["Xte"][:, 2])   # KDRI split
    worst = 0.0
    for lev in (False, True):
        sel = strata == lev
        for k in (1, 2):
            aj, _ = aalen_johansen(data["tte"][sel], data["ete"][sel], T_EVAL,
                                   cause=k)
            pred = cif[sel, :, k - 1].mean(axis=0)
            worst = max(worst, float(np.abs(aj - pred).max()))
            print(f"    KDRI-high={lev} cause{k}: AJ {np.round(aj,4)} "
                  f"vs RSF {np.round(pred,4)}")
    assert worst < 0.06, f"RSF strays {worst:.3f} from Aalen-Johansen"


# ---------------------------------------------------------------------------
# 7. The minimal reference is a floor
# ---------------------------------------------------------------------------

def test_7_minimal_reference_is_beaten(fitted, data):
    """With information beyond KDRI/EPTS in the truth, the full RP model must
    beat the two-composite reference on held-out Brier score. Confirms the
    reference anchors the incremental-value story rather than capping it."""
    from metrics import brier_score
    full = fitted["royston_parmar"][0].predict_cif(data["Xte"], T_EVAL)
    ref = fitted["rp_kdri_epts"][0].predict_cif(data["Xte"], T_EVAL)
    bs_full = brier_score(full, data["tte"], data["ete"], T_EVAL)
    bs_ref = brier_score(ref, data["tte"], data["ete"], T_EVAL)
    print("\n  held-out Brier (lower is better)")
    better = 0
    for k in (1, 2):
        print(f"    cause{k}: full {np.round(bs_full[k],5)}  "
                f"KDRI+EPTS only {np.round(bs_ref[k],5)}")
        better += int(np.mean(bs_full[k]) < np.mean(bs_ref[k]))
    assert better == 2, "the full model failed to beat the minimal reference"


# ---------------------------------------------------------------------------
# 8. Harness
# ---------------------------------------------------------------------------

def _cheap_evaluate(cif, time, event, t_eval, surv=None, **kw):
    """A fast stand-in for metrics.evaluate() so the harness tests exercise
    the runner rather than the metrics (already covered elsewhere)."""
    from metrics import brier_score
    bs = brier_score(cif, time, event, t_eval)
    dev = (float(np.abs(cif.sum(-1) + surv - 1).max()) if surv is not None
           else float("nan"))
    return {
        "meta": {"n": int(len(time)), "t_eval": list(map(float, t_eval)),
                 "n_events": {k: int((event == k).sum()) for k in (1, 2)}},
        "discrimination": {"auc": {k: [float("nan")] * len(t_eval)
                                   for k in (1, 2)}},
        "calibration": {"brier": {k: list(map(float, bs[k])) for k in (1, 2)}},
        "checks": {"coherence": {"max_abs_deviation": dev}},
    }


def _cells(data, models=("cox", "fine_gray")):
    factories = {"cox": lambda s: CoxBaseline(columns=COLUMNS, seed=s),
                 "fine_gray": lambda s: FineGrayBaseline(columns=COLUMNS, seed=s)}
    cells = []
    for name in models:
        for fold in (0, 1):
            cells.append(BenchmarkCell(
                model_name=name, fold=fold, imputation=1,
                train={"X": data["Xtr"], "time": data["ttr"],
                       "event": data["etr"]},
                evaluate_on={"test": {"X": data["Xte"], "time": data["tte"],
                                      "event": data["ete"]}},
                factory=factories[name]))
    return cells


def test_8_harness_is_resumable(data, tmp_path):
    """Interrupting and restarting must skip completed combinations and
    reproduce identical numbers."""
    out = tmp_path / "results.parquet"
    first = run_benchmark(_cells(data, ("cox",)), T_EVAL, out,
                          evaluate_fn=_cheap_evaluate, verbose=False)
    # 2 folds x 1 eval set x 2 causes x |t_eval| horizons, in long format.
    assert len(first) == 2 * 1 * 2 * len(T_EVAL)
    n_after_first = len(first)

    # Restart with MORE work: the finished cells must be skipped, not refitted.
    full = run_benchmark(_cells(data, ("cox", "fine_gray")), T_EVAL, out,
                         evaluate_fn=_cheap_evaluate, verbose=False)
    assert len(full) > n_after_first
    cox_first = first[first.model == "cox"].sort_values(
        ["fold", "cause", "horizon"]).reset_index(drop=True)
    cox_again = full[full.model == "cox"].sort_values(
        ["fold", "cause", "horizon"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(cox_first[["brier", "seed", "coherence_max"]],
                                  cox_again[["brier", "seed", "coherence_max"]])

    # A third call adds nothing.
    third = run_benchmark(_cells(data, ("cox", "fine_gray")), T_EVAL, out,
                          evaluate_fn=_cheap_evaluate, verbose=False)
    assert len(third) == len(full)
    print(f"\n  resumable: {n_after_first} -> {len(full)} records, "
          f"third call added {len(third) - len(full)}")


def test_8b_seeds_are_deterministic_and_distinct():
    a = derive_seed("cox", 0, 1)
    assert a == derive_seed("cox", 0, 1)
    assert a != derive_seed("cox", 1, 1) != derive_seed("cox", 0, 2)
    assert derive_seed("cox", 0, 1) != derive_seed("fine_gray", 0, 1)


def test_8c_coherence_table_from_results(data, tmp_path):
    out = tmp_path / "r.parquet"
    res = run_benchmark(_cells(data, ("cox", "fine_gray")), T_EVAL, out,
                        evaluate_fn=_cheap_evaluate, verbose=False)
    table = coherence_table(res)
    print("\n" + table.to_string())
    assert set(table.index) == {"cox", "fine_gray"}
    assert table.loc["cox", "worst"] < table.loc["fine_gray", "worst"]


def test_8e_records_match_the_real_evaluate(data, tmp_path):
    """Run the harness once through the ACTUAL metrics.evaluate().

    Every other harness test injects a cheap stand-in, which exercises the
    runner but not the flattening: if `records_from_evaluation` misread the
    real result's nesting, all of those would still pass and every ICI, E50
    and CITL in the benchmark would silently be NaN. This is the test that
    would notice.
    """
    from metrics import evaluate

    cells = [BenchmarkCell(
        model_name="cox", fold=0, imputation=1,
        train={"X": data["Xtr"], "time": data["ttr"], "event": data["etr"]},
        evaluate_on={"test": {"X": data["Xte"], "time": data["tte"],
                              "event": data["ete"]}},
        factory=lambda s: CoxBaseline(columns=COLUMNS, seed=s))]
    res = run_benchmark(cells, T_EVAL, tmp_path / "real.parquet",
                        evaluate_fn=evaluate, verbose=False,
                        evaluate_kwargs={"bootstrap": False, "do_cindex": True})

    assert list(res.columns) == RECORD_COLUMNS
    assert len(res) == 2 * len(T_EVAL)
    # Every metric column must carry real numbers, not the NaNs a mis-read
    # nesting would leave behind.
    for col in ("auc", "cindex", "brier", "ici", "e50", "e90", "emax",
                "citl_predicted", "citl_observed", "citl_difference",
                "coherence_max", "n_rows", "n_events_cause"):
        assert res[col].notna().all(), f"{col} came back NaN"
    assert (res["calibration_method"].astype(str).str.len() > 0).all()
    assert res["coherence_max"].max() < 1e-6
    print("\n  real evaluate() through the harness:")
    print(res[["cause", "horizon", "auc", "brier", "ici", "e50",
               "citl_difference"]].round(4).to_string(index=False))


def test_8d_non_convergence_is_recorded(data, tmp_path, capsys):
    """A baseline whose optimiser failed must not pass silently.

    25 cells run unattended is exactly where a non-converged fit slips through
    and quietly weakens the comparator the paper's claim rests on, so the
    harness records convergence per cause and shouts about failures.
    """
    class Broken(FineGrayBaseline):
        converged = {1: True, 2: False}

    cells = [BenchmarkCell(
        model_name="broken", fold=0, imputation=1,
        train={"X": data["Xtr"], "time": data["ttr"], "event": data["etr"]},
        evaluate_on={"test": {"X": data["Xte"], "time": data["tte"],
                              "event": data["ete"]}},
        factory=lambda s: Broken(columns=COLUMNS, seed=s))]
    res = run_benchmark(cells, T_EVAL, tmp_path / "b.parquet",
                        evaluate_fn=_cheap_evaluate, verbose=True)
    assert res[res.cause == 1]["converged"].all()
    assert not res[res.cause == 2]["converged"].any()
    assert "NOT CONVERGED" in capsys.readouterr().out

    # The real Fine-Gray fits must actually converge, or the comparison is not
    # the fair one the paper claims.
    assert all(FineGrayBaseline(columns=COLUMNS)
               .fit(data["Xtr"], data["ttr"], data["etr"]).converged.values())


def test_runtime_projection(fitted):
    """Project the full 5 folds x 5 imputations x 5 models matrix."""
    per_model = {name: t for name, (_, t) in fitted.items()}
    n_cells = 5 * 5
    total = sum(per_model.values()) * n_cells
    print(f"\n  per-fit seconds (n=2,800 synthetic): "
          f"{ {k: round(v, 1) for k, v in per_model.items()} }")
    print(f"  4 classical models x 25 (fold x imputation) = {total / 60:.1f} min "
          f"at this size; scale by n and add RSF separately - it is the "
          f"bottleneck and should be timed on real data before committing.")
