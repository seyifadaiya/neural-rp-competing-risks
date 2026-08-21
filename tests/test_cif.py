"""Validation of the cumulative incidence functions.

The CIF is the model's primary clinical output - absolute risk of each
competing event - and the one quantity that needs quadrature, since training
is closed form. The headline check is COHERENCE: F_1 + F_2 + S = 1 holds
exactly in the model by construction, so any measured deviation is quadrature
error and nothing else. Nothing in `cif.py` normalises or rescales to enforce
it, which is what makes the number meaningful.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))

import cif as C                                                        # noqa: E402
from gen_data import generate                                          # noqa: E402
from train import TrainConfig, set_seed, train_model                   # noqa: E402
from test_variant_a import as_batch, build_variant_a, knots_for        # noqa: E402
from test_variant_b import build_variant_b                             # noqa: E402

SEED = 42


def _train(x, time, event, variant="B", **kw):
    knots = {k: knots_for(time, event, k) for k in (1, 2)}
    batch = as_batch(time, event, x)
    set_seed(SEED)
    build = build_variant_b if variant == "B" else build_variant_a
    model = build(knots, x, hidden_dims=(64, 32), latent_dim=16, **kw)
    train_model(model, batch, batch,
                TrainConfig(max_epochs=300, batch_size=512, patience=40,
                            scheduler="cosine", seed=SEED, verbose=False))
    model.eval()
    return model, batch


@pytest.fixture(scope="module")
def fitted():
    time, event, x = generate()
    model, batch = _train(x, time, event)
    return model, batch, time, event, x


@pytest.fixture(scope="module")
def dense_grid():
    return torch.tensor(np.exp(np.linspace(np.log(0.05), np.log(8.0), 25)))


# ---------------------------------------------------------------------------
# 1. Coherence - the headline number
# ---------------------------------------------------------------------------

def test_1_coherence(fitted, dense_grid):
    """max |F_1(t) + F_2(t) + S(t) - 1| over 500 patients x a dense grid.

    Exact in the model by construction (d/dt[F_1+F_2+S] = 0), so this measures
    quadrature error alone. It is NOT enforced anywhere in cif.py.
    """
    model, batch, *_ = fitted
    x = batch["x_cont"][:500]
    dev = C.coherence_deviation(model, x, dense_grid)
    print(f"\n  COHERENCE: max |F1 + F2 + S - 1| = {dev:.3e} "
          f"({x.shape[0]} patients x {dense_grid.numel()} times, "
          f"n_nodes={C.DEFAULT_NODES})")
    assert dev < 1e-4


def test_1b_coherence_variant_a(dense_grid):
    """Coherence holds for Variant A too - the CIF takes dlogH/dv from the
    model, so the shared-shape and patient-specific-shape cases are handled
    without branching."""
    time, event, x = generate()
    model, batch = _train(x, time, event, variant="A")
    dev = C.coherence_deviation(model, batch["x_cont"][:500], dense_grid)
    print(f"  Variant A coherence: {dev:.3e}")
    assert dev < 1e-4


# ---------------------------------------------------------------------------
# 2. Node-count convergence - justifies DEFAULT_NODES
# ---------------------------------------------------------------------------

def test_2_node_count_convergence(fitted):
    """Sweep n_nodes and confirm the error falls then plateaus, and that
    DEFAULT_NODES is the smallest count on the plateau."""
    model, batch, *_ = fitted
    x = batch["x_cont"][:50]
    t_grid = torch.tensor([0.5, 1.0, 3.0, 5.0, 8.0], dtype=torch.float64)
    counts = [5, 10, 15, 20, 30, 40, 60, 80, 120]

    ref, _ = C.predict_cif(model, x, t_grid, n_nodes=240)
    errs, devs = {}, {}
    for n in counts:
        got, _ = C.predict_cif(model, x, t_grid, n_nodes=n)
        errs[n] = float((got - ref).abs().max())
        devs[n] = C.coherence_deviation(model, x, t_grid, n_nodes=n)
    print("\n  n_nodes   |GL(n)-GL(240)|   coherence")
    for n in counts:
        print(f"   {n:5d}      {errs[n]:.3e}      {devs[n]:.3e}")

    # Falls: coarse quadrature must be visibly worse than the default.
    assert errs[5] > errs[C.DEFAULT_NODES] * 10
    # Plateaus: beyond the default, nothing improves materially, because the
    # floor is float32 model output rather than the quadrature.
    plateau = [errs[n] for n in counts if n >= C.DEFAULT_NODES]
    assert max(plateau) < 1e-6, f"no plateau reached: {plateau}"
    # The default is ON the plateau, and the next count down is not.
    below = [n for n in counts if n < C.DEFAULT_NODES]
    assert errs[max(below)] > max(plateau), (
        f"DEFAULT_NODES={C.DEFAULT_NODES} is not the smallest on the plateau")


# ---------------------------------------------------------------------------
# 3. Independent integration method
# ---------------------------------------------------------------------------

def test_3_matches_fine_trapezoid(fitted):
    """Gauss-Legendre vs an independent 10,000-point trapezoid on the log
    scale, integrated in ONE pass (splitting the trapezoid into chunks and
    summing drops the linking sub-interval at each boundary, which silently
    degrades the reference as the point count grows)."""
    model, batch, *_ = fitted
    x = batch["x_cont"][:20]
    t_grid = torch.tensor([0.5, 1.0, 3.0, 5.0], dtype=torch.float64)
    v_lo = C._auto_lower_limit(model, x)
    n_pts, n_pat = 10_000, x.shape[0]

    ref = torch.zeros(n_pat, t_grid.numel(), 2, dtype=torch.float64)
    with torch.no_grad():
        for j, tt in enumerate(t_grid):
            vg = torch.linspace(v_lo, float(torch.log(tt)), n_pts,
                                dtype=torch.float64)
            log_H, dlogH = C._model_at(model, vg.repeat(n_pat),
                                       x.repeat_interleave(n_pts, dim=0))
            log_H = log_H.reshape(n_pat, n_pts, 2)
            dlogH = dlogH.reshape(n_pat, n_pts, 2)
            H = torch.exp(log_H)
            S = torch.exp(-H.sum(dim=-1))
            ref[:, j, :] = torch.trapezoid(H * dlogH * S.unsqueeze(-1), vg, dim=1)

    got, _ = C.predict_cif(model, x, t_grid)
    diff = float((got - ref).abs().max())
    print(f"\n  max |GL({C.DEFAULT_NODES}) - trapezoid(10,000)| = {diff:.3e}")
    assert diff < 1e-5


# ---------------------------------------------------------------------------
# 4. Lower truncation
# ---------------------------------------------------------------------------

def test_4_lower_truncation_is_safe(fitted):
    """The chosen lower limit must discard a negligible tail.

    Worth stating what this test caught: truncating at the smallest BOUNDARY
    KNOT - the natural choice, and what the design originally assumed - is
    NOT safe. sum_k H_k there is ~1e-4 for a fitted model, and since
    H_k dlogH_k/dv = dH_k/dv the discarded tail carries about that much
    probability mass. It showed up as a coherence deviation of 1.3e-4 that was
    completely FLAT in the node count - the signature of a domain error rather
    than a quadrature error. `_auto_lower_limit` now steps the limit down
    until the model itself reports the remaining mass below 1e-10.
    """
    model, batch, *_ = fitted
    x = batch["x_cont"][:20]
    t_grid = torch.tensor([0.5, 1.0, 3.0, 5.0], dtype=torch.float64)

    auto = C._auto_lower_limit(model, x)
    knot = min(float(model.basis[k].knots[0]) for k in (1, 2))
    default_cif, _ = C.predict_cif(model, x, t_grid)
    deeper_cif, _ = C.predict_cif(model, x, t_grid, v_lo=-60.0)
    knot_cif, _ = C.predict_cif(model, x, t_grid, v_lo=knot)

    tail_default = float((default_cif - deeper_cif).abs().max())
    tail_knot = float((knot_cif - deeper_cif).abs().max())
    print(f"\n  auto v_lo = {auto:.1f} (t = {np.exp(auto):.2e}); "
          f"boundary knot = {knot:.3f}")
    print(f"  tail discarded by auto limit:      {tail_default:.3e}")
    print(f"  tail discarded by boundary knot:   {tail_knot:.3e}  "
          f"(the assumption that failed)")
    assert tail_default < 1e-6


# ---------------------------------------------------------------------------
# 5. Single-cause reduction
# ---------------------------------------------------------------------------

def test_5_single_cause_reduction(fitted):
    """With cause 2 switched off, F_1(t) must reduce to 1 - exp(-H_1(t)),
    the standard single-event identity."""
    model, batch, *_ = fitted
    import copy
    solo = copy.deepcopy(model)
    with torch.no_grad():                     # push H_2 to ~0
        solo.a2.bias.fill_(-60.0)
        solo.a2.weight.zero_()
    solo.eval()

    x = batch["x_cont"][:100]
    t_grid = torch.tensor([0.5, 1.0, 3.0, 5.0, 8.0], dtype=torch.float64)
    cif, _ = C.predict_cif(solo, x, t_grid)
    log_H = C.predict_cumhaz(solo, x, t_grid)
    expected = 1.0 - torch.exp(-torch.exp(log_H[:, :, 0]))
    diff = float((cif[:, :, 0] - expected).abs().max())
    print(f"\n  max |F_1 - (1 - exp(-H_1))| with cause 2 off = {diff:.3e}")
    assert diff < 1e-6
    assert float(cif[:, :, 1].abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# 6. Monotonicity and bounds
# ---------------------------------------------------------------------------

def test_6_monotonic_and_bounded(fitted, dense_grid):
    model, batch, *_ = fitted
    x = batch["x_cont"][:500]
    cif, surv = C.predict_cif(model, x, dense_grid)
    dF = cif[:, 1:, :] - cif[:, :-1, :]
    dS = surv[:, 1:] - surv[:, :-1]
    print(f"\n  min dF = {float(dF.min()):.3e}, max dS = {float(dS.max()):.3e}, "
          f"F range [{float(cif.min()):.3e}, {float(cif.max()):.4f}]")
    assert float(dF.min()) >= -1e-9, "CIF decreased in t"
    assert float(dS.max()) <= 1e-9, "survival increased in t"
    assert float(cif.min()) >= -1e-9 and float(cif.max()) <= 1.0
    assert float(surv.min()) >= 0.0 and float(surv.max()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 7 & 8. Nonparametric reference (Aalen-Johansen)
# ---------------------------------------------------------------------------

def aalen_johansen(time, event, t_eval):
    """F_k(t) = sum_{t_i <= t} (d_ki / n_i) S(t_{i-1}), the nonparametric
    Aalen-Johansen estimator. Cross-checked against R's cmprsk::cuminc in
    test_7."""
    order = np.argsort(time)
    time, event = np.asarray(time)[order], np.asarray(event)[order]
    uniq = np.unique(time[event > 0])
    surv, cum = 1.0, {1: 0.0, 2: 0.0}
    ts, curves = [], {1: [], 2: []}
    for u in uniq:
        at_risk = (time >= u).sum()
        d = {k: ((time == u) & (event == k)).sum() for k in (1, 2)}
        for k in (1, 2):
            cum[k] += surv * d[k] / at_risk
        surv *= 1 - (d[1] + d[2]) / at_risk
        ts.append(u)
        for k in (1, 2):
            curves[k].append(cum[k])
    ts = np.asarray(ts)
    return {k: np.interp(t_eval, ts, np.asarray(curves[k]), left=0.0)
            for k in (1, 2)}


def _cmprsk_cuminc(time, event, t_eval):
    """R cmprsk::cuminc reference, or None if R/cmprsk is unavailable."""
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    rpath = path.replace("\\", "/")
    np.savetxt(path, np.column_stack([time, event]), delimiter=",",
               header="time,event", comments="")
    te = ",".join(str(float(v)) for v in t_eval)
    script = f'''
suppressMessages(library(cmprsk)); d <- read.csv("{rpath}")
ci <- cuminc(d$time, d$event, cencode = 0); te <- c({te})
f1 <- approx(ci[["1 1"]]$time, ci[["1 1"]]$est, te, rule = 2)$y
f2 <- approx(ci[["1 2"]]$time, ci[["1 2"]]$est, te, rule = 2)$y
cat(paste(c(f1, f2), collapse = ","))
'''
    try:
        r = subprocess.run(["Rscript", "-e", script], capture_output=True,
                           text=True, timeout=300)
        os.unlink(path)
        
        if r.returncode != 0:
            print(f"  [_cmprsk_cuminc] R failed:\n{r.stderr[-500:]}")
            return None
        if not r.stdout.strip():
            print("  [_cmprsk_cuminc] R produced no output")
            return None
        vals = np.array([float(z) for z in r.stdout.strip().split(",")])
        return {1: vals[:len(t_eval)], 2: vals[len(t_eval):]}
    except Exception as exc:
        print(f"  [_cmprsk_cuminc] {type(exc).__name__}: {exc}")
        return None
        
        """if r.returncode != 0 or not r.stdout.strip():
            return None
        vals = np.array([float(z) for z in r.stdout.strip().split(",")])
        return {1: vals[:len(t_eval)], 2: vals[len(t_eval):]}
    except Exception:
        return None"""


def test_7_against_r_reference():
    """Comparison against an independent published implementation.

    ROUTE TAKEN, AND WHY. `stpm2cr` is a Stata command; there is no R port,
    and rstpm2 (built from source here) does not export it. `flexsurv` is not
    installable in this environment either - CRAN's archive paths are blocked
    by the egress policy and it is not in the Ubuntu r-cran set, along with
    its mstate/muhaz dependencies. So the parametric R route is genuinely not
    workable here, and this test takes the sanctioned fallback: R's
    `cmprsk::cuminc`, the standard published implementation of the
    nonparametric Aalen-Johansen estimator.

    The fallback is strengthened by STRATIFYING on a binary covariate and
    comparing within strata, so it checks the model's covariate conditioning
    rather than only its marginal curve. The model's stratum CIF is the mean
    of F_k(t | x_i) over that stratum's patients - the correct comparison to a
    marginal nonparametric estimator.
    """
    time, event, x = generate(n=20_000, seed=11)
    model, batch = _train(x, time, event)
    t_eval = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0])

    ref = _cmprsk_cuminc(time, event, t_eval)
    if ref is None:
        pytest.skip("R/cmprsk unavailable")
    mine = aalen_johansen(time, event, t_eval)
    agree = max(np.abs(mine[k] - ref[k]).max() for k in (1, 2))
    print(f"\n  python AJ vs R cmprsk::cuminc: max abs diff = {agree:.3e}")
    assert agree < 1e-3, "the Python Aalen-Johansen implementation disagrees with R"

    # Stratified comparison: does the model condition on covariates correctly?
    strata = x[:, 2]
    for level in (0.0, 1.0):
        sel = strata == level
        ref_s = _cmprsk_cuminc(time[sel], event[sel], t_eval)
        cif, _ = C.predict_cif(model, batch["x_cont"][sel],
                               torch.tensor(t_eval), batch_size=2048)
        marg = cif.mean(dim=0)
        worst = max(float(np.abs(ref_s[k] - marg[:, k - 1].numpy()).max())
                    for k in (1, 2))
        print(f"  stratum x3={level:.0f} (n={int(sel.sum())}): "
              f"max |model - cmprsk AJ| = {worst:.3e}")
        assert worst < 1e-2


def test_8_aalen_johansen_sanity():
    """The model's marginal CIF must track the nonparametric estimator on
    covariate-free data - an assumption-free reference."""
    rng = np.random.default_rng(3)
    n = 20_000
    shape, scale = {1: 1.3, 2: 0.9}, {1: 0.042, 2: 0.085}
    latent = {k: (rng.exponential(size=n) / scale[k]) ** (1 / shape[k])
              for k in (1, 2)}
    t_ev = np.minimum(latent[1], latent[2])
    cause = np.where(latent[1] <= latent[2], 1, 2)
    c = rng.uniform(0, 9.0, size=n)
    time = np.minimum(t_ev, c)
    event = np.where(t_ev <= c, cause, 0)
    x = rng.standard_normal((n, 4)) * 0.0 + 1.0      # constant covariates

    model, batch = _train(x, time, event)
    t_eval = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    aj = aalen_johansen(time, event, t_eval)
    cif, _ = C.predict_cif(model, batch["x_cont"][:2000],
                           torch.tensor(t_eval), batch_size=2048)
    marg = cif.mean(dim=0)
    worst = 0.0
    print("\n     t     AJ_1    model_1  |   AJ_2    model_2")
    for j, tt in enumerate(t_eval):
        print(f"   {tt:4.1f}  {aj[1][j]:.5f}  {float(marg[j,0]):.5f}  |  "
              f"{aj[2][j]:.5f}  {float(marg[j,1]):.5f}")
        worst = max(worst, abs(aj[1][j] - float(marg[j, 0])),
                    abs(aj[2][j] - float(marg[j, 1])))
    print(f"   max abs diff = {worst:.3e}")
    assert worst < 1e-2


# ---------------------------------------------------------------------------
# 9. Batching, and the evaluation budget
# ---------------------------------------------------------------------------

def test_9_batching_invariance(fitted, dense_grid):
    model, batch, *_ = fitted
    x = batch["x_cont"][:600]
    a, sa = C.predict_cif(model, x, dense_grid, batch_size=128)
    b, sb = C.predict_cif(model, x, dense_grid, batch_size=10_000)
    d = float((a - b).abs().max())
    print(f"\n  batched(128) vs single pass: max abs diff = {d:.3e}")
    assert d < 1e-6 and float((sa - sb).abs().max()) < 1e-6


def test_timing_budget(fitted):
    """Report the evaluation budget for the eventual SRTR run: 50,000
    patients x 3 time points (1/3/5 years)."""
    model, batch, *_ = fitted
    pool = batch["x_cont"]
    idx = torch.randint(0, pool.shape[0], (50_000,),
                        generator=torch.Generator().manual_seed(SEED))
    x = pool[idx]
    t_grid = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    start = time.time()
    cif, surv = C.predict_cif(model, x, t_grid, batch_size=2048)
    elapsed = time.time() - start
    dev = float((cif.sum(dim=-1) + surv - 1.0).abs().max())
    print(f"\n  TIMING: 50,000 patients x 3 time points, "
          f"n_nodes={C.DEFAULT_NODES}: {elapsed:.1f} s "
          f"(coherence {dev:.3e})")
    assert cif.shape == (50_000, 3, 2)
