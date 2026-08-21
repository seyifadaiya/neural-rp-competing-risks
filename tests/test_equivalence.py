"""Module 4 — the equivalence check against R's rstpm2 (the point of this
package). Requires rp_core/reference/r_reference.json, produced once by:

    Rscript rp_core/reference/fit_reference.R

The torch model uses the truncated-power RCS basis; rstpm2 uses the nsx
natural-spline basis. With IDENTICAL knots these span the same function
space, so the fitted log H functions, the beta's, and the maximised
log-likelihoods must coincide even though the spline coefficient vectors
are not directly comparable. Function-level comparisons (dense grids)
are therefore the operative assertions, per the task spec.

Assertions:
  1. beta coefficients agree with R within 1e-3 (absolute), both causes.
  2. Baseline spline functions agree: max |Δ log H| < 1e-3 on a dense
     200-point log-time grid (robust to basis reparameterisation).
  3. Total log-likelihood: |torch − (R cause 1 + R cause 2)| < 0.1.
  4. log H_k(t) curves for 5 reference covariate profiles overlay the R
     predictions (max abs diff < 1e-3).

On failure, a torch-vs-R diagnostic table is printed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from rp_classical import ClassicalRP, fit_classical_rp

torch.set_default_dtype(torch.float64)

REF_PATH = HERE / "reference" / "r_reference.json"
DATA_PATH = HERE / "reference" / "sim_data.csv"

pytestmark = pytest.mark.skipif(
    not REF_PATH.exists(),
    reason="r_reference.json missing - run Rscript rp_core/reference/fit_reference.R once",
)


@pytest.fixture(scope="module")
def fitted():
    ref = json.loads(REF_PATH.read_text())
    data = np.genfromtxt(DATA_PATH, delimiter=",", names=True)
    time = torch.tensor(data["time"])
    event = torch.tensor(data["event"].astype(np.int64))
    x = torch.stack([torch.tensor(data[c]) for c in ("x1", "x2", "x3", "x4")], dim=1)

    knots = {}
    for k in (1, 2):
        r = ref[f"cause{k}"]
        ks = sorted([r["boundary_knots"][0]] + list(r["internal_knots"])
                    + [r["boundary_knots"][1]])
        knots[k] = torch.tensor(ks)

    model = ClassicalRP(knots, p=4)
    nll = fit_classical_rp(model, time, event, x)
    return ref, model, nll, time, event, x


def _r_betas(ref, k):
    coefs = ref[f"cause{k}"]["coefficients"]
    return np.array([coefs[c] for c in ("x1", "x2", "x3", "x4")])


def _diag_table(label, torch_vals, r_vals, names):
    rows = [f"    {n:14s} torch={t: .8f}  R={r: .8f}  diff={t - r: .2e}"
            for n, t, r in zip(names, torch_vals, r_vals)]
    return f"  {label}:\n" + "\n".join(rows)


def test_1_betas_match_r(fitted):
    ref, model, *_ = fitted
    for k in (1, 2):
        torch_beta = getattr(model, f"beta{k}").detach().numpy()
        r_beta = _r_betas(ref, k)
        diff = np.abs(torch_beta - r_beta).max()
        assert diff < 1e-3, (
            f"cause {k} beta mismatch (max abs diff {diff:.2e})\n"
            + _diag_table(f"cause {k} beta", torch_beta, r_beta,
                          ["x1", "x2", "x3", "x4"])
        )


def test_2_baseline_spline_functions_match(fitted):
    ref, model, *_ = fitted
    tgrid = torch.tensor(ref["time_grid"], dtype=torch.float64)
    v = torch.log(tgrid)
    x0 = torch.zeros(len(tgrid), 4, dtype=torch.float64)
    log_H, _ = model.log_H_and_deriv(v, x0)
    for k in (1, 2):
        r_curve = np.array(ref[f"cause{k}"]["log_cumhaz_baseline"])
        t_curve = log_H[:, k - 1].detach().numpy()
        diff = np.abs(t_curve - r_curve).max()
        assert diff < 1e-3, (
            f"cause {k} baseline log H mismatch: max abs diff {diff:.2e}\n"
            + _diag_table(f"cause {k} log H (first 10 grid points)",
                          t_curve[:10], r_curve[:10],
                          [f"t={t:.3g}" for t in tgrid[:10].tolist()])
        )


def test_3_total_loglik_matches_sum_of_r_fits(fitted):
    ref, model, nll, time, event, x = fitted
    n = len(time)
    torch_loglik = -nll * n
    r_total = ref["cause1"]["logLik"] + ref["cause2"]["logLik"]
    diff = abs(torch_loglik - r_total)
    assert diff < 0.1, (
        f"log-likelihood mismatch: torch {torch_loglik:.6f} vs "
        f"R {r_total:.6f} (cause1 {ref['cause1']['logLik']:.6f} + "
        f"cause2 {ref['cause2']['logLik']:.6f}), diff {diff:.4f}"
    )


def test_4_profile_curves_overlay_r(fitted):
    ref, model, *_ = fitted
    tgrid = torch.tensor(ref["time_grid"], dtype=torch.float64)
    v = torch.log(tgrid)
    profs = ref["profiles"]
    n_prof = len(profs["x1"])
    for i in range(n_prof):
        xi = torch.tensor([[profs["x1"][i], profs["x2"][i],
                            profs["x3"][i], profs["x4"][i]]],
                          dtype=torch.float64)
        x_rep = xi.repeat(len(tgrid), 1)
        log_H, _ = model.log_H_and_deriv(v, x_rep)
        for k in (1, 2):
            r_curve = np.array(ref[f"cause{k}"]["log_cumhaz_profiles"][i])
            t_curve = log_H[:, k - 1].detach().numpy()
            diff = np.abs(t_curve - r_curve).max()
            assert diff < 1e-3, (
                f"profile {i}, cause {k}: max |Δ log H| = {diff:.2e}\n"
                + _diag_table("log H (first 10 grid points)",
                              t_curve[:10], r_curve[:10],
                              [f"t={t:.3g}" for t in tgrid[:10].tolist()])
            )
