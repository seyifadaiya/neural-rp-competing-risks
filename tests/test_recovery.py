"""Module 5 — simulation recovery without R: simulate from the RP model
itself (known gamma, beta, df = 4), n = 20,000, fit Module 3, and confirm
parameter recovery within sampling error (|bias| < 2 SE, SE from the
inverse Hessian of the total NLL). Catches likelihood errors
independently of the R comparison.

Simulation: latent times per cause. For cause k draw E_k ~ Exp(1) and
solve log H_k(t | x) = log E_k for v = log t; log H_k is strictly
increasing in v (the true gammas keep s' > 0), so vectorised bisection
converges to machine precision. T = min(T_1, T_2), cause = argmin, plus
uniform censoring. Root-finding is simulation machinery only - the
likelihood itself stays closed-form.
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from rcs_basis import spline_eval
from rp_classical import (ClassicalRP, fit_classical_rp, flat_param_vector,
                          make_flat_nll)

torch.set_default_dtype(torch.float64)

SEED = 42
N = 20_000
P = 4

# df = 4: 3 internal + 2 boundary knots on the log-time scale.
KNOTS = {
    1: torch.tensor([-3.0, -1.2, -0.2, 0.8, 2.2]),
    2: torch.tensor([-3.2, -1.0, 0.0, 1.0, 2.4]),
}
GAMMA_TRUE = {
    1: torch.tensor([-1.8, 1.15, 0.045, -0.075, 0.030]),
    2: torch.tensor([-2.1, 0.95, 0.030, -0.060, 0.024]),
}
BETA_TRUE = {
    1: torch.tensor([0.5, -0.3, 0.4, -0.2]),
    2: torch.tensor([-0.2, 0.4, -0.3, 0.5]),
}
C_MAX = 14.0


def _invert_log_H(target, knots, gamma, lo=-14.0, hi=9.0, iters=200):
    """Solve s(v) = target for v by bisection (s strictly increasing)."""
    lo = torch.full_like(target, lo)
    hi = torch.full_like(target, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = spline_eval(mid, knots, gamma) < target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def simulate(n=N, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    x = torch.cat([
        torch.randn(n, 2, generator=g),
        (torch.rand(n, 2, generator=g) < 0.5).to(torch.float64),
    ], dim=1)
    latent = {}
    for k in (1, 2):
        e = -torch.log(torch.rand(n, generator=g))          # Exp(1)
        target = torch.log(e) - x @ BETA_TRUE[k] - GAMMA_TRUE[k][0]
        # spline_eval includes gamma_0; solve on the centred spline.
        gamma_no_int = GAMMA_TRUE[k].clone()
        gamma_no_int[0] = 0.0
        v = _invert_log_H(target, KNOTS[k], gamma_no_int)
        latent[k] = torch.exp(v)
    t_event = torch.minimum(latent[1], latent[2])
    cause = torch.where(latent[1] <= latent[2], 1, 2)
    c = torch.rand(n, generator=g) * C_MAX
    time = torch.minimum(t_event, c)
    event = torch.where(t_event <= c, cause, torch.zeros_like(cause))
    return time, event, x


def _se_from_hessian(model, time, event, x):
    """SE = sqrt(diag(H^-1)) with H the Hessian of the TOTAL (summed) NLL,
    i.e. the observed information at the fitted parameters."""
    flat0 = flat_param_vector(model)
    total_nll = make_flat_nll(model, torch.log(time), x, event, reduction="sum")
    H = torch.autograd.functional.hessian(total_nll, flat0)
    cov = torch.linalg.inv(H)
    return torch.sqrt(torch.diag(cov)), flat0, H


@pytest.fixture(scope="module")
def recovery_fit():
    time, event, x = simulate()
    frac = [(event == k).to(torch.float64).mean().item() for k in (0, 1, 2)]
    print(f"\nsimulated mix: censored {frac[0]:.3f}, "
          f"cause1 {frac[1]:.3f}, cause2 {frac[2]:.3f}")
    model = ClassicalRP(KNOTS, p=P)
    fit_classical_rp(model, time, event, x)
    return model, time, event, x


def test_parameter_recovery_within_sampling_error(recovery_fit):
    """Recovery within sampling error, assessed family-wise.

    With 18 parameters, requiring every |z| < 2 individually would fail
    ~60% of the time even for a perfectly correct likelihood (0.95^18).
    The rigorous joint criterion is the Wald statistic
    (theta_hat - theta)' H (theta_hat - theta) ~ chi2_18 under the truth
    (H = observed information); we require it below the 0.99 quantile.
    Per-parameter |z| < 3 (~Bonferroni at 18 tests) guards against a
    single wildly-biased coordinate hiding in an acceptable joint value.
    """
    from scipy.stats import chi2

    model, time, event, x = recovery_fit
    se, flat_hat, H = _se_from_hessian(model, time, event, x)
    truth = torch.cat([GAMMA_TRUE[1], BETA_TRUE[1], GAMMA_TRUE[2], BETA_TRUE[2]])
    names = ([f"gamma1[{i}]" for i in range(5)] + [f"beta1[{i}]" for i in range(4)]
             + [f"gamma2[{i}]" for i in range(5)] + [f"beta2[{i}]" for i in range(4)])

    bias = flat_hat - truth
    z = bias / se
    lines = [f"  {n_:11s} true={t: .4f}  est={e: .4f}  SE={s:.4f}  z={zi:+.2f}"
             for n_, t, e, s, zi in zip(names, truth, flat_hat, se, z)]
    print("\n".join(lines))

    # Joint Wald test against the observed information.
    wald = float(bias @ H @ bias)
    crit = chi2.ppf(0.99, df=len(bias))
    print(f"  joint Wald: {wald:.2f}  (chi2_{len(bias)} 0.99 critical value "
          f"{crit:.2f})")
    assert wald < crit, (
        f"joint Wald {wald:.2f} exceeds chi2 critical value {crit:.2f}\n"
        + "\n".join(lines)
    )
    # Bonferroni-level per-parameter guard.
    failures = [(n_, zi.item()) for n_, zi in zip(names, z) if abs(zi) >= 3]
    assert not failures, (
        "parameters grossly outside sampling error: " + str(failures) + "\n"
        + "\n".join(lines)
    )


def test_hessian_is_positive_definite(recovery_fit):
    model, time, event, x = recovery_fit
    se, _, _ = _se_from_hessian(model, time, event, x)
    assert torch.isfinite(se).all() and (se > 0).all()
