"""Module 1 unit tests: analytic derivative vs autograd, linearity beyond
the boundary knots, and C2 continuity at every knot."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcs_basis import rcs_design, rcs_deriv_design, spline_eval, spline_deriv

torch.set_default_dtype(torch.float64)

KNOTS = torch.tensor([-2.3, -0.7, 0.1, 0.9, 2.1], dtype=torch.float64)
GAMMA = torch.tensor([0.4, 1.1, 0.35, -0.6, 0.25], dtype=torch.float64)


def test_analytic_deriv_matches_autograd():
    torch.manual_seed(0)
    v = (torch.rand(100, dtype=torch.float64) * 8 - 4).requires_grad_(True)
    s = spline_eval(v, KNOTS, GAMMA)
    (auto,) = torch.autograd.grad(s.sum(), v)
    analytic = spline_deriv(v.detach(), KNOTS, GAMMA)
    assert torch.allclose(analytic, auto, rtol=1e-8, atol=1e-12)


def test_linear_beyond_boundary_knots():
    # Natural spline restriction: exactly linear outside [k_1, k_m].
    for lo, hi in [(-8.0, KNOTS[0].item() - 1e-9), (KNOTS[-1].item() + 1e-9, 8.0)]:
        v = torch.linspace(lo, hi, 200, dtype=torch.float64)
        s = spline_eval(v, KNOTS, GAMMA)
        second_diff = s[2:] - 2 * s[1:-1] + s[:-2]
        assert second_diff.abs().max() < 1e-9
        # Derivative is exactly constant there too.
        d = spline_deriv(v, KNOTS, GAMMA)
        assert (d - d[0]).abs().max() < 1e-9


@pytest.mark.parametrize("knot_idx", range(len(KNOTS)))
def test_continuity_at_knots(knot_idx):
    # Approach the knot from both sides and extrapolate to the knot itself;
    # a discontinuity in value / s' / s'' would appear as a mismatch between
    # the left and right extrapolations beyond the O(h^2) extrapolation error.
    k = KNOTS[knot_idx]
    h = 1e-5
    v = k + torch.tensor([-2 * h, -h, 0.0, h, 2 * h], dtype=torch.float64)
    s = spline_eval(v, KNOTS, GAMMA)
    d = spline_deriv(v, KNOTS, GAMMA)

    # Value continuity: linear extrapolation from each side to v = k.
    s_from_left = 2 * s[1] - s[0]
    s_from_right = 2 * s[3] - s[4]
    assert abs(s_from_left - s_from_right) < 1e-7

    # First-derivative continuity, same construction on the analytic s'.
    d_from_left = 2 * d[1] - d[0]
    d_from_right = 2 * d[3] - d[4]
    assert abs(d_from_left - d_from_right) < 1e-7

    # Second-derivative continuity: one-sided finite differences of s'
    # ending exactly at the knot.
    d2_left = (d[2] - d[1]) / h
    d2_right = (d[3] - d[2]) / h
    assert abs(d2_left - d2_right) < 1e-3


def test_design_shapes():
    v = torch.linspace(-3, 3, 17, dtype=torch.float64)
    m = KNOTS.numel()
    assert rcs_design(v, KNOTS).shape == (17, m - 1)
    assert rcs_deriv_design(v, KNOTS).shape == (17, m - 1)
    # First column of the derivative design is the constant 1 (d v / d v).
    assert (rcs_deriv_design(v, KNOTS)[:, 0] == 1).all()


# ---------------------------------------------------------------------------
# Orthonormalised working basis (RCSBasis)
# ---------------------------------------------------------------------------

import numpy as np                                                    # noqa: E402
from rcs_basis import RCSBasis                                        # noqa: E402
from rp_classical import ClassicalRP, fit_classical_rp                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from gen_data import generate                                         # noqa: E402


def _knots_from(time, event, k, n_internal=3):
    """Quantile-based knots on log time for cause k (deterministic)."""
    lt = np.log(time[event == k])
    qs = np.linspace(0, 1, n_internal + 2)
    return torch.tensor(np.quantile(lt, qs), dtype=torch.float64)


@pytest.fixture(scope="module")
def sim():
    time, event, x = generate()
    return (torch.tensor(time), torch.tensor(event), torch.tensor(x),
            {k: _knots_from(time, event, k) for k in (1, 2)})


def test_ortho_span_invariance(sim):
    """Fitting in the orthonormalised basis and mapping back reproduces the
    truncated-power fit: same function, different geometry."""
    time, event, x, knots = sim
    v = torch.log(time)

    plain = ClassicalRP(knots, p=x.shape[1])
    fit_classical_rp(plain, time, event, x)

    basis = {k: RCSBasis(knots[k]) for k in (1, 2)}
    ortho = ClassicalRP(knots, p=x.shape[1], basis=basis)
    fit_classical_rp(ortho, time, event, x)

    # (a) coefficients map back onto the truncated-power fit
    for k in (1, 2):
        alpha = getattr(ortho, f"gamma{k}").detach()
        mapped = basis[k].to_original_gamma(alpha)
        direct = getattr(plain, f"gamma{k}").detach()
        assert torch.allclose(mapped, direct, atol=1e-6), (
            f"cause {k}: mapped gamma {mapped.numpy()} vs {direct.numpy()}"
        )

    # (b) the fitted log H functions coincide on a dense grid
    grid = torch.linspace(v.min(), v.max(), 200, dtype=torch.float64)
    x0 = torch.zeros(200, x.shape[1], dtype=torch.float64)
    lh_plain, _ = plain.log_H_and_deriv(grid, x0)
    lh_ortho, _ = ortho.log_H_and_deriv(grid, x0)
    diff = (lh_plain - lh_ortho).abs().max().item()
    assert diff < 1e-8, f"max |delta log H| = {diff:.3e}"


def test_ortho_conditioning_improves(sim):
    _, _, _, knots = sim
    for k in (1, 2):
        basis = RCSBasis(knots[k])
        cond_B, cond_Q = basis.condition_numbers()
        print(f"  cause {k}: cond(B'B) = {cond_B:.3e} -> cond(Q'Q) = {cond_Q:.3e} "
              f"({cond_B / cond_Q:.2e}x better)")
        assert cond_Q < 1e3
        assert cond_B / cond_Q > 1e3, "expected orders-of-magnitude improvement"


def test_ortho_deriv_matches_autograd(sim):
    _, _, _, knots = sim
    basis = RCSBasis(knots[1])
    v = torch.linspace(knots[1][0].item(), knots[1][-1].item(), 100,
                       dtype=torch.float64).requires_grad_(True)
    alpha = torch.tensor([0.7, -0.2, 0.35, 0.1], dtype=torch.float64)
    s = basis.design(v) @ alpha
    (auto,) = torch.autograd.grad(s.sum(), v)
    analytic = basis.deriv_design(v.detach()) @ alpha
    assert torch.allclose(analytic, auto, rtol=1e-8, atol=1e-12)


def test_basis_serialisation_roundtrip(sim):
    """R must survive a checkpoint round-trip exactly - a model restored
    without its basis is silently wrong."""
    _, _, _, knots = sim
    basis = RCSBasis(knots[1])
    import json
    restored = RCSBasis.from_dict(json.loads(json.dumps(basis.to_dict())))
    v = torch.linspace(-4, 2, 77, dtype=torch.float64)
    assert torch.equal(basis.R, restored.R)
    assert (basis.design(v) - restored.design(v)).abs().max() < 1e-10
    assert (basis.deriv_design(v) - restored.deriv_design(v)).abs().max() < 1e-10


def test_gamma_alpha_roundtrip(sim):
    _, _, _, knots = sim
    basis = RCSBasis(knots[2])
    gamma = torch.tensor([-1.3, 0.9, 0.2, -0.15, 0.05], dtype=torch.float64)
    back = basis.to_original_gamma(basis.to_working_alpha(gamma))
    assert torch.allclose(back, gamma, atol=1e-10)
