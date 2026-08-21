"""
Cumulative incidence functions for the neural Royston-Parmar competing-risks
models. PREDICTION ONLY - the training likelihood is closed form and this
module never touches it.

    F_k(t | x) = int_0^t h_k(u | x) S(u | x) du,    S(t | x) = exp(-sum_j H_j)

Substituting u = e^v (du = e^v dv) and using h_k(u) u = H_k(u) dlogH_k/dv:

    F_k(t | x) = int_{-inf}^{log t} H_k(e^v) [dlogH_k/dv](v) S(e^v) dv

Integrating over LOG TIME is the right choice: the model is parameterised in
log time, the integrand is smooth on that scale, and the substitution removes
the 1/t singularity of the hazard as t -> 0.

WHY COHERENCE IS THE HEADLINE CHECK. Differentiating the identity,

    d/dt [F_1 + F_2 + S] = h_1 S + h_2 S - (h_1 + h_2) S = 0,

so F_1(t) + F_2(t) + S(t) = 1 exactly, for all t, BY CONSTRUCTION. Any
observed deviation is quadrature error and nothing else. Nothing in this
module normalises, softmaxes, or rescales to enforce it - doing so would
destroy the very property being claimed. If the identity fails, that is a
bug in the quadrature, never something to paper over.

The derivative dlogH_k/dv is taken FROM THE MODEL (its analytic derivative),
so Variant A (shared s'_k) and Variant B (patient-specific b_k(z) s~'_k) are
handled without branching.
"""

import numpy as np
import torch

# Gauss-Legendre nodes per sub-interval, chosen from the convergence study in
# tests/test_cif.py::test_2_node_count_convergence rather than assumed.
# Measured on a fitted Variant B model, |GL(n) - GL(240)| runs
#
#     n:      5        10       15       20       30       40       80
#     err:  6.4e-05  2.2e-06  2.2e-06  9.6e-08  1.5e-07  1.1e-07  1.8e-08
#
# and the coherence deviation behaves the same way. The integrand is smooth on
# the log-time scale, so Gauss-Legendre converges spectrally and the curve is
# flat from n=20 onward; the plateau is set by float32 model output, not by the
# quadrature (the same model in float64 reaches 8.7e-08 where float32 gives
# 2.1e-07). 20 is the smallest count on that plateau, so that is the default.
#
# Cost is mild if more margin is wanted: 50,000 patients x 3 time points takes
# ~4 s at 20 nodes and ~8 s at 40 on a 4-core CPU.
DEFAULT_NODES = 40

# The quadrature accumulates in float64 even though the model is float32:
# nodes/weights are exact in double, and the sums are cheap relative to the
# network evaluation. This keeps quadrature error separable from float32
# model noise, which is what makes the coherence number interpretable.
ACC_DTYPE = torch.float64


def _gauss_legendre(n_nodes: int):
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    return (torch.as_tensor(nodes, dtype=ACC_DTYPE),
            torch.as_tensor(weights, dtype=ACC_DTYPE))


TAIL_TOL = 1e-10          # required max_patients sum_k H_k at the lower limit
N_TAIL = 4                # sub-intervals covering the tail below the first grid time


@torch.no_grad()
def _auto_lower_limit(model, x, x_cat=None, tol: float = TAIL_TOL,
                      floor: float = -60.0) -> float:
    """Lower truncation of the log-time integral, chosen so the discarded
    tail is genuinely negligible.

    Truncating at the smallest boundary knot - the obvious choice, since
    H_k(t) -> 0 as t -> 0 - is NOT good enough in practice. Measured on a
    fitted model, sum_k H_k at that knot is around 1e-4, and because

        H_k (dlogH_k/dv) = dH_k/dv,

    the tail below v_lo carries approximately H_k(e^v_lo) of probability mass
    (S ~ 1 there). That mass is missing from F_k, and it shows up directly as
    a coherence deviation of the same order - flat in the node count, because
    it is a domain error, not a quadrature error.

    So instead of assuming, step the limit down until the model itself says
    the remaining mass is below `tol`. Below the boundary knot the spline is
    linear in v (the natural-spline restriction), so H_k decays as a power of
    t and a few steps suffice. Fixing the domain rather than adding a
    correction term keeps this pure quadrature: nothing here nudges
    F_1 + F_2 + S toward 1.
    """
    v = min(float(model.basis[k].knots[0]) for k in (1, 2))
    x = torch.as_tensor(x)
    n = x.shape[0]
    while v > floor:
        log_H, _ = _model_at(model, torch.full((n,), v, dtype=ACC_DTYPE), x, x_cat)
        if float(torch.exp(log_H).sum(dim=-1).max()) < tol:
            return v
        v -= 2.0
    return floor


def _quadrature_points(v_lo: float, log_t: torch.Tensor, n_nodes: int,
                       n_tail: int = N_TAIL):
    """Cumulative scheme: one Gauss-Legendre rule per sub-interval, then a
    cumulative sum gives every grid time in a single pass - far cheaper than
    T independent integrals over [v_lo, log t_i], and it reuses the partial
    sums exactly.

    The stretch below the first grid time is split into `n_tail` sub-intervals
    rather than integrated in one go. With the lower limit pushed down far
    enough to make the tail negligible, that stretch is wide (tens of log
    units) and its integrand is concentrated at the top end; a single rule
    over it would converge slowly for no good reason.

    Returns points (S, n_nodes), weights (S, n_nodes), and the indices of the
    cumulative sums corresponding to the requested grid times.
    """
    log_t = log_t.to(ACC_DTYPE)
    first = torch.clamp(log_t[:1], min=v_lo)
    tail_edges = torch.linspace(v_lo, float(first), n_tail + 1, dtype=ACC_DTYPE)
    edges = torch.cat([tail_edges, log_t[1:]])
    # Times at or below the truncation point get a zero-width interval, so
    # they contribute nothing rather than integrating backwards.
    edges = torch.cummax(edges, dim=0).values
    lo, hi = edges[:-1], edges[1:]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    nodes, weights = _gauss_legendre(n_nodes)
    take = n_tail - 1 + torch.arange(log_t.numel())
    return (mid[:, None] + half[:, None] * nodes[None, :],
            half[:, None] * weights[None, :], take)


@torch.no_grad()
def _model_at(model, v_flat: torch.Tensor, x_rep: torch.Tensor,
              x_cat_rep: torch.Tensor = None):
    """(log_H, dlogH_dv) from the model at matched (v, x) rows, in ACC_DTYPE."""
    log_H, dlogH = model(v_flat.to(model.dtype_), x_rep, x_cat_rep)
    return log_H.to(ACC_DTYPE), dlogH.to(ACC_DTYPE)


@torch.no_grad()
def predict_cif(model, x, t_grid, n_nodes: int = DEFAULT_NODES, cause=None,
                batch_size: int = 512, x_cat=None, v_lo: float = None):
    """F_k(t | x) on t_grid, plus event-free survival S(t | x).

    model:      VariantA or VariantB (exposes log_H and dlogH_dv at any v)
    x:          (n, p) covariates
    t_grid:     (T,) evaluation times in years, all > 0
    n_nodes:    Gauss-Legendre nodes per sub-interval (see DEFAULT_NODES)
    cause:      None for both causes, or 1 / 2 for a single one
    batch_size: patients per batch; the work is n x T x n_nodes model
                evaluations, so this bounds peak memory
    v_lo:       override the lower truncation (tests use this)

    returns:    (cif, surv) with cif (n, T, 2) - or (n, T) when `cause` is
                given - and surv (n, T).
    """
    model.eval()
    x = torch.as_tensor(x)
    t_grid = torch.as_tensor(t_grid, dtype=ACC_DTYPE).reshape(-1)
    assert (t_grid > 0).all(), "t_grid must be strictly positive"
    assert (t_grid[1:] >= t_grid[:-1]).all(), "t_grid must be non-decreasing"

    log_t = torch.log(t_grid)
    if v_lo is None:
        v_lo = _auto_lower_limit(model, x, x_cat)
    pts, wts, take = _quadrature_points(v_lo, log_t, n_nodes)
    n_sub, n_q = pts.shape
    n_t = log_t.numel()
    pts_flat = pts.reshape(-1)

    n = x.shape[0]
    cif_out = torch.empty(n, n_t, 2, dtype=ACC_DTYPE)
    surv_out = torch.empty(n, n_t, dtype=ACC_DTYPE)

    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size]
        cb = None if x_cat is None else x_cat[start:start + batch_size]
        b = xb.shape[0]

        # Every patient at every quadrature point, vectorised - no Python
        # loop over subjects.
        v_rep = pts_flat.repeat(b)                                # (b*S*n_q,)
        x_rep = xb.repeat_interleave(n_sub * n_q, dim=0)
        c_rep = None if cb is None else cb.repeat_interleave(n_sub * n_q, dim=0)
        log_H, dlogH = _model_at(model, v_rep, x_rep, c_rep)
        log_H = log_H.reshape(b, n_sub, n_q, 2)
        dlogH = dlogH.reshape(b, n_sub, n_q, 2)

        H = torch.exp(log_H)                                      # (b,S,n_q,2)
        surv = torch.exp(-H.sum(dim=-1))                          # (b,S,n_q)
        integrand = H * dlogH * surv.unsqueeze(-1)
        sub_integral = (integrand * wts[None, :, :, None]).sum(dim=2)   # (b,S,2)
        cif_out[start:start + b] = torch.cumsum(sub_integral, dim=1)[:, take, :]

        # S(t) evaluated directly at the grid times (not from the quadrature),
        # so the coherence check compares two independent computations.
        v_t = log_t.repeat(b)
        x_t = xb.repeat_interleave(n_t, dim=0)
        c_t = None if cb is None else cb.repeat_interleave(n_t, dim=0)
        log_H_t, _ = _model_at(model, v_t, x_t, c_t)
        surv_out[start:start + b] = torch.exp(
            -torch.exp(log_H_t.reshape(b, n_t, 2)).sum(dim=-1))

    if cause is not None:
        return cif_out[:, :, cause - 1], surv_out
    return cif_out, surv_out


@torch.no_grad()
def predict_survival(model, x, t_grid, batch_size: int = 512, x_cat=None):
    """Event-free survival S(t | x) = exp(-sum_k H_k(t | x)), shape (n, T)."""
    log_H = predict_cumhaz(model, x, t_grid, batch_size=batch_size, x_cat=x_cat)
    return torch.exp(-torch.exp(log_H).sum(dim=-1))


@torch.no_grad()
def predict_cumhaz(model, x, t_grid, batch_size: int = 512, x_cat=None):
    """Per-cause log cumulative hazard log H_k(t | x), shape (n, T, 2)."""
    model.eval()
    x = torch.as_tensor(x)
    t_grid = torch.as_tensor(t_grid, dtype=ACC_DTYPE).reshape(-1)
    log_t = torch.log(t_grid)
    n, n_t = x.shape[0], log_t.numel()
    out = torch.empty(n, n_t, 2, dtype=ACC_DTYPE)
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size]
        cb = None if x_cat is None else x_cat[start:start + batch_size]
        b = xb.shape[0]
        log_H, _ = _model_at(model, log_t.repeat(b),
                             xb.repeat_interleave(n_t, dim=0),
                             None if cb is None else cb.repeat_interleave(n_t, dim=0))
        out[start:start + b] = log_H.reshape(b, n_t, 2)
    return out


@torch.no_grad()
def coherence_deviation(model, x, t_grid, n_nodes: int = DEFAULT_NODES,
                        batch_size: int = 512, x_cat=None) -> float:
    """max |F_1 + F_2 + S - 1| over patients and times.

    The direct empirical measure of probability coherence. It is a pure
    quadrature-error diagnostic: the identity is exact in the model.
    """
    cif, surv = predict_cif(model, x, t_grid, n_nodes=n_nodes,
                            batch_size=batch_size, x_cat=x_cat)
    return float((cif.sum(dim=-1) + surv - 1.0).abs().max())
