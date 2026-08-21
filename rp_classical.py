"""
Classical cause-specific Royston-Parmar model as a degenerate network:
Variant A with a linear covariate map and no hidden layers,

    log H_k(t | x) = gamma_{k0} + s_k(log t; gamma_k) + beta_k^T x,

fitted jointly for both causes by full-batch L-BFGS on the Module 2
competing-risks loss. Because the parameters of the two causes do not
interact in the likelihood, the joint fit is exactly the pair of
classical censoring-based cause-specific fits.
"""

import torch

from rcs_basis import rcs_design, rcs_deriv_design
from rp_likelihood import rp_competing_loss


class ClassicalRP(torch.nn.Module):
    """Two cause-specific RP models with shared input data layout.

    knots: dict {1: tensor, 2: tensor} - per-cause knot vectors (log time),
           imported (e.g. from R), never re-derived here.
    p:     number of covariates.
    """

    def __init__(self, knots: dict, p: int, dtype=torch.float64, basis: dict = None):
        super().__init__()
        self.knots = {k: torch.as_tensor(v, dtype=dtype) for k, v in knots.items()}
        self.p = p
        # Optional orthonormalised working basis per cause (rcs_basis.RCSBasis).
        # None => the truncated-power basis, i.e. the verified default path.
        self.basis = basis
        for k in (1, 2):
            m = self.knots[k].numel()
            # gamma_k: [gamma_0, gamma_1, gamma_2..] (m entries);
            # start near log H = -1 + 1*v (a unit-rate exponential-ish start).
            g = torch.zeros(m, dtype=dtype)
            g[0], g[1] = -1.0, 1.0
            self.register_parameter(f"gamma{k}", torch.nn.Parameter(g))
            self.register_parameter(
                f"beta{k}", torch.nn.Parameter(torch.zeros(p, dtype=dtype))
            )

    def designs(self, v: torch.Tensor, k: int):
        """(design, deriv_design) for cause k at log-times v, each (n, m-1).

        Truncated-power basis by default; the orthonormalised working basis
        when one was supplied. Both span the same function space, so the
        fitted log H is identical either way - only the parameter geometry
        differs (see rcs_basis.RCSBasis).
        """
        if self.basis is not None:
            return self.basis[k].design(v), self.basis[k].deriv_design(v)
        return rcs_design(v, self.knots[k]), rcs_deriv_design(v, self.knots[k])

    def log_H_and_deriv(self, v: torch.Tensor, x: torch.Tensor):
        """Returns (log_H, dlogH_dv), each (n, 2)."""
        outs, derivs = [], []
        for k in (1, 2):
            gamma = getattr(self, f"gamma{k}")
            beta = getattr(self, f"beta{k}")
            design, ddesign = self.designs(v, k)
            outs.append(gamma[0] + design @ gamma[1:] + x @ beta)
            derivs.append(ddesign @ gamma[1:])
        return torch.stack(outs, dim=1), torch.stack(derivs, dim=1)

    def loss(self, v, x, event_type):
        log_H, dlogH = self.log_H_and_deriv(v, x)
        return rp_competing_loss(log_H, dlogH, event_type, v)


def flat_param_vector(model: ClassicalRP) -> torch.Tensor:
    """Current parameters as one flat vector (order = named_parameters)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def make_flat_nll(model: ClassicalRP,
                  v: torch.Tensor,
                  x: torch.Tensor,
                  event_type: torch.Tensor,
                  reduction: str = "mean"):
    """Build f(flat) -> NLL, differentiable in `flat`.

    Needed for Newton steps and for the observed information matrix; the
    design matrices do not depend on the parameters, so they are built once.
    """
    specs = [(n, p.shape, p.numel()) for n, p in model.named_parameters()]
    n_obs = v.numel()
    designs = {k: model.designs(v, k) for k in (1, 2)}

    def flat_nll(flat: torch.Tensor) -> torch.Tensor:
        parts, i = {}, 0
        for name, shape, numel in specs:
            parts[name] = flat[i:i + numel].reshape(shape)
            i += numel
        outs, derivs = [], []
        for k in (1, 2):
            gamma, beta = parts[f"gamma{k}"], parts[f"beta{k}"]
            design, ddesign = designs[k]
            outs.append(gamma[0] + design @ gamma[1:] + x @ beta)
            derivs.append(ddesign @ gamma[1:])
        loss = rp_competing_loss(torch.stack(outs, dim=1),
                                 torch.stack(derivs, dim=1), event_type, v)
        return loss * n_obs if reduction == "sum" else loss

    return flat_nll


def _write_flat(model: ClassicalRP, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(flat[i:i + p.numel()].reshape(p.shape))
            i += p.numel()


def fit_classical_rp(model: ClassicalRP,
                     time: torch.Tensor,
                     event_type: torch.Tensor,
                     x: torch.Tensor,
                     max_iter: int = 500,
                     tol: float = 1e-9,
                     gtol: float = 1e-11,
                     verbose: bool = False) -> float:
    """Fit to the MLE and return the final mean NLL. float64 throughout.

    Two stages. L-BFGS does the bulk of the descent, then damped Newton
    polishes to a stationary point. The polish is not optional: the
    truncated-power RCS basis is badly conditioned (the cubic columns span
    a far larger range than the linear one), and on that geometry L-BFGS
    stalls with a visibly non-zero gradient - short of the true MLE by
    enough to break a 1e-3 agreement check against R. Newton is
    affine-invariant, so the conditioning does not affect it, and with
    only a handful of parameters the exact Hessian is cheap.
    """
    v = torch.log(time)
    opt = torch.optim.LBFGS(
        model.parameters(), max_iter=100, history_size=50,
        tolerance_grad=1e-14, tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = model.loss(v, x, event_type)
        loss.backward()
        return loss

    prev = float("inf")
    for it in range(max(1, max_iter // 100)):
        loss = opt.step(closure).item()
        if verbose:
            print(f"  L-BFGS round {it}: mean NLL = {loss:.12f}")
        if abs(prev - loss) < tol:
            break
        prev = loss

    # ---- damped Newton polish ----
    f = make_flat_nll(model, v, x, event_type)
    flat = flat_param_vector(model)
    for it in range(50):
        fl = flat.detach().requires_grad_(True)
        loss_t = f(fl)
        (g,) = torch.autograd.grad(loss_t, fl)
        gmax = g.abs().max().item()
        if verbose:
            print(f"  Newton {it}: mean NLL = {loss_t.item():.14f}  "
                  f"|g|_inf = {gmax:.3e}")
        if gmax < gtol:
            break
        H = torch.autograd.functional.hessian(f, flat.detach())

        # Levenberg damping until we get a descent direction.
        lam, step = 0.0, None
        for _ in range(30):
            damped = H + lam * torch.eye(H.shape[0], dtype=H.dtype)
            try:
                cand = torch.linalg.solve(damped, -g)
            except RuntimeError:
                lam = max(lam * 10, 1e-10)
                continue
            if (g @ cand) < 0:
                step = cand
                break
            lam = max(lam * 10, 1e-10)
        if step is None:
            step = -g

        # Backtracking line search (Armijo).
        base, slope, t = loss_t.item(), (g @ step).item(), 1.0
        for _ in range(60):
            trial = flat.detach() + t * step
            if f(trial).item() <= base + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            break                      # no acceptable step: already at optimum
        flat = flat.detach() + t * step

    _write_flat(model, flat)
    return f(flat.detach()).item()
