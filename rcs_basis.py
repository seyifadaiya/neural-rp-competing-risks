"""
Royston-Parmar restricted cubic spline (RCS) basis in PyTorch.

Truncated-power natural cubic spline basis on v = log(t), with knots
k_1 < ... < k_m and lambda_j = (k_m - k_{j+1}) / (k_m - k_1):

    B_j(v)  = (v - k_{j+1})^3_+ - lambda_j (v - k_1)^3_+
              - (1 - lambda_j)(v - k_m)^3_+          for j = 1 .. m-2
    s(v; g) = g_0 + g_1 v + sum_j g_{j+1} B_j(v)

The analytic derivative (mandatory for the likelihood hot path):

    B'_j(v) = 3(v - k_{j+1})^2_+ - 3 lambda_j (v - k_1)^2_+
              - 3(1 - lambda_j)(v - k_m)^2_+
    s'(v; g) = g_1 + sum_j g_{j+1} B'_j(v)

All functions are batched, differentiable, and dtype-preserving (use
float64 tensors for verification work).
"""

import torch


def _check_knots(knots: torch.Tensor) -> torch.Tensor:
    knots = torch.as_tensor(knots)
    assert knots.ndim == 1 and knots.numel() >= 3, "need at least 3 knots"
    assert (knots[1:] > knots[:-1]).all(), "knots must be strictly increasing"
    return knots


def rcs_design(v: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Design matrix (n, m-1): columns [v, B_1(v), ..., B_{m-2}(v)].

    No intercept column; the intercept is gamma_0 in spline_eval.
    """
    knots = _check_knots(knots).to(v.dtype)
    k1, km = knots[0], knots[-1]
    inner = knots[1:-1]                                    # k_2 .. k_{m-1}
    lam = (km - inner) / (km - k1)                         # (m-2,)

    vv = v.reshape(-1, 1)                                  # (n, 1)
    cube = lambda z: torch.clamp(z, min=0.0) ** 3
    B = (cube(vv - inner) - lam * cube(vv - k1)
         - (1.0 - lam) * cube(vv - km))                    # (n, m-2)
    return torch.cat([vv, B], dim=1)                       # (n, m-1)


def rcs_deriv_design(v: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Derivative design matrix (n, m-1): columns [1, B'_1(v), ..., B'_{m-2}(v)]."""
    knots = _check_knots(knots).to(v.dtype)
    k1, km = knots[0], knots[-1]
    inner = knots[1:-1]
    lam = (km - inner) / (km - k1)

    vv = v.reshape(-1, 1)
    sq = lambda z: torch.clamp(z, min=0.0) ** 2
    dB = 3.0 * (sq(vv - inner) - lam * sq(vv - k1)
                - (1.0 - lam) * sq(vv - km))
    ones = torch.ones_like(vv)
    return torch.cat([ones, dB], dim=1)


def spline_eval(v: torch.Tensor, knots: torch.Tensor,
                gamma: torch.Tensor) -> torch.Tensor:
    """s(v; gamma) = gamma_0 + gamma_1 v + sum_j gamma_{j+1} B_j(v).

    gamma has m-1+1 = m entries: [gamma_0, gamma_1, gamma_2, ..., gamma_{m-1}].
    Returns shape (n,).
    """
    design = rcs_design(v, knots)
    return gamma[0] + design @ gamma[1:]


def spline_deriv(v: torch.Tensor, knots: torch.Tensor,
                 gamma: torch.Tensor) -> torch.Tensor:
    """s'(v; gamma) = gamma_1 + sum_j gamma_{j+1} B'_j(v). Shape (n,)."""
    deriv = rcs_deriv_design(v, knots)
    return deriv @ gamma[1:]


# ---------------------------------------------------------------------------
# Orthonormalised working basis
# ---------------------------------------------------------------------------

class RCSBasis:
    """Orthonormalised working basis for a fixed knot set.

    The truncated-power design B(v) = [v, B_1(v), ..., B_{m-2}(v)] is badly
    conditioned: with widely separated boundary knots the cubic columns span
    a range orders of magnitude larger than the linear column. That is
    survivable for the classical fit (damped Newton is affine-invariant) but
    not for the neural phase, where Adam's diagonal preconditioning is not
    affine-invariant and the global spline coefficients would learn far more
    slowly than the encoder weights - a quiet failure in which the baseline
    hazard shape underfits while the loss still descends.

    Fix: QR-factorise B on a FIXED DENSE GRID spanning [k_1, k_m] and work in
    Q(v) = B(v) R^-1. The grid is deterministic and depends only on the
    knots, so this is never a train/test-fitted transform - it is identical
    across folds by construction. The function space is unchanged; only the
    coordinate system differs, and `to_original_gamma` maps coefficients back
    for interpretable reporting.

    R must travel with any checkpoint: a model loaded without its basis is
    silently wrong. Use `to_dict()` / `from_dict()`.
    """

    def __init__(self, knots, n_grid: int = 1000, dtype=torch.float64,
                 _R: torch.Tensor = None):
        self.knots = _check_knots(torch.as_tensor(knots, dtype=torch.float64))
        self.n_grid = int(n_grid)
        if _R is None:
            grid = torch.linspace(self.knots[0], self.knots[-1], self.n_grid,
                                  dtype=torch.float64)
            B = rcs_design(grid, self.knots)
            _, R = torch.linalg.qr(B)
            # Fix the sign convention (positive diagonal) so the factorisation
            # is unique and reproducible across platforms/versions.
            signs = torch.sign(torch.diagonal(R))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            R = signs.unsqueeze(1) * R
            # Scale to UNIT RMS per column rather than unit norm. Plain QR
            # gives columns of norm 1, i.e. entries ~ 1/sqrt(n_grid), so the
            # coefficients needed to represent an O(1) function are ~sqrt(n_grid)
            # (~94 for a unit-slope spline at n_grid=1000). Adam takes a fixed
            # step per iteration, so that scale alone would need ~10k steps to
            # traverse - orthogonality without sane scaling would trade one
            # optimisation pathology for another. With unit RMS the working
            # coefficients live on the same scale as the original gammas.
            _R = R / (self.n_grid ** 0.5)
        self.R = _R.to(torch.float64)
        self.R_inv = torch.linalg.inv(self.R)
        self._dtype = dtype

    # -- basis evaluation ---------------------------------------------------

    def design(self, v: torch.Tensor) -> torch.Tensor:
        """Q(v) = B(v) R^-1, shape (n, m-1)."""
        knots = self.knots.to(v.dtype)
        return rcs_design(v, knots) @ self.R_inv.to(v.dtype)

    def deriv_design(self, v: torch.Tensor) -> torch.Tensor:
        """Q'(v) = B'(v) R^-1, shape (n, m-1)."""
        knots = self.knots.to(v.dtype)
        return rcs_deriv_design(v, knots) @ self.R_inv.to(v.dtype)

    # -- coefficient mapping ------------------------------------------------

    def to_original_gamma(self, alpha: torch.Tensor) -> torch.Tensor:
        """Working coefficients -> truncated-power gamma (gamma = R^-1 alpha).

        Accepts either the m-1 working coefficients alone, or the full m-vector
        [intercept, alpha...]; the intercept is unaffected by the transform and
        is passed through, so the return matches the input convention.
        """
        R_inv = self.R_inv.to(alpha.dtype)
        if alpha.numel() == R_inv.shape[0]:
            return R_inv @ alpha
        if alpha.numel() == R_inv.shape[0] + 1:
            return torch.cat([alpha[:1], R_inv @ alpha[1:]])
        raise ValueError(
            f"alpha has {alpha.numel()} entries; expected "
            f"{R_inv.shape[0]} or {R_inv.shape[0] + 1}"
        )

    def to_working_alpha(self, gamma: torch.Tensor) -> torch.Tensor:
        """Inverse of `to_original_gamma` (alpha = R gamma)."""
        R = self.R.to(gamma.dtype)
        if gamma.numel() == R.shape[0]:
            return R @ gamma
        if gamma.numel() == R.shape[0] + 1:
            return torch.cat([gamma[:1], R @ gamma[1:]])
        raise ValueError(
            f"gamma has {gamma.numel()} entries; expected "
            f"{R.shape[0]} or {R.shape[0] + 1}"
        )

    # -- housekeeping -------------------------------------------------------

    @property
    def n_basis(self) -> int:
        """Number of working columns (m-1); add 1 for the intercept."""
        return self.R.shape[0]

    def condition_numbers(self) -> tuple:
        """(cond(B^T B), cond(Q^T Q)) on the construction grid - the
        diagnostic that motivates this class."""
        grid = torch.linspace(self.knots[0], self.knots[-1], self.n_grid,
                              dtype=torch.float64)
        B = rcs_design(grid, self.knots)
        Q = B @ self.R_inv
        return (torch.linalg.cond(B.T @ B).item(),
                torch.linalg.cond(Q.T @ Q).item())

    def to_dict(self) -> dict:
        return {"knots": self.knots.tolist(), "n_grid": self.n_grid,
                "R": self.R.tolist()}

    @classmethod
    def from_dict(cls, d: dict, dtype=torch.float64) -> "RCSBasis":
        return cls(torch.tensor(d["knots"], dtype=torch.float64),
                   n_grid=d["n_grid"], dtype=dtype,
                   _R=torch.tensor(d["R"], dtype=torch.float64))

    def __repr__(self) -> str:
        cb, cq = self.condition_numbers()
        return (f"RCSBasis(m={self.knots.numel()} knots, n_basis={self.n_basis}, "
                f"cond(B'B)={cb:.3e} -> cond(Q'Q)={cq:.3e})")
