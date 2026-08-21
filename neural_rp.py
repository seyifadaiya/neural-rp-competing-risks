"""
Neural Royston-Parmar competing-risks models.

Variant A - global spline shape, encoder shifts it (proportional hazards):

    log H_k(t | x) = gamma_k0 + s_k(log t; alpha_k) + w_k^T z(x)

The spline shape s_k is GLOBAL - alpha_k are free nn.Parameters, not
produced by the network - so the baseline hazard shape is shared across
patients and the encoder only shifts it. That is proportional hazards
within each cause, and it means

    d log H_k / dv = s'_k(log t)

is independent of x, which is what `rp_competing_loss` receives.

Two design points carried over from the verified core:

* the spline uses the ORTHONORMALISED basis (rcs_basis.RCSBasis), because
  Adam is not affine-invariant and the truncated-power geometry would let
  the global spline coefficients underfit while the loss still descends;
* the derivative is ANALYTIC (Q'(v) alpha), never autograd - this is the
  hot loop.

Monotonicity (s'_k > 0) is a property of alpha_k alone, so it is checked on
a dense grid rather than per-observation, and a violation is surfaced
loudly rather than clamped away.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from encoder import CovariateEncoder, EncoderConfig
from rcs_basis import RCSBasis
from rp_likelihood import rp_competing_loss

CAUSES = (1, 2)


@dataclass
class VariantAConfig:
    encoder: EncoderConfig
    spline_lr: float = 1e-2
    encoder_lr: float = 1e-3

    def to_dict(self) -> dict:
        return {"encoder": self.encoder.to_dict(),
                "spline_lr": self.spline_lr,
                "encoder_lr": self.encoder_lr}

    @classmethod
    def from_dict(cls, d: dict) -> "VariantAConfig":
        return cls(encoder=EncoderConfig.from_dict(d["encoder"]),
                   spline_lr=d["spline_lr"], encoder_lr=d["encoder_lr"])


class VariantA(nn.Module):
    """Global-spline neural RP model. float32 by default (neural path)."""

    def __init__(self, basis: dict, config: VariantAConfig,
                 scaler_mean: torch.Tensor = None,
                 scaler_std: torch.Tensor = None,
                 dtype=torch.float32):
        super().__init__()
        assert set(basis) == set(CAUSES), "need one RCSBasis per cause"
        self.basis = basis
        self.config = config
        self.dtype_ = dtype
        self.encoder = CovariateEncoder(config.encoder, scaler_mean, scaler_std,
                                        dtype=dtype)

        for k in CAUSES:
            n_b = basis[k].n_basis
            # Initialise at s(v) ~ v, i.e. gamma = [0, 1, 0, ...] mapped into
            # the working basis, so training starts from a sane Weibull-like
            # baseline instead of a flat one.
            gamma0 = torch.zeros(n_b, dtype=torch.float64)
            gamma0[0] = 1.0
            alpha0 = basis[k].to_working_alpha(gamma0).to(dtype)
            self.register_parameter(f"alpha{k}", nn.Parameter(alpha0))
            self.register_parameter(
                f"intercept{k}", nn.Parameter(torch.tensor(-1.0, dtype=dtype))
            )
            self.add_module(f"w{k}", nn.Linear(config.encoder.latent_dim, 1,
                                               bias=False, dtype=dtype))
            nn.init.zeros_(getattr(self, f"w{k}").weight)

    # -- forward ------------------------------------------------------------

    def forward(self, log_t: torch.Tensor, x_cont: torch.Tensor,
                x_cat: torch.Tensor = None):
        """Returns (log_H, dlogH_dv), each (n, 2), for `rp_competing_loss`."""
        log_t = log_t.to(self.dtype_)
        z = self.encoder(x_cont, x_cat)
        outs, derivs = [], []
        for k in CAUSES:
            alpha = getattr(self, f"alpha{k}")
            design = self.basis[k].design(log_t)            # (n, m-1)
            ddesign = self.basis[k].deriv_design(log_t)     # analytic
            shift = getattr(self, f"w{k}")(z).squeeze(-1)
            outs.append(getattr(self, f"intercept{k}") + design @ alpha + shift)
            derivs.append(ddesign @ alpha)
        return torch.stack(outs, dim=1), torch.stack(derivs, dim=1)

    def loss(self, log_t, x_cont, event_type, x_cat=None):
        log_t = log_t.to(self.dtype_)
        log_H, dlogH = self(log_t, x_cont, x_cat)
        return rp_competing_loss(log_H, dlogH, event_type, log_t)

    # -- monotonicity -------------------------------------------------------

    @torch.no_grad()
    def min_spline_deriv(self, n_grid: int = 1000) -> dict:
        """min s'_k(v) over a dense grid spanning each cause's knot range.

        s'_k depends only on alpha_k, so this is a complete check of the
        fitted model's monotonicity - not a sample-dependent one.
        """
        out = {}
        for k in CAUSES:
            knots = self.basis[k].knots
            grid = torch.linspace(knots[0].item(), knots[-1].item(), n_grid,
                                  dtype=self.dtype_)
            d = self.basis[k].deriv_design(grid) @ getattr(self, f"alpha{k}")
            out[k] = d.min().item()
        return out

    def assert_monotone(self, n_grid: int = 1000) -> dict:
        mins = self.min_spline_deriv(n_grid)
        bad = {k: v for k, v in mins.items() if v <= 0}
        if bad:
            raise RuntimeError(
                f"cumulative hazard is non-monotone: min s'_k = {bad}. "
                "This is a hard failure - the fitted spline is not a valid "
                "cumulative hazard. Do not clamp it away."
            )
        return mins

    # -- reporting / persistence -------------------------------------------

    @torch.no_grad()
    def original_gamma(self, k: int) -> torch.Tensor:
        """Interpretable truncated-power RP coefficients for cause k,
        [gamma_0, gamma_1, ...] including the intercept."""
        alpha = getattr(self, f"alpha{k}").detach().to(torch.float64)
        gamma = self.basis[k].to_original_gamma(alpha)
        intercept = getattr(self, f"intercept{k}").detach().to(torch.float64)
        return torch.cat([intercept.reshape(1), gamma])

    def param_groups(self) -> list:
        """Spline coefficients and encoder weights in SEPARATE groups.

        The differential-learning-rate failure mode is the reason: with one
        shared rate the global spline coefficients move far more slowly than
        the encoder weights and the baseline shape underfits quietly.
        Orthonormalisation reduces this; separate groups make it
        controllable.
        """
        spline, other = [], []
        for name, p in self.named_parameters():
            (spline if name.startswith(("alpha", "intercept")) else other).append(p)
        return [
            {"params": spline, "lr": self.config.spline_lr, "name": "spline"},
            {"params": other, "lr": self.config.encoder_lr, "name": "encoder"},
        ]

    def checkpoint_dict(self) -> dict:
        """Everything needed to restore the model EXACTLY.

        The basis R matrices and the scaler are included deliberately: a
        model reloaded without them evaluates a different function while
        looking perfectly healthy.
        """
        return {
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "basis": {str(k): b.to_dict() for k, b in self.basis.items()},
            "dtype": str(self.dtype_).replace("torch.", ""),
        }

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "VariantA":
        dtype = getattr(torch, ckpt.get("dtype", "float32"))
        basis = {int(k): RCSBasis.from_dict(v) for k, v in ckpt["basis"].items()}
        model = cls(basis, VariantAConfig.from_dict(ckpt["config"]), dtype=dtype)
        model.load_state_dict(ckpt["state_dict"])
        return model


# ---------------------------------------------------------------------------
# Variant B - patient-specific hazard shape
# ---------------------------------------------------------------------------

SOFTPLUS_INV_1 = 0.541324854612918        # softplus^-1(1): b_k starts at 1


@dataclass
class VariantBConfig:
    encoder: EncoderConfig
    spline_lr: float = 1e-2
    encoder_lr: float = 1e-3
    head_lr: float = 1e-3

    def to_dict(self) -> dict:
        return {"encoder": self.encoder.to_dict(), "spline_lr": self.spline_lr,
                "encoder_lr": self.encoder_lr, "head_lr": self.head_lr}

    @classmethod
    def from_dict(cls, d: dict) -> "VariantBConfig":
        return cls(encoder=EncoderConfig.from_dict(d["encoder"]),
                   spline_lr=d["spline_lr"], encoder_lr=d["encoder_lr"],
                   head_lr=d.get("head_lr", 1e-3))


class VariantB(nn.Module):
    """Patient-specific hazard shape with a hard monotonicity guarantee:

        log H_k(t | x) = a_k(z) + b_k(z) * s~_k(log t; alpha_k),
        b_k(z) = softplus(.) > 0

    Unlike Variant A the encoder rescales the shape rather than only
    shifting it, so covariates modulate HOW STEEPLY risk accumulates over
    time - non-proportional hazards within cause.

    IDENTIFIABILITY. The naive form has two flat directions; both are
    removed structurally, never by regularisation:

    1. Additive. A spline intercept alpha_0 would enter as b_k * alpha_0 and
       duplicate the shift a_k(z) already supplies. So the spline carries NO
       intercept - it lives entirely in a_k.

    2. Multiplicative. (b_k, s~_k) and (c b_k, s~_k / c) are the same
       function, so the optimiser could drift along a ridge with b_k
       collapsing or inflating. So the LINEAR COEFFICIENT IS FIXED AT 1:

           s~_k(v) = v + sum_{j>=2} alpha_kj Q_j(v)

       Only the nonlinear working-basis coefficients are free parameters -
       `alpha_k` genuinely has n_basis - 1 entries; there is no alpha_k1 to
       drift. Q_j are the unit-RMS orthonormalised columns from RCSBasis,
       and because QR makes them orthogonal to the linear column, the
       nonlinear deviations are orthogonal to the part being held fixed.

    WEIBULL AT INITIALISATION. With every alpha_kj = 0 the model is exactly

           log H_k = a_k(z) + b_k(z) * v,

    i.e. a Weibull model with a_k the log scale and b_k(z) a patient-specific
    shape. Variant B is therefore "neural Weibull with spline deviations".
    The b_k head bias starts at softplus^-1(1) so training begins at shape 1.

    MONOTONICITY. b_k(z) > 0 by construction, so monotonicity of log H_k in v
    for EVERY patient follows from s~'_k(v) = 1 + sum_j alpha_kj Q'_j(v) > 0,
    which depends on alpha_k alone. It therefore stays a GLOBAL grid check,
    not a per-observation one.
    """

    def __init__(self, basis: dict, config: VariantBConfig,
                 scaler_mean: torch.Tensor = None,
                 scaler_std: torch.Tensor = None,
                 dtype=torch.float32,
                 constant_scale: bool = False):
        super().__init__()
        assert set(basis) == set(CAUSES), "need one RCSBasis per cause"
        self.basis = basis
        self.config = config
        self.dtype_ = dtype
        self.constant_scale = constant_scale
        self.encoder = CovariateEncoder(config.encoder, scaler_mean, scaler_std,
                                        dtype=dtype)
        latent = config.encoder.latent_dim

        for k in CAUSES:
            # Nonlinear working columns only - the linear coefficient is fixed
            # at 1 and is NOT a parameter (see class docstring).
            n_nl = basis[k].n_basis - 1
            self.register_parameter(
                f"alpha{k}", nn.Parameter(torch.zeros(n_nl, dtype=dtype))
            )
            a_head = nn.Linear(latent, 1, dtype=dtype)
            b_head = nn.Linear(latent, 1, dtype=dtype)
            nn.init.zeros_(a_head.weight)
            nn.init.constant_(a_head.bias, -1.0)
            nn.init.zeros_(b_head.weight)
            # Deliberate: start at shape 1 (a plain Weibull), not a default init.
            nn.init.constant_(b_head.bias, SOFTPLUS_INV_1)
            if constant_scale:
                # b_k(z) constant across patients: only the bias acts. Used by
                # the tests that nest Variant A.
                b_head.weight.requires_grad_(False)
            self.add_module(f"a{k}", a_head)
            self.add_module(f"b{k}", b_head)

    # -- pieces -------------------------------------------------------------

    def _shape_scale(self, z: torch.Tensor, k: int):
        a = getattr(self, f"a{k}")(z).squeeze(-1)
        b = torch.nn.functional.softplus(getattr(self, f"b{k}")(z)).squeeze(-1)
        return a, b

    def _spline(self, v: torch.Tensor, k: int):
        """(s~_k(v), s~'_k(v)) with the linear coefficient fixed at 1.

        Analytic derivative - the likelihood path never calls autograd.
        """
        alpha = getattr(self, f"alpha{k}")
        q_nl = self.basis[k].design(v)[:, 1:]           # drop the linear column
        dq_nl = self.basis[k].deriv_design(v)[:, 1:]
        return v + q_nl @ alpha, 1.0 + dq_nl @ alpha

    # -- forward ------------------------------------------------------------

    def forward(self, log_t: torch.Tensor, x_cont: torch.Tensor,
                x_cat: torch.Tensor = None):
        """Returns (log_H, dlogH_dv), each (n, 2).

        d log H_k / dv = b_k(z) * s~'_k(v) - patient-specific, unlike
        Variant A, and analytic.
        """
        log_t = log_t.to(self.dtype_)
        z = self.encoder(x_cont, x_cat)
        outs, derivs = [], []
        for k in CAUSES:
            a, b = self._shape_scale(z, k)
            s, ds = self._spline(log_t, k)
            outs.append(a + b * s)
            derivs.append(b * ds)
        return torch.stack(outs, dim=1), torch.stack(derivs, dim=1)

    def loss(self, log_t, x_cont, event_type, x_cat=None):
        log_t = log_t.to(self.dtype_)
        log_H, dlogH = self(log_t, x_cont, x_cat)
        return rp_competing_loss(log_H, dlogH, event_type, log_t)

    # -- monotonicity -------------------------------------------------------

    @torch.no_grad()
    def min_spline_deriv(self, n_grid: int = 1000) -> dict:
        """min s~'_k(v) over a dense grid. Because b_k(z) > 0, this is a
        complete monotonicity check for EVERY patient - no per-row scan."""
        out = {}
        for k in CAUSES:
            knots = self.basis[k].knots
            grid = torch.linspace(knots[0].item(), knots[-1].item(), n_grid,
                                  dtype=self.dtype_)
            _, ds = self._spline(grid, k)
            out[k] = ds.min().item()
        return out

    def assert_monotone(self, n_grid: int = 1000) -> dict:
        mins = self.min_spline_deriv(n_grid)
        bad = {k: v for k, v in mins.items() if v <= 0}
        if bad:
            raise RuntimeError(
                f"cumulative hazard is non-monotone: min s~'_k = {bad}. "
                "Hard failure - do not clamp it away."
            )
        return mins

    # -- diagnostics --------------------------------------------------------

    @torch.no_grad()
    def scale_stats(self, x_cont: torch.Tensor, x_cat: torch.Tensor = None) -> dict:
        """min / median / max of b_k(z) across a batch.

        A b_k distribution collapsing toward 0 or drifting large is the
        signature of an identifiability problem surviving the structural
        constraints. It should not happen; this is how it would show up.
        """
        z = self.encoder(x_cont, x_cat)
        out = {}
        for k in CAUSES:
            b = torch.nn.functional.softplus(getattr(self, f"b{k}")(z)).squeeze(-1)
            out[k] = {"min": b.min().item(), "median": b.median().item(),
                      "max": b.max().item()}
        return out

    def batch_diagnostics(self, batch: dict) -> dict:
        stats = self.scale_stats(batch["x_cont"], batch.get("x_cat"))
        return {f"b{k}": f"[{s['min']:.3f} {s['median']:.3f} {s['max']:.3f}]"
                for k, s in stats.items()}

    # -- optimiser / persistence -------------------------------------------

    def param_groups(self) -> list:
        """Three groups: spline coefficients, encoder weights, and the
        a_k / b_k heads. b_k sits inside a softplus, so its effective
        gradient scale differs from the rest and it needs its own control."""
        spline, heads, enc = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("alpha"):
                spline.append(p)
            elif name.startswith(("a1.", "a2.", "b1.", "b2.")):
                heads.append(p)
            else:
                enc.append(p)
        return [
            {"params": spline, "lr": self.config.spline_lr, "name": "spline"},
            {"params": enc, "lr": self.config.encoder_lr, "name": "encoder"},
            {"params": heads, "lr": self.config.head_lr, "name": "heads"},
        ]

    def checkpoint_dict(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "basis": {str(k): b.to_dict() for k, b in self.basis.items()},
            "dtype": str(self.dtype_).replace("torch.", ""),
            "constant_scale": self.constant_scale,
            "variant": "B",
        }

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "VariantB":
        dtype = getattr(torch, ckpt.get("dtype", "float32"))
        basis = {int(k): RCSBasis.from_dict(v) for k, v in ckpt["basis"].items()}
        model = cls(basis, VariantBConfig.from_dict(ckpt["config"]), dtype=dtype,
                    constant_scale=ckpt.get("constant_scale", False))
        model.load_state_dict(ckpt["state_dict"])
        return model
