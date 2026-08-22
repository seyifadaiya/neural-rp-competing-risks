"""Verification of Variant B — patient-specific hazard shape.

    log H_k(t | x) = a_k(z) + b_k(z) * s~_k(log t; alpha_k),  b_k = softplus > 0

Variant B extends the chain of trust one link further:

    Variant B  ->  Variant A  ->  fit_classical_rp  ->  rstpm2

Following the pattern established for Variant A, the nesting check is split
into a parameter-planting test that pins the MODEL (optimiser-independent)
and a training test that isolates the OPTIMISER. The substantive claim of
the model — non-proportional hazards — is tested in both directions: it must
buy real held-out likelihood when PH is false, and cost almost nothing when
PH is true.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))

from encoder import EncoderConfig, fit_scaler                          # noqa: E402
from gen_data import BETA, SCALE, SHAPE, generate                      # noqa: E402
from neural_rp import (SOFTPLUS_INV_1, VariantA, VariantB,             # noqa: E402
                       VariantBConfig)
from rcs_basis import RCSBasis                                         # noqa: E402
from train import TrainConfig, set_seed, train_model                   # noqa: E402
from test_variant_a import (as_batch, build_variant_a, comparison_grid,  # noqa: E402
                            knots_for)

SEED = 42
N_GRID = 200


def build_variant_b(knots, x_train, hidden_dims=(), latent_dim=None,
                    dropout=0.0, dtype=torch.float32, constant_scale=False,
                    spline_lr=1e-2, encoder_lr=1e-3, head_lr=1e-3):
    p = x_train.shape[1]
    cfg = VariantBConfig(
        encoder=EncoderConfig(n_continuous=p, hidden_dims=tuple(hidden_dims),
                              latent_dim=latent_dim if latent_dim else p,
                              dropout=dropout),
        spline_lr=spline_lr, encoder_lr=encoder_lr, head_lr=head_lr)
    mean, std = fit_scaler(torch.as_tensor(x_train, dtype=dtype))
    basis = {k: RCSBasis(knots[k]) for k in (1, 2)}
    return VariantB(basis, cfg, scaler_mean=mean, scaler_std=std, dtype=dtype,
                    constant_scale=constant_scale)


# ---------------------------------------------------------------------------
# 1. Weibull at initialisation
# ---------------------------------------------------------------------------

def test_1a_spline_is_exactly_linear_at_init():
    """With every alpha_kj = 0 the spline is EXACTLY s~(v) = v, s~'(v) = 1,
    so the model is exactly Weibull: log H_k = a_k(z) + b_k(z) v."""
    time, event, x = generate(n=2000)
    knots = {k: knots_for(time, event, k) for k in (1, 2)}
    set_seed(SEED)
    model = build_variant_b(knots, x)
    v = torch.linspace(-4, 2, 300, dtype=torch.float32)
    for k in (1, 2):
        s, ds = model._spline(v, k)
        assert torch.equal(s, v), "s~(v) != v at init"
        assert torch.equal(ds, torch.ones_like(v)), "s~'(v) != 1 at init"
    stats = model.scale_stats(torch.tensor(x, dtype=torch.float32))
    for k in (1, 2):
        # b_k head bias is softplus^-1(1), so training starts at shape 1.
        assert abs(stats[k]["median"] - 1.0) < 1e-6
        assert abs(getattr(model, f"b{k}").bias.item() - SOFTPLUS_INV_1) < 1e-6


def test_1b_weibull_parameter_recovery():
    """Fitted against Weibull-simulated competing-risks data with alpha
    frozen at 0, the model must recover the true shape and scale.

    This proves the parameterisation is what it claims: a_k is the log scale
    and b_k the Weibull shape. n is raised above the standard 5,000 so
    sampling error is small enough to assert against — at n=5,000 the MLE
    for this generator sits ~2 SE from truth on the binary covariates, which
    `fit_classical_rp` and rstpm2 both reproduce, so it is the data, not the
    model.
    """
    time, event, x = generate(n=40_000, seed=7)
    knots = {k: knots_for(time, event, k) for k in (1, 2)}
    batch = as_batch(time, event, x)
    set_seed(SEED)
    model = build_variant_b(knots, x, constant_scale=True,
                            encoder_lr=5e-2, head_lr=5e-2)
    model.alpha1.requires_grad_(False)      # force the exact Weibull submodel
    model.alpha2.requires_grad_(False)
    train_model(model, batch, batch,
                TrainConfig(max_epochs=5000, batch_size=len(time), patience=5000,
                            scheduler="cosine", seed=SEED, verbose=False))
    model.eval()
    mean = model.encoder.scaler_mean
    std = model.encoder.scaler_std
    for k in (1, 2):
        shape = torch.nn.functional.softplus(getattr(model, f"b{k}").bias).item()
        w = getattr(model, f"a{k}").weight.detach().squeeze(0)
        beta = w / std
        log_scale = getattr(model, f"a{k}").bias.item() - float((beta * mean).sum())
        print(f"  cause{k}: shape {shape:.4f} (true {SHAPE[k]}), "
              f"log-scale {log_scale:.4f} (true {np.log(SCALE[k]):.4f}), "
              f"beta {np.round(beta.numpy(), 3)} (true {BETA[k]})")
        assert abs(shape / SHAPE[k] - 1) < 0.05
        assert abs(log_scale - np.log(SCALE[k])) < 0.15
        assert np.abs(beta.numpy() - BETA[k]).max() < 0.12


# ---------------------------------------------------------------------------
# 2 & 3. Nesting Variant A
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nesting():
    """Variant A fit, plus the data/knots both nesting tests share."""
    time, event, x = generate()
    knots = {k: knots_for(time, event, k) for k in (1, 2)}
    batch = as_batch(time, event, x)
    set_seed(SEED)
    model_a = build_variant_a(knots, x, spline_lr=1e-1, encoder_lr=5e-2)
    train_model(model_a, batch, batch,
                TrainConfig(max_epochs=6000, batch_size=len(time), patience=6000,
                            scheduler="cosine", seed=SEED, verbose=False))
    model_a.eval()
    return model_a, knots, batch, time, event, x


def plant_variant_a_into_b(model_a, model_b):
    """Exact mapping of a Variant A solution into constrained Variant B.

    Variant A:  log H = i_A + sum_j alphaA_j Q_j(v) + w_A^T z
    Variant B:  log H = a_bias + w_a^T z + c [ v + sum_{j>=2} alphaB_j Q_j(v) ]

    QR makes the first working column proportional to v, Q_1 = v / R_00, so
    v = R_00 Q_1(v) and matching coefficients gives

        c         = alphaA_1 / R_00        (hence b bias = softplus^-1(c))
        alphaB_j  = alphaA_j / c   (j >= 2)
        a_bias    = i_A,   w_a = w_A
    """
    with torch.no_grad():
        for k in (1, 2):
            alpha_a = getattr(model_a, f"alpha{k}").detach().double()
            r00 = model_b.basis[k].R[0, 0]
            c = (alpha_a[0] / r00).item()
            assert c > 0, f"cause {k}: implied scale {c} is not positive"
            getattr(model_b, f"alpha{k}").copy_((alpha_a[1:] / c).float())
            getattr(model_b, f"b{k}").bias.fill_(float(np.log(np.expm1(c))))
            getattr(model_b, f"a{k}").weight.copy_(getattr(model_a, f"w{k}").weight)
            getattr(model_b, f"a{k}").bias.fill_(
                getattr(model_a, f"intercept{k}").item())


def test_2_nests_variant_a_by_planting(nesting):
    """Optimiser-independent: with b_k(z) constant, Variant B contains
    Variant A exactly, so planted parameters must agree at the float32 floor."""
    model_a, knots, batch, time, event, x = nesting
    set_seed(SEED)
    model_b = build_variant_b(knots, x, constant_scale=True)
    plant_variant_a_into_b(model_a, model_b)
    model_b.eval()

    v = comparison_grid(time)
    x0 = torch.zeros(N_GRID, x.shape[1], dtype=torch.float32)
    with torch.no_grad():
        lh_a, d_a = model_a(v, x0)
        lh_b, d_b = model_b(v, x0)
        nll_a = model_a.loss(batch["log_t"], batch["x_cont"], batch["event_type"]).item()
        nll_b = model_b.loss(batch["log_t"], batch["x_cont"], batch["event_type"]).item()
    diffs = {k: (lh_b[:, k - 1] - lh_a[:, k - 1]).abs().max().item() for k in (1, 2)}
    dderiv = {k: (d_b[:, k - 1] - d_a[:, k - 1]).abs().max().item() for k in (1, 2)}
    print(f"\n  planted: max|dlogH| cause1 {diffs[1]:.3e}, cause2 {diffs[2]:.3e}; "
          f"max|d dlogH/dv| {dderiv[1]:.3e}/{dderiv[2]:.3e}; "
          f"NLL gap {nll_b - nll_a:+.2e}")
    for k in (1, 2):
        assert diffs[k] < 1e-5, f"cause {k}: {diffs[k]:.3e}"
        assert dderiv[k] < 1e-5, f"cause {k} derivative: {dderiv[k]:.3e}"
    assert abs(nll_b - nll_a) < 1e-5


def test_3_trained_nesting_matches_variant_a(nesting):
    """Same constrained configuration, trained. With test 2 passing, any
    residual here is attributable to the optimiser alone."""
    model_a, knots, batch, time, event, x = nesting
    set_seed(SEED)
    model_b = build_variant_b(knots, x, constant_scale=True,
                              spline_lr=1e-1, encoder_lr=5e-2, head_lr=5e-2)
    history, _ = train_model(model_b, batch, batch,
                             TrainConfig(max_epochs=6000, batch_size=len(time),
                                         patience=6000, scheduler="cosine",
                                         seed=SEED, verbose=False))
    model_b.eval()
    v = comparison_grid(time)
    x0 = torch.zeros(N_GRID, x.shape[1], dtype=torch.float32)
    with torch.no_grad():
        lh_a, _ = model_a(v, x0)
        lh_b, _ = model_b(v, x0)
        nll_a = model_a.loss(batch["log_t"], batch["x_cont"], batch["event_type"]).item()
        nll_b = model_b.loss(batch["log_t"], batch["x_cont"], batch["event_type"]).item()
    diffs = {k: (lh_b[:, k - 1] - lh_a[:, k - 1]).abs().max().item() for k in (1, 2)}
    report = (f"\n  trained nesting: cause1 {diffs[1]:.3e}, cause2 {diffs[2]:.3e}"
              f"\n  mean NLL: B {nll_b:.8f} vs A {nll_a:.8f} (gap {nll_b - nll_a:+.2e})"
              f"\n  final grad norms: {history[-1].grad_norms}")
    print(report)
    for k in (1, 2):
        assert diffs[k] < 1e-2, f"cause {k} exceeds 1e-2{report}"


# ---------------------------------------------------------------------------
# 4 & 5. The substantive capability claim
# ---------------------------------------------------------------------------

def simulate_non_ph(n=12_000, seed=SEED, spread=0.35):
    """Competing risks where PH is FALSE by construction for cause 1.

    The Weibull shape itself depends on x1, shape_1(x) = 1.3 exp(0.35 x1), so
    the hazard ratio for x1 changes with time. Cause 2 stays proportional,
    which also gives the b_k diagnostics something to discriminate.
    """
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.standard_normal(n), rng.standard_normal(n),
                         rng.standard_normal(n), (rng.random(n) < 0.5).astype(float)])
    shape1 = 1.3 * np.exp(spread * x[:, 0])
    a1 = np.log(0.05) + 0.3 * x[:, 0] - 0.2 * x[:, 1]
    t1 = np.exp((np.log(rng.exponential(size=n)) - a1) / shape1)
    a2 = np.log(0.06) - 0.2 * x[:, 1] + 0.3 * x[:, 2]
    t2 = np.exp((np.log(rng.exponential(size=n)) - a2) / 0.9)
    t_ev = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2)
    c = rng.uniform(0, 9.0, size=n)
    time = np.minimum(t_ev, c)
    return time, np.where(t_ev <= c, cause, 0), x


def _fit_a_and_b(time, event, x, epochs=400):
    n = len(time)
    perm = np.random.default_rng(0).permutation(n)
    tr, te = perm[:int(0.7 * n)], perm[int(0.7 * n):]
    knots = {k: knots_for(time[tr], event[tr], k) for k in (1, 2)}
    b_tr, b_te = (as_batch(time[tr], event[tr], x[tr]),
                  as_batch(time[te], event[te], x[te]))
    cfg = dict(hidden_dims=(64, 32), latent_dim=16, dropout=0.0)
    tcfg = TrainConfig(max_epochs=epochs, batch_size=512, patience=40,
                       scheduler="cosine", seed=SEED, verbose=False)
    set_seed(SEED)
    model_a = build_variant_a(knots, x[tr], **cfg)
    train_model(model_a, b_tr, b_te, tcfg)
    model_a.eval()
    set_seed(SEED)
    model_b = build_variant_b(knots, x[tr], **cfg)
    train_model(model_b, b_tr, b_te, tcfg)
    model_b.eval()
    with torch.no_grad():
        nll_a = model_a.loss(b_te["log_t"], b_te["x_cont"], b_te["event_type"]).item()
        nll_b = model_b.loss(b_te["log_t"], b_te["x_cont"], b_te["event_type"]).item()
    return model_a, model_b, (nll_a - nll_b) * len(te), len(te), b_te


def test_4_non_proportionality_gain():
    """THE headline capability check: when PH is false, Variant B's
    patient-specific shape must buy real held-out likelihood over Variant A."""
    time, event, x = simulate_non_ph()
    model_a, model_b, gain, n_te, b_te = _fit_a_and_b(time, event, x)
    stats = model_b.scale_stats(b_te["x_cont"])
    print(f"\n  non-PH data: held-out log-lik gain (B - A) = {gain:+.2f} "
          f"over {n_te} test rows")
    print(f"  b_k spread: cause1 {stats[1]}, cause2 {stats[2]}")
    assert gain > 20.0, f"Variant B gained only {gain:+.2f} log-lik units"
    # Cause 1 is the non-PH one, so its shape multiplier should vary far more
    # across patients than cause 2's, which was simulated proportional.
    spread1 = stats[1]["max"] - stats[1]["min"]
    spread2 = stats[2]["max"] - stats[2]["min"]
    assert spread1 > spread2, (
        f"expected wider b_k spread on the non-PH cause: {spread1:.3f} "
        f"vs {spread2:.3f}")


def test_4b_bk_spread_is_not_encoder_capacity():
    """FALSIFICATION: b_k spread must track non-proportionality, not capacity.

    The per-cause reading of b_k on real data - "this cause is
    non-proportional, that one is not" - is only licensed if a PH truth
    produces a NARROW b_k under the same encoder that produced the wide one.
    Otherwise the spread is the network having capacity to spend, and the
    interpretation is unfounded.

    Measured as sd(log b_k): b_k multiplies the spline, so the meaningful
    scale is multiplicative, and percentile-based spread beats the min/max in
    `scale_stats`, which are noisy extremes over thousands of patients.

    Established by the fuller sweep this test compresses (3 seeds x 3 encoder
    sizes x 6 effect sizes):

      * PH truth gives sd(log b) in 0.02-0.08 at EVERY capacity from
        (64,32)->16 to (512,256)->128 - an 8x width increase moves it not at
        all, which is the result that kills the capacity explanation.
      * Non-PH truth at shape exponent 0.35 gives ~0.33, a 4-5x separation.
      * The response is near-linear, sd(log b) ~ 0.045 + 0.82 * exponent,
        detectable from an exponent of about 0.05.
      * The PH cause stays at 0.03-0.05 regardless of what the other cause
        does, so causes do not contaminate each other.
    """
    def sd_log_bk(model, x_cont):
        with torch.no_grad():
            z = model.encoder(x_cont, None)
            return {k: float(np.std(np.log(torch.nn.functional.softplus(
                        getattr(model, f"b{k}")(z)).squeeze(-1).numpy())))
                    for k in (1, 2)}

    out = {}
    for arm, sim in (("PH", lambda s: generate(n=8_000, seed=s)),
                     ("non-PH", lambda s: simulate_non_ph(n=8_000, seed=s))):
        for seed in (42, 43):
            time, event, x = sim(seed)
            n = len(time)
            perm = np.random.default_rng(0).permutation(n)
            tr, te = perm[:int(0.7 * n)], perm[int(0.7 * n):]
            knots = {k: knots_for(time[tr], event[tr], k) for k in (1, 2)}
            b_tr = as_batch(time[tr], event[tr], x[tr])
            b_te = as_batch(time[te], event[te], x[te])
            set_seed(seed)
            # Production capacity, not a small probe: capacity is the
            # hypothesis under test, so it must not be reduced here.
            m = build_variant_b(knots, x[tr], hidden_dims=(256, 128),
                                latent_dim=64, dropout=0.0)
            train_model(m, b_tr, b_te,
                        TrainConfig(max_epochs=300, batch_size=512, patience=40,
                                    scheduler="cosine", seed=seed, verbose=False))
            m.eval()
            for k, v in sd_log_bk(m, b_te["x_cont"]).items():
                out.setdefault((arm, k), []).append(v)

    print("\n  sd(log b_k) at production capacity (256,128)->64")
    for (arm, k), v in sorted(out.items()):
        print(f"    {arm:8s} cause{k}  {np.mean(v):.4f}  "
              f"({min(v):.4f}-{max(v):.4f})")

    # Every cell whose TRUTH is proportional must stay in the null band.
    null = out[("PH", 1)] + out[("PH", 2)] + out[("non-PH", 2)]
    alt = out[("non-PH", 1)]
    assert max(null) < 0.12, (
        f"a proportional truth produced sd(log b) = {max(null):.4f}; if PH "
        f"data can spread this far the per-cause reading is unfounded")
    assert min(alt) > 2 * max(null), (
        f"non-PH cause 1 ({min(alt):.4f}) must stand clear of the PH null "
        f"({max(null):.4f}) or the diagnostic cannot discriminate")


def test_5_no_material_loss_when_ph_holds():
    """The extra flexibility must cost little when it is not needed."""
    time, event, x = generate(n=12_000, seed=SEED)
    _, _, gain, n_te, _ = _fit_a_and_b(time, event, x)
    print(f"\n  PH data: held-out log-lik gain (B - A) = {gain:+.2f} "
          f"over {n_te} test rows")
    assert gain > -10.0, (
        f"Variant B lost {-gain:.2f} log-lik units where PH holds - the "
        f"extra capacity is being paid for without being used")


# ---------------------------------------------------------------------------
# 6-9. Monotonicity, identifiability, stability, checkpointing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_standard():
    time, event, x = generate()
    knots = {k: knots_for(time, event, k) for k in (1, 2)}
    batch = as_batch(time, event, x)
    set_seed(SEED)
    model = build_variant_b(knots, x, hidden_dims=(64, 32), latent_dim=16)
    train_model(model, batch, batch,
                TrainConfig(max_epochs=300, batch_size=512, patience=40,
                            scheduler="cosine", seed=SEED, verbose=False))
    model.eval()
    return model, batch


def test_6_monotonicity_global_grid(trained_standard):
    """min s~'_k > 0 on a 1,000-point grid, both causes, post-training.

    Per-observation checking is UNNECESSARY here, and that is the main
    reason Variant B is the primary model rather than Variant C. Because
    b_k(z) = softplus(.) > 0 for every patient by construction,

        d log H_k / dv = b_k(z) * s~'_k(v)

    is positive for EVERY patient exactly when s~'_k(v) > 0, and s~'_k
    depends only on the global alpha_k - not on z, not on the sample. So a
    single grid check over the knot range certifies monotonicity for all
    patients, including ones never seen in training.
    """
    model, _ = trained_standard
    mins = model.assert_monotone(n_grid=1000)
    print(f"\n  min s~'_k on 1000-point grid: {mins}")
    for k, val in mins.items():
        assert val > 0


def test_7_identifiability_diagnostics(trained_standard):
    """The structural constraints must actually hold after training:
    b_k neither collapsed nor exploded, and no free linear coefficient."""
    model, batch = trained_standard
    stats = model.scale_stats(batch["x_cont"])
    print(f"\n  b_k after training: {stats}")
    for k in (1, 2):
        assert 0.2 < stats[k]["median"] < 5.0, f"cause {k} median off-scale"
        assert stats[k]["min"] > 0.01, f"cause {k} b_k collapsed"
        assert stats[k]["max"] < 100.0, f"cause {k} b_k exploded"

    # The linear coefficient is genuinely FIXED, not merely initialised to 1:
    # alpha_k carries only the nonlinear columns, so there is no alpha_k1 to
    # drift along the multiplicative ridge.
    for k in (1, 2):
        alpha = getattr(model, f"alpha{k}")
        assert alpha.numel() == model.basis[k].n_basis - 1
        with torch.no_grad():
            saved = alpha.detach().clone()
            alpha.zero_()
            v = torch.linspace(-3, 2, 50, dtype=torch.float32)
            s, ds = model._spline(v, k)
            assert torch.equal(s, v) and torch.equal(ds, torch.ones_like(v))
            alpha.copy_(saved)
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.endswith("alpha_linear") or n.startswith("linear_coef")
                   for n in names)


def test_8_stability_100_steps_no_nan():
    """100 steps at the DEFAULT learning rates: no NaN/Inf in loss or grads."""
    time, event, x = generate()
    knots = {k: knots_for(time, event, k) for k in (1, 2)}
    set_seed(SEED)
    model = build_variant_b(knots, x, hidden_dims=(256, 128), latent_dim=64,
                            dropout=0.2)
    batch = as_batch(time, event, x)
    opt = torch.optim.Adam(model.param_groups())
    gen = torch.Generator().manual_seed(SEED)
    for step in range(100):
        idx = torch.randint(0, len(time), (512,), generator=gen)
        opt.zero_grad()
        loss = model.loss(batch["log_t"][idx], batch["x_cont"][idx],
                          batch["event_type"][idx])
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad {name}"
        opt.step()
    print(f"  100 steps completed, final mean NLL {loss.item():.6f}")


def test_9_checkpoint_roundtrip(trained_standard):
    """The basis R and the scaler must travel with the weights."""
    model, batch = trained_standard
    ckpt = model.checkpoint_dict()
    assert ckpt["variant"] == "B" and set(ckpt["basis"]) == {"1", "2"}
    restored = VariantB.from_checkpoint(ckpt)
    restored.eval()
    v = torch.linspace(-4.0, 2.0, 50, dtype=torch.float32)
    x0 = torch.zeros(50, model.encoder.config.n_continuous,
                     dtype=torch.float32)
    with torch.no_grad():
        a, da = model(v, x0)
        b, db = restored(v, x0)
    assert torch.equal(a, b) and torch.equal(da, db)
    assert torch.equal(model.encoder.scaler_mean, restored.encoder.scaler_mean)


def test_analytic_derivative_in_likelihood_path(trained_standard):
    """dlogH_dv must be b_k(z) * s~'_k(v), computed analytically."""
    model, batch = trained_standard
    v = batch["log_t"][:200]
    xc = batch["x_cont"][:200]
    with torch.no_grad():
        _, dlogH = model(v, xc)
        z = model.encoder(xc)
        for k in (1, 2):
            _, b = model._shape_scale(z, k)
            _, ds = model._spline(v, k)
            assert torch.equal(dlogH[:, k - 1], b * ds)
