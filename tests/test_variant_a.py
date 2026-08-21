"""Part 5 — verification of Variant A.

The primary check is degenerate-case equivalence: with an identity encoder,
Variant A *is* a classical linear-covariate Royston-Parmar model, so it must
reproduce `fit_classical_rp` — which is itself verified against rstpm2. That
gives the chain of trust

    Variant A  ->  fit_classical_rp  ->  rstpm2

Anything Variant A adds beyond the classical model (the encoder) is then the
only thing that can explain a later discrepancy.
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
from gen_data import generate                                          # noqa: E402
from neural_rp import VariantA, VariantAConfig                         # noqa: E402
from rcs_basis import RCSBasis                                         # noqa: E402
from rp_classical import ClassicalRP, fit_classical_rp                 # noqa: E402
from train import TrainConfig, set_seed, train_model                   # noqa: E402

SEED = 42
N_GRID = 200


def knots_for(time, event, k, n_internal=3):
    lt = np.log(time[event == k])
    return torch.tensor(np.quantile(lt, np.linspace(0, 1, n_internal + 2)),
                        dtype=torch.float64)


def as_batch(time, event, x, dtype=torch.float32):
    return {"log_t": torch.log(torch.as_tensor(time, dtype=dtype)),
            "x_cont": torch.as_tensor(x, dtype=dtype),
            "event_type": torch.as_tensor(event, dtype=torch.long),
            "x_cat": None}


def build_variant_a(knots, x_train, hidden_dims=(), latent_dim=None,
                    dropout=0.0, dtype=torch.float32, spline_lr=1e-2,
                    encoder_lr=1e-3):
    p = x_train.shape[1]
    cfg = VariantAConfig(
        encoder=EncoderConfig(n_continuous=p, hidden_dims=tuple(hidden_dims),
                              latent_dim=latent_dim if latent_dim else p,
                              dropout=dropout),
        spline_lr=spline_lr, encoder_lr=encoder_lr,
    )
    mean, std = fit_scaler(torch.as_tensor(x_train, dtype=dtype))
    basis = {k: RCSBasis(knots[k]) for k in (1, 2)}
    return VariantA(basis, cfg, scaler_mean=mean, scaler_std=std, dtype=dtype)


def comparison_grid(time, n=N_GRID):
    """1st-99th percentile of observed log times.

    The same convention the rp_core R-equivalence test uses. Beyond the
    observed range the spline extrapolates linearly and almost no data
    constrains it, so differences there measure the flatness of the
    likelihood rather than the agreement of two fits.
    """
    lt = np.log(time)
    return torch.linspace(float(np.quantile(lt, 0.01)),
                          float(np.quantile(lt, 0.99)), n,
                          dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. Degenerate-case equivalence (primary check)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def degenerate():
    time, event, x = generate()
    knots = {k: knots_for(time, event, k) for k in (1, 2)}

    # Reference: the verified classical fit (float64, Newton-polished).
    t64 = torch.tensor(time)
    e64 = torch.tensor(event)
    x64 = torch.tensor(x)
    classical = ClassicalRP(knots, p=x.shape[1])
    classical_nll = fit_classical_rp(classical, t64, e64, x64)

    # Variant A with an identity encoder == the same model class.
    #
    # Optimiser settings are raised above the library defaults (spline 1e-2 /
    # encoder 1e-3) *for this test only*, with cosine annealing. The defaults
    # are sized for minibatch training, where an epoch is many steps; this
    # comparison is full-batch against a Newton-polished reference, so one
    # epoch is one step and the defaults simply run out of steps before
    # reaching the optimum. Annealing matters too: at constant learning rate
    # Adam oscillates around the optimum rather than settling into it.
    set_seed(SEED)
    model = build_variant_a(knots, x, spline_lr=1e-1, encoder_lr=5e-2)
    batch = as_batch(time, event, x)
    cfg = TrainConfig(max_epochs=6000, batch_size=len(time), patience=6000,
                      scheduler="cosine", seed=SEED, log_every=2000, verbose=True)
    history, _ = train_model(model, batch, batch, cfg)
    return model, classical, classical_nll, history, batch, time


def test_1a_representational_equivalence_exact(degenerate):
    """Optimiser-independent half of the primary check.

    Plant the classical fit's parameters into Variant A through the
    documented mappings (gamma -> alpha via the basis, beta -> w through the
    scaler) and require the two models to agree to the float32 floor. This
    proves the *model* and its coefficient mappings are equivalent; whatever
    residual the trained comparison shows is then attributable to Adam
    alone, which is exactly the distinction the task asks for.
    """
    _, classical, classical_nll, _, batch, time = degenerate
    knots = {k: classical.knots[k] for k in (1, 2)}
    set_seed(SEED)
    planted = build_variant_a(knots, batch["x_cont"].numpy())
    mean = planted.encoder.scaler_mean.double()
    std = planted.encoder.scaler_std.double()

    with torch.no_grad():
        for k in (1, 2):
            gamma = getattr(classical, f"gamma{k}").detach()
            beta = getattr(classical, f"beta{k}").detach()
            alpha = planted.basis[k].to_working_alpha(gamma[1:])
            getattr(planted, f"alpha{k}").copy_(alpha.float())
            # beta^T x = (beta*std)^T z + beta^T mean; the constant folds
            # into the intercept.
            getattr(planted, f"w{k}").weight.copy_(
                (beta * std).reshape(1, -1).float())
            getattr(planted, f"intercept{k}").copy_((gamma[0] + beta @ mean).float())

    planted.eval()
    v = comparison_grid(time)
    x0_32 = torch.zeros(N_GRID, classical.p, dtype=torch.float32)
    x0_64 = torch.zeros(N_GRID, classical.p, dtype=torch.float64)
    with torch.no_grad():
        lh_planted, _ = planted(v.to(torch.float32), x0_32)
        nll_planted = planted.loss(batch["log_t"], batch["x_cont"],
                                   batch["event_type"]).item()
    lh_classical, _ = classical.log_H_and_deriv(v.to(torch.float64), x0_64)

    diffs = {k: (lh_planted[:, k - 1].to(torch.float64)
                 - lh_classical[:, k - 1]).abs().max().item() for k in (1, 2)}
    print(f"\n  planted-parameter agreement: cause1 {diffs[1]:.3e}, "
          f"cause2 {diffs[2]:.3e}; mean NLL gap "
          f"{nll_planted - classical_nll:+.2e}")
    for k in (1, 2):
        assert diffs[k] < 1e-5, f"cause {k}: {diffs[k]:.3e} (float32 floor ~2e-6)"
    assert abs(nll_planted - classical_nll) < 1e-5


def test_1_degenerate_equivalence_with_classical(degenerate):
    model, classical, classical_nll, history, batch, time = degenerate
    model.eval()

    v = comparison_grid(time)
    x0_32 = torch.zeros(N_GRID, classical.p, dtype=torch.float32)
    x0_64 = torch.zeros(N_GRID, classical.p, dtype=torch.float64)
    with torch.no_grad():
        lh_neural, _ = model(v.to(torch.float32), x0_32)
    lh_classical, _ = classical.log_H_and_deriv(v.to(torch.float64), x0_64)

    diffs = {k: (lh_neural[:, k - 1].to(torch.float64)
                 - lh_classical[:, k - 1]).abs().max().item() for k in (1, 2)}

    # Diagnostics that separate "optimiser problem" from "model problem".
    with torch.no_grad():
        neural_nll = model.loss(batch["log_t"], batch["x_cont"],
                                batch["event_type"]).item()
    last = history[-1]
    report = (
        f"\n  max |delta log H|: cause1 {diffs[1]:.3e}, cause2 {diffs[2]:.3e}"
        f"\n  mean NLL: VariantA {neural_nll:.8f} vs "
        f"fit_classical_rp {classical_nll:.8f} "
        f"(gap {neural_nll - classical_nll:+.2e})"
        f"\n  final grad norms: {last.grad_norms}"
        f"\n  epochs run: {len(history)}"
    )
    print(report)
    for k in (1, 2):
        assert diffs[k] < 1e-2, f"cause {k} exceeds 1e-2 tolerance{report}"


def test_4_monotonicity_after_training(degenerate):
    model = degenerate[0]
    mins = model.assert_monotone(n_grid=1000)
    print(f"  min s'_k on 1000-point grid: {mins}")
    for k, val in mins.items():
        assert val > 0


def test_checkpoint_roundtrip_preserves_predictions(degenerate):
    """A checkpoint must carry its basis: a model restored without R
    evaluates a different function while looking healthy."""
    model = degenerate[0]
    model.eval()
    ckpt = model.checkpoint_dict()
    assert "basis" in ckpt and "1" in ckpt["basis"] and "2" in ckpt["basis"]

    restored = VariantA.from_checkpoint(ckpt)
    restored.eval()
    v = torch.linspace(-4.0, 2.0, 50, dtype=torch.float32)
    x0 = torch.zeros(50, model.encoder.config.n_continuous, dtype=torch.float32)
    with torch.no_grad():
        a, _ = model(v, x0)
        b, _ = restored(v, x0)
    assert torch.equal(a, b)
    # The scaler travels too (it is a buffer in the state dict).
    assert torch.equal(model.encoder.scaler_mean, restored.encoder.scaler_mean)
    assert torch.equal(model.encoder.scaler_std, restored.encoder.scaler_std)


# ---------------------------------------------------------------------------
# 2. Chain of trust: the classical fit still matches the R reference
# ---------------------------------------------------------------------------

def test_2_chain_of_trust_classical_still_matches_r():
    """Regression guard on the link Variant A depends on. The rest of the
    rp_core suite (test_equivalence.py) covers this in full; this asserts the
    critical link has not drifted while the basis work was done."""
    import json
    ref_path = HERE / "reference" / "r_reference.json"
    if not ref_path.exists():
        pytest.skip("r_reference.json missing - run fit_reference.R once")
    ref = json.loads(ref_path.read_text())
    data = np.genfromtxt(HERE / "reference" / "sim_data.csv",
                         delimiter=",", names=True)
    time = torch.tensor(data["time"])
    event = torch.tensor(data["event"].astype(np.int64))
    x = torch.stack([torch.tensor(data[c]) for c in ("x1", "x2", "x3", "x4")],
                    dim=1)
    knots = {k: torch.tensor(sorted(
        [ref[f"cause{k}"]["boundary_knots"][0]]
        + list(ref[f"cause{k}"]["internal_knots"])
        + [ref[f"cause{k}"]["boundary_knots"][1]])) for k in (1, 2)}

    model = ClassicalRP(knots, p=4)
    nll = fit_classical_rp(model, time, event, x)
    total = -nll * len(time)
    r_total = ref["cause1"]["logLik"] + ref["cause2"]["logLik"]
    assert abs(total - r_total) < 0.1, f"torch {total} vs R {r_total}"
    for k in (1, 2):
        tb = getattr(model, f"beta{k}").detach().numpy()
        rb = np.array([ref[f"cause{k}"]["coefficients"][c]
                       for c in ("x1", "x2", "x3", "x4")])
        assert np.abs(tb - rb).max() < 1e-3


# ---------------------------------------------------------------------------
# 3. Nonlinearity gain
# ---------------------------------------------------------------------------

def simulate_nonlinear(n=8000, seed=SEED):
    """Cause 1 has a genuinely nonlinear covariate effect,
    eta_1(x) = sin(2 x1) + x2 x3; cause 2 stays linear."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([
        rng.standard_normal(n), rng.standard_normal(n),
        rng.standard_normal(n), (rng.random(n) < 0.5).astype(float),
    ])
    eta = {1: np.sin(2 * x[:, 0]) + x[:, 1] * x[:, 2],
           2: -0.2 * x[:, 0] + 0.4 * x[:, 1]}
    shape, scale = {1: 1.3, 2: 0.9}, {1: 0.05, 2: 0.06}
    latent = {}
    for k in (1, 2):
        e = rng.exponential(size=n)
        latent[k] = (e / (scale[k] * np.exp(eta[k]))) ** (1.0 / shape[k])
    t_ev = np.minimum(latent[1], latent[2])
    cause = np.where(latent[1] <= latent[2], 1, 2)
    c = rng.uniform(0, 9.0, size=n)
    time = np.minimum(t_ev, c)
    event = np.where(t_ev <= c, cause, 0)
    return time, event, x


def test_3_nonlinearity_gain():
    """With a nonlinear truth, Variant A's encoder must buy real held-out
    likelihood over the linear classical model. Sanity check that the
    encoder does something - not a headline result."""
    time, event, x = simulate_nonlinear()
    n = len(time)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    knots = {k: knots_for(time[tr], event[tr], k) for k in (1, 2)}

    # Linear reference on the training rows.
    classical = ClassicalRP(knots, p=x.shape[1])
    fit_classical_rp(classical, torch.tensor(time[tr]),
                     torch.tensor(event[tr]), torch.tensor(x[tr]))
    with torch.no_grad():
        cls_te = classical.loss(torch.tensor(np.log(time[te])),
                                torch.tensor(x[te]),
                                torch.tensor(event[te])).item()

    # Variant A with a real encoder body.
    set_seed(SEED)
    model = build_variant_a(knots, x[tr], hidden_dims=(64, 32), latent_dim=16,
                            dropout=0.0)
    batch_tr = as_batch(time[tr], event[tr], x[tr])
    batch_te = as_batch(time[te], event[te], x[te])
    cfg = TrainConfig(max_epochs=600, batch_size=512, patience=40,
                      scheduler="cosine", seed=SEED, log_every=100)
    train_model(model, batch_tr, batch_te, cfg)
    model.eval()
    with torch.no_grad():
        neu_te = model.loss(batch_te["log_t"], batch_te["x_cont"],
                            batch_te["event_type"]).item()

    n_te = len(te)
    print(f"\n  held-out total log-lik: classical {-cls_te * n_te:.2f}, "
          f"VariantA {-neu_te * n_te:.2f}  "
          f"(gain {(cls_te - neu_te) * n_te:+.2f})")
    assert neu_te < cls_te, (
        f"Variant A did not improve on the linear model: "
        f"mean NLL {neu_te:.6f} vs {cls_te:.6f}"
    )
    # "Materially better": at least a few log-likelihood units on held-out data.
    assert (cls_te - neu_te) * n_te > 10.0
    assert all(v > 0 for v in model.min_spline_deriv().values())


# ---------------------------------------------------------------------------
# 5. Stability
# ---------------------------------------------------------------------------

def test_5_stability_100_steps_no_nan():
    """100 steps at the default learning rates: no NaN/Inf in loss or grads."""
    time, event, x = generate()
    knots = {k: knots_for(time, event, k) for k in (1, 2)}
    set_seed(SEED)
    model = build_variant_a(knots, x, hidden_dims=(256, 128), latent_dim=64,
                            dropout=0.2)
    batch = as_batch(time, event, x)
    opt = torch.optim.Adam(model.param_groups())

    n = len(time)
    gen = torch.Generator().manual_seed(SEED)
    for step in range(100):
        idx = torch.randint(0, n, (512,), generator=gen)
        opt.zero_grad()
        loss = model.loss(batch["log_t"][idx], batch["x_cont"][idx],
                          batch["event_type"][idx])
        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss}"
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"non-finite grad in {name} at step {step}"
                )
        opt.step()
    print(f"  100 steps completed, final mean NLL {loss.item():.6f}")


def test_analytic_derivative_used_in_forward():
    """The likelihood path must use the analytic derivative, not autograd:
    dlogH_dv from forward() must equal Q'(v) alpha exactly."""
    time, event, x = generate(n=200)
    knots = {k: knots_for(time, event, k) for k in (1, 2)}
    set_seed(SEED)
    model = build_variant_a(knots, x)
    v = torch.log(torch.tensor(time, dtype=torch.float32))
    with torch.no_grad():
        _, dlogH = model(v, torch.tensor(x, dtype=torch.float32))
        for k in (1, 2):
            expected = (model.basis[k].deriv_design(v)
                        @ getattr(model, f"alpha{k}"))
            assert torch.equal(dlogH[:, k - 1], expected)
