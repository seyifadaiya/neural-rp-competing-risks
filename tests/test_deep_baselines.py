"""Validation of the deep-learning baselines and the shared tuning harness.

The concern here is the same one that governs the classical baselines: a
comparator that was misconfigured, undertrained, or tuned less carefully than
the proposed model is not a comparator. Two tests carry most of that weight -
test 3 (each baseline recovers a CIF whose generating model is known) and
test 4 (each beats an intercept-only predictor). A deep model that silently
fails to train still returns plausible-looking numbers, and finding that here
costs minutes where finding it in the results table costs the paper.

The coherence test is deliberately NOT a ranking. DeepHit and Neural Fine-Gray
construct survival as the complement of the CIFs, so their residual is zero by
definition, while the Royston-Parmar residual is quadrature error in a check
that could have failed. These are different quantities and the test asserts
each against its own expectation rather than against the others.
"""

import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from benchmark import BenchmarkCell, run_benchmark                      # noqa: E402
from benchmark.deep import (DeepConfig, DSMBaseline, DeepHitBaseline,   # noqa: E402
                            NeuralFineGrayBaseline, available_deep_baselines,
                            make_deep_baselines, onehot_design)
from benchmark.tuning import (SEARCH_SPACES, SearchSpace, assert_tunable,  # noqa: E402
                              make_cv_folds, random_search, score_fold,
                              search_space_table)
from metrics import aalen_johansen, brier_score                         # noqa: E402

SEED = 7
T_EVAL = np.array([1.0, 3.0, 5.0])
N_CONT = 5
CARDS = (3, 4)

# Small nets and short schedules: these tests check that the wrappers are
# correct, not that the models are state of the art.
FAST = DeepConfig(hidden=(64, 32), head=(32,), epochs=150, batch_size=256,
                  lr=1e-3, dropout=0.1, num_durations=50)


def simulate(n=4000, seed=SEED):
    """Competing risks with KNOWN cause-specific hazards.

    Both causes are exponential with a log-linear covariate effect, so the
    true CIF has a closed form and test 3 can compare against truth rather
    than against another estimator.
    """
    rng = np.random.default_rng(seed)
    cont = rng.normal(size=(n, N_CONT))
    cat = np.column_stack([rng.integers(0, c, n) for c in CARDS]).astype(float)
    X = np.hstack([cont, cat])
    eta1 = 0.8 * cont[:, 0] + 0.4 * cont[:, 1]
    eta2 = -0.5 * cont[:, 1] + 0.6 * cont[:, 2]
    h1 = 0.15 * np.exp(eta1)
    h2 = 0.10 * np.exp(eta2)
    l1 = rng.exponential(1.0 / h1)
    l2 = rng.exponential(1.0 / h2)
    t_ev = np.minimum(l1, l2)
    cause = np.where(l1 <= l2, 1, 2)
    c = rng.uniform(0, 12.0, n)
    time = np.minimum(t_ev, c)
    event = np.where(t_ev <= c, cause, 0)
    return X, time, event, h1, h2


def true_cif(h1, h2, t_eval):
    """Closed-form CIF for constant cause-specific hazards:

        F_k(t) = h_k / (h1 + h2) * (1 - exp(-(h1 + h2) t))
    """
    tot = h1 + h2
    out = np.zeros((len(h1), len(t_eval), 2))
    for j, t in enumerate(t_eval):
        frac = (1.0 - np.exp(-tot * t))
        out[:, j, 0] = h1 / tot * frac
        out[:, j, 1] = h2 / tot * frac
    return out


@pytest.fixture(scope="module")
def data():
    X, time, event, h1, h2 = simulate()
    n_tr = int(0.7 * len(time))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(time))
    tr, te = perm[:n_tr], perm[n_tr:]
    return {"Xtr": X[tr], "ttr": time[tr], "etr": event[tr],
            "Xte": X[te], "tte": time[te], "ete": event[te],
            "h1te": h1[te], "h2te": h2[te]}


def _skip_unless(cls):
    ok = cls.available()
    if not ok:
        pytest.skip(f"{cls.name} unavailable in this environment")


AVAILABLE = available_deep_baselines()
CLASSES = [DeepHitBaseline, NeuralFineGrayBaseline, DSMBaseline]


@pytest.fixture(scope="module")
def fitted(data):
    """One fit per available baseline, shared across tests."""
    models = {}
    for cls in CLASSES:
        if not cls.available():
            continue
        m = cls(n_continuous=N_CONT, cardinalities=CARDS, seed=SEED, config=FAST)
        t0 = _time.time()
        m.fit(data["Xtr"], data["ttr"], data["etr"])
        models[cls.name] = (m, _time.time() - t0)
    return models


def test_0_environment_report():
    """Which baselines this environment can run, and why not - printed so the
    reason is visible rather than inferred from a skip."""
    print("\n  deep baseline availability")
    for name, (ok, why) in AVAILABLE.items():
        print(f"    {name:18s} {'YES' if ok else 'NO ':3s}  {why}")
    assert set(AVAILABLE) == {"deephit", "neural_fine_gray", "dsm"}


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------

def test_1_protocol_conformance(fitted, data):
    """(n, T, 2), in [0,1], non-decreasing in t."""
    if not fitted:
        pytest.skip("no deep baseline available")
    n = len(data["tte"])
    for name, (m, _) in fitted.items():
        cif = m.predict_cif(data["Xte"], T_EVAL)
        assert cif.shape == (n, len(T_EVAL), 2), f"{name}: {cif.shape}"
        assert np.isfinite(cif).all(), f"{name}: non-finite CIF"
        assert cif.min() >= -1e-9 and cif.max() <= 1 + 1e-9, name
        d = np.diff(cif, axis=1)
        assert d.min() >= -1e-6, f"{name}: CIF decreased in t ({d.min():.2e})"
        print(f"\n  {name:18s} ok  F1+F2 max {cif.sum(-1).max():.4f}")


def test_1b_accepted_by_run_benchmark(data, tmp_path):
    """The harness must drive these unchanged - no protocol edits."""
    factories = make_deep_baselines(N_CONT, CARDS, seed=SEED,
                                    configs={k: FAST for k in AVAILABLE})
    if not factories:
        pytest.skip("no deep baseline available")
    cells = [BenchmarkCell(
        model_name=name, fold=0, imputation=1,
        train={"X": data["Xtr"], "time": data["ttr"], "event": data["etr"]},
        evaluate_on={"test": {"X": data["Xte"], "time": data["tte"],
                              "event": data["ete"]}},
        factory=f) for name, f in factories.items()]
    res = run_benchmark(cells, T_EVAL, tmp_path / "deep.parquet",
                        evaluate_fn=_cheap_evaluate, verbose=False)
    assert set(res.model) == set(factories)
    assert len(res) == len(factories) * 2 * len(T_EVAL)


def _cheap_evaluate(cif, time, event, t_eval, surv=None, **kw):
    bs = brier_score(cif, time, event, t_eval)
    dev = (float(np.abs(cif.sum(-1) + surv - 1).max()) if surv is not None
           else float("nan"))
    return {"meta": {"n": int(len(time)), "t_eval": list(map(float, t_eval)),
                     "n_events": {k: int((event == k).sum()) for k in (1, 2)}},
            "discrimination": {"auc": {k: [float("nan")] * len(t_eval)
                                       for k in (1, 2)}},
            "calibration": {"brier": {k: list(map(float, bs[k])) for k in (1, 2)}},
            "checks": {"coherence": {"max_abs_deviation": dev}}}


# ---------------------------------------------------------------------------
# 2. Coherence, per model, against its OWN expectation
# ---------------------------------------------------------------------------

def test_2_coherence_by_construction(fitted, data):
    """Extends the classical coherence table.

    This is not a ranking. DeepHit imposes coherence with a softmax over
    (cause, interval) plus a no-event class; Neural Fine-Gray defines
    F_k = beta_k (1 - exp(log s_k)) with sum_k beta_k = 1. For both, survival
    IS the complement and the residual is zero by definition. DSM fits one
    mixture per cause with nothing coupling them, so F1 + F2 may exceed 1 and
    the excess is the honest measurement.
    """
    if not fitted:
        pytest.skip("no deep baseline available")
    from benchmark.base import _survival_for

    print("\n  COHERENCE  max |F1 + F2 + S - 1|")
    for name, (m, _) in fitted.items():
        cif = m.predict_cif(data["Xte"], T_EVAL)
        surv = _survival_for(m, data["Xte"], T_EVAL, cif)
        dev = float(np.abs(cif.sum(-1) + surv - 1).max())
        total = cif.sum(-1)
        own = hasattr(m, "predict_survival")
        kind = "own S" if own else "complement"
        print(f"    {name:18s} {dev:.3e}  ({kind})  "
              f"F1+F2 max {total.max():.4f}")

        if name == "deephit":
            # Softmax imposes it; only floating-point error should remain.
            assert dev < 1e-5, "DeepHit's softmax should impose coherence"
            assert total.max() <= 1 + 1e-6
        elif name == "neural_fine_gray":
            assert dev < 1e-9, "NFG's survival is the complement by definition"
            assert total.max() <= 1 + 1e-6
        elif name == "dsm":
            # DSM's heads are cause-specific hazards (its loss censors the
            # competing event), so the wrapper integrates them jointly exactly
            # as CoxBaseline does. Coherence is then structural, and the
            # residual is the product-limit recursion's rounding.
            assert dev < 1e-9, (
                "DSM's CIFs are built by joint integration of its own "
                "cause-specific hazards; F1+F2+S=1 should hold to machine "
                "precision. A large residual means the integration regressed "
                "or predict_cif went back to using 1 - S_k directly.")
            assert total.max() <= 1 + 1e-6


def test_2b_dsm_upstream_returns_a_net_probability_not_a_cif(data):
    """Pin the upstream behaviour the DSM wrapper compensates for.

    auton-survival's `predict_risk(risk=k)` is `1 - exp(-H_k)` with `H_k` a
    CAUSE-SPECIFIC cumulative hazard - its loss censors every subject whose
    event is not k, including those who had the competing event. That makes it
    a net (1 - KM) probability, so the two causes each tend to 1 and the pair
    tends to 2, which no CIF pair can do.

    This asserts the upstream quantity IS unbounded and that our CIF is not.
    If auton-survival ever changes `predict_risk` to return a true cumulative
    incidence, this test fails and the wrapper's integration should be revisited
    rather than left silently doing the conversion twice.
    """
    _skip_unless(DSMBaseline)
    m = DSMBaseline(n_continuous=N_CONT, cardinalities=CARDS, seed=SEED,
                    config=FAST)
    m.fit(data["Xtr"], data["ttr"], data["etr"])

    far = np.array([1.0, 200.0])
    D = m._design(data["Xte"]).astype(np.float64)
    raw = np.stack([1.0 - np.asarray(m._net_survival(D, far)[k])
                    for k in (1, 2)], axis=-1)
    ours = m.predict_cif(data["Xte"], far)
    print(f"\n  upstream 1-S_k at t=200: sum max {raw.sum(-1).max():.4f}")
    print(f"  our integrated CIF     : sum max {ours.sum(-1).max():.6f}")

    assert raw.sum(-1).max() > 1.5, (
        "upstream no longer returns an unbounded net probability - if "
        "predict_risk now gives a real CIF, the wrapper's integration is "
        "double-converting and must be revisited")
    assert (raw[:, :, 0] <= 1 + 1e-9).all() and (raw[:, :, 1] <= 1 + 1e-9).all(), (
        "each cause alone must stay under 1; if not, the defect is in the "
        "single-cause path, not in the competing-risks combination")
    assert ours.sum(-1).max() <= 1 + 1e-9


def test_2c_deephit_bins_come_from_a_binary_kaplan_meier(data):
    """DeepHit's quantile cuts must be placed from a BINARY Kaplan-Meier.

    pycox builds them from its own KM, whose inner loop is `di[idx] +=
    events[i]` - it SUMS the event codes, and its docstring says "events array
    0/1". Handing it competing-risks codes 0/1/2 counts every cause-2 event as
    two deaths: the curve falls too fast, the bins bunch up early, and where
    enough cause-2 events coincide `di` exceeds `ni`, survival goes negative
    and the cut search runs off the end with an IndexError.

    This asserts the fitted cuts match the binary-KM cuts and NOT the
    raw-code ones, so the two can never be confused again.
    """
    _skip_unless(DeepHitBaseline)
    from pycox.preprocessing.discretization import cuts_quantiles

    m = DeepHitBaseline(n_continuous=N_CONT, cardinalities=CARDS, seed=SEED,
                        config=FAST)
    m.fit(data["Xtr"], data["ttr"], data["etr"])
    fitted = np.asarray(m._duration_index, dtype=np.float64)

    t = np.maximum(np.asarray(data["ttr"], dtype=np.float64), 1e-8)
    e = np.asarray(data["etr"]).astype(np.int64)
    want = cuts_quantiles(t, (e > 0).astype(np.int64), FAST.num_durations,
                          0.0, "float64")
    print(f"\n  fitted cuts   {np.round(fitted[:5], 4)} ... {fitted[-1]:.4f}")
    print(f"  binary-KM     {np.round(want[:5], 4)} ... {want[-1]:.4f}")
    assert np.allclose(fitted, want), (
        "DeepHit's bins are not the binary-KM quantile cuts; the event codes "
        "are reaching pycox's Kaplan-Meier unbinarised")

    # And the corrupted placement must be genuinely different, or this test
    # would pass for the wrong reason.
    try:
        raw = cuts_quantiles(t, e, FAST.num_durations, 0.0, "float64")
    except Exception:
        return                      # raw codes crashed outright - also fine
    if len(raw) == len(want):
        assert not np.allclose(raw, want), (
            "raw-code and binary cuts coincide on this data, so the test "
            "cannot detect the bug - use data with more cause-2 events")


# ---------------------------------------------------------------------------
# 3. Recovers a known CIF
# ---------------------------------------------------------------------------

def test_3_recovers_known_cif(fitted, data):
    """Each baseline's CIF tracks the truth from a known generating model.

    The data are exponential with log-linear covariate effects, so
    F_k(t) = h_k/(h1+h2) * (1 - exp(-(h1+h2)t)) exactly. A baseline that
    cannot recover this is misconfigured, and that is much cheaper to learn
    here than from the results table.

    Three checks, because a single pointwise tolerance conflates two very
    different things. Mean |error| measures per-patient DISPERSION, which is
    mostly estimator variance at this sample size and net width and would make
    the threshold a proxy for how long the model trained. The properties that
    indicate correct configuration are that the model ORDERS patients like the
    truth does, and that it is unbiased IN THE LARGE - the latter being the
    quantity this whole project is about.
    """
    if not fitted:
        pytest.skip("no deep baseline available")
    truth = true_cif(data["h1te"], data["h2te"], T_EVAL)
    print("\n  recovery of a known CIF")
    print(f"    {'model':18s} {'mean|err|':>10s} {'bias c1':>9s} {'bias c2':>9s}"
          f" {'corr c1':>8s} {'corr c2':>8s}")
    for name, (m, _) in fitted.items():
        cif = m.predict_cif(data["Xte"], T_EVAL)
        err = float(np.abs(cif - truth).mean())
        bias = (cif - truth).mean(axis=(0, 1))
        corr = [float(np.corrcoef(cif[:, -1, k], truth[:, -1, k])[0, 1])
                for k in (0, 1)]
        print(f"    {name:18s} {err:10.4f} {bias[0]:+9.4f} {bias[1]:+9.4f}"
              f" {corr[0]:8.3f} {corr[1]:8.3f}")

        # Gross failure: predictions nowhere near the truth at all.
        assert err < 0.15, f"{name} strays {err:.3f} from truth"
        # Ordering: catches a model that predicts the marginal for everyone,
        # which a small mean error can otherwise hide.
        assert min(corr) > 0.5, (
            f"{name} barely tracks the truth's ordering (corr {corr}); it is "
            f"probably predicting close to the marginal for every patient")
        # Calibration in the large - the project's endpoint. One tolerance for
        # all three now.
        #
        # DSM used to need a looser bound, because the wrapper read its
        # per-cause `1 - S_k` as a CIF. That quantity is a net (1 - KM)
        # probability, so it over-stated both causes by ~5 percentage points.
        # Integrating DSM's cause-specific hazards jointly - the same thing
        # CoxBaseline does - removed the bias (+0.045/+0.054 -> -0.007/-0.002)
        # and cut the error against the known truth from 0.067 to 0.037, so
        # DSM is now held to the same 0.05 as the others.
        assert np.abs(bias).max() < 0.05, (
            f"{name} is biased in the large by {bias}; discrimination can look "
            f"fine while absolute risk is wrong, which is the failure this "
            f"project exists to detect")


# ---------------------------------------------------------------------------
# 4. Beats a null model
# ---------------------------------------------------------------------------

def test_4_beats_null_model(fitted, data):
    """Materially better than an intercept-only Aalen-Johansen predictor.

    Silent training failure is the characteristic deep-learning bug: the model
    converges to the marginal, every downstream number looks reasonable, and
    nothing else in the pipeline notices.
    """
    if not fitted:
        pytest.skip("no deep baseline available")
    marginal = np.zeros((len(data["tte"]), len(T_EVAL), 2))
    for k in (1, 2):
        est, _ = aalen_johansen(data["ttr"], data["etr"], T_EVAL, cause=k,
                               variance=False)
        marginal[:, :, k - 1] = np.asarray(est, dtype=np.float64)[None, :]
    bs_null = brier_score(marginal, data["tte"], data["ete"], T_EVAL)
    null_mean = np.mean([np.mean(bs_null[k]) for k in (1, 2)])

    print(f"\n  held-out Brier vs null (null = {null_mean:.5f})")
    for name, (m, _) in fitted.items():
        cif = m.predict_cif(data["Xte"], T_EVAL)
        bs = brier_score(cif, data["tte"], data["ete"], T_EVAL)
        mean = np.mean([np.mean(bs[k]) for k in (1, 2)])
        print(f"    {name:18s} {mean:.5f}  ({100 * (1 - mean / null_mean):+.1f}%)")
        assert mean < null_mean, f"{name} is no better than the marginal"


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", CLASSES, ids=[c.name for c in CLASSES])
def test_5_determinism(cls, data):
    """Same seed, same data, same predictions."""
    _skip_unless(cls)
    quick = DeepConfig(hidden=(32,), head=(16,), epochs=30, batch_size=256)
    out = []
    for _ in range(2):
        m = cls(n_continuous=N_CONT, cardinalities=CARDS, seed=123, config=quick)
        m.fit(data["Xtr"][:1500], data["ttr"][:1500], data["etr"][:1500])
        out.append(m.predict_cif(data["Xte"][:200], T_EVAL))
    diff = float(np.abs(out[0] - out[1]).max())
    print(f"\n  {cls.name}: max |run1 - run2| = {diff:.3e}")
    assert diff < 1e-8, f"{cls.name} is not deterministic ({diff:.2e})"


# ---------------------------------------------------------------------------
# 6. Tuning harness
# ---------------------------------------------------------------------------

def test_6_search_spaces_are_comparable():
    """Budgets declared in one place, and visibly of similar scale."""
    table = search_space_table()
    print("\n" + table[["model", "criterion", "n_hyperparameters",
                        "grid_size"]].to_string(index=False))
    assert set(table.model) >= {"deephit", "neural_fine_gray", "dsm",
                                "variant_a", "variant_b"}
    # Every model gets a grid far larger than the 30-trial budget, so no model
    # is quietly searched exhaustively while another is sampled.
    assert (table.grid_size > 100).all()
    assert set(table.criterion) <= {"loglik", "ibs"}


def test_6b_tuning_refuses_held_out_windows():
    """The guard raises rather than warns - a search that touched the temporal
    window would invalidate every number later computed on it."""
    assert_tunable(np.array(["development"] * 10), imputation=1)
    for bad in ("temporal", "exploratory"):
        with pytest.raises(ValueError, match="held-out window"):
            assert_tunable(np.array(["development"] * 5 + [bad] * 2))
    with pytest.raises(ValueError, match="imputation"):
        assert_tunable(np.array(["development"]), imputation=3)


def test_6c_budget_respected_and_trials_logged(data, tmp_path):
    """Budget honoured, criterion recorded, every trial in the parquet."""
    space = SearchSpace("toy", {"lr": [1e-3, 3e-3], "dropout": [0.0, 0.1, 0.2]},
                        criterion="ibs")
    calls = {"n": 0}

    class Toy:
        name = "toy"

        def __init__(self, cfg, seed):
            self.cfg, self.seed = cfg, seed

        def fit(self, X, t, e):
            calls["n"] += 1
            self.base = np.array([0.1, 0.2, 0.3]) * (1 + self.cfg["dropout"])
            return self

        def predict_cif(self, X, t_eval):
            t_eval = np.atleast_1d(t_eval)
            f = 1 - np.exp(-0.1 * t_eval)
            out = np.zeros((len(X), len(t_eval), 2))
            out[:, :, 0] = f * (1 + self.cfg["dropout"])
            out[:, :, 1] = f * 0.5
            return np.clip(out, 0, 1)

    folds = make_cv_folds(data["Xtr"], data["ttr"], data["etr"],
                          np.arange(len(data["ttr"])) % 3, folds=(0, 1, 2))
    assert len(folds) == 3
    out = tmp_path / "trials.parquet"
    res = random_search("toy", lambda cfg, s: Toy(cfg, s), folds, T_EVAL,
                        budget=4, space=space, output_path=out, verbose=False,
                        window=np.array(["development"]), imputation=1)

    assert res.n_evaluated == 4, "budget not respected"
    assert res.criterion == "ibs"
    assert calls["n"] == 4 * 3, "each trial must be scored on every fold"
    logged = pd.read_parquet(out)
    assert len(logged) == 4
    assert logged.config.nunique() == 4, "the budget was spent on duplicates"
    assert (logged.fit_seconds >= 0).all()
    print(f"\n  budget 4 -> {res.n_evaluated} trials, best {res.best_score:+.5f} "
          f"at {res.best_config}")


def test_6d_small_grid_is_enumerated_not_resampled():
    """A grid smaller than the budget is enumerated exactly once.

    Sampling with replacement would report a budget of 30 while evaluating six
    distinct configurations, which would overstate the search in the paper.
    """
    space = SearchSpace("tiny", {"lr": [1e-3, 3e-3], "k": [1, 2, 3]},
                        criterion="ibs")

    class Const:
        def __init__(self, cfg, seed):
            self.cfg = cfg

        def fit(self, X, t, e):
            return self

        def predict_cif(self, X, t_eval):
            t_eval = np.atleast_1d(t_eval)
            out = np.zeros((len(X), len(t_eval), 2))
            out[:, :, 0] = 0.1 * self.cfg["k"]
            out[:, :, 1] = 0.1
            return out

    rng = np.random.default_rng(0)
    n = 400
    folds = [{"train": {"X": rng.normal(size=(n, 3)),
                        "time": rng.uniform(0.1, 5, n),
                        "event": rng.integers(0, 3, n)},
              "validate": {"X": rng.normal(size=(n, 3)),
                           "time": rng.uniform(0.1, 5, n),
                           "event": rng.integers(0, 3, n)}}]
    res = random_search("tiny", lambda c, s: Const(c, s), folds, T_EVAL,
                        budget=30, space=space, verbose=False)
    assert res.n_evaluated == space.size() == 6
    assert res.trials.config.nunique() == 6


def test_6e_failed_trials_are_recorded_not_dropped(data, tmp_path):
    """A diverging config must not silently shrink the budget."""
    space = SearchSpace("brk", {"bad": [True, False]}, criterion="ibs")

    class Sometimes:
        def __init__(self, cfg, seed):
            self.cfg = cfg

        def fit(self, X, t, e):
            if self.cfg["bad"]:
                raise FloatingPointError("diverged")
            return self

        def predict_cif(self, X, t_eval):
            t_eval = np.atleast_1d(t_eval)
            return np.full((len(X), len(t_eval), 2), 0.1)

    folds = make_cv_folds(data["Xtr"][:600], data["ttr"][:600],
                          data["etr"][:600], np.arange(600) % 2, folds=(0, 1))
    out = tmp_path / "t.parquet"
    res = random_search("brk", lambda c, s: Sometimes(c, s), folds, T_EVAL,
                        budget=2, space=space, output_path=out, verbose=False)
    logged = pd.read_parquet(out)
    assert len(logged) == 2, "the failed trial vanished from the log"
    assert logged.failed.sum() == 1
    assert logged.loc[logged.failed, "error"].iloc[0].startswith("FloatingPointError")
    assert res.best_config == {"bad": False}


def test_6f_selection_criterion_direction(data):
    """Higher is better for both criteria, so selection cannot flip sign."""
    class Fixed:
        def __init__(self, scale):
            self.scale = scale

        def predict_cif(self, X, t_eval):
            t_eval = np.atleast_1d(t_eval)
            out = np.zeros((len(X), len(t_eval), 2))
            out[:, :, 0] = np.clip(self.scale * (1 - np.exp(-0.15 * t_eval)), 0, 1)
            out[:, :, 1] = np.clip(0.10 * (1 - np.exp(-0.10 * t_eval)), 0, 1)
            return out

    good = score_fold(Fixed(1.0), data["Xte"], data["tte"], data["ete"],
                      T_EVAL, "ibs")
    bad = score_fold(Fixed(8.0), data["Xte"], data["tte"], data["ete"],
                     T_EVAL, "ibs")
    print(f"\n  ibs score: sane {good:+.5f}  absurd {bad:+.5f}")
    assert good > bad, "IBS is not oriented so that higher is better"


@pytest.mark.parametrize("cls", CLASSES, ids=[c.name for c in CLASSES])
def test_6g_declared_criterion_is_supported(cls, data):
    """A model declaring 'loglik' must actually expose a likelihood."""
    _skip_unless(cls)
    space = SEARCH_SPACES[cls.name]
    m = cls(n_continuous=N_CONT, cardinalities=CARDS, seed=SEED,
            config=DeepConfig(hidden=(32,), head=(16,), epochs=20))
    m.fit(data["Xtr"][:1200], data["ttr"][:1200], data["etr"][:1200])
    s = score_fold(m, data["Xte"][:400], data["tte"][:400], data["ete"][:400],
                   T_EVAL, space.criterion)
    print(f"\n  {cls.name:18s} criterion={space.criterion:7s} score {s:+.5f}")
    assert np.isfinite(s)
    if space.criterion == "loglik":
        assert hasattr(m, "compute_nll")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def test_7_onehot_design_preserves_information():
    """The one-hot adapter must not invent or lose anything."""
    codes = np.array([[0, 3], [2, 0], [1, 1]], dtype=float)
    cont = np.array([[0.5], [-1.0], [2.0]])
    X = np.hstack([cont, codes])
    D = onehot_design(X, 1, (3, 4))
    assert D.shape == (3, 1 + 3 + 4)
    assert np.array_equal(D[:, 0], cont[:, 0])
    assert np.array_equal(D[:, 1:4].sum(1), np.ones(3))
    assert np.array_equal(D[:, 4:8].sum(1), np.ones(3))
    assert D[0, 1] == 1.0 and D[0, 4 + 3] == 1.0

    # A code outside the declared cardinality means the spec and the data
    # disagree, which must fail loudly rather than one-hot into the wrong slot.
    with pytest.raises(ValueError, match="outside"):
        onehot_design(np.hstack([cont, np.array([[9, 0], [0, 0], [0, 0.]])]),
                      1, (3, 4))
    with pytest.raises(ValueError, match="cardinalities"):
        onehot_design(X, 1, (3,))


def test_7b_fitting_leaves_no_files_behind(data, tmp_path, monkeypatch):
    """No baseline may write into the working directory.

    torchtuples' EarlyStopping checkpoints the best weights to a file in the
    PROCESS WORKING DIRECTORY and was observed leaving them behind. One stray
    file per fit is invisible in a test run and becomes 25 scattered
    checkpoints through the user's repository over a benchmark matrix, so the
    wrapper hands it a temporary path it owns.
    """
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    quick = DeepConfig(hidden=(32,), head=(16,), epochs=25, batch_size=256)
    for cls in CLASSES:
        if not cls.available():
            continue
        m = cls(n_continuous=N_CONT, cardinalities=CARDS, seed=3, config=quick)
        m.fit(data["Xtr"][:1200], data["ttr"][:1200], data["etr"][:1200])
        m.predict_cif(data["Xte"][:200], T_EVAL)
    leftovers = sorted(p.name for p in work.iterdir())
    assert leftovers == [], f"fitting left files behind: {leftovers}"


@pytest.mark.parametrize("cls", CLASSES, ids=[c.name for c in CLASSES])
def test_7c_robust_to_process_wide_float64(cls, data):
    """Fitting must not depend on the process-wide default dtype.

    `test_equivalence.py` sets `torch.set_default_dtype(float64)` globally, and
    it leaks: a network built while that is in force gets Double parameters,
    meets the Float design matrix, and raises

        mat1 and mat2 must have the same dtype, but got Float and Double

    This passed in isolation and failed in the full suite - the failure
    depended on which test ran first, which is the worst kind. rp_core's own
    models already guard against this; the wrapped packages assume float32, so
    the wrapper pins the default while their layers are created.
    """
    _skip_unless(cls)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        m = cls(n_continuous=N_CONT, cardinalities=CARDS, seed=5,
                config=DeepConfig(hidden=(32,), head=(16,), epochs=20))
        m.fit(data["Xtr"][:1000], data["ttr"][:1000], data["etr"][:1000])
        cif = m.predict_cif(data["Xte"][:200], T_EVAL)
        assert cif.shape == (200, len(T_EVAL), 2)
        assert np.isfinite(cif).all()
        # and the wrapper must put the default back where it found it
        assert torch.get_default_dtype() is torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_8_runtime_projection(fitted):
    """Per-fit wall time, and what it implies at 37,000 training rows.

    37,000 is the size that matters: the development window holds 47,367 rows
    and 4/5 of that is what each CV fold trains on.
    """
    if not fitted:
        pytest.skip("no deep baseline available")
    n_train = 2800
    print(f"\n  per-fit seconds at n={n_train:,} (config: {FAST.hidden}, "
          f"{FAST.epochs} epochs)")
    for name, (_, secs) in sorted(fitted.items(), key=lambda kv: kv[1][1]):
        print(f"    {name:18s} {secs:6.1f}s")
    print("  These are SMALL-net, short-schedule fits; see the README for the "
          "measured 37,000-row timings at full width.")
