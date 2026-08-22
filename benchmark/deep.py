"""
Deep-learning competing-risks baselines - the nearest published competitors.

Neural Fine-Gray is the closest architectural neighbour to this project's
model: shared embedding, per-risk heads, exact likelihood, coherent CIFs. It
differs only in that its heads are generic constrained-monotonic networks
rather than Royston-Parmar splines. Its absence would be the first thing a
reviewer flags, so it is the primary deep comparator here.

PROVENANCE, which belongs in the paper's methods:

  DeepHit            authors' method via `pycox` 0.3.0 (Kvamme's package, the
                     reference Python implementation). The NETWORK is ours,
                     because pycox requires the caller to supply one - built
                     to match Variant B's encoder.
  Neural Fine-Gray   the authors' own repository, unmodified, imported from a
                     local clone. Not on PyPI.
  DSM (competing)    authors' method via `auton-survival` 0.1.0.

None of these is a reimplementation. Where a package could not be installed
the baseline reports itself unavailable rather than silently substituting
something else.

WHAT COHERENCE MEANS HERE, because the extended table is easy to misread.
DeepHit's `F1 + F2 + S = 1` holds because a softmax over (cause, interval)
plus a no-event class IMPOSES it; Neural Fine-Gray's holds because it defines
`F_k = beta_k (1 - exp(log s_k))` with `sum_k beta_k = 1`. In both cases the
survival is the complement BY CONSTRUCTION and the residual is exactly zero -
definitionally, not as evidence of anything. The Royston-Parmar residual
(~1e-06) is a different quantity: there S is obtained by independently
integrating the fitted cumulative hazards, so a nonzero value is quadrature
error in a check that COULD have failed. A table showing "RP 3e-06, NFG 0.0"
therefore does not show NFG doing better, and the README says so.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn

CAUSES = (1, 2)

# Variant B's encoder, so "comparable capacity" is a fact about the code
# rather than an assertion in the paper.
DEFAULT_HIDDEN = (256, 128)
DEFAULT_HEAD = (64,)
DEFAULT_DROPOUT = 0.2


# ---------------------------------------------------------------------------
# Availability - probed, never assumed
# ---------------------------------------------------------------------------

def pycox_available() -> bool:
    try:
        import pycox  # noqa: F401
        import torchtuples  # noqa: F401
        return True
    except Exception:
        return False


def dsm_available() -> bool:
    try:
        from auton_survival.models.dsm import DeepSurvivalMachines  # noqa: F401
        return True
    except Exception:
        return False


NFG_ENV_VAR = "NFG_PATH"


def nfg_root(path: str | None = None) -> str | None:
    """Locate the Neural Fine-Gray clone.

    The authors do not publish to PyPI, so the code has to come from a clone:

        git clone --recursive https://github.com/Jeanselme/NeuralFineGray.git

    Pass the path explicitly or set `NFG_PATH`. Returning None (rather than
    raising or quietly swapping in something else) is what lets the baseline
    report itself unavailable.
    """
    candidate = path or os.environ.get(NFG_ENV_VAR)
    if not candidate:
        return None
    candidate = os.path.abspath(os.path.expanduser(candidate))
    if os.path.isdir(os.path.join(candidate, "nfg")):
        return candidate
    return None


def nfg_available(path: str | None = None) -> bool:
    root = nfg_root(path)
    if root is None:
        return False
    try:
        _import_nfg(root)
        return True
    except Exception:
        return False


def _import_nfg(root: str):
    """Import `NeuralFineGray` from the clone, including its DSM submodule.

    The repository imports `dsm.*` by absolute name from a git submodule, so
    both directories go on the path. Entries are inserted once and left there;
    removing them would break the lazily-imported internals mid-fit.
    """
    for extra in (root, os.path.join(root, "DeepSurvivalMachines")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from nfg import NeuralFineGray
    return NeuralFineGray


# ---------------------------------------------------------------------------
# Shared input handling
# ---------------------------------------------------------------------------

def onehot_design(X, n_continuous: int, cardinalities) -> np.ndarray:
    """`[continuous | integer codes]` -> a single float matrix.

    This is the `embedding` representation Variants A and B consume, adapted
    to networks that have no embedding layers. It is NOT a re-encoding: the
    codes come from the same fitted `FeatureSpec`, and no level mapping,
    scaling, or reference choice is recomputed here.

    Nor is it a handicap. A one-hot block followed by a linear layer computes
    exactly an embedding lookup whose dimension is the layer width, so the
    baselines lose no expressive power relative to Variant B's learned
    embeddings - if anything they have more, since the lookup is unconstrained.
    """
    X = np.asarray(X, dtype=np.float64)
    cont = X[:, :n_continuous]
    codes = X[:, n_continuous:]
    if codes.shape[1] != len(cardinalities):
        raise ValueError(
            f"X has {codes.shape[1]} categorical columns but {len(cardinalities)} "
            f"cardinalities were given; pass n_continuous and cardinalities "
            f"from the same FeatureSpec that produced X")
    blocks = [cont]
    for j, card in enumerate(cardinalities):
        c = codes[:, j].astype(np.int64)
        if c.size and (c.min() < 0 or c.max() >= card):
            raise ValueError(f"categorical column {j} has codes outside "
                             f"[0, {card}); the spec and X disagree")
        eye = np.zeros((len(c), int(card)), dtype=np.float64)
        eye[np.arange(len(c)), c] = 1.0
        blocks.append(eye)
    return np.hstack(blocks) if len(blocks) > 1 else blocks[0]


def mlp(d_in: int, hidden, d_out: int, dropout: float,
        final_activation: bool = False) -> nn.Sequential:
    layers, d = [], int(d_in)
    for h in hidden:
        layers += [nn.Linear(d, int(h)), nn.BatchNorm1d(int(h)), nn.ReLU(),
                   nn.Dropout(dropout)]
        d = int(h)
    layers.append(nn.Linear(d, int(d_out)))
    if final_activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@dataclass
class DeepConfig:
    """Knobs shared across the deep baselines, so the tuning budgets are
    visibly comparable rather than comparable by assertion."""
    hidden: tuple = DEFAULT_HIDDEN
    head: tuple = DEFAULT_HEAD
    dropout: float = DEFAULT_DROPOUT
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    patience: int = 5
    # DeepHit
    num_durations: int = 50
    alpha: float = 0.2
    sigma: float = 0.1
    # DSM
    k: int = 3
    distribution: str = "Weibull"
    # NFG
    act: str = "Tanh"

    def to_dict(self) -> dict:
        return {"hidden": list(self.hidden), "head": list(self.head),
                "dropout": self.dropout, "lr": self.lr,
                "batch_size": self.batch_size, "epochs": self.epochs,
                "patience": self.patience, "num_durations": self.num_durations,
                "alpha": self.alpha, "sigma": self.sigma, "k": self.k,
                "distribution": self.distribution, "act": self.act}

    def merged(self, **kw) -> "DeepConfig":
        known = {f: v for f, v in kw.items() if f in self.__dataclass_fields__}
        for seq in ("hidden", "head"):
            if seq in known:
                known[seq] = tuple(known[seq])
        return replace(self, **known)


def seed_everything(seed: int) -> None:
    np.random.seed(seed % (2 ** 31 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def _float32_default():
    """Build networks in float32 whatever the process-wide default dtype is.

    The verification tests set `torch.set_default_dtype(float64)` globally, and
    that leaks: a net constructed while it is in force gets Double parameters,
    which then meet the Float design matrix and raise

        mat1 and mat2 must have the same dtype, but got Float and Double

    Only in the full suite, never when the deep tests run alone - the failure
    depends on which other test happened to run first. `rp_core`'s own models
    already defend against this by casting inputs to their own dtype; the
    wrapped packages assume float32 throughout, so the fix here is to pin the
    default while their layers are created.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def _quiet():
    """Silence the vendored packages' progress bars and warnings.

    They write to stdout unconditionally, which turns a 25-cell benchmark log
    into thousands of tqdm lines and hides the harness's own output.
    """
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield buf


class _DeepBaseline:
    """Shared scaffolding: seeding, the one-hot design, and standardisation.

    Standardisation is fitted on TRAINING rows and reused at predict time -
    the same fit/apply discipline the feature builder and the classical
    baselines follow.
    """

    name = "deep"
    provenance = "unspecified"

    def __init__(self, columns=None, n_continuous=None, cardinalities=(),
                 seed=42, config: DeepConfig | None = None, **kw):
        self.columns = list(columns) if columns is not None else None
        self.cardinalities = tuple(int(c) for c in cardinalities)
        self.n_continuous = (int(n_continuous) if n_continuous is not None
                             else None)
        self.seed = int(seed)
        self.config = (config or DeepConfig()).merged(**kw)
        self.model = None
        self._mu = None
        self._sd = None

    # -- input -------------------------------------------------------------
    def _n_continuous(self, X) -> int:
        if self.n_continuous is not None:
            return self.n_continuous
        # No categoricals declared: every column is continuous.
        return np.asarray(X).shape[1] - len(self.cardinalities)

    def _design(self, X, fit: bool = False) -> np.ndarray:
        D = onehot_design(X, self._n_continuous(X), self.cardinalities)
        if fit:
            self._mu = D.mean(axis=0)
            sd = D.std(axis=0)
            self._sd = np.where(sd > 1e-12, sd, 1.0)
        if self._mu is None:
            raise RuntimeError("fit() must be called before predict")
        return ((D - self._mu) / self._sd).astype(np.float32)

    @staticmethod
    def _clean(time, event):
        t = np.asarray(time, dtype=np.float64)
        e = np.asarray(event).astype(np.int64)
        # Every wrapped package assumes strictly positive durations.
        t = np.maximum(t, 1e-8)
        return t, e

    # -- protocol ----------------------------------------------------------
    def fit(self, X, time, event, **kw):
        raise NotImplementedError

    def predict_cif(self, X, t_eval) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"model": self.name, "provenance": self.provenance,
                "seed": self.seed, **self.config.to_dict()}


# ---------------------------------------------------------------------------
# 1. DeepHit (Lee et al., AAAI 2018) via pycox
# ---------------------------------------------------------------------------

class _CauseSpecificNet(nn.Module):
    """Shared trunk, one head per cause, output `(batch, causes, durations)`.

    pycox's competing-risks loss requires the 3-D layout; the architecture
    follows the authors' cause-specific design, sized to match Variant B's
    encoder so capacity is comparable.
    """

    def __init__(self, d_in, hidden, head, n_causes, n_durations, dropout):
        super().__init__()
        hidden = tuple(hidden) or (64,)
        self.shared = mlp(d_in, hidden[:-1], hidden[-1], dropout,
                          final_activation=True)
        self.heads = nn.ModuleList([
            mlp(hidden[-1], head, n_durations, dropout)
            for _ in range(n_causes)])

    def forward(self, x):
        z = self.shared(x)
        return torch.stack([h(z) for h in self.heads], dim=1)


class DeepHitBaseline(_DeepBaseline):
    """Discrete-time joint PMF over (cause, interval) with a softmax output.

    The time discretisation is a MODELLING CHOICE that materially changes the
    answer, not an implementation detail, so `num_durations` is tuned
    (20/50/100) and reported alongside alpha and sigma. Bins are placed at
    equidistant quantiles of the observed event times, so each carries a
    similar number of events rather than a similar width.

    Coherence is exact here because softmax imposes it - see the module
    docstring. This model is the reason the paper's claim has to be about
    structural coherence in CONTINUOUS time, not about competitors producing
    incoherent probabilities.
    """

    name = "deephit"
    provenance = "authors' method via pycox 0.3.0; network supplied by us"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lab_trans = None
        self._duration_index = None

    @staticmethod
    def available() -> bool:
        return pycox_available()

    def fit(self, X, time, event, **kw):
        import torchtuples as tt
        from pycox.models import DeepHit
        from pycox.preprocessing.label_transforms import LabTransDiscreteTime

        seed_everything(self.seed)
        t, e = self._clean(time, event)
        D = self._design(X, fit=True)
        cfg = self.config

        class _LT(LabTransDiscreteTime):
            """Discretise durations while keeping the CAUSE label.

            The base class collapses events to 0/1; competing risks needs the
            cause index preserved, which is what `transform` restores.

            `fit` MUST be given binary events. pycox places quantile cuts from
            a Kaplan-Meier estimate whose inner loop is

                di[idx] += events[i]

            - it SUMS the event codes, and its docstring says "events array
            0/1". Passing competing-risks codes 0/1/2 therefore counts every
            cause-2 event as two deaths. The KM falls too fast (measured: 0.44
            at t = 5 where the truth was 0.58), the bins are packed too early
            (first cuts 0.48/0.89/1.40 instead of 0.70/1.33/2.00), and where
            enough cause-2 events coincide `di` exceeds `ni`, the survival goes
            negative, and the cut search runs off the end of the array with an
            IndexError.

            The cause labels are restored in `transform`, which is the only
            place the competing-risks loss needs them.
            """

            def fit(self, dur, ev):
                return super().fit(dur, (np.asarray(ev) > 0).astype("int64"))

            def transform(self, dur, ev):
                idx, _ = super().transform(dur, ev > 0)
                return idx, ev.astype("int64")

        self.lab_trans = _LT(int(cfg.num_durations), scheme="quantiles")
        y = self.lab_trans.fit_transform(t, e)
        self._duration_index = np.asarray(self.lab_trans.cuts, dtype=np.float64)

        with _float32_default():
            net = _CauseSpecificNet(D.shape[1], cfg.hidden, cfg.head,
                                    len(CAUSES), int(cfg.num_durations),
                                    cfg.dropout).float()
        self.model = DeepHit(net, tt.optim.Adam(cfg.lr), alpha=cfg.alpha,
                             sigma=cfg.sigma, duration_index=self._duration_index)

        # Early stopping needs a VALIDATION signal. Passing the callback with
        # val_data=None lets it watch the training loss instead, which on a
        # noisy minibatch objective fires within a few epochs and leaves the
        # model barely better than the marginal - a silent undertraining bug
        # that every downstream number would have absorbed without complaint.
        # NFG and DSM each carve an internal validation split for exactly this
        # reason, so DeepHit gets one too, on the same 15% and by the same seed.
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(D))
        n_val = max(1, int(round(0.15 * len(D))))
        va, tr = perm[:n_val], perm[n_val:]
        y_tr = (y[0][tr], y[1][tr])
        y_va = (y[0][va], y[1][va])
        # torchtuples' EarlyStopping checkpoints the best weights to a file and
        # is supposed to delete it afterwards. Observed leaving them behind, in
        # the PROCESS WORKING DIRECTORY, one per fit - which over a 25-cell
        # benchmark would quietly scatter checkpoint files through the user's
        # repository. Give it a path we own and delete the directory ourselves,
        # so nothing lands outside regardless of whether its cleanup runs.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="deephit_ckpt_") as tmp:
            callbacks = [tt.callbacks.EarlyStopping(
                patience=int(cfg.patience),
                file_path=os.path.join(tmp, "weights.pt"))]
            with _quiet(), _float32_default():
                self.model.fit(D[tr], y_tr, batch_size=int(cfg.batch_size),
                               epochs=int(cfg.epochs), callbacks=callbacks,
                               verbose=False, val_data=(D[va], y_va))
        return self

    def _grid_predictions(self, X):
        D = self._design(X)
        with _quiet(), _float32_default():
            cif = np.asarray(self.model.predict_cif(D), dtype=np.float64)
            surv = np.asarray(self.model.predict_surv(D), dtype=np.float64)
        # pycox returns (causes, times, n) and (times, n).
        return np.transpose(cif, (2, 1, 0)), surv.T

    def _interp(self, grid_vals, t_eval):
        """Step-function lookup onto `t_eval`.

        Deliberately NOT linear interpolation: the model is discrete-time, so
        the CIF is a step function and interpolating between bins would invent
        continuity the model does not claim.
        """
        idx = np.searchsorted(self._duration_index, t_eval, side="right") - 1
        out = np.empty((grid_vals.shape[0], len(t_eval)) + grid_vals.shape[2:])
        for j, i in enumerate(idx):
            if i < 0:
                out[:, j] = 0.0 if grid_vals.ndim == 3 else 0.0
            else:
                out[:, j] = grid_vals[:, min(i, grid_vals.shape[1] - 1)]
        return out

    def predict_cif(self, X, t_eval) -> np.ndarray:
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        cif, _ = self._grid_predictions(X)
        return np.clip(self._interp(cif, t_eval), 0.0, 1.0)

    def predict_survival(self, X, t_eval) -> np.ndarray:
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        _, surv = self._grid_predictions(X)
        out = np.ones((surv.shape[0], len(t_eval)))
        idx = np.searchsorted(self._duration_index, t_eval, side="right") - 1
        for j, i in enumerate(idx):
            if i >= 0:
                out[:, j] = surv[:, min(i, surv.shape[1] - 1)]
        return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Neural Fine-Gray (Jeanselme et al., CHIL 2023) - the primary comparator
# ---------------------------------------------------------------------------

class NeuralFineGrayBaseline(_DeepBaseline):
    """Shared embedding -> per-risk monotone networks -> balancing network.

    The closest architectural neighbour to this project's model: same
    decomposition, exact likelihood by automatic differentiation, coherent
    CIFs by construction. The difference that matters is the head - generic
    constrained-monotonic networks here, Royston-Parmar splines in Variant B -
    which is exactly the comparison the paper needs to make.

    Runs the AUTHORS' OWN CODE from their repository, unmodified. It is not on
    PyPI, so the clone location comes from `path=` or `NFG_PATH`; without it
    the baseline reports itself unavailable rather than being silently
    replaced by something else.
    """

    name = "neural_fine_gray"
    provenance = "authors' own repository (Jeanselme/NeuralFineGray), unmodified"

    def __init__(self, *a, path: str | None = None, **kw):
        super().__init__(*a, **kw)
        self.path = path
        self._root = None

    @staticmethod
    def available(path: str | None = None) -> bool:
        return nfg_available(path)

    def fit(self, X, time, event, **kw):
        self._root = nfg_root(self.path)
        if self._root is None:
            raise RuntimeError(
                "Neural Fine-Gray not found. Clone it and point at it:\n"
                "  git clone --recursive "
                "https://github.com/Jeanselme/NeuralFineGray.git\n"
                f"  export {NFG_ENV_VAR}=/path/to/NeuralFineGray")
        NeuralFineGray = _import_nfg(self._root)

        seed_everything(self.seed)
        t, e = self._clean(time, event)
        D = self._design(X, fit=True).astype(np.float64)
        cfg = self.config
        self.model = NeuralFineGray(
            layers=list(cfg.hidden), layers_surv=list(cfg.head),
            act=cfg.act, dropout=cfg.dropout)
        with _quiet():
            self.model.fit(D, t, e, n_iter=int(cfg.epochs), lr=cfg.lr,
                           bs=int(cfg.batch_size), random_state=self.seed)
        return self

    def predict_cif(self, X, t_eval) -> np.ndarray:
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        D = self._design(X).astype(np.float64)
        with _quiet():
            cols = [np.asarray(self.model.predict_risk(D, list(t_eval), risk=k),
                               dtype=np.float64) for k in CAUSES]
        return np.clip(np.stack(cols, axis=-1), 0.0, 1.0)

    def compute_nll(self, X, time, event) -> float:
        """Held-out negative log-likelihood, for the tuning criterion."""
        t, e = self._clean(time, event)
        D = self._design(X).astype(np.float64)
        with _quiet():
            return float(self.model.compute_nll(D, t, e))

    # NOTE: no `predict_survival`. The model DEFINES
    #   F_k = beta_k (1 - exp(log s_k)),  sum_k beta_k = 1,
    # so its survival is exactly 1 - sum_k F_k and the harness's clipped
    # complement is the same quantity. Exposing it would dress a definitional
    # identity up as an independent check.


# ---------------------------------------------------------------------------
# 3. Deep Survival Machines, competing-risks variant (Nagpal et al., 2021)
# ---------------------------------------------------------------------------

class DSMBaseline(_DeepBaseline):
    """Mixture of parametric distributions with a neural parameterisation -
    the parametric-mixture contrast to the spline approach.

    The number of mixture components `k` is tuned, as is the base
    distribution (Weibull / log-normal).

    WHAT auton-survival ACTUALLY RETURNS, because it is not a CIF and reading
    it as one was a mistake this wrapper originally made. Per risk k, the loss
    (`losses._conditional_weibull_loss`) is

        uncens = e == k        -> density term
        cens   = e != k        -> survival term

    so every subject whose event is NOT k, INCLUDING those who had the
    competing event, enters as censored. That is the textbook "censor the
    competing risk" construction, and it estimates the CAUSE-SPECIFIC HAZARD.
    `predict_survival(risk=k)` therefore returns a NET survival
    `S_k = exp(-H_k)`, and `predict_risk(risk=k) = 1 - S_k` is the 1-KM
    analogue: a latent-time probability, not a cumulative incidence.

    Two consequences follow, and both were measured. Each cause tends to 1 as
    t grows, so the pair tends to 2 - 1.9999 at t = 200 through the authors'
    own unmodified API. And `1 - S_k` over-states the CIF, in the same
    direction and for the same reason 1 - KM over-states cumulative incidence.

    So the CIF is built here the way it is built for every other
    cause-specific model in this benchmark; `CoxBaseline._integrate` does the
    identical thing with Breslow hazards:

        H_k(t) = -log S_k(t)                    from DSM's own fitted survival
        F_k(t) = sum_{u <= t} S(u-) dH_k(u),    S = prod (1 - sum_j dH_j)

    This is an INTEGRATION, not a normalisation: no free constant, no
    rescaling, and only quantities DSM itself estimated. Treating Cox this way
    and DSM the other way would have made DSM look incoherent for a reason
    that was ours, and would have scored it on a different estimand from every
    model it is compared against.
    """

    name = "dsm"
    provenance = "authors' method via auton-survival 0.1.0"

    # Points used to integrate the cause-specific hazards. auton-survival
    # evaluates one time point per forward pass, so predict cost is linear in
    # this; 100 is ample for horizons in years and keeps the discretisation
    # residual far below anything the coherence table resolves.
    n_integration_grid = 100

    @staticmethod
    def available() -> bool:
        return dsm_available()

    def fit(self, X, time, event, **kw):
        from auton_survival.models.dsm import DeepSurvivalMachines

        seed_everything(self.seed)
        t, e = self._clean(time, event)
        D = self._design(X, fit=True).astype(np.float64)
        cfg = self.config
        self.model = DeepSurvivalMachines(
            k=int(cfg.k), layers=list(cfg.hidden),
            distribution=cfg.distribution, random_seed=self.seed)
        with _quiet():
            self.model.fit(D, t, e, iters=int(cfg.epochs),
                           learning_rate=cfg.lr, batch_size=int(cfg.batch_size))
        return self

    def _net_survival(self, D, grid):
        """`S_k = exp(-H_k)` per cause on `grid`, straight from DSM."""
        with _quiet():
            return {k: np.asarray(self.model.predict_survival(
                        D, [float(g) for g in grid], risk=k), dtype=np.float64)
                    for k in CAUSES}

    def _integrate(self, D, t_eval):
        """Cause-specific hazards -> (CIF, event-free survival).

        Same product-limit recursion as `CoxBaseline._integrate`, for the same
        reason: with `S(t_j) = prod (1 - dH)` the per-cause sums telescope to
        `1 - S(t)` exactly, so F1 + F2 + S = 1 holds to machine precision
        rather than to O(dH^2).
        """
        t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
        t_max = float(np.max(t_eval))
        n_grid = max(int(self.n_integration_grid), 2 * len(t_eval))
        grid = np.unique(np.concatenate([
            np.linspace(t_max / n_grid, t_max, n_grid), t_eval]))

        surv_k = self._net_survival(D, grid)
        inc = {}
        for k in CAUSES:
            # H_k = -log S_k. The clip only keeps the log finite; it does not
            # rescale the estimate.
            H = -np.log(np.clip(surv_k[k], 1e-300, 1.0))
            H = np.maximum.accumulate(H, axis=1)      # H_k is non-decreasing
            inc[k] = np.diff(np.concatenate(
                [np.zeros((H.shape[0], 1)), H], axis=1), axis=1)

        total = np.clip(sum(inc[k] for k in CAUSES), 0.0, 1.0 - 1e-12)
        keep = 1.0 - total
        surv = np.cumprod(keep, axis=1)
        surv_prev = surv / np.maximum(keep, 1e-300)
        raw = sum(inc[k] for k in CAUSES)
        scale = np.where(raw > 0, total / np.maximum(raw, 1e-300), 0.0)
        cif = {k: np.cumsum(surv_prev * inc[k] * scale, axis=1) for k in CAUSES}

        idx = np.clip(np.searchsorted(grid, t_eval), 0, len(grid) - 1)
        out = np.stack([cif[k][:, idx] for k in CAUSES], axis=-1)
        return np.clip(out, 0.0, 1.0), surv[:, idx]

    def predict_cif(self, X, t_eval) -> np.ndarray:
        return self._integrate(self._design(X).astype(np.float64), t_eval)[0]

    def predict_survival(self, X, t_eval) -> np.ndarray:
        """Event-free survival implied by the same integration.

        DSM has no joint survival of its own - `predict_survival(risk=k)` is
        per-cause - but once the cause-specific hazards are integrated jointly
        there is one, and it is exactly what the CIFs were built against.
        """
        return self._integrate(self._design(X).astype(np.float64), t_eval)[1]

    def compute_nll(self, X, time, event) -> float:
        t, e = self._clean(time, event)
        D = self._design(X).astype(np.float64)
        with _quiet():
            return float(self.model.compute_nll(D, t, e))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEEP_BASELINES = {
    "deephit": DeepHitBaseline,
    "neural_fine_gray": NeuralFineGrayBaseline,
    "dsm": DSMBaseline,
}


def available_deep_baselines(nfg_path: str | None = None) -> dict:
    """Which deep baselines this environment can actually run, and why not."""
    return {
        "deephit": (DeepHitBaseline.available(),
                    "" if DeepHitBaseline.available() else "pycox not installed"),
        "neural_fine_gray": (
            NeuralFineGrayBaseline.available(nfg_path),
            "" if NeuralFineGrayBaseline.available(nfg_path)
            else f"clone the repository and set {NFG_ENV_VAR}"),
        "dsm": (DSMBaseline.available(),
                "" if DSMBaseline.available() else "auton-survival not installed"),
    }


def make_deep_baselines(n_continuous, cardinalities, columns=None, seed=42,
                        configs: dict | None = None, nfg_path: str | None = None,
                        include=None):
    """`factory(seed)` per available deep baseline, for `run_benchmark`.

    Unavailable baselines are OMITTED rather than replaced, so a missing
    package shows up as an absent row in the results instead of a silent
    substitution.
    """
    configs = configs or {}
    wanted = set(include) if include is not None else set(DEEP_BASELINES)
    out = {}
    for name, cls in DEEP_BASELINES.items():
        if name not in wanted:
            continue
        ok = (cls.available(nfg_path) if name == "neural_fine_gray"
              else cls.available())
        if not ok:
            continue
        cfg = configs.get(name)
        extra = {"path": nfg_path} if name == "neural_fine_gray" else {}
        out[name] = (lambda s, cls=cls, cfg=cfg, extra=extra: cls(
            columns=columns, n_continuous=n_continuous,
            cardinalities=cardinalities, seed=s, config=cfg, **extra))
    return out
