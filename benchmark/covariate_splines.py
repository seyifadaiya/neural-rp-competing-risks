"""
Restricted cubic spline expansion of CONTINUOUS COVARIATES for the classical
baselines.

This is distinct from the time splines inside the Royston-Parmar model - it
expands covariates, not log time - but it reuses the same verified basis, so
the two agree by construction rather than by coincidence.

WHY THIS EXISTS. The paper claims neural representation learning improves
calibrated absolute risk beyond the best classical models. That claim is only
defensible if the baselines can express nonlinearity too. Entering donor age
or cold ischemia linearly while the encoder learns an arbitrary function
would rig the comparison, and a reviewer would say so. Every continuous
covariate therefore gets 4 knots at Harrell's quantiles, adding two columns
each.

Knots are fitted on TRAINING rows only and serialise with the model: a model
restored without its knots evaluates a different function while looking
perfectly healthy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# Harrell's recommended placement for 4 knots.
HARRELL_QUANTILES_4 = (0.05, 0.35, 0.65, 0.95)

# A variable needs enough distinct values for a cubic spline to mean anything.
# NPKID (a 0-4 count) falls below this and stays linear, as specified.
MIN_DISTINCT_FOR_SPLINE = 20


@dataclass
class CovariateSplineSpec:
    """Which columns are expanded and at which knots. Fitted on training rows."""
    knots: dict = field(default_factory=dict)        # column name -> knot list
    expanded: list = field(default_factory=list)     # names, in expansion order
    input_columns: list = field(default_factory=list)
    quantiles: tuple = HARRELL_QUANTILES_4
    min_distinct: int = MIN_DISTINCT_FOR_SPLINE

    def output_columns(self) -> list:
        """Design-matrix column names after expansion, in array order."""
        cols = []
        for c in self.input_columns:
            cols.append(c)
            if c in self.knots:
                n_extra = len(self.knots[c]) - 2
                cols += [f"{c}_rcs{j + 1}" for j in range(n_extra)]
        return cols

    @property
    def n_output(self) -> int:
        return len(self.output_columns())

    def to_json(self, path=None) -> str:
        payload = {"knots": {k: list(map(float, v)) for k, v in self.knots.items()},
                   "expanded": list(self.expanded),
                   "input_columns": list(self.input_columns),
                   "quantiles": list(self.quantiles),
                   "min_distinct": int(self.min_distinct)}
        text = json.dumps(payload, indent=2)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_json(cls, text_or_path) -> "CovariateSplineSpec":
        text = text_or_path
        if isinstance(text_or_path, str) and not text_or_path.lstrip().startswith("{"):
            with open(text_or_path) as fh:
                text = fh.read()
        d = json.loads(text)
        return cls(knots={k: list(v) for k, v in d["knots"].items()},
                   expanded=list(d["expanded"]),
                   input_columns=list(d["input_columns"]),
                   quantiles=tuple(d["quantiles"]),
                   min_distinct=int(d["min_distinct"]))


def _rcs_columns(v: np.ndarray, knots) -> np.ndarray:
    """The k-2 nonlinear RCS columns for one variable (the linear term is the
    original column, kept separately).

    Same truncated-power basis as `rp_core.rcs_basis`; `test_baselines.py`
    asserts the two agree.
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1, 1)
    k = np.asarray(knots, dtype=np.float64)
    k1, km, inner = k[0], k[-1], k[1:-1]
    lam = (km - inner) / (km - k1)
    cube = lambda z: np.clip(z, 0.0, None) ** 3
    return cube(v - inner) - lam * cube(v - k1) - (1.0 - lam) * cube(v - km)


def fit_covariate_splines(X_train, columns, quantiles=HARRELL_QUANTILES_4,
                          min_distinct=MIN_DISTINCT_FOR_SPLINE,
                          only: list | None = None) -> CovariateSplineSpec:
    """Choose which columns to expand and place their knots, from TRAINING rows.

    `only` restricts expansion to a named subset (the design matrix already
    contains one-hot indicators, which must not be expanded).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    columns = list(columns)
    assert X_train.shape[1] == len(columns), "X and column names disagree"

    knots = {}
    expanded = []
    for j, c in enumerate(columns):
        if only is not None and c not in only:
            continue
        col = X_train[:, j]
        col = col[np.isfinite(col)]
        if np.unique(col).size < min_distinct:
            continue                       # counts and indicators stay linear
        k = np.unique(np.quantile(col, quantiles))
        if k.size < 3:
            continue                       # too little spread to place knots
        knots[c] = k.tolist()
        expanded.append(c)
    return CovariateSplineSpec(knots=knots, expanded=expanded,
                               input_columns=columns, quantiles=tuple(quantiles),
                               min_distinct=int(min_distinct))


def apply_covariate_splines(X, spec: CovariateSplineSpec) -> np.ndarray:
    """Expand `X` using an already-fitted spec. Nothing is learned here."""
    X = np.asarray(X, dtype=np.float64)
    assert X.shape[1] == len(spec.input_columns), (
        f"X has {X.shape[1]} columns, spec expects {len(spec.input_columns)}")
    blocks = []
    for j, c in enumerate(spec.input_columns):
        blocks.append(X[:, [j]])
        if c in spec.knots:
            blocks.append(_rcs_columns(X[:, j], spec.knots[c]))
    return np.hstack(blocks)


def events_per_variable(n_events: int, n_columns: int) -> float:
    """Events per estimated parameter - the number to report alongside the
    design width, since RCS expansion buys flexibility at the cost of
    parameters."""
    return float(n_events) / max(n_columns, 1)


def describe(spec: CovariateSplineSpec, n_events: dict | None = None) -> str:
    lines = [f"covariate splines: {len(spec.expanded)} of "
             f"{len(spec.input_columns)} columns expanded "
             f"({spec.quantiles}, min {spec.min_distinct} distinct values)",
             f"  design width {len(spec.input_columns)} -> {spec.n_output} "
             f"(+{spec.n_output - len(spec.input_columns)})"]
    for c in spec.expanded:
        k = np.round(spec.knots[c], 4)
        lines.append(f"    {c:24s} knots {list(k)}")
    if n_events:
        for cause, n in n_events.items():
            epv = events_per_variable(n, spec.n_output)
            flag = "  *** thin ***" if epv < 10 else ""
            lines.append(f"  cause {cause}: {n} events, "
                         f"{epv:.1f} events per variable{flag}")
    return "\n".join(lines)
