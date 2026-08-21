# `rp_core` — verified mathematical core of a Neural Royston-Parmar competing-risks model

Restricted cubic spline basis, competing-risks likelihood, and a classical
Royston-Parmar fit in PyTorch, checked against R's `rstpm2`. No neural
network training on real data and no SRTR data — this package exists so that
everything built on top of it inherits a verified core.

The model places restricted cubic splines on the log cumulative hazard as a
function of log time, per competing cause k ∈ {1, 2}:

```
log H_k(t | x) = s_k(log t; γ_k) + β_kᵀx
log h_k(t)     = log H_k(t) + log s′_k(log t) − log t
log L_i        = 1[δ_i > 0] · log h_{δ_i}(t_i) − Σ_k H_k(t_i)
```

Training is closed form — no numerical integration anywhere. The cumulative
incidence functions are the one quantity that needs quadrature, and they are
prediction-only (see `cif.py` below).

## Layout

| Path | What it is |
|---|---|
| `rcs_basis.py` | RCS basis, analytic derivative, spline evaluation, `RCSBasis` (orthonormalised working basis) |
| `rp_likelihood.py` | Vectorised competing-risks negative log-likelihood |
| `rp_classical.py` | Classical cause-specific RP model + MLE fitter |
| `encoder.py` | `CovariateEncoder` — shared covariate encoder for the neural variants |
| `neural_rp.py` | Variant A (global spline, encoder shifts it) and Variant B (patient-specific shape) |
| `train.py` | Generic training loop (variant-agnostic) |
| `cif.py` | Cumulative incidence functions (prediction only) |
| `metrics/` | Competing-risks evaluation metrics (model-agnostic; arrays in, numbers out) |
| `features/` | Feature builder: imputed parquet → model inputs, plus `build.ipynb` |
| `benchmark/` | Classical baselines and the resumable benchmark harness |
| `benchmark/deep.py` | Deep-learning baselines (DeepHit, Neural Fine-Gray, DSM) |
| `benchmark/tuning.py` | The shared hyperparameter search every model is tuned by |
| `reference/gen_data.py` | Synthetic competing-risks dataset (seed 42) |
| `reference/fit_reference.R` | R reference fit via `rstpm2`, exports JSON |
| `reference/sim_data.csv` | The dataset both implementations are fitted to |
| `reference/r_reference.json` | Cached R knots, coefficients, log-liks, curves |
| `tests/` | The test modules described below |

## Chain of trust

Each link is asserted by a test, so a failure localises to the layer that
introduced it:

```
Variant B  ->  Variant A  ->  fit_classical_rp  ->  rstpm2
 (test_variant_b.py)  (test_variant_a.py)   (test_equivalence.py)
```

Variant A with an identity encoder *is* a classical linear-covariate RP
model, so it must reproduce `fit_classical_rp`, which is itself verified
against `rstpm2`. Variant B with `b_k(z)` constrained to a constant *is*
Variant A. Anything each layer adds beyond the one below (the encoder, then
the patient-specific shape) is the only thing that can explain a new
discrepancy.

Every link is checked twice, in the pattern established for Variant A: a
**parameter-planting** test that pins the model and is independent of any
optimiser, and a **training** test that therefore isolates the optimiser.

## Running everything

```bash
uv run pytest rp_core/tests/ -q          # all tests (~8 min; the neural fits dominate)
uv run pytest rp_core/tests/ -q -s       # with recovery / equivalence / b_k / CIF tables
```

The R step is separate and only needs to run once — its output is cached in
`reference/r_reference.json`, which is committed. Regenerate it only if the
dataset or the reference model changes:

```bash
uv run python rp_core/reference/gen_data.py     # writes sim_data.csv
Rscript rp_core/reference/fit_reference.R       # writes r_reference.json
```

If `r_reference.json` is absent, `test_equivalence.py` skips rather than
fails, so the rest of the suite still runs without R installed.

Requires R with `rstpm2` (and its dependencies `Rcpp`, `RcppArmadillo`,
`bbmle`, `mgcv`, `survival`, `fastGHQuad`, `jsonlite`). Python deps are
managed with `uv`; `torch` and `pytest` are in the project `pyproject.toml`.

The baseline tests additionally use `survival`, `cmprsk` and
`randomForestSRC`; each check skips rather than fails when its package is
missing, so the suite runs without them. `randomForestSRC` declares
`data.tree` and `DiagrammeR` in `Imports`, but both are used only by
`get.tree` for browser tree visualisation. Where those are unavailable, drop
them from `DESCRIPTION: Imports` and delete the two matching `importFrom`
lines in `NAMESPACE` before `R CMD INSTALL`; nothing on the fitting or
prediction path touches them.

## What each test proves

**`test_basis.py` — the basis is a correct natural cubic spline.**
The analytic derivative `spline_deriv` agrees with `torch.autograd.grad` of
`spline_eval` at 100 random points to rtol 1e-8. This matters because the
likelihood path uses the analytic derivative (it will be the hot loop once
the neural variants exist) and must never silently drift from the function
it claims to differentiate. Also checks the natural-spline restriction —
exactly linear beyond both boundary knots — and continuity of value, first,
and second derivative at every knot.

**`test_likelihood.py` — the likelihood is arithmetically correct.**
The vectorised loss is compared against a hand-computed likelihood on a
5-subject toy dataset (one censored, two cause-1, two cause-2) with fixed
γ and β, written out term by term from the formulas above, agreeing to
1e-10. A second test confirms the clamps make pathological inputs
(negative derivative, extreme `log_H`) finite rather than NaN.

**`test_equivalence.py` — the core matches an independent published
implementation.** This is the point of the package. The same 5,000-subject
dataset is fitted by `rstpm2::stpm2` in R (cause-specific: competing events
censored, `df = 4`) and by `rp_classical.py` in PyTorch **using the knots
imported from R** — knot placement is a modelling choice, so the comparison
is only meaningful with identical knots. Four assertions:

1. β coefficients agree within 1e-3 (absolute), both causes.
2. Baseline log H curves agree to < 1e-3 max absolute difference on a dense
   200-point grid.
3. Total log-likelihood agrees within 0.1. The cause-specific
   factorisation means the joint competing-risks likelihood equals the sum
   of the two censoring-based fits, so this identity is itself a check that
   the Module 2 loss is right.
4. log H_k(t) curves for 5 reference covariate profiles overlay R's
   predictions to < 1e-3.

Assertions 2 and 4 compare *functions on a grid* rather than coefficient
vectors, because the two implementations use different parameterisations of
the same function space: `rcs_basis.py` uses the truncated-power RCS basis
from the Royston-Parmar papers, `rstpm2` uses its `nsx` natural-spline
basis. Grid comparison is invariant to that choice. (Confirmed empirically:
projecting R's fitted curve onto this basis leaves a residual of 4e-15, so
the spaces are identical to machine precision.) Failures print a
torch-vs-R diagnostic table rather than a bare assertion error.

**`test_recovery.py` — the likelihood is right independently of R.**
Simulates 20,000 subjects from the RP model itself with known γ and β
(df = 4), fits, and checks parameter recovery against the observed
information. This catches likelihood errors that a comparison against
another implementation could in principle miss.

Recovery is judged **jointly**, not per parameter: with 18 parameters,
demanding every |z| < 2 individually would fail roughly 60% of the time
(0.95¹⁸) even for a perfectly correct likelihood. The test uses the Wald
statistic `(θ̂ − θ)ᵀ H (θ̂ − θ) ~ χ²₁₈` against its 0.99 quantile, plus a
per-parameter |z| < 3 guard (roughly Bonferroni at 18 tests) so a single
grossly biased coordinate cannot hide inside an acceptable joint value.

**`test_variant_a.py` — Variant A behaves like the model it claims to be.**
The primary check is degenerate-case equivalence, split into two tests that
answer different questions:

* `test_1a_representational_equivalence_exact` plants the classical fit's
  parameters into Variant A through the documented mappings (γ → α via the
  basis, β → w through the scaler) and requires agreement at the float32
  floor (achieved 2.1e-06). This is optimiser-independent: it proves the
  *model* and its coefficient mappings are equivalent.
* `test_1_degenerate_equivalence_with_classical` trains Variant A with Adam
  and requires max |Δ log H| < 1e-2 (achieved 2.1e-03). Because 1a already
  pins the model, any residual here is attributable to Adam alone — which is
  the distinction that matters when this check is repeated for later
  variants.

Also covered: monotonicity (min s′_k > 0 on a 1,000-point grid after
training), the nonlinearity gain (an encoder with hidden layers must buy
real held-out likelihood against the linear model when the truth is
nonlinear — achieved +286 log-likelihood units), 100-step stability at the
default learning rates, checkpoint round-tripping, and a test that the
forward pass uses the analytic derivative rather than autograd.

## Variant B — patient-specific hazard shape

```
log H_k(t | x) = a_k(z) + b_k(z) · s̃_k(log t; α_k),    b_k(z) = softplus(·) > 0
```

Variant A holds the spline shape fixed across patients and lets the encoder
shift it — proportional hazards within cause. Variant B lets the encoder
*rescale* the shape, so covariates modulate how steeply risk accumulates over
time, not merely its level. That is the substantive claim of the project, and
`test_4_non_proportionality_gain` is the test that demonstrates it.

### Identifiability

The naive form has two flat directions. Both are removed **structurally**,
never by regularisation:

1. **Additive.** A spline intercept α₀ would enter as `b_k · α₀` and duplicate
   the shift `a_k(z)` already supplies. So the spline carries no intercept —
   it lives entirely in `a_k`.
2. **Multiplicative.** `(b_k, s̃_k)` and `(c·b_k, s̃_k/c)` are the same
   function, so the optimiser could drift along a ridge with `b_k` collapsing
   or inflating. So the linear coefficient is **fixed at 1**:

   ```
   s̃_k(v) = v + Σ_{j≥2} α_kj · Q_j(v)
   ```

   `α_k` genuinely has `n_basis − 1` entries — there is no α_k1 to drift, and
   `test_7_identifiability_diagnostics` asserts this rather than trusting the
   initialisation. Because QR makes the working columns orthogonal to the
   linear column, the free nonlinear deviations are orthogonal to the part
   being held fixed.

`train.py` logs the distribution of `b_k(z)` across the batch (min/median/max)
every epoch: a `b_k` collapsing toward 0 or drifting large is the signature of
an identifiability problem surviving the constraints. On the non-PH test data
this diagnostic is also informative in a useful way — `b₁` spreads across
patients (0.48 → 4.12) on the cause simulated non-proportional, while `b₂`
stays pinned near 1.0 on the cause simulated proportional. The model finds
the non-proportionality where it actually is.

### Is `b_k(z)` spread real, or just encoder capacity?

The per-cause reading of `b_k` on real data — *this cause is non-proportional,
that one is not* — is only licensed if a **proportional** truth produces a
narrow `b_k` **under the same encoder** that produced the wide one. Otherwise
the spread is the network having capacity to spend, and the interpretation is
unfounded. Measured as `sd(log b_k)`, since `b_k` multiplies the spline and the
meaningful scale is multiplicative (`scale_stats`' min/max are noisy extremes
over thousands of patients and understate nothing but overstate readily).

**Capacity is not the explanation.** Three seeds at each of three encoder
sizes, PH truth versus a non-PH cause 1:

| Encoder | PH null band | non-PH cause 1 | Separation |
|---|---|---|---|
| (64,32)→16 | 0.010–0.066 | 0.333–0.354 | 5.0× |
| **(256,128)→64** (production) | 0.025–0.077 | 0.307–0.344 | 4.0× |
| (512,256)→128 | 0.020–0.060 | 0.326–0.349 | 5.4× |

An **eightfold** increase in encoder width leaves the null band where it is.
Extra capacity has nothing to gain by spreading `b_k` when the truth is
proportional, because the identifiability constraints and the likelihood pin
it — which is exactly what the structural constraints in this section were for.

**The response is near-linear, so a fitted value inverts to an effect size.**
Cause 1 given a Weibull shape exponent `s` on x1, at production capacity:

| exponent `s` | 0.00 | 0.05 | 0.10 | 0.15 | 0.25 | 0.35 |
|---|---|---|---|---|---|---|
| cause 1 `sd(log b)` | 0.047 | 0.087 | 0.120 | 0.151 | 0.241 | 0.329 |
| cause 2 (PH throughout) | 0.045 | 0.034 | 0.031 | 0.032 | 0.047 | 0.048 |

`sd(log b) ≈ 0.045 + 0.82 s`. The detection floor is around `s = 0.05` (0.075–
0.098 against a null of 0.039–0.055 — separated, but only just) and clean from
`s = 0.10`. Cause 2 holds at 0.030–0.048 in every column, so **causes do not
contaminate each other** — which is what makes a per-cause claim legitimate
rather than one cause absorbing the other's misfit.

Reading a real fit, therefore: below ~0.08 is indistinguishable from
proportional; above ~0.12 is a real departure; in between, quote the number and
the null band rather than a verdict. `test_4b_bk_spread_is_not_encoder_capacity`
asserts the null band and the separation at production capacity; the full
sweeps are `reference/bk_falsify.py`, `bk_capacity.py`, and `bk_dose.py`.

### Weibull at initialisation

With every α_kj = 0 the model is exactly `log H_k = a_k(z) + b_k(z)·v` — a
Weibull model with `a_k` the log scale and `b_k(z)` a patient-specific shape.
Variant B is therefore "neural Weibull with spline deviations", which is both
a sensible starting point and a clean interpretation. The `b_k` head bias is
initialised at `softplus⁻¹(1) ≈ 0.5413` so training begins at shape 1, and
`test_1b_weibull_parameter_recovery` confirms the parameterisation is what it
claims by recovering the true shape and scale from Weibull-simulated data
(shape within 1.2%, log-scale within 0.05).

### Why monotonicity stays a global check

Because `b_k(z) > 0` by construction,

```
∂ log H_k / ∂v = b_k(z) · s̃′_k(v)
```

is positive for **every** patient exactly when `s̃′_k(v) = 1 + Σ α_kj Q′_j(v) > 0`,
and `s̃′_k` depends only on the global `α_k` — not on `z`, not on the sample.
So a single 1,000-point grid check over the knot range certifies monotonicity
for all patients, including ones never seen in training. No per-observation
scan, and no clamping: a violation is a hard failure, surfaced. This is the
main reason Variant B is the primary model rather than a variant whose shape
is fully network-produced.

## Cumulative incidence

`cif.py` is prediction-only — the training likelihood is closed form and is
untouched by it. The CIF is the model's primary clinical output and the one
quantity that needs quadrature:

```
F_k(t | x) = ∫_{−∞}^{log t} H_k(eᵛ) · [∂ log H_k/∂v](v) · S(eᵛ) dv
```

Integrating over log time is the right choice: the model is parameterised in
log time, the integrand is smooth there, and the substitution removes the
`1/t` singularity of the hazard as `t → 0`. `∂ log H_k/∂v` is taken from the
model's own analytic derivative, so Variant A (shared `s′_k`) and Variant B
(patient-specific `b_k(z)·s̃′_k`) need no branching.

### Coherence — the headline result

Differentiating `d/dt[F₁ + F₂ + S] = h₁S + h₂S − (h₁+h₂)S = 0` shows that

```
F₁(t) + F₂(t) + S(t) = 1
```

holds **exactly**, for all t, by construction — a structural property of the
model, unlike discrete-time competing-risks models that normalise numerically.
Nothing in `cif.py` softmaxes, normalises, or rescales to enforce it, so any
measured deviation is quadrature error and nothing else.

**Measured: `max |F₁ + F₂ + S − 1| = 3.1e-07`** over 500 patients × 25 time
points (Variant B; 1.9e-07 for Variant A). This is the number for the paper.

### Node count, chosen from evidence

`DEFAULT_NODES = 20`, not assumed. Measured `|GL(n) − GL(240)|`:

| n | 5 | 10 | 15 | **20** | 30 | 40 | 80 | 120 |
|---|---|---|---|---|---|---|---|---|
| err | 7.3e-05 | 3.1e-06 | 2.6e-06 | **1.0e-07** | 1.8e-07 | 1.4e-07 | 2.2e-08 | 9.3e-09 |

The integrand is smooth on the log-time scale, so Gauss-Legendre converges
spectrally; the curve is flat from n=20 onward and the floor is float32 model
output rather than the quadrature (the same model in float64 reaches 8.7e-08
where float32 gives 2.1e-07). 20 is the smallest count on that plateau.

### The lower limit is not where you would guess

Truncating the integral at the smallest **boundary knot** — the obvious
choice, since `H_k(t) → 0` as `t → 0` — is **not safe**. Because
`H_k · ∂logH_k/∂v = dH_k/dv`, the discarded tail carries approximately
`H_k(e^{v_lo})` of probability mass, and for a fitted model `Σ_k H_k` at that
knot is ~1e-4. That mass goes missing from `F_k` and shows up as a coherence
deviation of **1.3e-04 that is completely flat in the node count** — the
signature of a domain error rather than a quadrature error, and the reason
adding nodes appeared to do nothing.

`_auto_lower_limit` instead steps the limit down until the model itself
reports the remaining mass below 1e-10 (reaching `v ≈ −24.4`, i.e.
`t ≈ 3e-11`), and the stretch below the first grid time is split into several
sub-intervals so the wide tail does not degrade the Gauss-Legendre rule.
Fixing the *domain* rather than adding a correction term keeps this pure
quadrature — nothing nudges `F₁ + F₂ + S` toward 1. Discarded tail after the
fix: **5.1e-09**, versus 8.4e-05 at the boundary knot.

### Evaluation budget

50,000 patients × 3 time points takes **3.3 s** at 20 nodes on a 4-core CPU
(~8 s at 40). The cost is `n × T × n_nodes` model evaluations, batched over
patients; a cumulative scheme integrates all grid times in one pass and reuses
the partial sums.

## Feature builder

`features/` is the bridge from the imputed SRTR parquet files to model inputs.
It sits **last** in the pipeline deliberately: category→index maps, reference
levels and continuous scalers are all *learned*, so this is the final place a
train/test boundary can be violated.

### Fit/apply discipline

| Fitted on | Applied to |
|---|---|
| `dev_fold{f}_train` | `dev_fold{f}_train`, `dev_fold{f}_test` |
| `dev_full` | `dev_full`, `temporal`, `exploratory` |
| `pretrain` (masked) | `pretrain` only |

One `FeatureSpec` per (fit set × imputation), serialised to JSON alongside the
arrays so any result can be reproduced or audited. The pretraining spec drops
the five features absent before 2014 (`END_CPRA`, `EPTS_FINAL`,
`INTERSTITIAL_FIBROSIS`, `VASCULAR_CHANGES`, `GLOMERUL_COUNT`) and retains
`GLOMERUL`.

### One spec, two representations

The same fitted spec produces both:

* **`embedding`** — standardised continuous block + integer codes per
  categorical, plus the cardinality tuple `CovariateEncoder` expects.
* **`onehot`** — standardised continuous block + one-hot with the reference
  level dropped, a design matrix for the classical baselines.

That single object is the mechanism guaranteeing every model receives
identical information, and `test_2_information_equivalence` asserts it by
recovering the categorical values from both representations and requiring an
exact match. If that test fails, no benchmark result means anything.

Feature counts on the SRTR schema **as shipped**: embedding = 18 continuous +
27 categorical codes = 45; one-hot = 18 continuous + Σ(levels − 1).

With `BMI_CALC` and `BMI_DON_CALC` commented out of `CONTINUOUS` — which is
the configuration this project runs — those become **16 continuous + 27
categorical codes = 43**, and the one-hot design narrows by two columns plus
the four RCS columns the two would have contributed. Nothing else changes:
`derive_bmi` simply has no consumer, height and weight remain in the design in
their own right, and no fitted quantity moves.

### Three schema points worth knowing

* **`END_CPRA`** appears in the pretrain mask list but in neither the
  continuous nor the categorical list of the task spec. It is a percentage,
  so it is treated as continuous — giving 18 + 27 = 45, which reconciles with
  the stated 45-feature total (the spec's "17 continuous + 28 categorical"
  splits the same 45 differently).
* **`BMI_CALC` / `BMI_DON_CALC`** are not in the imputed files. They would be
  derived from height and weight at build time — deterministic arithmetic on
  existing columns, so no fitted quantity and no leakage. **They are commented
  out of `CONTINUOUS` in `spec.py` and therefore excluded**, on the grounds
  that BMI is a deterministic function of two covariates already in the design:
  including it adds a collinear column that buys flexibility only in the
  particular ratio `w/h²`, which the splines on height and weight can express
  anyway. `derive_bmi` is left in place, unused, so the decision is reversible
  by uncommenting two entries.
* **`DIAG`'s hypertensive/vascular reference code** is not discoverable from
  the data, so it is a parameter. The notebook prints the observed DIAG levels
  with counts; until it is set, the builder falls back to the modal level and
  **warns**, because that is a valid fit but not the intended contrast.

### Reference levels are pinned across imputations, and why

The reference level is the modal level of the training rows — a *sample*
quantity. With two levels of similar frequency it flips between imputations,
so a different column gets dropped. Predictions are invariant to which level
is dropped, so no metric is biased. But **pooled baseline coefficients across
the five imputations would be meaningless** if imputation 1 dropped one level
and imputation 3 another.

The build notebook therefore fits imputation 1 of a fit set and pins its
references for imputations 2–5. Those are the same patients, so this is one
fitted choice applied consistently, not leakage. `test_7_column_order_is_stable`
verifies both halves: identical column names across imputations with pinning,
and stable feature order and one-hot width across independently fitted folds.

### Running it

`features/build.ipynb` is meant to be run where the data lives. Set
`IMPUTED_DIR`, `OUTPUT_DIR`, `RP_CORE_DIR` and `DIAG_REFERENCE` in the first
cell; the notebook checks the schema, builds every (fit set × imputation),
and reports rows and feature counts per applied set, cardinality per
categorical, scaler kind per continuous column, and any unseen levels mapped
to the reference.

## Evaluation metrics

`metrics/` scores predictions, never models. Every function takes the same
three arrays — `cif (n, T, K)`, `time (n,)`, `event (n,)` — plus the horizons,
and imports nothing from `neural_rp.py`. That is what makes the neural-vs-
classical comparison fair by construction rather than by care: the same code
path scores all of them. `evaluate(...)` returns one structured record per
model per imputation for the benchmark harness.

**Calibration is the priority, not discrimination.** The JASN ten-country
study found competing and non-competing models achieve near-identical
discrimination for graft failure, so a discrimination-led result would be
weak. `test_2_miscalibration_is_detected` puts that argument in code:
multiplying the true CIF by 1.5 is a monotone transform, so AUC and the
C-index move by < 5e-5 while ICI rises roughly tenfold. A model can rank
patients perfectly and still be badly wrong about absolute risk.

### What is implemented

| Metric | Definition used |
|---|---|
| Time-dependent AUC | Blanche et al., IPCW. Cases = cause-k events by t; controls = event-free at t **plus competing events by t** |
| Time-dependent C-index | Competing-risks form, IPCW, truncated at the horizon |
| Brier score | IPCW with the standard three-group weights; IBS by trapezoid |
| Calibration curve | **Austin, Putter, Lee & van Buuren (2022)**, *Diagnostic and Prognostic Research* **6**:2 — Fine-Gray model with a restricted cubic spline of the predicted CIF as sole covariate → ICI, E50, E90, Emax, plus the curve itself |
| Grouped calibration | Deciles of predicted risk; observed risk by **Aalen-Johansen** with CI |
| Calibration-in-the-large | Mean predicted CIF vs the overall Aalen-Johansen estimate |
| Subgroups | All of the above within pre-specified strata, with an event-count warning below ~50 |
| Coherence | `max |F₁ + F₂ + S − 1|`, reported as a metric, not only asserted |

Two rules are enforced rather than assumed, because both are easy to get
wrong and neither announces itself:

* **Aalen-Johansen, never Kaplan-Meier**, for observed risk wherever competing
  events exist. KM treats a competing event as censoring — as if that patient
  could still fail — and so overstates absolute risk. This is the exact error
  the project exists to characterise, and the test reports its magnitude.
* **Group-level predictions are the mean of individual CIFs**,
  `mean_i F_k(t|xᵢ)`, never `F_k(t|x̄)`. By Jensen these differ, and the gap
  widens with covariate spread; the test constructs a case where it exceeds
  0.01 in absolute risk.

### Validated against published implementations

The pieces that could silently be wrong are checked against R rather than
against themselves:

| Component | Reference | Agreement |
|---|---|---|
| Fine-Gray fit | `cmprsk::crr` | 2.6e-09 |
| Aalen-Johansen | `cmprsk::cuminc` | see test output |
| AUC and Brier | `riskRegression::Score` | see test output |
| RCS basis (numpy) | `rp_core/rcs_basis.py` (itself verified against rstpm2) | < 1e-12 |

The Fine-Gray fit needed the same two numerical lessons that the neural
phase did. Its covariate columns are **centred and scaled** before fitting —
a truncated-power spline basis evaluated on predicted risks in [0, 1] has
cubic columns orders of magnitude smaller than the linear one — and each
Newton step is accepted only if the log partial likelihood improves, with
halving otherwise. Without both, the fit failed to converge on roughly half
the cases and silently fell back to the cruder smoother, which is exactly
the kind of quiet degradation the fallback is supposed to make visible.

Risk-set sums use cumulative sums rather than an explicit
(events × subjects) matrix, since the weight of a subject who had a competing
event at `T_j` is `G(t)/G(T_j)` and both parts of the sum are running totals.
That is what keeps the Fine-Gray fit usable at benchmark scale.

### Evaluation budget

Measured at 50,000 patients × 3 horizons on a 4-core CPU:

| Component | Time |
|---|---|
| AUC (2 causes × 3 horizons) | 0.1 s |
| Brier | < 0.1 s |
| Time-dependent C-index | 2.4 s |
| Austin calibration curve (per cause-horizon) | 0.2 s |
| Grouped calibration + CITL | < 0.1 s |
| 500 bootstrap replicates (AUC + Brier) | 59 s |

Getting there took profiling rather than intuition, and the intuition was
wrong twice. The first suspect — a Python loop over ~12,000 event times in
the Fine-Gray fit — was real but minor. The actual cost was the
**Aalen-Johansen variance**, whose textbook double loop is O(M²) over the
~35,000 distinct event times and took 57 s for a *single* cause-horizon.
Expanding `[F(t) − F(t_i)]²` makes every term a prefix sum, so the whole
variance is O(M) cumulative sums; that one change took calibration-in-the-
large from 57 s to under 0.1 s and the full point-estimate pass from ~390 s
to a few seconds. The C-index, the component that looks most expensive
because it is the only 2-D dominance count and the only Python-level loop,
was never the problem at 2.4 s.

Bootstrapping the C-index is the one genuinely costly option (it scales
~n log n with a Python-level Fenwick loop and cannot ride on cumulative
sums); `evaluate(do_cindex=...)` and `bootstrap_ci(subsample=...)` expose
the choice. Given that discrimination is not this project's endpoint,
bootstrapping AUC/Brier/calibration and reporting the C-index as a point
estimate is the sensible default.

## Classical baselines and the benchmark harness

`benchmark/` holds the comparators the paper's central claim is measured
against. The organising concern is **fairness**: the claim is that neural
representation learning improves *calibrated absolute risk* beyond the best
classical models, and a claim like that is only worth making if the classical
models were given every advantage. A weak baseline produces a strong-looking
result that means nothing.

| Model | `name` | What it is |
|---|---|---|
| Cause-specific Royston-Parmar | `royston_parmar` | The primary comparator; `fit_classical_rp`, already verified against `rstpm2` |
| Cause-specific Cox | `cox` | Breslow baseline hazards, CIF by product-limit integration |
| Fine-Gray | `fine_gray` | Two independent subdistribution models with IPCW weights |
| Competing-risks RSF | `rsf` | `randomForestSRC`, `splitrule = "logrankCR"`, via subprocess |
| Minimal reference | `rp_kdri_epts` | RP on `KDRI_RAO` and `EPTS_FINAL` only — the floor every richer model must clear |

### What "fair" means concretely

1. **Splines on every continuous covariate.** `covariate_splines.py` fits
   4-knot restricted cubic splines at Harrell quantiles (0.05, 0.35, 0.65,
   0.95) on the *training rows only*, and skips columns with fewer than 20
   distinct values so counts and indicators stay linear. Without this the
   linear models would be handicapped against a network that can bend, and
   any difference would measure functional form rather than representation
   learning. The expansion reuses the same `rcs_design` verified against
   `rstpm2` — test 6b asserts the two agree to 1e-9.

   With no `spline_columns` argument every eligible column is expanded, and
   the ≥20-distinct-values gate is what keeps one-hot indicators and small
   counts linear. That gate is a safety net, not a specification: **pass
   `spline_columns=` the feature builder's continuous list** so the choice is
   explicit rather than inferred from cardinality, which would expand a
   many-levelled ordinal by accident.
2. **The validated composites are supplied.** `KDRI_RAO` and `EPTS_FINAL`
   go in as covariates, so the baselines start from what the field already
   knows rather than being asked to rediscover it.
3. **Identical scoring code.** Every model is scored by the same
   `metrics.evaluate` on the same held-out rows, and every fit gets a seed
   from `derive_seed(model, fold, imputation)` — a hash, so a model's random
   state cannot depend on how many fits preceded it in the run.
4. **Knots from training rows only.** `fit_covariate_splines` / 
   `apply_covariate_splines` split the way the feature builder does. Test 6
   confirms that corrupting the held-out rows changes nothing, *and* that
   mixing them in does change the spec — otherwise the test could not detect
   a leak.

### The coherence table — a paper result, not merely a test

`coherence_table(results)` reports `max |F₁ + F₂ + S − 1|` per model. It
separates the formulations rather than the fits:

| Model | Typical worst deviation | Why |
|---|---|---|
| `cox` | ~5e-15 | Exact, but only because the integration uses the product-limit survival `S(t_j) = ∏(1 − dH)`, which makes the sum telescope. The obvious `exp(−Σ dH)` gives ~1e-03. |
| `rp_kdri_epts`, `royston_parmar` | ~1e-06 to 3e-06 **at 20 nodes** | Structural: the two causes share one survival function, so the residual is Gauss-Legendre quadrature error alone. |
| `fine_gray` | ~8e-02 | No guarantee at all. The two subdistribution models are fitted independently and nothing couples them; `F₁ + F₂` is observed above 1. |
| `rsf` | ~8e-02 | The ensemble averages CIFs and cumulative hazards separately, and averaging does not commute with the Aalen-Johansen recursion. |

**Never quote the Royston-Parmar figure without its node count.** It is
quadrature error, and it converges — same model, same data:

| `n_nodes` | 10 | 20 (default) | 40 | 80 | 160 |
|---|---|---|---|---|---|
| coherence | 3.0e-05 | 3.0e-06 | 6.8e-08 | 1.2e-07 | 1.2e-08 |

Roughly an order of magnitude per doubling until it reaches the float64
round-off floor near 1e-08, where it stops improving monotonically. So RP
figures quoted from different tasks — 3.1e-07 for the CIF work, 3.0e-06 here —
are **not** in conflict and neither is more correct: they are different fitted
models on different data, both at 20 nodes. `test_2b` asserts the convergence,
which is what licenses calling the residual quadrature error rather than a
defect.

**`fine_gray` and `rsf` land within ~1% of each other by coincidence, not by a
shared mechanism.** They arrive by opposite routes, and `test_2c` pins this
down because two unrelated estimators agreeing to two significant figures
otherwise looks like an artefact of the clipped-complement path:

| | own `S`? | `F₁+F₂` max | entries above 1 | residual | residual if forced through the complement |
|---|---|---|---|---|---|
| `fine_gray` | no | 1.0838 | 20 | 8.38e-02 | 8.38e-02 |
| `rsf` | yes | 0.8490 | **0** | 8.46e-02 | **0** |

Fine-Gray has no joint survival, so its residual *is* the excess mass. RSF's
CIFs never exceed 1, so the clip never engages and that path would report
exactly zero — its residual comes wholly from its own reconstructed survival.
Had both been riding the clip floor, both would show entries above 1.

RSF's figure does depend on the reconstruction, but weakly: across a tenfold
change in `ntime` (100 → 1000) it moves only 8.49e-02 → 8.24e-02, about 3%. It
is a property of the bagged estimator, not of the grid it is read off.

This is a substantive point in favour of the Royston-Parmar formulation, so
the harness is careful not to fake it. `run_benchmark` takes `S` from the
model where the model has one (`predict_survival`) and only falls back to the
clipped complement `clip(1 − F₁ − F₂, 0, 1)` where it does not. Deriving `S`
as `1 − F₁ − F₂` unconditionally would make the deviation identically zero for
every model and the column would measure nothing. Under the fallback the
residual is exactly the excess mass `max(0, F₁ + F₂ − 1)`, which is the
quantity worth reporting for a model with no joint distribution to speak of.

For the same reason, `F₁ + F₂ ≤ 1` is asserted only for the models where it
holds by construction. For Fine-Gray and RSF the excess is **reported**, not
asserted away.

### Running the benchmark

```python
from benchmark import BenchmarkCell, make_default_baselines, run_benchmark

cells = [BenchmarkCell(model_name=name, fold=f, imputation=m,
                       train={"X": Xtr, "time": ttr, "event": etr},
                       evaluate_on={"test": {"X": Xte, "time": tte,
                                             "event": ete}},
                       factory=factory)
         for f in folds for m in imputations
         for name, factory in make_default_baselines(
                 columns=COLS, spline_columns=CONTINUOUS).items()]

results = run_benchmark(cells, t_eval=[1, 3, 5], output_path="results.parquet")
```

Results are written in **long format** — one row per
(model, fold, imputation, eval_set, cause, horizon) — because every
downstream question (pool over imputations, compare models within a horizon,
subgroup tables) is then a `groupby` rather than a reshape.

Every record carries a `converged` flag, per cause, for models that expose
one, and the runner prints a loud `!! NOT CONVERGED` line when a fit fails.
Twenty-five cells run unattended is precisely where a failed optimiser slips
through and quietly weakens the comparator the paper's claim rests on —
**filter on `converged` before pooling.**

The run is **resumable**: completed
(model, fold, imputation, eval_set) combinations are skipped and the parquet
is rewritten after every cell, so an interrupted run continues instead of
restarting. Because seeds come from `derive_seed`, a resumed run reproduces
exactly what an uninterrupted one would have produced — test 8 checks this by
running a subset, restarting with more work, and asserting the first subset's
numbers are unchanged.

### Runtime, measured

Single-threaded (`torch.set_num_threads(1)`) on an otherwise idle machine.
Pinning matters: the same cells measured 30-60x slower when other jobs were
competing for cores, so any timing taken on a loaded machine is noise.

Small synthetic data, to show the SHAPE of the scaling:

| n | design width | RP fit | RP pred | Cox fit | Cox pred | FG fit |
|---|---|---|---|---|---|---|
| 2,000 | 18 | 2.8 s | 0.2 s | 0.1 s | 0.1 s | 0.1 s |
| 2,000 | 54 | 2.4 s | 0.2 s | 0.5 s | 0.1 s | 0.7 s |
| 8,000 | 18 | 2.9 s | 0.3 s | 0.3 s | 1.1 s | 0.4 s |
| 8,000 | 54 | 4.8 s | 0.3 s | 1.8 s | 1.0 s | 2.7 s |
| 32,000 | 18 | 8.6 s | 0.2 s | 2.1 s | 6.2 s | 2.6 s |
| 32,000 | 54 | 14.0 s | 0.4 s | 22.3 s | 7.4 s | 24.5 s |

Royston-Parmar scales roughly linearly in `n` and mildly in width, because
the cost is a Newton solve on a parameter count that grows only additively
with the design. Cox and Fine-Gray are superlinear in both: at width 54,
quadrupling `n` costs 12x, not 4x.

**An anchor at realistic scale**, because extrapolating across a superlinear
`n` term and a quadratic `p` term is guesswork. `features/spec.py` lists 18
continuous (16 with the BMI pair excluded) and 27 categorical inputs; one-hot
expanded and splined that is a
design in the 130-160 column range, so: n = 50,000, 96 inputs, **162 expanded
columns**, 13,476 cause-1 and 9,366 cause-2 events.

| Model | Fit | Predict (all 50,000 rows) | Per cell |
|---|---|---|---|
| `royston_parmar` | 79.9 s | 8.3 s | ~1.5 min |
| `cox` | 346.8 s | 79.6 s | ~7.1 min |
| `fine_gray` | 426.6 s | 0.7 s | ~7.1 min |

Two things worth reading off this table. The primary comparator is the
**cheapest** of the three - Royston-Parmar is 4x faster to fit than Cox at
this width, so the model the paper is about is not the one that constrains
the schedule. And Cox is the only model whose PREDICTION is expensive
(79.6 s against Fine-Gray's 0.7 s), because it integrates over every distinct
event time while Fine-Gray evaluates a closed form.

Peak memory holds flat at ~1.0 GB across all of these. That is the point of
accumulating the information matrix one covariate at a time: the textbook
`(n, p, p)` form needs `50,000 x 162^2` doubles - **10.5 GB** for the
temporary alone - and would simply not run.

#### RSF is the bottleneck, and by how much

Measured with `randomForestSRC` 3.6.2, competing-risks splitting
(`logrankCR`), timing the whole build-and-predict cycle because the wrapper
rebuilds the forest in the scoring session - `rfsrc` has no portable
cross-process serialisation, so RSF pays its full cost once per cell with no
way to amortise it.

| n | inputs | ntree | nodesize | build + predict |
|---|---|---|---|---|
| 5,000 | 8 | 300 | 15 | 44.3 s |
| 5,000 | 96 | 300 | 15 | 80.4 s |
| 10,000 | 96 | 300 | 15 | 267.2 s |
| 20,000 | 96 | 300 | 15 | 1,002.8 s |
| 20,000 | 96 | 100 | 100 | 335.3 s |
| 50,000 | 96 | 100 | 100 | 2,147.5 s |

Two things fall out. **Width is nearly free** - twelve times the inputs (8 to
96) costs 1.8x, which is what `mtry = sqrt(p)` predicts. **Rows are not**:
the fitted exponent is about `n^1.9`, near-quadratic, which is what
competing-risks log-rank splitting costs when every candidate split rescans
its node. Extrapolating the full-settings curve to n = 50,000 gives ~96
minutes per cell, and two attempts at exactly that cell were killed at the
50- and 90-minute caps, so treat 300 trees at `nodesize = 15` as out of reach
at this size.

Dropping to 100 trees and `nodesize = 100` costs 3x less - almost all of it
from the tree count - and n = 50,000 then completes in **35.8 minutes per
cell**. That is still ~15 hours for a 25-cell matrix against ~6-7 hours for
the other three models combined.

**This is a judgement call, not a tuning problem, so it is left open.** The
defensible options, in rough order of cost:

1. Run RSF on one imputation across all five folds (5 cells, ~3 hours). The
   fold variation is what a comparator needs to show; the imputation
   variation is already characterised by the other four models.
2. Run the full matrix at reduced settings and note them (~15 hours, shardable
   by fold).
3. Drop RSF and say why. It is a discrimination-oriented comparator whose
   coherence is among the worst in the table (~8e-02), and this project's
   endpoint is calibrated absolute risk.

Whichever is chosen, report the settings: an RSF at 100 trees is a weaker
comparator than one at 1,000, and a reviewer will ask.

`ntime` is set to 250 by default and is **not** a tuning knob. Left unset,
`rfsrc` evaluates the CIFs at every distinct event time: at n = 50,000 that
is a `(50000, ~23000, 2)` array for `cif` and another for `chf`, about 36 GB.
Capping the grid removes that without measurable cost, since the CIFs are
interpolated onto `t_eval` regardless - test 5's agreement with
Aalen-Johansen is unchanged. It was not, however, what made the full-settings
cell infeasible; the tree count and node size were.

**Projection for the full matrix.** At this scale the three models above cost
about 16 minutes per (fold, imputation) cell, so 5 folds x 5 imputations is
roughly **6-7 hours**, plus the minimal reference (cheap - a 6-column design)
and RSF (a further ~15 hours even at reduced settings, per the table above).

That is an overnight job run serially, and it does not have to be. The 25
cells are **independent fits**, and each cell's seed comes from
`derive_seed(model, fold, imputation)` rather than from execution order, so
sharding by fold across processes and merging the parquets afterwards gives
bit-identical results to a serial run. Combined with per-cell resume, the
run can be taken in pieces without losing work or reproducibility.

## Deep-learning baselines

`benchmark/deep.py` wraps the nearest published competitors behind the same
`SurvivalBaseline` protocol the classical models satisfy, so `run_benchmark`
drives them unchanged and `metrics.evaluate` scores them with identical code.

### Provenance — which is whose code

This belongs in the paper's methods, so it is recorded per model rather than
summarised.

| Model | Implementation | Reimplemented? |
|---|---|---|
| DeepHit (Lee et al., AAAI 2018) | `pycox` 0.3.0, the reference Python implementation | **No.** The *network* is ours, because pycox requires the caller to supply one — built to match Variant B's encoder. |
| Neural Fine-Gray (Jeanselme et al., CHIL 2023) | the authors' own repository, unmodified | **No.** |
| DSM, competing risks (Nagpal et al., IEEE JBHI 2021) | `auton-survival` 0.1.0 | **No.** Its per-cause output is a net probability, converted to a CIF by the same integration `cox` uses — see the coherence section. |

Nothing here is a reimplementation, which is worth stating plainly because
the spec allowed one for Neural Fine-Gray if the repository would not install.
It installed.

Two installation notes, both reproducible:

* `auton-survival` pins `torch==1.13.1`, which has no CPython 3.12 wheels, so
  the dependency resolver refuses it. Installed with `--no-deps`; the code
  itself runs unmodified on torch 2.13.
* Neural Fine-Gray is not published to PyPI. Clone it and point at the clone:

  ```bash
  git clone --recursive https://github.com/Jeanselme/NeuralFineGray.git
  export NFG_PATH=/path/to/NeuralFineGray
  ```

  Without it the baseline reports itself **unavailable** and is omitted from
  the factory dict. It is never silently replaced by something else, and
  `available_deep_baselines()` prints the reason.

**Optional models not attempted:** DeSurv (neural-ODE CIF) and SurvTRACE
(transformer encoder). Both were listed as optional; neither was implemented.
The Neural Fine-Gray repository vendors a DeSurv implementation, so adding it
is mostly wrapper work if it becomes wanted.

### Fairness

The same four rules the classical baselines follow:

1. **Same features.** Every deep baseline consumes the `embedding`
   representation from the same fitted `FeatureSpec` as Variants A and B —
   `[continuous | integer codes]` — and never re-encodes. The codes are
   one-hot expanded inside the wrapper because these networks have no
   embedding layers. That is an adapter, not a re-encoding: no level mapping,
   scaling, or reference choice is recomputed. It is also not a handicap, since
   a one-hot block followed by a linear layer computes exactly an embedding
   lookup of the layer's width.
2. **Comparable capacity.** All widths are drawn from one ladder
   (`(128,64) / (256,128) / (512,256)`) shared with Variants A and B, and the
   default matches Variant B's encoder.
3. **Same folds, imputations, seeds** — `derive_seed(model, fold, imputation)`.
4. **Identical scoring** — `(n, T, 2)` CIFs into `metrics.evaluate`.

### The extended coherence table, and how to read it

`F₁ + F₂ + S = 1` now spans models for which it means three different things,
and the table is easy to read backwards:

| Model | Residual | Why |
|---|---|---|
| `neural_fine_gray` | 0 exactly | **Definitional.** `F_k = β_k(1 − e^{log s_k})` with `Σβ_k = 1`, so S *is* the complement. |
| `deephit` | ~1e-8 | **Definitional.** A softmax over (cause, interval) plus a no-event class imposes it; the residual is float error. |
| `royston_parmar` | ~1e-6 | **Derived.** S comes from independently integrating the fitted cumulative hazards, so this is quadrature error in a check that *could* have failed. |
| `cox` | ~5e-15 | **Derived**, exact because the integration uses the product-limit form. |
| `dsm` | ~2e-15 | **Derived**, same product-limit integration as `cox` — see below. |
| `fine_gray`, `rsf` | ~1e-1 | **Not guaranteed.** Independent per-cause fits with nothing coupling them. |

A table reading "RP 3e-06, NFG 0.0" does **not** show NFG doing better. Zero
by construction and small-by-measurement are different quantities, and the
paper's claim has to be about *structural coherence in continuous time*, not
about competitors producing incoherent probabilities — which is false for
DeepHit and Neural Fine-Gray and would be caught immediately.

#### DSM was in this table's bottom row, and that was our error

DSM's CIFs originally summed to 1.77, and the obvious reading — independent
per-cause mixtures with nothing coupling them — was wrong. `auton-survival`'s
`predict_risk(risk=k)` is not a cumulative incidence. Its loss
(`losses._conditional_weibull_loss`) censors every subject whose event is not
*k*, competing events included:

```python
uncens = np.where(e == int(risk))[0]     # density term
cens   = np.where(e != int(risk))[0]     # survival term
```

That is the textbook "censor the competing risk" construction, which estimates
the **cause-specific hazard**. So `predict_survival(risk=k)` returns a net
survival `S_k = exp(−H_k)` and `predict_risk` returns `1 − S_k` — the 1−KM
analogue. Each cause therefore tends to 1 and the pair tends to 2; measured
through the authors' own unmodified API, `(1−S₁)+(1−S₂) = 1.9999` at t = 200.

The fix is the same integration `CoxBaseline` already applies to Breslow
hazards, using only quantities DSM itself estimates:

```
H_k(t) = −log S_k(t)
F_k(t) = Σ_{u ≤ t} S(u⁻) dH_k(u),    S = ∏ (1 − Σ_j dH_j)
```

No normalisation, no rescaling, no free constant. That it is *correct* rather
than merely *tidier* is settled by an independent criterion — error against the
closed-form true CIF, which nothing was fitted to:

| | raw `1 − S_k` | integrated CIF |
|---|---|---|
| F₁+F₂ max | 1.7749 | 1.000000 |
| coherence residual | 7.7e-01 | 1.6e-15 |
| mean \|err\| vs known truth | 0.0672 | **0.0370** |
| bias, cause 1 / 2 | +0.045 / +0.054 | **−0.007 / −0.002** |
| correlation @5y | 0.940 / 0.895 | **0.971 / 0.965** |
| held-out Brier vs null | +14.4% | **+18.1%** |

Two lessons worth carrying. Cox and DSM are both cause-specific models, so
post-processing one and not the other would have scored DSM on a different
estimand from every model it is compared against — the incoherence would have
been ours, reported as theirs. And a baseline can look healthy on
discrimination while answering the wrong question: DSM's correlations were
already 0.94/0.90 *before* the fix.

### Hyperparameter search

`benchmark/tuning.py` applies one procedure to every model, Variants A and B
included, because a baseline tuned less carefully than the proposed model is
not a baseline. Random search, default 30 configurations, fold 0–4 CV on the
development window at imputation m = 1 only.

Selection uses held-out **log-likelihood** where the model has one and
**integrated Brier score** where it does not, and which one was used is
recorded per model. DeepHit is the exception that forces this: its training
objective mixes likelihood with a ranking term and is therefore not a
likelihood, so it is selected on IBS. Both criteria are oriented so that
higher is better, so a selection rule cannot silently flip direction between
models.

| Model | Criterion | Hyperparameters | Grid size |
|---|---|---|---|
| `deephit` | ibs | 8 | 8,748 |
| `dsm` | loglik | 6 | 1,080 |
| `neural_fine_gray` | loglik | 6 | 648 |
| `variant_a` | loglik | 5 | 324 |
| `variant_b` | loglik | 6 | 972 |

Every grid is far larger than the 30-trial budget, so no model is quietly
searched exhaustively while another is sampled. Where a grid *is* smaller than
the budget the harness enumerates it exactly once rather than resampling — a
reported budget of 30 must not mean six configurations evaluated five times.
Failed trials (a diverging learning rate, say) are logged with their exception
rather than dropped, so the budget stays honest.

Two guards raise rather than warn:

* `assert_tunable` refuses any data containing **temporal** or **exploratory**
  rows. A search that touched them would invalidate every number later
  computed on them, and that is undetectable after the fact.
* It also refuses imputations other than m = 1, so the budget is spent on
  hyperparameters rather than on imputation noise.

Every trial — config, per-fold scores, wall time, failure — is logged to
parquet. That log is a supplementary table, not a debugging aid: it is what
shows a reviewer the comparators got a real search.

### Runtime, measured — and why the default budget does not survive it

At **37,000 training rows** (the size that matters: the development window
holds 47,367 and each CV fold trains on 4/5 of it), width 112, single-threaded,
full-size nets:

| Model | Fit | Predict (37,000 rows) |
|---|---|---|
| `dsm` | 10.3 s | 0.7 s |
| `deephit` | 25.0 s | 0.4 s |
| `neural_fine_gray` | **1,209.8 s** (20.2 min) | 11.4 s |

Neural Fine-Gray is 48x DeepHit and 117x DSM, and it is the primary
comparator — the one that most needs a real search. The consequences:

| | DeepHit | DSM | Neural Fine-Gray |
|---|---|---|---|
| Full matrix (25 cells) | ~11 min | ~5 min | **~8.5 h** |
| Tuning (30 configs x 5 folds) | ~1 h | ~25 min | **~50 h** |

The wall times reflect each package's own early-stopping behaviour rather than
a fixed step count: DeepHit stops early on its validation split, while Neural
Fine-Gray runs close to its full schedule. NFG is therefore getting *more*
training, not being handicapped — but the comparison is of configured models,
not of equal compute, and that should be stated.

**Where Neural Fine-Gray's cost actually comes from**, since the mitigation
depends on it and the obvious guess is wrong:

| n | `n_iter` | Fit |
|---|---|---|
| 5,000 | 1000 | 33.1 s |
| 10,000 | 1000 | 34.1 s |
| 10,000 | **200** | 33.0 s |
| 20,000 | 1000 | 109.0 s |
| 37,000 | 1000 | 1,209.8 s |

At n = 10,000 the iteration cap is **not binding** — 200 and 1000 give the
same wall time, so early stopping fires well before 200 either way. The cost
is therefore not a per-row constant and not a fixed schedule: the number of
iterations before the validation loss plateaus GROWS with n, roughly 9.5x
between 10,000 and 37,000, on top of the extra minibatches per iteration.
Larger data keeps offering improvements that `patience_max = 3` keeps
accepting.

**Recommended mitigation.** Run the *search* on a subsample (10,000 rows costs
34 s per fit against 1,210 s, so 30 configurations x 5 folds falls from ~50
hours to ~85 minutes) and fit the selected configuration at full size, where
early stopping is free to take all the iterations it wants. This keeps the
procedure identical across models — same 30 configurations, same 5 folds, same
criterion — which the alternatives do not: cutting folds or capping the
schedule during the search weakens the comparator in ways a reviewer can
reasonably object to, whereas subsampling weakens only the precision of the
ranking.

The caveat worth stating: a configuration selected at 10,000 rows is selected
under a shorter effective training run than it will get at 37,000. That is
acceptable for choosing *architecture and learning rate*, which is what the
grid varies, and it is why the final fit must keep a generous `n_iter` and let
early stopping decide — not inherit an iteration count from the search.

Trials are independent, so the search also parallelises across processes
without changing any result.

## The orthonormalised basis

`RCSBasis` QR-factorises the truncated-power design on a **fixed dense grid**
spanning `[k₁, k_m]` and works in `Q(v) = B(v) R⁻¹`. The grid depends only on
the knots — never on observed data — so the transform is deterministic and
identical across folds by construction, and can never become another
train/test-fitted quantity. `to_original_gamma` maps coefficients back for
interpretable reporting.

Measured on this data: `cond(BᵀB) ≈ 4.2e+06` → `cond(QᵀQ) = 1.00`.

Two scaling subtleties, both discovered by the tests rather than by
inspection:

* Plain QR gives columns of unit **norm**, so entries are ~`1/√n_grid` and
  the coefficients needed to represent an O(1) function are ~`√n_grid`
  (≈ 94 for a unit-slope spline at `n_grid = 1000`). Adam moves a roughly
  fixed distance per step, so orthogonality *without* sane scaling just
  trades one optimisation pathology for another. `RCSBasis` therefore scales
  to unit **RMS** per column, putting the working coefficients on the same
  scale as the original γ.
* The degenerate-equivalence test uses learning rates above the library
  defaults with cosine annealing. This is not a fudge: the defaults are
  sized for minibatch training where one epoch is many steps, whereas that
  test is full-batch (one step per epoch) against a Newton-polished
  reference. At a constant rate Adam also oscillates around the optimum
  instead of settling into it.

**`test_variant_b.py` — Variant B does what it claims and nothing it
shouldn't.** Weibull-at-init (exact `s̃(v) = v`, `s̃′ = 1`) and Weibull
parameter recovery; nesting Variant A by planting (9.5e-07, float32 floor)
and by training (3.5e-03 against a 1e-2 tolerance); the non-proportionality
gain (+90 held-out log-likelihood units when PH is false) and its converse
(±0 when PH holds, so the extra capacity is not paid for when unused);
global monotonicity; identifiability diagnostics; 100-step stability at the
default learning rates; and checkpoint round-tripping.

**`test_cif.py` — the CIF integrates correctly.** Coherence (above);
node-count convergence, which justifies `DEFAULT_NODES` rather than assuming
it; agreement with an independent 10,000-point trapezoid (1.4e-07); the
lower-truncation check that caught the domain error above; single-cause
reduction to `1 − exp(−H₁)` when cause 2 is switched off (1.4e-07);
monotonicity and bounds; batching invariance (5.7e-09); and the timing budget.

The R comparison needs a note. `stpm2cr` is a Stata command with no R port —
`rstpm2`, built from source here, does not export it — and `flexsurv` is not
installable in this environment (CRAN archive paths are blocked by the egress
policy, and neither it nor its `mstate`/`muhaz` dependencies are in the Ubuntu
r-cran set). The parametric R route is therefore genuinely unavailable, and
the tests take the sanctioned fallback rather than skipping: R's
`cmprsk::cuminc`, the standard published implementation of the nonparametric
Aalen-Johansen estimator. It is strengthened by **stratifying on a binary
covariate**, so it checks the model's covariate conditioning rather than only
a marginal curve. The Python Aalen-Johansen implementation is itself validated
against `cmprsk` (1.1e-04) before being used. Model vs Aalen-Johansen: 4.3e-03
and 6.7e-03 within strata, 7.1e-03 on covariate-free data — all within Monte
Carlo error of a nonparametric estimator at this sample size.

**`test_metrics.py` — the metrics reward correctness and detect the failure
the project cares about.** Perfect-prediction limit (true CIFs in: Brier at
its irreducible minimum, ICI small and shown to shrink with n, so the
residual is smoother noise rather than bias); miscalibration detected while
discrimination is not; IPCW recovery of uncensored values under independent
censoring; the competing-event-controls distinction, shown to change the AUC
materially rather than cosmetically; the Kaplan-Meier upward bias against
Aalen-Johansen, with its magnitude reported; the Jensen gap between averaging
predictions and predicting at the mean covariate; agreement with
`riskRegression::Score`, `cmprsk::crr` and `cmprsk::cuminc`; bootstrap
coverage; and the benchmark timing budget.

**`test_features.py` — nothing is fitted on held-out data, and every model
sees the same information.** No-leakage (perturbing held-out rows leaves the
fitted spec byte-identical, and a spec fitted on train+test differs, so the
test has teeth); information equivalence between the two representations;
unseen levels folding onto the reference without creating a column; reference
levels; `not_biopsied` surviving as its own level; JSON round-trip; column
order stability; scaler choice recorded; and pretrain masking.

**`test_baselines.py` — the comparators are strong, and the coherence result
is real.** Interface conformance for all five models (shape, range,
monotonicity in `t`, and `F₁ + F₂ ≤ 1` for the formulations that guarantee
it); the coherence table; Cox coefficients against `survival::coxph` to
1.2e-09 (the design is scaled with `ddof=1` to match R's `scale()` — with
`ddof=0` the two designs differ by `sqrt(n/(n−1))` per column and the
agreement caps out near 1e-04, which would hide a real discrepancy of that
size); Fine-Gray against `cmprsk::crr` to 9.0e-09; RSF CIFs tracking
subgroup Aalen-Johansen curves to within 0.06; the RCS expansion's knots
coming from training rows only and its basis matching the `rstpm2`-verified
one to 1e-09; the minimal reference being beaten by the full model on
held-out Brier at every horizon and both causes; and the harness resuming
without refitting or changing a number.

## Implementation notes

**The fitter needs a Newton polish, and this is not incidental.** The
truncated-power RCS basis is badly conditioned: with boundary knots far
apart, the cubic columns span a range orders of magnitude larger than the
linear column. On that geometry L-BFGS stalls at a visibly non-zero
gradient (|g|∞ ≈ 1e-4) — close enough to look converged by loss-change
criteria, but 0.012 short of the true optimum in total log-likelihood,
which is enough to break the 1e-3 curve agreement against R. Newton is
affine-invariant, so conditioning does not affect it, and with 18
parameters the exact Hessian is cheap. `fit_classical_rp` therefore runs
L-BFGS for bulk descent and then damped Newton (Levenberg damping,
Armijo backtracking) to |g|∞ < 1e-11. After the polish, torch and R agree
on the total log-likelihood to ~2e-8.

**Models are robust to the process-wide default dtype.** `torch`'s default
dtype is global state, and the float64 verification tests set it. The neural
modules therefore cast their inputs to their own dtype rather than relying on
it — without that, running the verification and neural suites in one pytest
session produced an opaque `Float`/`Double` matmul error from deep inside the
encoder body, even though every test passed in isolation.

**Precision.** The verification path (`rp_classical.py`, the equivalence and
recovery tests) is float64. The neural path (`encoder.py`, `neural_rp.py`,
`train.py`) is float32; `RCSBasis` always computes and stores `R` in float64
and casts on use, so the basis itself is never the source of precision loss.

**Checkpoints carry the basis and the scaler.** `VariantA.checkpoint_dict()`
serialises `R` per cause alongside the weights, and the scaler is a buffer in
the state dict. A model restored without either evaluates a different
function while looking perfectly healthy — `test_checkpoint_roundtrip_
preserves_predictions` guards this.

**Monotonicity is surfaced, never clamped.** In Variant A, `s′_k > 0` is a
property of the global α_k alone, so it is checked on a dense grid rather
than per observation. `train.py` logs the minimum each epoch and warns
immediately on a violation; `assert_monotone()` raises. Transient violations
early in training (large first Adam steps) are normal and recover; a
violation at the end of training is a real failure.

**Clamps in the likelihood** (`dlogH_dv` at 1e-8 before the log, `log_H` to
[−30, 20] before `exp`) are guards against numerical excursions, not
modelling choices. Monotonicity of `H_k` is the architecture's
responsibility upstream.
