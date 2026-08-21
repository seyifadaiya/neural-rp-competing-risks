"""
Benchmark harness and classical baselines.

Every model - classical here, neural elsewhere, deep-learning later -
satisfies one narrow protocol: fit on training rows, return a CIF in the
layout `rp_core.metrics.evaluate()` consumes. Identical scoring code for all
of them is what makes the comparison fair rather than merely careful.
"""

from .base import (BenchmarkCell, RECORD_COLUMNS, SurvivalBaseline,
                   coherence_table, derive_seed, records_from_evaluation,
                   run_benchmark)
from .classical import (CoxBaseline, FineGrayBaseline, MinimalReferenceBaseline,
                        RoystonParmarBaseline, RSFBaseline,
                        make_default_baselines, time_knots)
from .covariate_splines import (CovariateSplineSpec, apply_covariate_splines,
                                describe, events_per_variable,
                                fit_covariate_splines)

from .tuning import (SEARCH_SPACES, assert_tunable, make_cv_folds,
                     random_search, score_fold, search_space_table,
                     tuning_summary)

__all__ = [
    "SurvivalBaseline", "BenchmarkCell", "run_benchmark", "derive_seed",
    "records_from_evaluation", "coherence_table", "RECORD_COLUMNS",
    "RoystonParmarBaseline", "CoxBaseline", "FineGrayBaseline",
    "RSFBaseline", "MinimalReferenceBaseline", "make_default_baselines",
    "time_knots",
    "CovariateSplineSpec", "fit_covariate_splines", "apply_covariate_splines",
    "events_per_variable", "describe",     "SEARCH_SPACES", "assert_tunable", 
    "make_cv_folds", "random_search",
    "score_fold", "search_space_table", "tuning_summary",
]
