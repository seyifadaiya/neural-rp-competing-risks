"""
Feature builder: the bridge between the imputed SRTR parquet files and the
model inputs.

It sits last in the pipeline on purpose. Category maps and continuous scalers
are learned from data, so they must be fitted on training rows and applied to
held-out rows; and the same information needs two shapes - integer codes for
the neural variants' embeddings, one-hot for the classical baselines. Both
come from one fitted `FeatureSpec`, which is what makes the model comparison
fair by construction rather than by care.
"""

from .spec import (CATEGORICAL, CONTINUOUS, DROPPED, FIXED_REFERENCES,
                   PASSTHROUGH, PRETRAIN_MASKED, ROBUST_COLUMNS, FeatureSpec,
                   Scaler, check_schema, derive_bmi, expected_columns,
                   fit_feature_spec)
from .transform import (TransformResult, encoder_config_kwargs,
                        recover_categoricals, summarise, transform)

__all__ = [
    "FeatureSpec", "Scaler", "fit_feature_spec", "transform",
    "recover_categoricals", "encoder_config_kwargs", "summarise",
    "check_schema", "derive_bmi", "expected_columns", "TransformResult",
    "CONTINUOUS", "CATEGORICAL", "ROBUST_COLUMNS", "PASSTHROUGH",
    "PRETRAIN_MASKED", "DROPPED", "FIXED_REFERENCES",
]
