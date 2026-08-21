"""
Shared covariate encoder for the neural Royston-Parmar variants.

Continuous features are standardised with a scaler fitted on TRAINING rows
only and passed in (never fitted here - that would be a leak). Categorical
features get embeddings. The body is a configurable MLP.

The encoder is deliberately variant-agnostic: Variant A maps its output z
through a per-cause linear map, later variants will use z differently.
"""

from dataclasses import dataclass, field
from math import ceil

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    """Configuration for `CovariateEncoder`.

    n_continuous:               number of continuous features
    categorical_cardinalities:  cardinality per categorical feature (may be empty)
    hidden_dims:                MLP widths; empty () means no hidden layers
    latent_dim:                 dimension of the returned z
    dropout:                    dropout probability in the MLP body
    """
    n_continuous: int
    categorical_cardinalities: tuple = ()
    hidden_dims: tuple = (256, 128)
    latent_dim: int = 64
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return {"n_continuous": self.n_continuous,
                "categorical_cardinalities": list(self.categorical_cardinalities),
                "hidden_dims": list(self.hidden_dims),
                "latent_dim": self.latent_dim,
                "dropout": self.dropout}

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        return cls(n_continuous=d["n_continuous"],
                   categorical_cardinalities=tuple(d["categorical_cardinalities"]),
                   hidden_dims=tuple(d["hidden_dims"]),
                   latent_dim=d["latent_dim"],
                   dropout=d["dropout"])


def embedding_dim(cardinality: int) -> int:
    """dim = min(16, ceil(cardinality / 2))."""
    return min(16, ceil(cardinality / 2))


class CovariateEncoder(nn.Module):
    """x_continuous (+ optional x_categorical) -> z of dim latent_dim.

    scaler_mean / scaler_std must come from the training rows. They are
    registered as buffers so they travel with the checkpoint: a model
    restored with a different scaler is silently wrong in the same way a
    model restored without its spline basis is.

    Degenerate configuration: `hidden_dims=()` with `latent_dim` equal to the
    total input width gives an exact identity pass-through (standardisation
    only, no learned map), which reduces Variant A to a classical
    linear-covariate RP model. That is what the degenerate-case equivalence
    test exercises.
    """

    def __init__(self, config: EncoderConfig,
                 scaler_mean: torch.Tensor = None,
                 scaler_std: torch.Tensor = None,
                 dtype=torch.float32):
        super().__init__()
        self.config = config

        n_cont = config.n_continuous
        if scaler_mean is None:
            scaler_mean = torch.zeros(n_cont, dtype=dtype)
        if scaler_std is None:
            scaler_std = torch.ones(n_cont, dtype=dtype)
        scaler_std = torch.as_tensor(scaler_std, dtype=dtype).clone()
        scaler_std[scaler_std <= 0] = 1.0            # constant column guard
        self.register_buffer("scaler_mean", torch.as_tensor(scaler_mean, dtype=dtype))
        self.register_buffer("scaler_std", scaler_std)

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim(card))
            for card in config.categorical_cardinalities
        ])
        emb_total = sum(embedding_dim(c) for c in config.categorical_cardinalities)
        in_dim = n_cont + emb_total
        self.input_dim = in_dim

        layers, d = [], in_dim
        for h in config.hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(config.dropout)]
            d = h
        if d != config.latent_dim:
            layers.append(nn.Linear(d, config.latent_dim))
        elif not config.hidden_dims:
            # Exact identity pass-through (see class docstring).
            pass
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.to(dtype)

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    def forward(self, x_cont: torch.Tensor,
                x_cat: torch.Tensor = None) -> torch.Tensor:
        # Cast to the module's own dtype. torch's default dtype is
        # process-wide state that any caller can change (the float64
        # verification tests do), and a silent Float/Double mix here would
        # surface as an opaque matmul error deep in the body.
        x_cont = x_cont.to(self.scaler_mean.dtype)
        z = (x_cont - self.scaler_mean) / self.scaler_std
        if self.embeddings:
            if x_cat is None:
                raise ValueError("encoder configured with categoricals but "
                                 "x_cat is None")
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            z = torch.cat([z] + embs, dim=1)
        return self.body(z)


def fit_scaler(x_cont: torch.Tensor) -> tuple:
    """Mean/SD of the TRAINING continuous features. Call on training rows only."""
    return x_cont.mean(dim=0), x_cont.std(dim=0)
