"""
Competing-risks likelihood for Royston-Parmar models on the log
cumulative hazard scale.

For subject i with follow-up t_i and event indicator delta_i in {0, 1, 2}:

    log L_i = 1[delta_i > 0] * log h_{delta_i}(t_i) - sum_k H_k(t_i)

with, on v = log t,

    log h_k(t) = log H_k(t) + log( d log H_k / dv ) - log t.

Closed form throughout - no numerical integration.
"""

import torch

EPS_DERIV = 1e-8
LOG_H_MIN, LOG_H_MAX = -30.0, 20.0


def rp_competing_loss(log_H: torch.Tensor,
                      dlogH_dv: torch.Tensor,
                      event_type: torch.Tensor,
                      log_t: torch.Tensor) -> torch.Tensor:
    """
    log_H:      (n, 2)  log H_k(t_i | x_i) for k = 1, 2
    dlogH_dv:   (n, 2)  d log H_k / dv at v = log t_i  [= s'_k(log t_i) in Variant A]
    event_type: (n,)    0 censored, 1 graft failure, 2 death
    log_t:      (n,)
    returns:    scalar mean negative log-likelihood
    """
    log_H = torch.clamp(log_H, LOG_H_MIN, LOG_H_MAX)
    H = torch.exp(log_H)                                    # (n, 2)

    # Survival part: -sum_k H_k for every subject.
    loglik = -H.sum(dim=1)                                  # (n,)

    # Event part, fully vectorised via one-hot masks per cause.
    log_deriv = torch.log(torch.clamp(dlogH_dv, min=EPS_DERIV))
    for k in (1, 2):
        is_k = (event_type == k).to(log_H.dtype)            # (n,)
        loglik = loglik + is_k * (log_H[:, k - 1]
                                  + log_deriv[:, k - 1]
                                  - log_t)
    return -loglik.mean()
