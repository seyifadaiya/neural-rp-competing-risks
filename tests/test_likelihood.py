"""Module 2 unit test: the vectorised loss reproduces a hand-computed
likelihood on a 5-subject toy dataset to 1e-10."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcs_basis import spline_deriv, spline_eval
from rp_likelihood import rp_competing_loss

torch.set_default_dtype(torch.float64)


def test_toy_dataset_hand_computed():
    # Fixed spline setup per cause (3 knots -> gamma has 3 entries) and
    # one covariate with fixed beta.
    knots = torch.tensor([-1.0, 0.2, 1.5], dtype=torch.float64)
    gamma = {1: torch.tensor([-0.5, 0.9, 0.15], dtype=torch.float64),
             2: torch.tensor([-1.1, 1.2, -0.08], dtype=torch.float64)}
    beta = {1: torch.tensor([0.7], dtype=torch.float64),
            2: torch.tensor([-0.4], dtype=torch.float64)}

    time = torch.tensor([0.8, 1.7, 0.4, 2.6, 1.1], dtype=torch.float64)
    event = torch.tensor([0, 1, 1, 2, 2])
    x = torch.tensor([[0.3], [-1.2], [0.5], [2.0], [-0.7]], dtype=torch.float64)
    v = torch.log(time)

    # Model quantities for the vectorised loss.
    log_H = torch.stack(
        [spline_eval(v, knots, gamma[k]) + (x @ beta[k]) for k in (1, 2)],
        dim=1,
    )
    dlogH = torch.stack([spline_deriv(v, knots, gamma[k]) for k in (1, 2)], dim=1)
    loss = rp_competing_loss(log_H, dlogH, event, v)

    # Hand computation, subject by subject, straight from the formulas:
    #   log L_i = 1[d>0] (log H_d + log s'_d(v) - v) - H_1 - H_2
    def s(vi, k):
        lam = (knots[2] - knots[1]) / (knots[2] - knots[0])
        B = (max(vi - knots[1].item(), 0.0) ** 3
             - lam.item() * max(vi - knots[0].item(), 0.0) ** 3
             - (1 - lam.item()) * max(vi - knots[2].item(), 0.0) ** 3)
        g = gamma[k]
        return g[0].item() + g[1].item() * vi + g[2].item() * B

    def sp(vi, k):
        lam = (knots[2] - knots[1]) / (knots[2] - knots[0])
        dB = 3 * (max(vi - knots[1].item(), 0.0) ** 2
                  - lam.item() * max(vi - knots[0].item(), 0.0) ** 2
                  - (1 - lam.item()) * max(vi - knots[2].item(), 0.0) ** 2)
        g = gamma[k]
        return g[1].item() + g[2].item() * dB

    total = 0.0
    for i in range(5):
        vi = math.log(time[i].item())
        xi = x[i, 0].item()
        logH = {k: s(vi, k) + beta[k][0].item() * xi for k in (1, 2)}
        li = -(math.exp(logH[1]) + math.exp(logH[2]))
        d = event[i].item()
        if d > 0:
            li += logH[d] + math.log(sp(vi, d)) - vi
        total += li
    hand_loss = -total / 5

    assert abs(loss.item() - hand_loss) < 1e-10


def test_clamps_guard_pathological_inputs():
    # A negative derivative or extreme log_H must not produce nan/inf.
    log_H = torch.tensor([[50.0, -50.0]], dtype=torch.float64)
    dlogH = torch.tensor([[-3.0, 1.0]], dtype=torch.float64)
    event = torch.tensor([1])
    log_t = torch.tensor([0.0], dtype=torch.float64)
    loss = rp_competing_loss(log_H, dlogH, event, log_t)
    assert torch.isfinite(loss)
