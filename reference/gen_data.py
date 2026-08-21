"""Generate the synthetic competing-risks dataset for the R equivalence
check (Module 4). Latent-failure-times construction: per cause k draw
E_k ~ Exp(1) and invert the Weibull cause-specific cumulative hazard

    H_k(t | x) = scale_k * t^shape_k * exp(beta_k^T x)
    =>  T_k = (E_k / (scale_k exp(beta_k^T x)))^(1/shape_k)

then T = min(T_1, T_2), cause = argmin, censored by C ~ U(0, C_MAX).
This yields exactly those cause-specific hazards. Seed 42; targets
roughly 25% cause 1 / 25% cause 2 / 50% censored.

Run:  uv run python rp_core/reference/gen_data.py
Writes rp_core/reference/sim_data.csv
"""

from pathlib import Path

import numpy as np

SEED = 42
N = 5_000
SHAPE = {1: 1.3, 2: 0.9}
SCALE = {1: 0.042, 2: 0.085}
BETA = {1: np.array([0.5, -0.3, 0.4, -0.2]),
        2: np.array([-0.2, 0.4, -0.3, 0.5])}
C_MAX = 9.0


def generate(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.column_stack([
        rng.standard_normal(n),
        rng.standard_normal(n),
        (rng.random(n) < 0.5).astype(float),
        (rng.random(n) < 0.3).astype(float),
    ])
    latent = {}
    for k in (1, 2):
        e = rng.exponential(size=n)
        rate = SCALE[k] * np.exp(x @ BETA[k])
        latent[k] = (e / rate) ** (1.0 / SHAPE[k])
    t_event = np.minimum(latent[1], latent[2])
    cause = np.where(latent[1] <= latent[2], 1, 2)
    c = rng.uniform(0, C_MAX, size=n)
    time = np.minimum(t_event, c)
    event = np.where(t_event <= c, cause, 0)
    return time, event, x


def main():
    time, event, x = generate()
    out = Path(__file__).parent / "sim_data.csv"
    header = "time,event,x1,x2,x3,x4"
    np.savetxt(out, np.column_stack([time, event, x]),
               delimiter=",", header=header, comments="", fmt="%.12g")
    frac = [(event == k).mean() for k in (0, 1, 2)]
    print(f"wrote {out}")
    print(f"censored {frac[0]:.3f}, cause1 {frac[1]:.3f}, cause2 {frac[2]:.3f}")
    print(f"time range [{time.min():.4g}, {time.max():.4g}]")


if __name__ == "__main__":
    main()
