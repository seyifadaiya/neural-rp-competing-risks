"""
Cause-specific time-dependent discrimination: AUC (Blanche et al.) and the
competing-risks C-index. Both IPCW-weighted, both taking arrays only.

Note on priority: the project's primary endpoint is calibrated absolute
risk, not discrimination - competing and non-competing models are known to
achieve near-identical discrimination for graft failure. These are reported
for completeness; `calibration.py` carries the argument.
"""

import numpy as np

from .ipcw import CensoringDistribution, make_censoring


def _case_control_weights(time, event, cause, t, cens):
    """Cases, controls and their IPCW weights at horizon t.

    CASES    : an observed cause-`cause` event by t.
    CONTROLS : event-free at t, PLUS subjects who had a COMPETING event by t.

    Including the competing-event group among the controls is the standard
    competing-risks definition and is exactly where naive implementations go
    wrong - dropping them inflates the AUC, because those subjects typically
    carry high predicted risk and would otherwise be scored as cases.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)

    is_case = (time <= t) & (event == cause)
    is_competing = (time <= t) & (event > 0) & (event != cause)
    is_free = time > t
    is_control = is_competing | is_free

    w = np.zeros_like(time)
    ev_by_t = is_case | is_competing
    w[ev_by_t] = 1.0 / cens.G_minus(time[ev_by_t])
    w[is_free] = 1.0 / cens.G(np.full(is_free.sum(), t))
    return is_case, is_control, w


def time_dependent_auc(cif, time, event, t_eval, cens=None, causes=None):
    """IPCW cause-specific time-dependent AUC.

    cif (n, T, K), time (n,), event (n,), t_eval (T,). Returns {cause: array(T)}.

    O(n log n) per (cause, horizon) via a sort and cumulative weights - no
    O(n^2) pair enumeration.
    """
    cif = np.asarray(cif, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
    n_causes = cif.shape[2]
    causes = causes or list(range(1, n_causes + 1))
    if cens is None:
        cens = make_censoring(time, event)

    out = {}
    for k in causes:
        vals = np.full(len(t_eval), np.nan)
        for j, t in enumerate(t_eval):
            is_case, is_ctrl, w = _case_control_weights(time, event, k, t, cens)
            if is_case.sum() == 0 or is_ctrl.sum() == 0:
                continue
            marker = cif[:, j, k - 1]
            order = np.argsort(marker, kind="mergesort")
            m, wc = marker[order], w[order]
            case_w = wc * is_case[order]
            ctrl_w = wc * is_ctrl[order]

            # Weighted controls strictly below each position, plus half the
            # tied mass (ties split evenly, as in the standard estimator).
            cum_ctrl = np.concatenate([[0.0], np.cumsum(ctrl_w)])
            # Group boundaries for ties in the marker.
            uniq, first = np.unique(m, return_index=True)
            grp = np.searchsorted(uniq, m)
            below = cum_ctrl[first][grp]                       # strictly less
            tot_grp = np.add.reduceat(ctrl_w, first)[grp]      # tied mass
            conc = below + 0.5 * tot_grp

            num = float((case_w * conc).sum())
            den = float(case_w.sum() * ctrl_w.sum())
            vals[j] = num / den if den > 0 else np.nan
        out[k] = vals
    return out


class _Fenwick:
    """Weighted prefix sums with point updates - used for the O(n log n)
    C-index, which is a 2-D dominance count and has no cumulative-sum form."""

    __slots__ = ("n", "tree")

    def __init__(self, n):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.float64)

    def add(self, i, w):
        i += 1
        while i <= self.n:
            self.tree[i] += w
            i += i & (-i)

    def prefix(self, i):
        """Sum of weights at positions < i."""
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s


def time_dependent_cindex(cif, time, event, t_eval, cens=None, causes=None):
    """IPCW cause-specific time-dependent C-index (competing-risks form).

    Comparable pairs for cause k, truncated at horizon t: subject i has an
    observed cause-k event at T_i <= t, and subject j is either
      (a) still at risk after T_i, weight 1 / (G(T_i-) G(T_i-)), or
      (b) known to have had a COMPETING event at T_j <= T_i, weight
          1 / (G(T_i-) G(T_j-)).
    Concordant when the predicted cause-k risk is higher for i.

    Both pair types are 2-D dominance counts (a condition on T and one on the
    marker), so this uses a Fenwick tree: O(n log n) rather than O(n^2).
    """
    cif = np.asarray(cif, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
    causes = causes or list(range(1, cif.shape[2] + 1))
    if cens is None:
        cens = make_censoring(time, event)
    n = len(time)
    g_minus = cens.G_minus(time)

    out = {}
    for k in causes:
        vals = np.full(len(t_eval), np.nan)
        for j_t, t in enumerate(t_eval):
            marker = cif[:, j_t, k - 1]
            uniq_m, inv_m = np.unique(marker, return_inverse=True)
            n_rank = len(uniq_m)

            cases = np.flatnonzero((time <= t) & (event == k))
            if cases.size == 0:
                continue
            competing = np.flatnonzero((event > 0) & (event != k))

            # --- type (a): j at risk after T_i -------------------------------
            # Insert every subject, process cases from the largest T_i down, so
            # the tree always holds exactly {j : T_j > T_i}.
            order_desc = np.argsort(-time, kind="mergesort")
            t_desc = time[order_desc]
            bit_a = _Fenwick(n_rank)
            tot_a = np.zeros(n_rank + 1)
            num = den = 0.0
            case_order = cases[np.argsort(-time[cases], kind="mergesort")]
            ptr, inserted = 0, 0.0
            for i in case_order:
                while ptr < n and t_desc[ptr] > time[i]:
                    jj = order_desc[ptr]
                    bit_a.add(inv_m[jj], 1.0)
                    inserted += 1.0
                    ptr += 1
                wi = 1.0 / (g_minus[i] * g_minus[i])
                lower = bit_a.prefix(inv_m[i])
                tied = bit_a.prefix(inv_m[i] + 1) - lower
                num += wi * (lower + 0.5 * tied)
                den += wi * inserted

            # --- type (b): j had a competing event at T_j <= T_i --------------
            if competing.size:
                bit_b = _Fenwick(n_rank)
                comp_order = competing[np.argsort(time[competing], kind="mergesort")]
                t_comp = time[comp_order]
                case_asc = cases[np.argsort(time[cases], kind="mergesort")]
                ptr_b, inserted_b = 0, 0.0
                for i in case_asc:
                    while ptr_b < len(comp_order) and t_comp[ptr_b] <= time[i]:
                        jj = comp_order[ptr_b]
                        wj = 1.0 / g_minus[jj]
                        bit_b.add(inv_m[jj], wj)
                        inserted_b += wj
                        ptr_b += 1
                    wi = 1.0 / g_minus[i]
                    lower = bit_b.prefix(inv_m[i])
                    tied = bit_b.prefix(inv_m[i] + 1) - lower
                    num += wi * (lower + 0.5 * tied)
                    den += wi * inserted_b

            vals[j_t] = num / den if den > 0 else np.nan
        out[k] = vals
    return out
