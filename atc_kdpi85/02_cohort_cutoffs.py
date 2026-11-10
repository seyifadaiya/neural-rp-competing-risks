"""Step 2 - cohort and KDPI 85/86 cut-off check.

    python atc_kdpi85/02_cohort_cutoffs.py --star PATH/TO/KIDPAN_DATA.sas7bdat \
        --donor-exclusions PATH/TO/exclusions.py   # or: --donor-exclusions none

Cohort: deceased donor, kidney-alone, single-kidney (no dual / en bloc, no
multi-organ), adult recipient (AGE >= 18), TX_DATE >= 2012-03-26, then the
donor exclusions used by the existing cohort code. That code is NOT in this
repository (the repo starts from already-imputed files), so the exclusions must
be supplied as a Python file defining `apply(df, cols) -> df`. Passing `none`
runs without them and says so in every output.

Mapping periods start as calendar years of TX_DATE. Inside each period the
cut-off is the KDRI_RAO threshold that puts the fewest kidneys on the wrong side
of the displayed label; when that number is not zero the period is split at the
transplant date that best separates two mapping tables (binary segmentation),
because a mid-year table change shows up exactly as wrong-side kidneys.

Outputs (atc_kdpi85/out/): cutoffs_by_period.csv, cohort_flow.csv,
era_counts.csv, step2.md, and wrong_side_kidneys.csv (row-level, git-ignored).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import star_io as sio

OUT = Path(__file__).resolve().parent / "out"
LABEL_KDPI = 86          # KDPI >= 86 carries the "high-KDPI" label


# ---------------------------------------------------------------- cut-off maths
def best_threshold(kdri: np.ndarray, high: np.ndarray) -> tuple[float, int, int, int]:
    """Threshold c (a KDRI value; KDRI >= c -> predicted label 86+) minimising
    wrong-side kidneys. Returns (c, n_wrong, n_low_above, n_high_below)."""
    o = np.argsort(kdri, kind="stable")
    k, h = kdri[o], high[o].astype(int)
    uniq, first = np.unique(k, return_index=True)
    # for c = uniq[j]: high below = high among k < c; low above = low among k >= c
    high_below = np.concatenate([[0], np.cumsum(h)])[first]
    low_total = len(h) - h.sum()
    low_below = first - high_below
    low_above = low_total - low_below
    # also allow c above all values (everything predicted low)
    cand_wrong = np.append(high_below + low_above, h.sum())
    j = int(np.argmin(cand_wrong))
    if j == len(uniq):
        return np.inf, int(h.sum()), 0, int(h.sum())
    return float(uniq[j]), int(cand_wrong[j]), int(low_above[j]), int(high_below[j])


def summarise(seg: pd.DataFrame) -> dict:
    lo, hi = seg[~seg.high], seg[seg.high]
    c, wrong, la, hb = best_threshold(seg.kdri.to_numpy(), seg.high.to_numpy())
    return {"start": seg.tx_date.min().date(), "end": seg.tx_date.max().date(),
            "n": len(seg), "n_kdpi_le85": len(lo), "n_kdpi_ge86": len(hi),
            "max_kdri_kdpi_le85": lo.kdri.max(), "min_kdri_kdpi_ge86": hi.kdri.min(),
            "cutoff": c, "n_wrong_side": wrong,
            "n_le85_at_or_above_cutoff": la, "n_ge86_below_cutoff": hb}


def segment(seg: pd.DataFrame, min_side: int, depth: int = 0, max_depth: int = 4,
            splits: list | None = None) -> list[pd.DataFrame]:
    """Split a period at the date that most reduces wrong-side kidneys. Every
    split date that ties for best is recorded in `splits` as a window: the
    table changed somewhere inside it."""
    splits = [] if splits is None else splits
    base = best_threshold(seg.kdri.to_numpy(), seg.high.to_numpy())[1]
    if base == 0 or depth >= max_depth:
        return [seg]
    seg = seg.sort_values("tx_date")
    dates = seg.tx_date.to_numpy()
    near = seg.kdpi.between(76, 95).to_numpy()            # kidneys that pin the cut-off
    near_cum = np.cumsum(near)
    best, ties = (base, None), []
    for d in np.unique(dates)[1:]:
        i = np.searchsorted(dates, d)                       # left = rows [0, i)
        if near_cum[i - 1] < min_side or near_cum[-1] - near_cum[i - 1] < min_side:
            continue
        L, R = seg.iloc[:i], seg.iloc[i:]
        cost = (best_threshold(L.kdri.to_numpy(), L.high.to_numpy())[1] +
                best_threshold(R.kdri.to_numpy(), R.high.to_numpy())[1])
        if cost < best[0]:
            best, ties = (cost, i), [d]
        elif cost == best[0] and best[1] is not None:
            ties.append(d)
    cost, i = best
    # accept only a split that removes at least half of the wrong-side kidneys;
    # a handful of stray kidneys is a data issue, not a table change
    if i is None or cost > base / 2:
        return [seg]
    splits.append({"first_new_table_date_earliest": pd.Timestamp(ties[0]).date(),
                   "first_new_table_date_latest": pd.Timestamp(ties[-1]).date(),
                   "wrong_side_before": base, "wrong_side_after": cost})
    return (segment(seg.iloc[:i], min_side, depth + 1, max_depth, splits) +
            segment(seg.iloc[i:], min_side, depth + 1, max_depth, splits))


# ---------------------------------------------------------------- cohort
def build_cohort(df: pd.DataFrame, f: dict, single_codes: set, excl_path: str):
    flow = [("STAR kidney file", len(df))]

    def keep(mask, label):
        nonlocal df
        df = df[mask.fillna(False).to_numpy()]
        flow.append((label, len(df)))

    up = lambda c: sio.blank_to_na(df[c]).astype("string").str.upper()
    keep(up(f["don_ty"]) == "C", "deceased donor (DON_TY = C)")
    df = df.assign(tx_date=sio.to_date(df[f["tx_date"]]))
    keep(df.tx_date >= sio.STUDY_START, "TX_DATE >= 2012-03-26")
    if f["organ"]:
        keep(up(f["organ"]) == "KI", "kidney alone (ORGAN = KI)")
    if f["multiorg"]:
        keep(up(f["multiorg"]).fillna("N") != "Y", "not multi-organ (MULTIORG != Y)")
    if f["tx_procedure"]:
        proc = pd.to_numeric(df[f["tx_procedure"]], errors="coerce")
        keep(proc.isin(single_codes), f"single kidney ({f['tx_procedure']} in {sorted(single_codes)})")
    else:
        flow.append(("WARNING: no procedure-type field; dual/en bloc NOT excluded", len(df)))
    keep(pd.to_numeric(df[f["rec_age"]], errors="coerce") >= 18, "adult recipient (AGE >= 18)")

    if excl_path.lower() == "none":
        flow.append(("WARNING: existing donor exclusions NOT applied (--donor-exclusions none)", len(df)))
    else:
        spec = importlib.util.spec_from_file_location("donor_exclusions", excl_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        df = mod.apply(df, f)
        flow.append(("existing donor exclusions", len(df)))

    kdpi = pd.to_numeric(df[f["kdpi"]], errors="coerce")
    if kdpi.max() <= 1.0:                                   # stored as a fraction
        kdpi = kdpi * 100
    df = df.assign(kdpi=kdpi.round(6), kdri=pd.to_numeric(df[f["kdri_rao"]], errors="coerce"))
    keep(df.kdpi.notna() & df.kdri.notna(), "KDPI and KDRI_RAO present")
    df = df.assign(high=df.kdpi >= LABEL_KDPI,
                   era=np.where(df.tx_date < sio.KAS_START, "pre-KAS", "post-KAS"))
    return df, pd.DataFrame(flow, columns=["step", "n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--header-file")
    ap.add_argument("--donor-exclusions", required=True,
                    help="python file defining apply(df, cols), or 'none'")
    ap.add_argument("--single-kidney-codes", default="101,102",
                    help="TX_PROCEDUR_TY codes for one kidney; check audit.md first")
    ap.add_argument("--min-side", type=int, default=30,
                    help="min kidneys with KDPI 76-95 on each side of a split date")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    raw = sio.load_star(a.star, a.header_file)
    f = sio.resolve(raw)
    need = ["tx_date", "kdpi", "kdri_rao", "don_ty", "rec_age"]
    if any(f[r] is None for r in need):
        raise SystemExit(f"missing required fields: {[r for r in need if f[r] is None]}")
    codes = {int(c) for c in a.single_kidney_codes.split(",")}
    if f["tx_procedure"]:
        print("procedure codes in file:\n",
              pd.to_numeric(raw[f["tx_procedure"]], errors="coerce").value_counts(dropna=False))
    df, flow = build_cohort(raw, f, codes, a.donor_exclusions)
    flow.to_csv(OUT / "cohort_flow.csv", index=False)

    # periods: calendar years, split where the mapping changes mid-year
    periods, cal, splits = [], [], []
    for yr, g in df.groupby(df.tx_date.dt.year):
        cal.append({"year": yr, **summarise(g)})
        for s in segment(g, a.min_side, splits=splits):
            periods.append(s)
    cal = pd.DataFrame(cal)
    tab = pd.DataFrame([{"period": i + 1, **summarise(s)} for i, s in enumerate(periods)])
    tab.to_csv(OUT / "cutoffs_by_period.csv", index=False)
    cal.to_csv(OUT / "cutoffs_by_calendar_year.csv", index=False)

    # row-level wrong-side kidneys (git-ignored) for inspection
    wrong = []
    for (_, r), s in zip(tab.iterrows(), periods):
        w = s[(s.high & (s.kdri < r.cutoff)) | (~s.high & (s.kdri >= r.cutoff))]
        wrong.append(w.assign(period=r.period, cutoff=r.cutoff))
    keepcols = [c for c in ["period", "tx_date", f["donor_id"], "kdpi", "kdri", "cutoff"] if c]
    pd.concat(wrong)[keepcols].to_csv(OUT / "wrong_side_kidneys.csv", index=False)

    # does the cut-off move? pooled single threshold
    pooled = best_threshold(df.kdri.to_numpy(), df.high.to_numpy())
    single_table = pooled[1] == 0
    era = (df.assign(kdpi_70_100=df.kdpi.between(70, 100))
             .groupby("era", sort=False)
             .agg(n=("kdpi", "size"), n_kdpi_70_100=("kdpi_70_100", "sum"),
                  first_tx=("tx_date", "min"), last_tx=("tx_date", "max")))
    era.to_csv(OUT / "era_counts.csv")
    straddle = tab[(pd.to_datetime(tab.start) < sio.KAS_START) & (pd.to_datetime(tab.end) >= sio.KAS_START)]

    md = ["# Step 2 - cohort and cut-off check\n", "## Cohort flow\n", flow.to_markdown(index=False),
          "\n## N per era\n", era.to_markdown(),
          "\n## Cut-off by calendar year (before splitting)\n", cal.to_markdown(index=False),
          "\n## Mid-year mapping changes found\n",
          pd.DataFrame(splits).to_markdown(index=False) if splits else "none",
          "\n## Cut-off by mapping period\n", tab.to_markdown(index=False),
          f"\nPooled single threshold over all years: cut-off {pooled[0]:.6f}, "
          f"{pooled[1]:,} kidneys on the wrong side.\n",
          f"Cut-off range across periods: {tab.cutoff.min():.6f} to {tab.cutoff.max():.6f}.\n"]
    if single_table:
        md.append("**STOP: one KDRI threshold separates KDPI <= 85 from >= 86 in every year. "
                  "The KDPI field was probably recomputed with a single mapping table, so it "
                  "is not the label shown on the offer.**\n")
    if tab.n_wrong_side.sum():
        md.append(f"{int(tab.n_wrong_side.sum()):,} kidneys remain on the wrong side after "
                  "splitting (listed in wrong_side_kidneys.csv). KDPI is set at the match run, "
                  "which can precede TX_DATE by days, so a few near each change date are expected.\n")
    if len(straddle):
        md.append(f"Periods straddling the KAS start (4 Dec 2014): {straddle.period.tolist()}. "
                  "Era is assigned by TX_DATE; the cut-off is by period.\n")
    (OUT / "step2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
