"""Step 1 - data audit of the STAR kidney file.

    python atc_kdpi85/01_audit.py --star PATH/TO/KIDPAN_DATA.sas7bdat [--header-file names.txt]

Writes atc_kdpi85/out/audit_fields.csv and atc_kdpi85/out/audit.md.
Missingness is reported in the audit scope: deceased-donor kidney transplants
on or after 26 Mar 2012 (before the age/multi-organ/dual and donor exclusions),
so it reflects the population the analysis draws from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import star_io as sio

OUT = Path(__file__).resolve().parent / "out"
# Codes STAR uses for "unknown" in Y/N fields; counted separately from blank.
UNKNOWN_CODES = {"U", "UNK", "998", "999"}


def describe(s: pd.Series) -> dict:
    s = sio.blank_to_na(s)
    n = len(s)
    miss = int(s.isna().sum())
    unk = int(s.astype("string").str.upper().isin(UNKNOWN_CODES).sum()) if n else 0
    d = {"dtype": str(s.dtype), "n": n, "n_missing": miss,
         "pct_missing": round(100 * miss / n, 2) if n else np.nan,
         "n_unknown_code": unk}
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= 0.95 * s.notna().sum() and s.nunique() > 12:
        q = num.quantile([0, .01, .5, .99, 1]).round(4).tolist()
        d["coding"] = f"numeric; min/p1/median/p99/max = {q}"
    else:
        vc = s.value_counts(dropna=True).head(12)
        d["coding"] = "; ".join(f"{k}={v}" for k, v in vc.items())
    return d


def death_handling(df: pd.DataFrame, f: dict) -> list[str]:
    """Does GSTATUS count death with a functioning graft as failure?"""
    out = []
    g, p = f["gstatus"], f["pstatus"]
    if not (g and p):
        return ["cannot check: GSTATUS or PSTATUS missing"]
    gs = pd.to_numeric(df[g], errors="coerce")
    ps = pd.to_numeric(df[p], errors="coerce")
    out.append("GSTATUS x PSTATUS:\n\n" + pd.crosstab(gs, ps, dropna=False).to_markdown())
    if f["grf_fail_date"] and f["death_date"]:
        gd, dd = sio.to_date(df[f["grf_fail_date"]]), sio.to_date(df[f["death_date"]])
        both = (gs == 1) & (ps == 1)
        same = both & gd.notna() & dd.notna() & ((dd - gd).abs().dt.days <= 1)
        nofaildate = both & gd.isna()
        out.append(f"GSTATUS=1 & PSTATUS=1: {int(both.sum()):,}; failure date within 1 day "
                   f"of death: {int(same.sum()):,}; no failure date: {int(nofaildate.sum()):,}. "
                   "Many same-day or missing failure dates would mean GSTATUS counts death "
                   "as failure; then death-censored failure must use the failure date "
                   "(or failure cause) rather than GSTATUS.")
    if f["grf_fail_cause"]:
        out.append(f"{f['grf_fail_cause']} among GSTATUS=1:\n\n" +
                   df.loc[gs == 1, f["grf_fail_cause"]].value_counts(dropna=False)
                   .head(15).to_frame().to_markdown())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--header-file")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df = sio.load_star(a.star, a.header_file)
    f = sio.resolve(df)
    lines = [f"# Step 1 - data audit\n\nFile: `{Path(a.star).name}`, "
             f"{len(df):,} rows x {df.shape[1]} columns.\n"]

    # Audit scope
    scope = pd.Series(True, index=df.index)
    if f["don_ty"]:
        scope &= sio.blank_to_na(df[f["don_ty"]]).astype("string").str.upper() == "C"
    if f["tx_date"]:
        scope &= sio.to_date(df[f["tx_date"]]) >= sio.STUDY_START
    if f["organ"]:
        scope &= sio.blank_to_na(df[f["organ"]]).astype("string").str.upper().isin(["KI"])
    d = df[scope.fillna(False)]
    lines.append(f"Audit scope (DON_TY='C', ORGAN='KI' if present, TX_DATE >= 2012-03-26): "
                 f"{len(d):,} rows.\n")

    rows = []
    for role in sio.FIELDS:
        col = f[role]
        r = {"role": role, "column": col, "needed": role in sio.NEEDED_FOR_ANALYSIS,
             "lookalikes": ", ".join(c for c in sio.lookalikes(df, role) if c != col)}
        if col:
            r.update(describe(d[col]))
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "audit_fields.csv", index=False)
    lines.append("## Fields\n\n" + tab.drop(columns=["dtype"]).to_markdown(index=False))

    missing = tab.loc[tab.column.isna(), "role"].tolist()
    lines.append("\n## Not found\n\n" + ("\n".join(f"- {m} (look-alikes: "
                 f"{tab.set_index('role').loc[m, 'lookalikes'] or 'none'})" for m in missing)
                 if missing else "none"))

    if f["share_ty"]:
        lines.append("\n## SHARE_TY by year\n\nSTAR codes: 3 = local, 4 = regional, "
                     "5 = national, 6 = foreign. After the Mar 2021 circle-based policy "
                     "'local' no longer means DSA, so non-local share is defined only "
                     "up to that date unless distance is used instead.\n\n" +
                     pd.crosstab(sio.to_date(d[f["tx_date"]]).dt.year,
                                 sio.blank_to_na(d[f["share_ty"]])).to_markdown())
    if f["kdpi"]:
        k = pd.to_numeric(d[f["kdpi"]], errors="coerce")
        nonint = int((k.dropna() % 1 != 0).sum())
        lines.append(f"\n## KDPI field\n\nrange {k.min()}-{k.max()}, non-integer values "
                     f"{nonint:,}. If KDPI is stored as a fraction (0-1), the 85/86 cut-off "
                     "is 0.85/0.86.")
    lines.append("\n## Death handling in graft status\n\n" + "\n\n".join(death_handling(d, f)))
    (OUT / "audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
