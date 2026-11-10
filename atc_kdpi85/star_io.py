"""Load the UNOS STAR kidney (KIDPAN) file and name the fields this study needs.

Standalone: imports nothing from the Neural RP code. Names below are the usual
STAR KIDPAN_DATA names; 01_audit.py checks each one against the actual file and
also lists any other column whose name looks related, so a renamed field is
found rather than silently missed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Era boundaries (transplant date).
STUDY_START = pd.Timestamp("2012-03-26")
KAS_START = pd.Timestamp("2014-12-04")

# role -> (candidate column names in priority order, regex used to list look-alikes)
FIELDS: dict[str, tuple[list[str], str]] = {
    "tx_date":        (["TX_DATE"], r"^TX_DATE|TXDATE"),
    "kdpi":           (["KDPI"], r"KDPI"),
    "kdri_rao":       (["KDRI_RAO"], r"KDRI"),
    "donor_id":       (["DONOR_ID"], r"DONOR_ID|DON_ID"),
    "cold_isch":      (["COLD_ISCH_KI", "COLD_ISCH"], r"COLD_ISCH|CIT"),
    "pump":           (["PUMP_KI", "MACHINE_PERF"], r"PUMP|PERF"),
    "share_ty":       (["SHARE_TY"], r"SHARE"),
    "distance":       (["DISTANCE"], r"DIST"),
    "rec_age":        (["AGE"], r"^AGE$|^INIT_AGE$|^AGE_TX"),
    "biopsy":         (["BIOPSY_DGN", "KI_BIOPSY", "BIOPSY"], r"BIOP"),
    "first_wk_dial":  (["FIRST_WK_DIAL"], r"FIRST_WK|DGF"),
    "gstatus":        (["GSTATUS_KI", "GSTATUS"], r"GSTATUS"),
    "gtime":          (["GTIME_KI", "GTIME"], r"GTIME"),
    "grf_fail_date":  (["GRF_FAIL_DATE_KI", "GRF_FAIL_DATE"], r"GRF_FAIL|FAIL_DATE"),
    "grf_fail_cause": (["GRF_FAIL_CAUSE_TY_KI", "GRF_FAIL_CAUSE_TY"], r"FAIL_CAUSE"),
    "pstatus":        (["PSTATUS"], r"PSTATUS"),
    "ptime":          (["PTIME"], r"PTIME"),
    "death_date":     (["COMPOSITE_DEATH_DATE", "PX_STAT_DATE"], r"DEATH|PX_STAT"),
    # cohort definition
    "don_ty":         (["DON_TY"], r"DON_TY"),
    "multiorg":       (["MULTIORG"], r"MULTIORG|MULTI_ORG"),
    "organ":          (["ORGAN"], r"^ORGAN$"),
    "tx_procedure":   (["TX_PROCEDUR_TY_KI", "TX_PROCEDUR_TY"], r"PROCEDUR|EN_?BLOC|DUAL"),
    "rec_id":         (["TRR_ID_CODE", "PT_CODE"], r"TRR_ID|PT_CODE"),
    # balance checks (not KDRI inputs, not affected by the label)
    "don_sex":        (["GENDER_DON"], r"GENDER_DON|SEX_DON"),
    "don_abo":        (["ABO_DON"], r"ABO_DON"),
}

NEEDED_FOR_ANALYSIS = ["tx_date", "kdpi", "kdri_rao", "donor_id", "cold_isch",
                       "pump", "share_ty", "distance", "rec_age", "biopsy",
                       "first_wk_dial", "gstatus", "gtime"]


def load_star(path: str | Path, header_file: str | Path | None = None) -> pd.DataFrame:
    """Read the STAR kidney file. Supports .sas7bdat, .dta, .parquet, .csv/.tsv,
    and the tab-delimited .DAT release (column names from `header_file`, one per
    line, or from the STAR .htm/.xls data dictionary exported to text)."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".sas7bdat":
        df = pd.read_sas(p, encoding="latin-1")
    elif suf == ".dta":
        df = pd.read_stata(p, convert_categoricals=False)
    elif suf == ".parquet":
        df = pd.read_parquet(p)
    elif suf in (".csv",):
        df = pd.read_csv(p, low_memory=False)
    elif suf in (".tsv", ".dat", ".txt"):
        names = None
        if header_file is not None:
            names = [l.strip() for l in Path(header_file).read_text().splitlines() if l.strip()]
        df = pd.read_csv(p, sep="\t", header=None if names else 0, names=names,
                         low_memory=False, encoding="latin-1", na_values=["", "."])
    else:
        raise ValueError(f"unsupported file type: {p}")
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def resolve(df: pd.DataFrame) -> dict[str, str | None]:
    """role -> first candidate column present in df (None if none)."""
    cols = set(df.columns)
    return {role: next((c for c in cands if c in cols), None)
            for role, (cands, _) in FIELDS.items()}


def lookalikes(df: pd.DataFrame, role: str) -> list[str]:
    rx = re.compile(FIELDS[role][1], re.I)
    return [c for c in df.columns if rx.search(c)]


def blank_to_na(s: pd.Series) -> pd.Series:
    if s.dtype == object or str(s.dtype).startswith("string"):
        s = s.astype("string").str.strip().replace({"": pd.NA, ".": pd.NA})
    return s


def to_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):          # SAS/Stata day counts from 1960-01-01
        return pd.to_datetime(s, unit="D", origin="1960-01-01", errors="coerce")
    return pd.to_datetime(blank_to_na(s), errors="coerce")
