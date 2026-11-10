# ATC 2027 – KDPI 85/86 regression discontinuity

A separate study from the Neural RP paper. Nothing here imports or uses Neural RP
code, models or outputs.

Runs on the raw UNOS STAR kidney file (KIDPAN_DATA). The file is not in this
repository and must not be committed; row-level outputs are git-ignored.

| Step | Script | Outputs in `out/` |
|---|---|---|
| 1. Data audit | `01_audit.py --star FILE` | `audit_fields.csv`, `audit.md` |
| 2. Cohort + cut-offs | `02_cohort_cutoffs.py --star FILE --donor-exclusions EXCL.py` | `cohort_flow.csv`, `era_counts.csv`, `cutoffs_by_calendar_year.csv`, `cutoffs_by_period.csv`, `step2.md` |

Run scripts from inside `atc_kdpi85/`. `.DAT` releases need `--header-file`
(one column name per line). Check `audit.md` for the `TX_PROCEDUR_TY` codes
before step 2 and pass them with `--single-kidney-codes` (default `101,102`
assumes left/right single kidney).

`--donor-exclusions` takes a Python file defining `apply(df, cols) -> df` with
the donor exclusions from the original cohort code, which lives outside this
repo. `none` runs without them and every output says so.

`tests/` checks the cut-off and split-date logic on synthetic data.
