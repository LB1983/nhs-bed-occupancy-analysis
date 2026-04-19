#!/usr/bin/env python
"""
MASTER QUARTERLY DATASET MERGE (MINIMAL MASTER - option A)
=========================================================

Goal
----
Produce a publication-grade *minimal* master dataset, keeping only:
- Keys + core metadata: org_code, org_name, period, financial_year, fy_quarter
- Clean KH03 bed-base fields (average beds, computed rates; no Excel artefacts)
- Core outcome fields from AE / Cancelled Ops / UEC NCTR

This avoids accidental use of ambiguous KH03 columns (e.g. _13, _19, % occupied, unnamed_*).

Inputs (repo-relative)
---------------------
- KH03:           02-data-interim/kh03-cleaned/kh03-all-quarters.csv
- A&E:            02-data-interim/ae-performance/ae-quarterly-trust.csv
- Cancelled Ops:  02-data-interim/cancelled-ops/cancelled-ops-quarterly-trust.csv
- UEC NCTR daily: 02-data-interim/uec-daily-cleaned/uec-nctr-table2-daily-icb-trust.csv

Outputs
-------
- 03-data-final/master-quarterly-trust.csv
- 03-data-final/merge-summary-stats.txt
- 03-data-final/data-dictionary.csv

Run (from repo root)
--------------------
python 04-analysis/04-merge-quarterly-datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
KH03_PATH = Path("02-data-interim/kh03-cleaned/kh03-all-quarters.csv")
AE_PATH = Path("02-data-interim/ae-performance/ae-quarterly-trust.csv")
CANCELLED_OPS_PATH = Path("02-data-interim/cancelled-ops/cancelled-ops-quarterly-trust.csv")
UEC_NCTR_DAILY_PATH = Path("02-data-interim/uec-daily-cleaned/uec-nctr-table2-daily-icb-trust.csv")

OUT_DIR = Path("03-data-final")
OUT_MASTER = OUT_DIR / "master-quarterly-trust.csv"
OUT_SUMMARY = OUT_DIR / "merge-summary-stats.txt"
OUT_DICT = OUT_DIR / "data-dictionary.csv"

MERGE_KEYS = ["org_code", "period"]
CORE_META = ["org_name", "financial_year", "fy_quarter"]

FY_Q_BY_MONTH = {
    1: "Q4", 2: "Q4", 3: "Q4",
    4: "Q1", 5: "Q1", 6: "Q1",
    7: "Q2", 8: "Q2", 9: "Q2",
    10: "Q3", 11: "Q3", 12: "Q3",
}

# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------
def banner(title: str) -> str:
    line = "=" * 80
    return f"\n{line}\n{title}\n{line}\n"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def norm_org_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def norm_period(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def ensure_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}. Found: {list(df.columns)}")

def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    n = pd.to_numeric(n, errors="coerce")
    d = pd.to_numeric(d, errors="coerce")
    out = n / d
    out[(d <= 0) | d.isna()] = np.nan
    return out

def assert_unique_keys(df: pd.DataFrame, keys: List[str], name: str) -> pd.DataFrame:
    dup_mask = df.duplicated(subset=keys, keep=False)
    if not dup_mask.any():
        return df

    print(f"  ⚠️  {name}: {dup_mask.sum():,} duplicate-key rows detected on {keys}. Collapsing...")

    numeric_cols = [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])]
    other_cols = [c for c in df.columns if c not in keys and c not in numeric_cols]

    agg_spec: Dict[str, str] = {c: "sum" for c in numeric_cols}
    for c in other_cols:
        agg_spec[c] = "first"

    collapsed = df.groupby(keys, as_index=False).agg(agg_spec)
    print(f"  ✅ {name}: collapsed to {collapsed.shape[0]:,} unique key rows")
    return collapsed

def keep_only(df: pd.DataFrame, cols: List[str], name: str) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError(f"{name}: keep_only() ended up with 0 columns. Check your keep list.")
    return df[cols].copy()

def prefix_except(df: pd.DataFrame, prefix: str, keep: List[str]) -> pd.DataFrame:
    keep_set = set(keep)
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in keep_set}
    return df.rename(columns=rename)

def write_data_dictionary(df: pd.DataFrame, out_path: Path) -> None:
    d = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null": df.notna().sum().values,
        "nulls": df.isna().sum().values,
    })
    d.to_csv(out_path, index=False)

# -----------------------------------------------------------------------------
# KH03 (minimal, analysis-ready)
# -----------------------------------------------------------------------------
def build_kh03_bed_bases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build explicit bed-base fields using KH03 interim conventions:
      available G&A:        general_&_acute
      occupied  G&A:        general_&_acute_13
      available maternity:  maternity
      occupied  maternity:  maternity_15
      all available avg:    avg_available_beds
      all occupied  avg:    avg_occupied_beds
    """
    required = [
        "avg_available_beds", "avg_occupied_beds",
        "general_&_acute", "general_&_acute_13",
        "maternity", "maternity_15",
    ]
    ensure_columns(df, required, "KH03 (bed-base build)")

    out = df.copy()

    # All overnight (avg beds)
    out["avail_all_avg_beds"] = pd.to_numeric(out["avg_available_beds"], errors="coerce")
    out["occ_all_avg_beds"] = pd.to_numeric(out["avg_occupied_beds"], errors="coerce")
    out["occ_all_rate"] = safe_div(out["occ_all_avg_beds"], out["avail_all_avg_beds"])

    # General & Acute (avg beds)
    out["avail_ga_avg_beds"] = pd.to_numeric(out["general_&_acute"], errors="coerce")
    out["occ_ga_avg_beds"] = pd.to_numeric(out["general_&_acute_13"], errors="coerce")
    out["occ_ga_rate"] = safe_div(out["occ_ga_avg_beds"], out["avail_ga_avg_beds"])

    # Adult G&A proxy == G&A (no adult/paeds split at trust level in KH03)
    out["occ_adult_ga_rate"] = out["occ_ga_rate"]

    # Maternity (avg beds)
    out["avail_mat_avg_beds"] = pd.to_numeric(out["maternity"], errors="coerce")
    out["occ_mat_avg_beds"] = pd.to_numeric(out["maternity_15"], errors="coerce")

    # Acute proxy = G&A + maternity (avg beds)
    out["avail_acute_avg_beds"] = out["avail_ga_avg_beds"] + out["avail_mat_avg_beds"]
    out["occ_acute_avg_beds"] = out["occ_ga_avg_beds"] + out["occ_mat_avg_beds"]
    out["occ_acute_rate"] = safe_div(out["occ_acute_avg_beds"], out["avail_acute_avg_beds"])

    keep_cols = MERGE_KEYS + CORE_META + [
        "avail_all_avg_beds", "occ_all_avg_beds", "occ_all_rate",
        "avail_ga_avg_beds", "occ_ga_avg_beds", "occ_ga_rate",
        "avail_mat_avg_beds", "occ_mat_avg_beds",
        "avail_acute_avg_beds", "occ_acute_avg_beds", "occ_acute_rate",
        "occ_adult_ga_rate",
    ]
    out = keep_only(out, keep_cols, "KH03 (post-build keep list)")
    return out

def load_kh03_minimal(path: Path) -> pd.DataFrame:
    print(banner("LOADING: KH03 (MINIMAL)"))
    if not path.exists():
        raise FileNotFoundError(f"KH03 file not found: {path}")

    df = pd.read_csv(path)
    print(f"  File: {path}")
    print(f"  ✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    ensure_columns(df, ["org_code", "org_name", "financial_year", "fy_quarter"], "KH03")

    df["org_code"] = norm_org_code(df["org_code"])
    df["org_name"] = df["org_name"].astype(str).str.strip()
    df["financial_year"] = df["financial_year"].astype(str).str.strip()
    df["fy_quarter"] = df["fy_quarter"].astype(str).str.strip().str.upper()

    df["period"] = norm_period(df["financial_year"] + " " + df["fy_quarter"])

    df = build_kh03_bed_bases(df)
    df = assert_unique_keys(df, MERGE_KEYS, "KH03_MINIMAL")

    print(f"  ✅ Minimal KH03 columns: {df.shape[1]}" )
    print(f"  Unique trusts: {df['org_code'].nunique()}" )
    print(f"  Unique periods: {df['period'].nunique()}" )
    print(f"  Period range: {df['period'].min()} to {df['period'].max()}" )
    return df

# -----------------------------------------------------------------------------
# AE + Cancelled Ops (keep only analysis fields)
# -----------------------------------------------------------------------------
def load_ae_minimal(path: Path) -> pd.DataFrame:
    print(banner("LOADING: AE (MINIMAL)"))
    if not path.exists():
        raise FileNotFoundError(f"AE file not found: {path}")
    df = pd.read_csv(path)
    print(f"  File: {path}")
    print(f"  ✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    ensure_columns(df, ["org_code", "period"], "AE")
    df["org_code"] = norm_org_code(df["org_code"])
    df["period"] = norm_period(df["period"])

    keep_cols = MERGE_KEYS + [c for c in ["ae_type1_pct_4hr", "ae_waits_12hr_decision_to_admit", "ae_type1_attendances"] if c in df.columns]
    if len(keep_cols) <= len(MERGE_KEYS):
        # fall back: keep any numeric columns as outcomes if names differ
        numeric = [c for c in df.columns if c not in MERGE_KEYS and pd.api.types.is_numeric_dtype(df[c])]
        keep_cols = MERGE_KEYS + numeric

    df = keep_only(df, keep_cols, "AE (keep list)")
    df = assert_unique_keys(df, MERGE_KEYS, "AE_MINIMAL")
    return df

def load_cancelled_ops_minimal(path: Path) -> pd.DataFrame:
    print(banner("LOADING: Cancelled Ops (MINIMAL)"))
    if not path.exists():
        raise FileNotFoundError(f"Cancelled ops file not found: {path}")
    df = pd.read_csv(path)
    print(f"  File: {path}")
    print(f"  ✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    ensure_columns(df, ["org_code", "period"], "Cancelled_Ops")
    df["org_code"] = norm_org_code(df["org_code"])
    df["period"] = norm_period(df["period"])

    keep_cols = MERGE_KEYS + [c for c in ["canc_cancelled_ops", "canc_breaches_28d", "canc_pct_not_treated_28_days"] if c in df.columns]
    if len(keep_cols) <= len(MERGE_KEYS):
        numeric = [c for c in df.columns if c not in MERGE_KEYS and pd.api.types.is_numeric_dtype(df[c])]
        keep_cols = MERGE_KEYS + numeric

    df = keep_only(df, keep_cols, "Cancelled_Ops (keep list)")
    df = assert_unique_keys(df, MERGE_KEYS, "Cancelled_Ops_MINIMAL")
    return df

# -----------------------------------------------------------------------------
# UEC NCTR daily -> quarterly minimal
# -----------------------------------------------------------------------------
def aggregate_uec_nctr_daily_to_quarterly_minimal(path: Path) -> pd.DataFrame:
    print(banner("LOADING: UEC NCTR (DAILY -> QUARTERLY MINIMAL)"))
    if not path.exists():
        raise FileNotFoundError(f"UEC NCTR daily file not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  File: {path}")
    print(f"  ✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    ensure_columns(df, ["geo_level", "org_code", "date"], "UEC NCTR daily")

    df["geo_level"] = df["geo_level"].astype(str).str.strip().str.upper()
    df = df[df["geo_level"] == "TRUST"].copy()
    print(f"  Filtered to TRUST level: {df.shape[0]:,} rows")

    df["org_code"] = norm_org_code(df["org_code"])

    # Period derivation
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["fy_quarter"] = df["month"].map(FY_Q_BY_MONTH)

    df["fy_start_year"] = np.where(df["month"] >= 4, df["year"], df["year"] - 1)
    df["financial_year"] = (
        df["fy_start_year"].astype(int).astype(str)
        + "-"
        + (df["fy_start_year"] + 1).astype(int).astype(str).str[-2:]
    )
    df["period"] = norm_period(df["financial_year"] + " " + df["fy_quarter"])

    # Metric selection
    metric = None
    if "nctr_total" in df.columns:
        metric = "nctr_total"
    else:
        cand = [c for c in df.columns if "nctr" in c.lower()]
        if cand:
            metric = cand[0]
    if metric is None:
        raise ValueError("UEC NCTR daily: no NCTR metric column found (expected nctr_total or *nctr*).")

    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    print(banner("AGGREGATING UEC NCTR TO QUARTERLY"))
    q = (
        df.groupby(["org_code", "financial_year", "fy_quarter", "period"], as_index=False)
        .agg(
            uec_nctr_mean=(metric, "mean"),
            uec_nctr_max=(metric, "max"),
            uec_days=("date", "nunique"),
        )
    )

    q = assert_unique_keys(q, MERGE_KEYS, "UEC_NCTR_QTR_MINIMAL")
    return q

# -----------------------------------------------------------------------------
# Merge helper
# -----------------------------------------------------------------------------
def merge_left(base: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    before = base.shape[0]
    merged = base.merge(other, how="left", on=MERGE_KEYS)
    if merged.shape[0] != before:
        raise RuntimeError(
            f"{label}: merge changed row count from {before:,} to {merged.shape[0]:,}. " 
            "RHS merge keys are not unique."
        )
    return merged

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print(banner("MASTER QUARTERLY DATASET MERGE (MINIMAL MASTER)"))
    print(banner("LOADING DATASETS"))

    kh03 = load_kh03_minimal(KH03_PATH)
    ae = load_ae_minimal(AE_PATH)
    canc = load_cancelled_ops_minimal(CANCELLED_OPS_PATH)
    uec = aggregate_uec_nctr_daily_to_quarterly_minimal(UEC_NCTR_DAILY_PATH)

    print(banner("PREFIXING NON-KEY COLUMNS"))
    preserve = MERGE_KEYS + CORE_META
    kh03_p = prefix_except(kh03, "kh03_", preserve)
    ae_p = prefix_except(ae, "ae_", MERGE_KEYS)       # AE has no meta in minimal form
    canc_p = prefix_except(canc, "canc_", MERGE_KEYS) # Cancelled has no meta in minimal form
    uec_p = prefix_except(uec, "uec_", MERGE_KEYS)    # UEC has no meta in minimal form

    print(banner("MERGING"))
    master = kh03_p.copy()
    print(f"Using KH03 as base dataset...\n  Base: {master.shape[0]:,} rows" )

    master = merge_left(master, ae_p, "AE")
    master = merge_left(master, canc_p, "Cancelled_Ops")
    master = merge_left(master, uec_p, "UEC_NCTR" )

    # Coverage (numeric-only, so metadata doesn't inflate matches)
    def has_any_numeric(prefix: str) -> pd.Series:
        cols = [c for c in master.columns if c.startswith(prefix)]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(master[c])]
        return master[num_cols].notna().any(axis=1) if num_cols else pd.Series([False] * len(master))

    has_ae = has_any_numeric("ae_")
    has_canc = has_any_numeric("canc_")
    has_uec = has_any_numeric("uec_")
    complete = has_ae & has_canc & has_uec

    print(banner("SUMMARY"))
    print(f"Rows: {master.shape[0]:,}" )
    print(f"Columns: {master.shape[1]}" )
    print(f"Unique trusts: {master['org_code'].nunique()}" )
    print(f"Unique periods: {master['period'].nunique()}" )
    print(f"Period range: {master['period'].min()} to {master['period'].max()}" )
    print("\nCoverage (share of KH03 rows with at least one joined numeric value):")
    print(f"  AE: {has_ae.mean()*100:.1f}%" )
    print(f"  Cancelled ops: {has_canc.mean()*100:.1f}%" )
    print(f"  UEC NCTR: {has_uec.mean()*100:.1f}%" )
    print(f"\nComplete rows (AE + Cancelled + UEC): {int(complete.sum()):,} ({complete.mean()*100:.1f}%)" )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(banner("SAVING OUTPUT"))
    master.to_csv(OUT_MASTER, index=False)
    print(f"✅ Saved: {OUT_MASTER} ({master.shape[0]:,} rows × {master.shape[1]} cols)" )

    summary_lines = [
        "MASTER QUARTERLY DATASET MERGE - MINIMAL MASTER (A)",
        "",
        f"Master dataset: {OUT_MASTER}",
        f"Rows: {master.shape[0]:,}",
        f"Columns: {master.shape[1]}",
        f"Unique trusts: {master['org_code'].nunique()}",
        f"Unique periods: {master['period'].nunique()}",
        f"Period range: {master['period'].min()} to {master['period'].max()}",
        "",
        "Coverage (share of KH03 rows with at least one joined numeric value):",
        f"AE: {has_ae.mean()*100:.1f}%",
        f"Cancelled ops: {has_canc.mean()*100:.1f}%",
        f"UEC NCTR: {has_uec.mean()*100:.1f}%",
        "",
        f"Complete rows (AE + Cancelled + UEC): {int(complete.sum()):,} ({complete.mean()*100:.1f}%)",
        "",
        "KH03 bed-base fields are average-beds based and computed (no raw % occupied columns retained)."
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")
    write_data_dictionary(master, OUT_DICT)
    print(f"✅ Summary: {OUT_SUMMARY}" )
    print(f"✅ Data dictionary: {OUT_DICT}" )
    print(banner("✅ COMPLETE"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ ERROR:", e)
        sys.exit(1)
