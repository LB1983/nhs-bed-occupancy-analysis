#!/usr/bin/env python3
"""
05b-regressions_DEBUGGED.py

IMPROVEMENTS FROM v8:
1. Flexible column name detection (handles prefixed columns from merge)
2. Derives missing outcome metrics (canc_per_100_beds, nctr_per_100_beds)
3. Makes proximity features OPTIONAL (script works without them)
4. Better error messages and diagnostics
5. Validates data availability before each regression
6. Handles both period formats: "2018-19 Q1" and "2018-Q2"

RUN WITH:
    python 04-analysis/05b-regressions_DEBUGGED.py

If it fails, run with --diagnose flag:
    python 04-analysis/05b-regressions_DEBUGGED.py --diagnose
"""

from __future__ import annotations

import os
import re
import sys
import inspect
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from linearmodels.panel import PanelOLS
except Exception:
    print("❌ Could not import linearmodels. Install with:")
    print("   pip install linearmodels --break-system-packages")
    raise

# =============================================================================
# CONFIGURATION
# =============================================================================

_BASE = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis"
MASTER_PATH = os.path.join(_BASE, "03-data-final", "master-quarterly-trust.csv")
PROX_PATH = os.path.join(_BASE, "02-data-interim", "geo", "trust_proximity_features.csv")
OUT_DIR = os.path.join(_BASE, "05-outputs")

# Proximity features (OPTIONAL - script works without them)
PROX_BASE = ["nearest_trust_km", "trusts_within_10km", "trusts_within_25km"]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def zscore(s: pd.Series) -> pd.Series:
    x = safe_num(s).astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (x - mu) / sd

def drop_all_na_or_constant(df: pd.DataFrame, cols: List[str]) -> Tuple[List[str], List[str]]:
    kept, dropped = [], []
    for c in cols:
        if c not in df.columns:
            dropped.append(c)
            continue
        v = df[c]
        if v.notna().sum() == 0:
            dropped.append(c)
            continue
        if v.dropna().nunique() <= 1:
            dropped.append(c)
            continue
        kept.append(c)
    return kept, dropped

def parse_fy_quarter_to_period_end(period: str) -> pd.Timestamp:
    """
    Handles both formats:
    - "2018-19 Q1" (A&E/Cancelled Ops format)
    - "2018-Q2" (KH03 format)
    """
    s = str(period).strip()
    
    # Try format: "2018-19 Q1"
    m = re.match(r"(\d{4})-\d{2}\s+Q([1-4])", s)
    if m:
        start_year = int(m.group(1))
        q = int(m.group(2))
    else:
        # Try format: "2018-Q2" or "2018 Q2"
        m2 = re.match(r"(\d{4})[-\s]*Q([1-4])", s, re.IGNORECASE)
        if m2:
            start_year = int(m2.group(1))
            q = int(m2.group(2))
        else:
            raise ValueError(f"Unrecognised period format: {period!r}")
    
    # Q1 = Apr-Jun, Q2 = Jul-Sep, Q3 = Oct-Dec, Q4 = Jan-Mar
    if q == 1:
        return pd.Timestamp(start_year, 6, 30)
    if q == 2:
        return pd.Timestamp(start_year, 9, 30)
    if q == 3:
        return pd.Timestamp(start_year, 12, 31)
    return pd.Timestamp(start_year + 1, 3, 31)

def coerce_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize merge keys and create period_end for panel models."""
    if "org_code" not in df.columns:
        raise KeyError("Expected org_code in master.")
    if "period" not in df.columns:
        raise KeyError("Expected period in master.")
    
    df["org_code"] = df["org_code"].astype(str).str.strip()
    df["period"] = df["period"].astype(str).str.strip()
    
    # Create period_end for PanelOLS time index
    df["period_end"] = df["period"].apply(parse_fy_quarter_to_period_end)
    
    return df

# =============================================================================
# COLUMN DETECTION
# =============================================================================

def find_column(df: pd.DataFrame, candidates: List[str], desc: str = "") -> Optional[str]:
    """
    Find first matching column from candidates list.
    Tries exact match first, then case-insensitive.
    """
    # Exact match
    for c in candidates:
        if c in df.columns:
            return c
    
    # Case-insensitive match
    lower_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    
    return None

def detect_occupancy_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect available occupancy rate columns.
    Returns dict like: {"all": "kh03_occupancy_rate_total", ...}
    """
    occ_map = {}
    
    # Total/All beds
    all_candidates = [
        "kh03_occupancy_rate_total",
        "kh03_occ_all_rate",
        "occupancy_rate_total",
        "occ_rate_total"
    ]
    col = find_column(df, all_candidates, "total occupancy")
    if col:
        occ_map["all"] = col
    
    # General & Acute
    ga_candidates = [
        "kh03_occupancy_rate_general_acute",
        "kh03_occ_ga_rate",
        "occupancy_rate_ga",
        "occ_rate_ga"
    ]
    col = find_column(df, ga_candidates, "G&A occupancy")
    if col:
        occ_map["ga"] = col
    
    return occ_map

def detect_capacity_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect available bed capacity columns.
    Returns dict like: {"all": "kh03_available_total", ...}
    """
    cap_map = {}
    
    # Total beds
    all_candidates = [
        "kh03_available_total",
        "kh03_avail_all_avg_beds",
        "available_total",
        "beds_total"
    ]
    col = find_column(df, all_candidates, "total beds")
    if col:
        cap_map["all"] = col
    
    # G&A beds
    ga_candidates = [
        "kh03_available_general_acute",
        "kh03_avail_ga_avg_beds",
        "available_ga",
        "beds_ga"
    ]
    col = find_column(df, ga_candidates, "G&A beds")
    if col:
        cap_map["ga"] = col
    
    return cap_map

def detect_outcome_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect available outcome columns.
    Returns dict like: {"ae_12h_per_1k": "ae_12h_per_1k_att", ...}
    """
    outcomes = {}
    
    # A&E 4-hour performance
    ae_4hr_candidates = ["ae_type1_pct_4hr", "ae_4hr_pct", "ae_pct_4hr"]
    col = find_column(df, ae_4hr_candidates)
    if col:
        outcomes["ae_4hr_pct"] = col
    
    # A&E 12-hour waits (will be derived if needed)
    ae_12h_candidates = ["ae_12h_per_1k_att", "ae_12h_per_1k"]
    col = find_column(df, ae_12h_candidates)
    if col:
        outcomes["ae_12h_per_1k"] = col
    
    # Cancelled ops % breach
    canc_pct_candidates = ["canc_pct_not_treated_28_days", "canc_pct28"]
    col = find_column(df, canc_pct_candidates)
    if col:
        outcomes["canc_pct28"] = col
    
    # Cancelled ops per 100 beds (will be derived if needed)
    canc_rate_candidates = ["canc_per_100_beds", "cancelled_ops_per_100_beds"]
    col = find_column(df, canc_rate_candidates)
    if col:
        outcomes["canc_per_100"] = col
    
    # UEC NCTR per 100 beds (will be derived if needed)
    nctr_candidates = ["nctr_max_per_100_beds", "uec_nctr_max_per_100_beds"]
    col = find_column(df, nctr_candidates)
    if col:
        outcomes["nctr_per_100"] = col
    
    return outcomes

# =============================================================================
# DERIVED OUTCOMES
# =============================================================================

def derive_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived outcome metrics if they don't exist.
    """
    df = df.copy()
    
    # 1. A&E 12h waits per 1,000 type 1 attendances
    if "ae_12h_per_1k_att" not in df.columns:
        waits_col = find_column(df, ["ae_waits_12hr_decision_to_admit", "ae_12hr_waits"])
        att_col = find_column(df, ["ae_type1_attendances", "ae_attendances"])
        
        if waits_col and att_col:
            num = safe_num(df[waits_col])
            den = safe_num(df[att_col])
            df["ae_12h_per_1k_att"] = np.where(den > 0, 1000.0 * num / den, np.nan)
            print(f"  ✅ Derived: ae_12h_per_1k_att from {waits_col} / {att_col}")
    
    # 2. Cancelled ops per 100 beds
    if "canc_per_100_beds" not in df.columns:
        canc_col = find_column(df, ["canc_cancelled_ops", "cancelled_ops"])
        beds_col = find_column(df, ["kh03_available_total", "kh03_avail_all_avg_beds"])
        
        if canc_col and beds_col:
            num = safe_num(df[canc_col])
            den = safe_num(df[beds_col])
            df["canc_per_100_beds"] = np.where(den > 0, 100.0 * num / den, np.nan)
            print(f"  ✅ Derived: canc_per_100_beds from {canc_col} / {beds_col}")
    
    # 3. UEC NCTR per 100 beds
    if "nctr_max_per_100_beds" not in df.columns:
        nctr_col = find_column(df, ["uec_nctr_peak_daily", "uec_nctr_max_daily", "nctr_peak"])
        beds_col = find_column(df, ["kh03_available_total", "kh03_avail_all_avg_beds"])
        
        if nctr_col and beds_col:
            num = safe_num(df[nctr_col])
            den = safe_num(df[beds_col])
            df["nctr_max_per_100_beds"] = np.where(den > 0, 100.0 * num / den, np.nan)
            print(f"  ✅ Derived: nctr_max_per_100_beds from {nctr_col} / {beds_col}")
    
    return df

# =============================================================================
# PROXIMITY DATA (OPTIONAL)
# =============================================================================

def load_proximity() -> Optional[pd.DataFrame]:
    """
    Load proximity features if file exists.
    Returns None if file not found (script continues without proximity).
    """
    if not os.path.exists(PROX_PATH):
        print(f"\n⚠️  Proximity file not found: {PROX_PATH}")
        print("   Continuing without proximity features...")
        return None
    
    try:
        prox = pd.read_csv(PROX_PATH)
        
        # Find org_code column
        key = find_column(prox, ["org_code", "ods_code", "ods  code", "ods"])
        if key is None:
            print(f"⚠️  Could not find org_code column in proximity file")
            return None
        
        prox = prox.rename(columns={key: "org_code"})
        prox["org_code"] = prox["org_code"].astype(str).str.strip()
        
        # Find proximity columns
        keep_cols = ["org_code"]
        for base in PROX_BASE:
            col = find_column(prox, [base])
            if col:
                if col != base:
                    prox = prox.rename(columns={col: base})
                prox[base] = safe_num(prox[base])
                keep_cols.append(base)
        
        if len(keep_cols) == 1:
            print(f"⚠️  No proximity columns found in file")
            return None
        
        print(f"✅ Loaded proximity features: {', '.join(keep_cols[1:])}")
        return prox[keep_cols]
        
    except Exception as e:
        print(f"⚠️  Error loading proximity file: {e}")
        return None

# =============================================================================
# PANEL REGRESSION
# =============================================================================

def build_design(df: pd.DataFrame, occ_col: str, cap_col: str, y_col: str,
                 has_proximity: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build regression design with standardized variables and interactions.
    """
    work = df.copy()
    
    # Standardize main variables
    work[occ_col] = safe_num(work[occ_col])
    work[cap_col] = safe_num(work[cap_col])
    work[y_col] = safe_num(work[y_col])
    
    # Z-scores
    work["occ_z"] = zscore(work[occ_col])
    work["cap_z"] = zscore(work[cap_col])
    
    # Occupancy × Capacity interaction
    work["occ_x_cap_z"] = work["occ_z"] * work["cap_z"]
    
    # Base regressors
    X = ["occ_z", "occ_x_cap_z"]
    
    # Add proximity interactions if available
    if has_proximity:
        for base in PROX_BASE:
            if base in work.columns:
                work[f"{base}_z"] = zscore(work[base])
                work[f"occ_x_{base}_z"] = work["occ_z"] * work[f"{base}_z"]
                X.append(f"occ_x_{base}_z")
    
    # Set panel index
    panel = work.set_index(["org_code", "period_end"]).sort_index()
    
    return panel, X

def fit_panel(panel: pd.DataFrame, y_col: str, X_cols: List[str]) -> Tuple[object, List[str]]:
    """
    Fit panel regression with entity and time fixed effects.
    """
    # Prepare data
    need = [y_col] + X_cols
    work = panel[need].copy()
    
    # Drop rows with missing outcome
    sample = work.dropna(subset=[y_col])
    
    # Drop constant or all-NA regressors
    kept_X, dropped_X = drop_all_na_or_constant(sample, X_cols)
    
    # Final sample
    final = panel[[y_col] + kept_X].dropna()
    y = final[y_col]
    Xmat = final[kept_X].copy()

    # Fit model
    model = PanelOLS(y, Xmat, entity_effects=True, time_effects=True, check_rank=False)
    
    # Clustered standard errors
    fit_sig = inspect.signature(model.fit).parameters
    kwargs = {}
    if "cov_type" in fit_sig:
        kwargs["cov_type"] = "clustered"
    if "cluster_entity" in fit_sig:
        kwargs["cluster_entity"] = True
    elif "clusters" in fit_sig:
        ent = final.index.get_level_values(0)
        kwargs["clusters"] = pd.Series(ent, index=final.index)
    
    res = model.fit(**kwargs)
    
    # Track dropped regressors
    params = set(res.params.index.astype(str).tolist())
    expected = set(Xmat.columns.astype(str).tolist())
    dropped_by_est = sorted(list(expected - params))
    dropped = sorted(list(set(dropped_X + dropped_by_est)))
    
    return res, dropped

def extract_stats(res, var: str) -> Tuple[float, float, float]:
    """Extract coefficient, SE, and p-value for a variable."""
    coef = float(res.params.get(var, np.nan)) if hasattr(res, "params") else np.nan
    se = float(res.std_errors.get(var, np.nan)) if hasattr(res, "std_errors") else np.nan
    p = float(res.pvalues.get(var, np.nan)) if hasattr(res, "pvalues") else np.nan
    return coef, se, p

# =============================================================================
# DIAGNOSTIC MODE
# =============================================================================

def run_diagnostics(df: pd.DataFrame) -> None:
    """
    Print detailed diagnostics about available data.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC MODE")
    print("="*80)
    
    print(f"\nDataset shape: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Unique trusts: {df['org_code'].nunique()}")
    print(f"Unique periods: {df['period'].nunique()}")
    print(f"Period range: {df['period'].min()} to {df['period'].max()}")
    
    print("\n" + "-"*80)
    print("OCCUPANCY COLUMNS DETECTED:")
    print("-"*80)
    occ_cols = detect_occupancy_columns(df)
    if occ_cols:
        for key, col in occ_cols.items():
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {key:15s}: {col:40s} ({non_null:,} / {len(df):,} = {pct:.1f}%)")
    else:
        print("  ❌ No occupancy columns found!")
    
    print("\n" + "-"*80)
    print("CAPACITY COLUMNS DETECTED:")
    print("-"*80)
    cap_cols = detect_capacity_columns(df)
    if cap_cols:
        for key, col in cap_cols.items():
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {key:15s}: {col:40s} ({non_null:,} / {len(df):,} = {pct:.1f}%)")
    else:
        print("  ❌ No capacity columns found!")
    
    print("\n" + "-"*80)
    print("OUTCOME COLUMNS DETECTED:")
    print("-"*80)
    outcome_cols = detect_outcome_columns(df)
    if outcome_cols:
        for key, col in outcome_cols.items():
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {key:20s}: {col:40s} ({non_null:,} / {len(df):,} = {pct:.1f}%)")
    else:
        print("  ⚠️  No outcome columns detected (will try to derive)")
    
    print("\n" + "-"*80)
    print("SAMPLE OF COLUMNS (first 30):")
    print("-"*80)
    for i, col in enumerate(df.columns[:30]):
        print(f"  {i+1:2d}. {col}")
    if len(df.columns) > 30:
        print(f"  ... and {len(df.columns) - 30} more columns")
    
    print("\n" + "="*80)
    print("Run without --diagnose flag to proceed with regressions")
    print("="*80)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_dir(OUT_DIR)
    
    print("="*80)
    print("PANEL REGRESSIONS - DEBUGGED VERSION")
    print("="*80)
    
    # Load master dataset
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(f"Missing master file: {MASTER_PATH}")
    
    print(f"\nLoading: {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)
    df = coerce_keys(df)
    print(f"  ✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
    
    # Diagnostic mode?
    if "--diagnose" in sys.argv:
        run_diagnostics(df)
        return
    
    # Derive outcomes
    print("\nDeriving outcome metrics...")
    df = derive_outcomes(df)
    
    # Load proximity (optional)
    print("\nLoading proximity features...")
    prox = load_proximity()
    has_proximity = False
    if prox is not None:
        df = df.merge(prox, on="org_code", how="left")
        matched = df["nearest_trust_km"].notna().mean() if "nearest_trust_km" in df.columns else 0.0
        print(f"  Proximity merge: {matched:.1%} matched")
        has_proximity = True
    
    # Detect available columns
    print("\nDetecting available columns...")
    occ_cols = detect_occupancy_columns(df)
    cap_cols = detect_capacity_columns(df)
    outcome_cols = detect_outcome_columns(df)
    
    if not occ_cols:
        raise ValueError("No occupancy columns found! Check column names.")
    if not cap_cols:
        raise ValueError("No capacity columns found! Check column names.")
    if not outcome_cols:
        print("⚠️  No outcomes detected - check data availability")
    
    print(f"  Found {len(occ_cols)} occupancy measures")
    print(f"  Found {len(cap_cols)} capacity measures")
    print(f"  Found {len(outcome_cols)} outcomes")
    
    # Save sample
    df.head(2000).to_csv(os.path.join(OUT_DIR, "sample_dataset.csv"), index=False)
    print(f"\n✅ Saved sample dataset to {OUT_DIR}/sample_dataset.csv")
    
    # Run regressions
    print("\n" + "="*80)
    print("RUNNING REGRESSIONS")
    print("="*80)
    
    results = []
    
    for bed_key in occ_cols.keys():
        if bed_key not in cap_cols:
            continue
        
        occ_col = occ_cols[bed_key]
        cap_col = cap_cols[bed_key]
        
        print(f"\n--- Bed measure: {bed_key} ---")
        print(f"    Occupancy: {occ_col}")
        print(f"    Capacity: {cap_col}")
        
        for outcome_key, y_col in outcome_cols.items():
            print(f"  Outcome: {outcome_key} ({y_col})")
            
            try:
                panel, X = build_design(df, occ_col, cap_col, y_col, has_proximity)
                res, dropped = fit_panel(panel, y_col=y_col, X_cols=X)
                
                rec = {
                    "bed_measure": bed_key,
                    "outcome": outcome_key,
                    "occ_col": occ_col,
                    "cap_col": cap_col,
                    "y_col": y_col,
                    "nobs": int(res.nobs),
                    "r_squared": float(res.rsquared),
                    "dropped_regressors": ";".join(dropped) if dropped else "",
                }
                
                # Extract coefficients
                for var in X:
                    coef, se, p = extract_stats(res, var)
                    rec[f"coef_{var}"] = coef
                    rec[f"se_{var}"] = se
                    rec[f"p_{var}"] = p
                
                results.append(rec)
                print(f"    ✅ N={res.nobs}, R²={res.rsquared:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
    
    # Save results
    if results:
        out_file = os.path.join(OUT_DIR, "regression_results.csv")
        pd.DataFrame(results).to_csv(out_file, index=False)
        print(f"\n✅ Saved {len(results)} regression results to: {out_file}")
    else:
        print(f"\n⚠️  No regressions completed successfully")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
