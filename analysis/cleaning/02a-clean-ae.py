#!/usr/bin/env python
"""
A&E Attendances and Emergency Admissions Data Cleaning Script
=============================================================

Author: Lauren Gavaghan (The Bevan Briefing)
Date: February 2026
Purpose: Clean NHS England monthly A&E performance data for bed occupancy analysis

This script:
1. Reads monthly Excel publications (various naming formats)
2. Extracts trust-level Type 1 A&E performance metrics
3. Aggregates monthly data to quarterly level
4. Produces clean analysis-ready dataset

Key metrics:
- Type 1 A&E attendances (major A&E departments)
- Emergency admissions via Type 1 A&E
- % Type 1 seen within 4 hours (national standard)
- >12 hour waits from decision to admit (exit block indicator)

File formats handled:
- YYYY-MM_ae-monthly.xls (e.g., 2018-04_ae-monthly.xls)
- Month-YYYY-AE.xls (e.g., April-2018-AE.xls)
- Auto-detects header row and sheet name

Outputs:
- 02-data-interim/ae-performance/ae-monthly-trust.csv (monthly data)
- 02-data-interim/ae-performance/ae-quarterly-trust.csv (aggregated to quarters)

Run from repo root:
  python 04-analysis/01-data-cleaning/02a-clean-ae.py
  
Or test mode:
  python 04-analysis/01-data-cleaning/02a-clean-ae.py --test
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from calendar import month_name

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = Path("01-data-raw/ae-performance")
OUTPUT_DIR = Path("02-data-interim/ae-performance")

# File patterns
# Pattern 1: YYYY-MM_ae-monthly.xls
PATTERN1 = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})_ae-monthly\.xls$", re.IGNORECASE)
# Pattern 2: Month-YYYY-AE.xls
PATTERN2 = re.compile(r"^(?P<month_name>[A-Za-z]+)-(?P<year>\d{4})-AE\.xls$", re.IGNORECASE)

# Month name to number mapping
MONTH_NAMES = {name.lower(): idx for idx, name in enumerate(month_name) if name}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_num(x):
    """Convert value to numeric, handling NHS data quirks."""
    if pd.isna(x):
        return pd.NA
    if isinstance(x, str):
        s = x.strip()
        if s in {"", "-", "*", "~"}:
            return pd.NA
        s = s.replace(",", "")
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(x, errors="coerce")


def parse_filename(filename: str) -> dict:
    """
    Extract year and month from filename.
    
    Handles multiple naming patterns:
    - 2018-04_ae-monthly.xls
    - April-2018-AE.xls
    
    Returns:
        dict with year, month, year_month, financial_year, fy_quarter, period
    """
    # Try pattern 1
    m = PATTERN1.match(filename)
    if m:
        year = int(m.group("year"))
        month = int(m.group("month"))
    else:
        # Try pattern 2
        m = PATTERN2.match(filename)
        if m:
            year = int(m.group("year"))
            month_name_str = m.group("month_name").lower()
            month = MONTH_NAMES.get(month_name_str)
            if month is None:
                raise ValueError(f"Unknown month name in filename: {filename}")
        else:
            raise ValueError(f"Filename doesn't match any known pattern: {filename}")
    
    # Financial year (April-March)
    if month >= 4:
        fy = f"{year}-{str(year+1)[2:]}"
    else:
        fy = f"{year-1}-{str(year)[2:]}"
    
    # FY Quarter
    if month in (4, 5, 6):
        fy_quarter = "Q1"
    elif month in (7, 8, 9):
        fy_quarter = "Q2"
    elif month in (10, 11, 12):
        fy_quarter = "Q3"
    else:  # 1, 2, 3
        fy_quarter = "Q4"
    
    return {
        "year": year,
        "month": month,
        "year_month": f"{year}-{month:02d}",
        "financial_year": fy,
        "fy_quarter": fy_quarter,
        "period": f"{fy} {fy_quarter}"
    }


def find_header_row(df: pd.DataFrame, max_rows: int = 50) -> int:
    """
    Find header row by looking for 'Code' and 'Type' in same row.
    """
    for i in range(min(max_rows, len(df))):
        row_text = " ".join(df.iloc[i].astype(str).tolist()).lower()
        if "code" in row_text and "type" in row_text:
            return i
    raise ValueError(f"Could not find header row in first {max_rows} rows")


def find_provider_sheet(xl_file) -> str:
    """
    Find sheet containing provider-level data.
    """
    sheet_names = xl_file.sheet_names
    
    # Try common names
    for pattern in ["provider level data", "provider level", "provider", "trust level"]:
        for sheet in sheet_names:
            if pattern in sheet.lower():
                return sheet
    
    # Fallback: use first non-metadata sheet
    for sheet in sheet_names:
        if sheet.lower() not in ["cover", "contents", "notes", "guidance", "system", "stp", "mapping", "footprint"]:
            return sheet
    
    raise ValueError(f"Could not find provider sheet. Available: {sheet_names}")


# =============================================================================
# MAIN READER FUNCTION
# =============================================================================

def read_ae_monthly(path: Path) -> pd.DataFrame:
    """
    Read monthly A&E Excel file and extract Type 1 metrics.
    
    Returns:
        DataFrame with trust-level Type 1 A&E data
    """
    # Parse filename
    file_info = parse_filename(path.name)
    
    # Open Excel file
    xl_file = pd.ExcelFile(path)
    sheet_name = find_provider_sheet(xl_file)
    
    # Read raw data
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    
    # Find header row
    header_row = find_header_row(raw)
    
    # Extract data
    cols = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1:].copy()
    data.columns = cols
    
    # Remove duplicate columns (pandas adds .1, .2 suffixes)
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated(keep='first')]
    
    # Standardize column names
    # Column structure (based on inspection):
    # - Code, Region, Name
    # - Type 1/2/3 Attendances (cols 4-7)
    # - Type 1/2/3 Attendances <4hr (cols 8-11)
    # - Type 1/2/3 Attendances >4hr (cols 12-15)
    # - % in 4 hours (all, type 1, type 2, type 3) (cols 16-19)
    # - Emergency Admissions via Type 1/2/3 (cols 20-23)
    # - Total Emergency Admissions (cols 24-25)
    # - >4 hours from decision to admit (col 26)
    # - >12 hours from decision to admit (col 27)
    
    rename_map = {}
    for col in data.columns:
        c_str = str(col).strip()
        c_lower = c_str.lower()
        
        # Org identifiers
        if c_lower == "code":
            rename_map[col] = "org_code"
        elif c_lower == "name":
            rename_map[col] = "org_name"
        elif c_lower == "region":
            rename_map[col] = "region"
        
        # Type 1 metrics (checking exact column structure from inspection)
        elif "type 1" in c_lower and "major a&e" in c_lower:
            # This appears 3 times - attendances, <4hr, >4hr
            # We need to distinguish by position or check for other keywords
            if rename_map.get(col) is None:  # First occurrence
                rename_map[col] = "type1_attendances"
        
        # Percentage in 4 hours or less (type 1)
        elif "percentage" in c_lower and "4 hour" in c_lower and "type 1" in c_lower:
            rename_map[col] = "type1_pct_4hr"
        
        # Emergency admissions via Type 1
        elif "emergency" in c_lower and "admit" in c_lower and "type 1" in c_lower:
            rename_map[col] = "type1_admissions"
        
        # >12 hour waits from decision to admit
        elif ">12" in c_str or "12 hour" in c_lower:
            if "decision to admit" in c_lower:
                rename_map[col] = "waits_12hr_decision_to_admit"
    
    data = data.rename(columns=rename_map)
    
    # Validate minimum required columns
    if "org_code" not in data.columns:
        raise ValueError(
            f"{path.name}: Could not find org_code column. "
            f"Found columns: {list(data.columns)[:15]}"
        )
    
    # Clean org codes
    data["org_code"] = data["org_code"].astype(str).str.strip()
    
    # Remove England totals and blank rows
    data = data[data["org_code"].notna()]
    data = data[data["org_code"] != ""]
    data = data[data["org_code"] != "-"]
    data = data[data["org_code"] != "nan"]
    
    # Convert metrics to numeric
    for col in data.columns:
        col_str = str(col)  # Convert to string first
        if col_str.startswith("type1_") or col_str.startswith("waits_"):
            data[col] = data[col].apply(to_num)
    
    # Select output columns
    output_cols = ["org_code"]
    
    # Add optional columns if they exist
    if "org_name" in data.columns:
        output_cols.append("org_name")
    if "region" in data.columns:
        output_cols.append("region")
    
    # Add metrics
    metric_cols = [
        "type1_attendances", 
        "type1_pct_4hr", 
        "type1_admissions",
        "waits_12hr_decision_to_admit"
    ]
    for col in metric_cols:
        if col in data.columns:
            output_cols.append(col)
    
    out = data[output_cols].copy()
    
    # Add file metadata
    out["year"] = file_info["year"]
    out["month"] = file_info["month"]
    out["year_month"] = file_info["year_month"]
    out["financial_year"] = file_info["financial_year"]
    out["fy_quarter"] = file_info["fy_quarter"]
    out["period"] = file_info["period"]
    out["source_file"] = path.name
    
    # Filter to trusts only (R codes)
    out = out[out["org_code"].str.match(r"^R[A-Z0-9]{2,4}$", na=False)].copy()
    
    if out.empty:
        raise ValueError(f"{path.name}: No trust records after filtering")
    
    return out


# =============================================================================
# AGGREGATION TO QUARTERLY
# =============================================================================

def aggregate_to_quarterly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly data to quarterly level.
    
    For counts (attendances, admissions, waits): sum
    For percentages: weighted average by attendances
    """
    # Group by trust + FY quarter
    group_cols = ["org_code", "financial_year", "fy_quarter", "period"]
    
    # Add org_name and region if available
    if "org_name" in monthly_df.columns:
        # Take first non-null value for each group
        name_map = monthly_df.groupby("org_code")["org_name"].first()
    
    if "region" in monthly_df.columns:
        region_map = monthly_df.groupby("org_code")["region"].first()
    
    # Aggregate metrics
    agg_dict = {}
    
    # Counts: sum
    for col in ["type1_attendances", "type1_admissions", "waits_12hr_decision_to_admit"]:
        if col in monthly_df.columns:
            agg_dict[col] = "sum"
    
    # Percentage: we need weighted average
    # Store attendances for weighting
    if "type1_pct_4hr" in monthly_df.columns and "type1_attendances" in monthly_df.columns:
        # Calculate weighted percentage manually after grouping
        pass
    
    # Basic aggregation
    quarterly = monthly_df.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # Calculate weighted average for percentage
    if "type1_pct_4hr" in monthly_df.columns and "type1_attendances" in monthly_df.columns:
        # For each quarter, calculate: sum(pct * attendances) / sum(attendances)
        # Only for rows where both values are non-null
        valid_mask = (
            monthly_df["type1_pct_4hr"].notna() & 
            monthly_df["type1_attendances"].notna() &
            (monthly_df["type1_attendances"] > 0)
        )
        
        monthly_df["weighted_4hr"] = pd.NA
        monthly_df.loc[valid_mask, "weighted_4hr"] = (
            monthly_df.loc[valid_mask, "type1_pct_4hr"] * 
            monthly_df.loc[valid_mask, "type1_attendances"] / 100
        )
        
        weighted_sum = (
            monthly_df.groupby(group_cols)["weighted_4hr"]
            .sum()
            .reset_index()
        )
        
        quarterly = quarterly.merge(
            weighted_sum,
            on=group_cols,
            how="left"
        )
        
        # Calculate percentage, avoiding division by zero
        quarterly["type1_pct_4hr"] = pd.NA
        valid_q = (quarterly["type1_attendances"] > 0) & (quarterly["weighted_4hr"].notna())
        quarterly.loc[valid_q, "type1_pct_4hr"] = (
            quarterly.loc[valid_q, "weighted_4hr"] / 
            quarterly.loc[valid_q, "type1_attendances"] * 100
        )
        quarterly = quarterly.drop(columns=["weighted_4hr"])
    
    # Add org_name and region back
    if "org_name" in monthly_df.columns:
        quarterly["org_name"] = quarterly["org_code"].map(name_map)
    
    if "region" in monthly_df.columns:
        quarterly["region"] = quarterly["org_code"].map(region_map)
    
    # Reorder columns
    col_order = ["org_code"]
    if "org_name" in quarterly.columns:
        col_order.append("org_name")
    if "region" in quarterly.columns:
        col_order.append("region")
    
    col_order.extend(["financial_year", "fy_quarter", "period"])
    
    # Add metrics
    for col in quarterly.columns:
        col_str = str(col)  # Convert to string first
        if col_str.startswith("type1_") or col_str.startswith("waits_"):
            if col not in col_order:
                col_order.append(col)
    
    quarterly = quarterly[col_order]
    
    return quarterly


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main data cleaning pipeline.
    """
    print("="*80)
    print("A&E PERFORMANCE DATA CLEANING PIPELINE")
    print("="*80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    frames = []
    skipped = []
    
    # Find all .xls files
    print("Scanning for A&E files...")
    xls_files = sorted(INPUT_DIR.glob("*.xls"))
    matched_files = []
    
    for f in xls_files:
        if PATTERN1.match(f.name) or PATTERN2.match(f.name):
            matched_files.append(f)
    
    print(f"  Found {len(matched_files)} files matching patterns")
    print()
    
    # Process each file
    for path in matched_files:
        print(f"Processing: {path.name}")
        try:
            df = read_ae_monthly(path)
            frames.append(df)
            print(f"  ✅ Loaded {len(df):,} trust records")
        except PermissionError as e:
            print(f"  ⚠️  SKIP (file locked): {e}")
            skipped.append((path.name, "File locked"))
        except AssertionError as e:
            print(f"  ❌ SKIP (corrupted Excel file)")
            skipped.append((path.name, "Corrupted Excel file - xlrd cannot parse"))
        except Exception as e:
            print(f"  ❌ SKIP (error): {e}")
            skipped.append((path.name, str(e)))
    
    # Validate we got data
    if not frames:
        raise RuntimeError(
            f"No A&E files successfully processed.\n"
            f"Check input directory: {INPUT_DIR}\n"
            f"Expected files like: 2018-04_ae-monthly.xls or April-2018-AE.xls"
        )
    
    # Combine monthly data
    print("\n" + "="*80)
    print("COMBINING MONTHLY DATA")
    print("="*80)
    
    monthly = pd.concat(frames, ignore_index=True)
    print(f"Combined {len(frames)} files into {len(monthly):,} monthly records")
    
    # Save monthly data
    print("\n" + "="*80)
    print("SAVING MONTHLY OUTPUT")
    print("="*80)
    
    monthly_file = OUTPUT_DIR / "ae-monthly-trust.csv"
    monthly.to_csv(monthly_file, index=False)
    print(f"✅ Saved: {monthly_file}")
    
    # Aggregate to quarterly
    print("\n" + "="*80)
    print("AGGREGATING TO QUARTERLY")
    print("="*80)
    
    quarterly = aggregate_to_quarterly(monthly)
    print(f"Aggregated to {len(quarterly):,} quarterly records")
    
    # Save quarterly data
    quarterly_file = OUTPUT_DIR / "ae-quarterly-trust.csv"
    quarterly.to_csv(quarterly_file, index=False)
    print(f"✅ Saved: {quarterly_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nMonthly data:")
    print(f"  Records: {len(monthly):,}")
    print(f"  Unique trusts: {monthly['org_code'].nunique()}")
    print(f"  Date range: {monthly['year_month'].min()} to {monthly['year_month'].max()}")
    print(f"  Months covered: {monthly['year_month'].nunique()}")
    
    print(f"\nQuarterly data:")
    print(f"  Records: {len(quarterly):,}")
    print(f"  Unique trusts: {quarterly['org_code'].nunique()}")
    print(f"  FY range: {quarterly['financial_year'].min()} to {quarterly['financial_year'].max()}")
    print(f"  Quarters covered: {quarterly.groupby(['financial_year', 'fy_quarter']).ngroups}")
    
    # Data quality
    print(f"\nData quality (monthly):")
    if "type1_attendances" in monthly.columns:
        print(f"  Records with missing attendances: {monthly['type1_attendances'].isna().sum():,}")
    if "type1_pct_4hr" in monthly.columns:
        print(f"  Records with missing 4hr %: {monthly['type1_pct_4hr'].isna().sum():,}")
    if "waits_12hr_decision_to_admit" in monthly.columns:
        print(f"  Records with missing 12hr waits: {monthly['waits_12hr_decision_to_admit'].isna().sum():,}")
    
    # Performance summary
    if "type1_pct_4hr" in quarterly.columns:
        print(f"\n4-hour standard performance:")
        recent = quarterly[quarterly["financial_year"] >= "2023-24"]
        if len(recent) > 0:
            print(f"  Average % seen in 4hr (2023-24 onwards): {recent['type1_pct_4hr'].mean():.1f}%")
            print(f"  National target: 95%")
    
    # Skipped files
    if skipped:
        print("\n" + "="*80)
        print("SKIPPED FILES")
        print("="*80)
        skip_csv = OUTPUT_DIR / "ae-skipped-files.csv"
        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(skip_csv, index=False)
        print(f"⚠️  {len(skipped)} files skipped - see: {skip_csv}")
        for file, reason in skipped[:10]:
            print(f"   - {file}: {reason[:100]}")
        if len(skipped) > 10:
            print(f"   ... and {len(skipped)-10} more")
    
    print("\n" + "="*80)
    print("✅ CLEANING COMPLETE!")
    print("="*80)


def test_mode():
    """
    Test mode - process single file.
    """
    print("="*80)
    print("TEST MODE")
    print("="*80)
    
    # Find first matching file
    test_files = list(INPUT_DIR.glob("*.xls"))
    test_files = [f for f in test_files if PATTERN1.match(f.name) or PATTERN2.match(f.name)]
    
    if not test_files:
        print("❌ No test files found")
        return
    
    test_file = test_files[0]
    print(f"\nTesting with: {test_file.name}")
    
    try:
        df = read_ae_monthly(test_file)
        print(f"\n✅ Successfully read {len(df)} rows")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        if "type1_attendances" in df.columns:
            print(f"\nSample statistics:")
            print(df[["type1_attendances", "type1_pct_4hr"]].describe())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        test_mode()
    else:
        main()
