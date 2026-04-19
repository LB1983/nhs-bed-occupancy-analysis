"""
KH03 Bed Occupancy Data Cleaning Script
========================================

Author: Lauren Gavaghan (The Bevan Briefing)
Date: February 2026
Purpose: Clean NHS England KH03 quarterly bed occupancy data for bed occupancy analysis

This script:
1. Reads KH03 quarterly Excel files (FY 2018-19 to 2024-25)
2. Extracts trust-level bed data by specialty
3. Calculates occupancy rates and metrics
4. Produces clean analysis-ready dataset

File naming convention: YYYYFY-QN_KH03_Overnight.xlsx (e.g., 201819-Q1_KH03_Overnight.xlsx)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - EDIT THESE to match your setup
RAW_DATA_DIR = Path("01-data-raw/kh03")
OUTPUT_DIR = Path("02-data-interim/kh03-cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Financial years to process (your downloaded data)
FINANCIAL_YEARS = [
    "201819", "201920", "202021", "202122", 
    "202223", "202324", "202425"
]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

# Expected sheet name in Excel files
SHEET_NAME = "Beds by Organisation"  # Most common, but we'll verify

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_fy_quarter(filename):
    """
    Extract financial year and quarter from filename.
    
    Example: "201920-Q1_KH03_Overnight.xlsx" -> ("2019-20", "Q1")
    """
    parts = filename.stem.split("-")
    
    # First part is YYYYFY format (e.g., "201920")
    fy_concat = parts[0]
    year1 = fy_concat[:4]  # "2019"
    year2 = fy_concat[4:]  # "20"
    fy = f"{year1}-{year2}"  # "2019-20"
    
    # Second part contains quarter
    quarter = parts[1].split("_")[0]  # "Q1"
    
    return fy, quarter


def fy_quarter_to_period(fy, quarter):
    """
    Convert FY + Quarter to calendar quarter for time series analysis.
    
    NHS FY Q1 (Apr-Jun) = Calendar Q2
    NHS FY Q2 (Jul-Sep) = Calendar Q3
    NHS FY Q3 (Oct-Dec) = Calendar Q4
    NHS FY Q4 (Jan-Mar) = Calendar Q1 (next year)
    """
    fy_start = int(fy.split("-")[0])
    quarter_num = int(quarter[1])
    
    if quarter_num == 1:  # Apr-Jun
        return f"{fy_start}-Q2"
    elif quarter_num == 2:  # Jul-Sep
        return f"{fy_start}-Q3"
    elif quarter_num == 3:  # Oct-Dec
        return f"{fy_start}-Q4"
    else:  # Q4: Jan-Mar (next calendar year)
        return f"{fy_start + 1}-Q1"


def find_data_sheet(xl_file):
    """
    Find the sheet containing trust-level bed data.
    """
    sheet_names = xl_file.sheet_names
    
    # Try common names (in priority order)
    possible_names = [
        "Beds by Organisation",
        "Beds by Organization",
        "NHS Trust by Sector",  # Common in your files
        "Organisation",
        "Trust Level",
        "Provider Level",
        "Data"
    ]
    
    for name in possible_names:
        if name in sheet_names:
            return name
    
    # Exclude certain sheets we know are NOT data
    exclude = ['cover', 'contents', 'notes', 'cover sheet', 'data quality', 'guidance']
    
    # Find first sheet that's not in exclude list
    for name in sheet_names:
        if name.lower() not in exclude:
            return name
    
    return sheet_names[0]  # Fallback


def load_kh03_file(filepath):
    """
    Load a single KH03 Excel file and extract bed data.
    KH03 files have metadata rows at the top and multi-level headers.
    Row 14 has section headers (Available, Occupied, % Occupied)
    Row 15 has column headers (Total, General & Acute, etc.)
    We need to combine these to create unique column names.
    """
    fy, quarter = parse_fy_quarter(filepath)
    period = fy_quarter_to_period(fy, quarter)
    
    print(f"\n{'='*80}")
    print(f"Processing: {filepath.name}")
    print(f"  FY: {fy} | Quarter: {quarter} | Period: {period}")
    
    try:
        # Open file and find data sheet
        xl_file = pd.ExcelFile(filepath)
        sheet_name = find_data_sheet(xl_file)
        print(f"  Using sheet: '{sheet_name}'")
        
        # Read the full sheet first to find header row
        df_raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
        
        # Find the row with "Org Code" - this is the header row
        header_row = None
        for i in range(min(30, len(df_raw))):
            row_text = ' '.join([str(v) for v in df_raw.iloc[i, :10] if pd.notna(v)]).lower()
            if 'org code' in row_text:
                header_row = i
                break
        
        if header_row is None:
            print(f"  ⚠️ Could not find header row, using default")
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=0)
        else:
            print(f"  Found data header at row {header_row}")
            
            # Read with multi-level header (section row + column row)
            # Section row is one above the header row
            section_row = header_row - 1
            
            # Read both rows
            section_labels = df_raw.iloc[section_row].fillna('')
            column_labels = df_raw.iloc[header_row].fillna('')
            
            # Combine section + column to create unique names
            combined_cols = []
            for i, (section, col) in enumerate(zip(section_labels, column_labels)):
                section = str(section).strip()
                col = str(col).strip()
                
                if section and section not in ['nan', 'None', '']:
                    # Has a section label
                    combined_cols.append(f"{section}_{col}" if col and col not in ['nan', 'None', ''] else section)
                elif col and col not in ['nan', 'None', '']:
                    # Just column label
                    combined_cols.append(col)
                else:
                    # Empty - use generic name
                    combined_cols.append(f"Unnamed_{i}")
            
            # Read data starting after header row
            df = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=header_row+1, header=None)
            df.columns = combined_cols
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
        
        # Add metadata
        df['financial_year'] = fy
        df['fy_quarter'] = quarter
        df['period'] = period
        df['source_file'] = filepath.name
        
        return df, sheet_name
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def standardize_columns(df):
    """
    Standardize column names across years (NHS sometimes changes names).
    """
    # Strip whitespace
    df.columns = df.columns.str.strip()
    
    # Create mapping
    column_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Organization identifiers
        if any(x in col_lower for x in ['org code', 'organisation code', 'ods code']) and 'name' not in col_lower:
            column_mapping[col] = 'org_code'
        elif any(x in col_lower for x in ['org name', 'organisation name', 'trust name']) and 'code' not in col_lower:
            column_mapping[col] = 'org_name'
        
        # Sector
        elif col_lower == 'sector':
            column_mapping[col] = 'sector'
        
        # Keep specialty/metric columns as-is but standardize format
        else:
            standardized = col.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
            column_mapping[col] = standardized
    
    df = df.rename(columns=column_mapping)
    
    return df


def identify_bed_metrics(df):
    """
    Identify columns containing bed data (available/occupied by specialty).
    """
    # Exclude metadata columns
    metadata_cols = ['org_code', 'org_name', 'sector', 'financial_year', 'fy_quarter', 'period', 'source_file']
    
    # Find all numeric columns that aren't metadata
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    bed_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Separate available vs occupied
    available_cols = [col for col in bed_cols if 'available' in col.lower()]
    occupied_cols = [col for col in bed_cols if 'occupied' in col.lower()]
    
    return {
        'available': available_cols,
        'occupied': occupied_cols,
        'all_beds': bed_cols
    }


def calculate_total_occupancy(df):
    """
    Calculate total bed metrics if not already present.
    """
    bed_metrics = identify_bed_metrics(df)
    
    # If we have specialty-level data, sum to get totals
    if len(bed_metrics['available']) > 0 and 'total_available_bed_days' not in df.columns:
        df['total_available_bed_days'] = df[bed_metrics['available']].sum(axis=1)
    
    if len(bed_metrics['occupied']) > 0 and 'total_occupied_bed_days' not in df.columns:
        df['total_occupied_bed_days'] = df[bed_metrics['occupied']].sum(axis=1)
    
    # Calculate occupancy rate
    if 'total_available_bed_days' in df.columns and 'total_occupied_bed_days' in df.columns:
        df['occupancy_rate'] = (
            df['total_occupied_bed_days'] / df['total_available_bed_days']
        ).round(4)
        
        # Average beds per day (quarter = ~91 days)
        days_in_quarter = 91
        df['avg_available_beds'] = (df['total_available_bed_days'] / days_in_quarter).round(1)
        df['avg_occupied_beds'] = (df['total_occupied_bed_days'] / days_in_quarter).round(1)
    
    return df


def flag_data_quality_issues(df, filepath):
    """
    Identify data quality problems.
    """
    issues = []
    
    if 'occupancy_rate' in df.columns:
        # Occupancy > 100%
        over_100 = (df['occupancy_rate'] > 1.0).sum()
        if over_100 > 0:
            issues.append(f"⚠️  {over_100} records with occupancy >100%")
        
        # Very low occupancy (<20%)
        under_20 = (df['occupancy_rate'] < 0.2).sum()
        if under_20 > 0:
            issues.append(f"⚠️  {under_20} records with occupancy <20%")
    
    # Missing org codes
    if 'org_code' in df.columns:
        missing = df['org_code'].isna().sum()
        if missing > 0:
            issues.append(f"⚠️  {missing} records with missing org codes")
    
    # Zero available beds
    if 'total_available_bed_days' in df.columns:
        zero_beds = (df['total_available_bed_days'] == 0).sum()
        if zero_beds > 0:
            issues.append(f"⚠️  {zero_beds} records with zero available beds")
    
    return issues


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main cleaning pipeline.
    """
    print("="*80)
    print("KH03 DATA CLEANING PIPELINE")
    print("="*80)
    print(f"Input directory: {RAW_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    all_data = []
    all_issues = []
    processed_files = []
    
    # Process each file
    for fy in FINANCIAL_YEARS:
        for quarter in QUARTERS:
            filename = f"{fy}-{quarter}_KH03_Overnight.xlsx"
            filepath = RAW_DATA_DIR / filename
            
            if not filepath.exists():
                print(f"⏭️  Skipping {filename} (not found)")
                continue
            
            # Load file
            df, sheet_name = load_kh03_file(filepath)
            
            if df is None:
                continue
            
            # Standardize columns
            df = standardize_columns(df)
            
            # Calculate metrics
            df = calculate_total_occupancy(df)
            
            # Check data quality
            issues = flag_data_quality_issues(df, filepath)
            if issues:
                print(f"  Data quality issues:")
                for issue in issues:
                    print(f"    {issue}")
                all_issues.extend([(filename, issue) for issue in issues])
            else:
                print(f"  ✅ No data quality issues")
            
            all_data.append(df)
            processed_files.append(filename)
    
    # Combine all quarters
    print("\n" + "="*80)
    print("COMBINING DATA")
    print("="*80)
    
    if not all_data:
        print("❌ No data loaded. Check file paths and names.")
        return
    
    # Use outer join to handle different columns across quarters
    combined = pd.concat(all_data, ignore_index=True, join='outer', sort=False)
    print(f"✅ Combined {len(all_data)} files")
    print(f"   Files processed: {processed_files[:3]}... and {len(processed_files)-3} more")
    print(f"   Total records: {len(combined):,}")
    print(f"   Date range: {combined['period'].min()} to {combined['period'].max()}")
    
    if 'org_code' in combined.columns:
        print(f"   Unique trusts: {combined['org_code'].nunique()}")
    
    # Save combined dataset
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # CSV first (always works, human-readable)
    csv_file = OUTPUT_DIR / "kh03-all-quarters.csv"
    combined.to_csv(csv_file, index=False)
    print(f"✅ CSV: {csv_file}")
    
    # Parquet (compressed, efficient) - with error handling
    try:
        output_file = OUTPUT_DIR / "kh03-all-quarters.parquet"
        combined.to_parquet(output_file, index=False)
        print(f"✅ Parquet: {output_file}")
    except Exception as e:
        print(f"⚠️  Parquet save failed: {str(e)[:100]}")
        print(f"   CSV file is still available and contains all data")
    
    # Sample for inspection
    sample_file = OUTPUT_DIR / "sample-first-file.csv"
    all_data[0].head(20).to_csv(sample_file, index=False)
    print(f"✅ Sample: {sample_file}")
    
    # Data quality issues
    if all_issues:
        issues_df = pd.DataFrame(all_issues, columns=['file', 'issue'])
        issues_file = OUTPUT_DIR / "data-quality-issues.csv"
        issues_df.to_csv(issues_file, index=False)
        print(f"⚠️  Issues: {issues_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if 'occupancy_rate' in combined.columns:
        summary = combined.groupby('period').agg({
            'org_code': 'count',
            'occupancy_rate': ['mean', 'median', 'std'],
            'total_available_bed_days': 'sum',
            'total_occupied_bed_days': 'sum'
        }).round(3)
        
        print("\nOccupancy by period:")
        print(summary)
        
        summary_file = OUTPUT_DIR / "summary-by-period.csv"
        summary.to_csv(summary_file)
        print(f"\n✅ Summary: {summary_file}")
    
    # Column documentation
    print("\n" + "="*80)
    print("COLUMN DOCUMENTATION")
    print("="*80)
    
    print(f"\nFinal dataset has {len(combined.columns)} columns:")
    
    # Core columns
    core_cols = ['org_code', 'org_name', 'sector', 'financial_year', 'fy_quarter', 'period']
    print(f"\n📋 Core columns: {[c for c in core_cols if c in combined.columns]}")
    
    # Calculated metrics
    metric_cols = ['total_available_bed_days', 'total_occupied_bed_days', 'occupancy_rate', 
                   'avg_available_beds', 'avg_occupied_beds']
    print(f"📊 Calculated metrics: {[c for c in metric_cols if c in combined.columns]}")
    
    # Specialty columns
    bed_metrics = identify_bed_metrics(combined)
    print(f"🛏️  Specialty columns: {len(bed_metrics['all_beds'])} columns")
    print(f"   Available: {len(bed_metrics['available'])}")
    print(f"   Occupied: {len(bed_metrics['occupied'])}")
    
    print("\n" + "="*80)
    print("✅ CLEANING COMPLETE!")
    print("="*80)
    
    print(f"""
Next steps:
1. Review: {sample_file}
2. Check summary: {summary_file}
3. Inspect any issues: {OUTPUT_DIR / 'data-quality-issues.csv' if all_issues else 'None'}
4. Ready for analysis!

Main output: {csv_file}
    """)


if __name__ == "__main__":
    main()
