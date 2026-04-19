"""
00_clean_eric.py
================
Stack annual ERIC Site CSV files, aggregate site->trust, build IV instruments.

SETUP:
    DB_PATH  = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/nhs_occupancy.duckdb"
    ERIC_DIR = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/01-data-raw/ERIC"     
    Expected: ERIC201819Site.csv ... ERIC2024_25Site.csv

VERIFIED: ERIC2022_23Site.csv (2,849 sites, 211 trusts)

INSTRUMENTS:
    Z1  backlog_hs_per_m2     (high+significant backlog / GIA m2)
        log_backlog_hs_per_m2 (log transform for right-skewed dist)
    Z2  pct_pre1985           (% floor area built pre-1985)
    +   backlog_total_per_m2, pct_pre1975 (robustness)

IDENTIFICATION:
    Estate distress/age -> unplanned ward closures -> higher occupancy
    on remaining beds. No plausible direct effect on A&E waits or
    cancellations conditional on trust FE + time FE + trust size.

OUTPUTS (DuckDB tables):
    eric_sites_raw, eric_trust_annual, eric_quarterly, eric_instruments
"""

import os, re, glob, warnings
import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH     = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/nhs_occupancy.duckdb"
ERIC_DIR    = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/01-data-raw/ERIC"
QUARTER_COL = "quarter"

# ── Canonical names ────────────────────────────────────────────────────────────
COL_TRUST_CODE    = "org_code"
COL_TRUST_NAME    = "trust_name"
COL_SITE_CODE     = "site_code"
COL_GIA           = "gia_m2"
COL_BACKLOG_HIGH  = "backlog_high"
COL_BACKLOG_SIG   = "backlog_significant"
COL_BACKLOG_MOD   = "backlog_moderate"
COL_BACKLOG_LOW   = "backlog_low"
COL_AGE_2015_2024 = "age_pct_2015_2024"
COL_AGE_2005_2014 = "age_pct_2005_2014"
COL_AGE_1995_2004 = "age_pct_1995_2004"
COL_AGE_1985_1994 = "age_pct_1985_1994"
COL_AGE_1975_1984 = "age_pct_1975_1984"
COL_AGE_1965_1974 = "age_pct_1965_1974"
COL_AGE_1955_1964 = "age_pct_1955_1964"
COL_AGE_1948_1954 = "age_pct_1948_1954"
COL_AGE_PRE1948   = "age_pct_pre1948"

ALL_BACKLOG_COLS = [COL_BACKLOG_HIGH, COL_BACKLOG_SIG, COL_BACKLOG_MOD, COL_BACKLOG_LOW]
ALL_AGE_COLS = [
    COL_AGE_2015_2024, COL_AGE_2005_2014, COL_AGE_1995_2004, COL_AGE_1985_1994,
    COL_AGE_1975_1984, COL_AGE_1965_1974, COL_AGE_1955_1964, COL_AGE_1948_1954,
    COL_AGE_PRE1948,
]

# ── Column map ─────────────────────────────────────────────────────────────────
# Two backlog naming eras:
#   2019+    : high / significant / moderate / low
#   pre-2019 : critical / high / significant / low
#              critical (most severe) now called "high"
#              high (second tier)    now called "significant"
COLUMN_MAP = {
    "trust code"                                       : COL_TRUST_CODE,
    "trust name"                                       : COL_TRUST_NAME,
    "site code"                                        : COL_SITE_CODE,
    "gross internal floor area"                        : COL_GIA,
    # Backlog current
    "cost to eradicate high risk backlog"              : COL_BACKLOG_HIGH,
    "cost to eradicate significant risk backlog"       : COL_BACKLOG_SIG,
    "cost to eradicate moderate risk backlog"          : COL_BACKLOG_MOD,
    "cost to eradicate low risk backlog"               : COL_BACKLOG_LOW,
    # Backlog legacy
    "cost to eradicate critical risk backlog"          : COL_BACKLOG_HIGH,
    "total cost to eradicate critical risk backlog"    : COL_BACKLOG_HIGH,
    "total cost to eradicate high risk backlog"        : COL_BACKLOG_SIG,
    "total cost to eradicate significant risk backlog" : COL_BACKLOG_SIG,
    "cost to eradicate high risk backlog maintenance"  : COL_BACKLOG_SIG,
    # Age current (% of GIA)
    "age profile - 2015 to 2024"                      : COL_AGE_2015_2024,
    "age profile - 2005 to 2014"                      : COL_AGE_2005_2014,
    "age profile - 1995 to 2004"                      : COL_AGE_1995_2004,
    "age profile - 1985 to 1994"                      : COL_AGE_1985_1994,
    "age profile - 1975 to 1984"                      : COL_AGE_1975_1984,
    "age profile - 1965 to 1974"                      : COL_AGE_1965_1974,
    "age profile - 1955 to 1964"                      : COL_AGE_1955_1964,
    "age profile - 1948 to 1954"                      : COL_AGE_1948_1954,
    "age profile - pre 1948"                          : COL_AGE_PRE1948,
    # Age legacy (absolute m2, pre-2019 — converted after load)
    "floor area of buildings built post 2000"          : "_leg_post2000",
    "floor area of buildings built 1985 to 2000"       : "_leg_1985_2000",
    "floor area of buildings built before 1985"        : "_leg_pre1985",
    "floor area built after 2000"                      : "_leg_post2000",
    "floor area built before 1985"                     : "_leg_pre1985",
}


def normalise_col(col):
    s = str(col).lower().strip()
    for bad, good in [("ï»¿",""),("â²",""),("mâ²","m2"),("m²","m2"),("â£",""),("£","")]:
        s = s.replace(bad, good)
    s = re.sub(r"\s*-\s*(pfi total|contractor provided.*|trust provided.*)$","",s)
    s = re.sub(r"\s*\(%\)\s*$","",s)
    s = re.sub(r"\s*\([^)]*\)\s*$","",s)
    return s.strip()


def detect_header_row(raw, max_rows=8):
    for i in range(max_rows):
        vals = [normalise_col(v) for v in raw.iloc[i].values if pd.notna(v)]
        if any(v in ("trust code","trust name","organisation code") for v in vals):
            return i
    return 0


def load_eric_site_file(path, year):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx",".xls"):
        raw = pd.read_excel(path, header=None, sheet_name=0)
    else:
        raw = pd.read_csv(path, header=None, encoding="latin1", low_memory=False)

    hrow   = detect_header_row(raw)
    df     = raw.iloc[hrow+1:].copy()
    df.columns = [str(c) for c in raw.iloc[hrow].values]
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    rename = {}
    seen   = set()
    for col in df.columns:
        target = COLUMN_MAP.get(normalise_col(col))
        if target and target not in seen:
            rename[col] = target
            seen.add(target)
    df.rename(columns=rename, inplace=True)

    wanted = set(COLUMN_MAP.values()) | {COL_TRUST_CODE, COL_TRUST_NAME, COL_SITE_CODE}
    df     = df[[c for c in df.columns if c in wanted]].copy()

    id_cols = {COL_TRUST_CODE, COL_TRUST_NAME, COL_SITE_CODE}
    for col in df.columns:
        if col in id_cols:
            continue
        df[col] = pd.to_numeric(
            df[col].astype(str)
                   .str.replace(r"[£,\s%]","",regex=True)
                   .str.replace("Not Applicable","",regex=False)
                   .str.replace("NotApplicable","",regex=False),
            errors="coerce"
        )

    if COL_TRUST_CODE in df.columns:
        df[COL_TRUST_CODE] = df[COL_TRUST_CODE].astype(str).str.upper().str.strip()
        df = df[df[COL_TRUST_CODE].str.match(r"^[A-Z][A-Z0-9]{2,5}$", na=False)]

    # Legacy m2 floor areas -> % of GIA
    if "_leg_pre1985" in df.columns and COL_GIA in df.columns:
        df["_age_pct_pre1985_legacy"] = (
            df["_leg_pre1985"] / df[COL_GIA].replace(0, np.nan) * 100
        )
    for leg in ["_leg_post2000","_leg_1985_2000","_leg_pre1985"]:
        if leg in df.columns:
            df.drop(columns=[leg], inplace=True)

    df["eric_year"] = year
    return df


def infer_year(fname):
    m = re.search(r"20(\d{2})[\-_]?(\d{2})", fname)
    if m:
        yr = int(m.group(2))
        return 2000+yr if yr < 50 else 1900+yr
    m = re.search(r"(20\d{2})", fname)
    return int(m.group(1)) if m else None


def aggregate_to_trust(df):
    present_backlog = [c for c in ALL_BACKLOG_COLS if c in df.columns]
    present_age     = [c for c in ALL_AGE_COLS + ["_age_pct_pre1985_legacy"]
                       if c in df.columns]
    df = df.copy()
    if COL_GIA not in df.columns:
        df[COL_GIA] = np.nan
    df[COL_GIA] = df[COL_GIA].fillna(0)

    rows = []
    for (trust_code, year), grp in df.groupby([COL_TRUST_CODE, 'eric_year']):
        row = {
            COL_TRUST_CODE : trust_code,
            COL_TRUST_NAME : grp[COL_TRUST_NAME].iloc[0] if COL_TRUST_NAME in grp.columns else "",
            "eric_year"    : year,
            "n_sites"      : len(grp),
            COL_GIA        : grp[COL_GIA].sum(),
        }
        for col in present_backlog:
            row[col] = grp[col].fillna(0).sum()
        for col in present_age:
            valid = grp[[COL_GIA, col]].dropna(subset=[col])
            if len(valid) > 0 and valid[COL_GIA].sum() > 0:
                row[col] = (valid[col] * valid[COL_GIA]).sum() / valid[COL_GIA].sum()
            else:
                row[col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def construct_instruments(df):
    df = df.copy()

    hs = (df.get(COL_BACKLOG_HIGH, pd.Series(0,index=df.index)).fillna(0)
        + df.get(COL_BACKLOG_SIG,  pd.Series(0,index=df.index)).fillna(0))
    df["backlog_hs_per_m2"]     = np.where(df[COL_GIA]>0, hs/df[COL_GIA], np.nan)
    df["log_backlog_hs_per_m2"] = np.log1p(
        df["backlog_hs_per_m2"].fillna(0)
    ).where(df["backlog_hs_per_m2"].notna(), np.nan)

    total = sum(df.get(c,pd.Series(0,index=df.index)).fillna(0) for c in ALL_BACKLOG_COLS)
    df["backlog_total_per_m2"] = np.where(df[COL_GIA]>0, total/df[COL_GIA], np.nan)
    df["backlog_high_per_m2"]  = np.where(
        df[COL_GIA]>0,
        df.get(COL_BACKLOG_HIGH,pd.Series(0,index=df.index)).fillna(0)/df[COL_GIA],
        np.nan
    )

    pre1985_bands = [COL_AGE_1975_1984,COL_AGE_1965_1974,
                     COL_AGE_1955_1964,COL_AGE_1948_1954,COL_AGE_PRE1948]
    if "_age_pct_pre1985_legacy" in df.columns:
        df["pct_pre1985"] = df["_age_pct_pre1985_legacy"]
    else:
        raw_sum = sum(df.get(c,pd.Series(0,index=df.index)).fillna(0)
                     for c in pre1985_bands)
        df["pct_pre1985"] = raw_sum.where(raw_sum > 0, np.nan)

    df["pct_pre1975"] = sum(
        df.get(c,pd.Series(0,index=df.index)).fillna(0)
        for c in [COL_AGE_1965_1974,COL_AGE_1955_1964,COL_AGE_1948_1954,COL_AGE_PRE1948]
    )
    df["pct_pre1975"] = df["pct_pre1975"].where(df["pct_pre1975"]>0, np.nan)

    for col in ("backlog_hs_per_m2","backlog_total_per_m2","backlog_high_per_m2",
                "log_backlog_hs_per_m2","pct_pre1985","pct_pre1975"):
        s = df[col].dropna()
        if len(s) > 10:
            lo, hi = s.quantile([0.01,0.99])
            df[col] = df[col].clip(lo, hi)
    return df


def align_to_quarters(df_annual):
    rows = []
    for _, row in df_annual.iterrows():
        fy = int(row["eric_year"])
        for q in [f"{fy-1}Q2",f"{fy-1}Q3",f"{fy-1}Q4",f"{fy}Q1"]:
            r = row.to_dict()
            r[QUARTER_COL] = q
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    files = sorted(
        glob.glob(os.path.join(ERIC_DIR,"**","*.csv"),  recursive=True) +
        glob.glob(os.path.join(ERIC_DIR,"**","*.xlsx"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(f"No ERIC files found in {ERIC_DIR}")

    site_frames = []
    for path in files:
        fname = os.path.basename(path)
        if "pfi" in fname.lower():
            print(f"  SKIP (PFI): {fname}"); continue
        year = infer_year(fname)
        if year is None:
            print(f"  SKIP (no year): {fname}"); continue
        try:
            df = load_eric_site_file(path, year)
            b  = sum(1 for c in ALL_BACKLOG_COLS if c in df.columns)
            a  = sum(1 for c in ALL_AGE_COLS     if c in df.columns)
            lg = "_age_pct_pre1985_legacy" in df.columns
            print(f"  OK  {fname} | year={year} sites={len(df)} "
                  f"backlog={b}/4 age_bands={a}/9 legacy={lg}")
            site_frames.append(df)
        except Exception as e:
            print(f"  ERR {fname} -> {e}")

    if not site_frames:
        raise RuntimeError("No files loaded.")

    sites_all    = pd.concat(site_frames, ignore_index=True)
    trust_annual = aggregate_to_trust(sites_all)
    trust_annual = construct_instruments(trust_annual)
    trust_qtr    = align_to_quarters(trust_annual)

    print(f"\nSites total   : {len(sites_all):,}")
    print(f"Trust-years   : {len(trust_annual):,}")
    print(f"Trust-quarters: {len(trust_qtr):,}")

    con = duckdb.connect(DB_PATH)
    for tbl, tdf in [("eric_sites_raw",    sites_all),
                     ("eric_trust_annual", trust_annual),
                     ("eric_quarterly",    trust_qtr)]:
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
        con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM tdf")
        print(f"Written: {tbl}")

    inst_cols = [COL_TRUST_CODE, QUARTER_COL, "eric_year", "n_sites", COL_GIA,
                 "backlog_hs_per_m2","log_backlog_hs_per_m2",
                 "backlog_total_per_m2","backlog_high_per_m2",
                 "pct_pre1985","pct_pre1975"]
    inst_df = trust_qtr[[c for c in inst_cols if c in trust_qtr.columns]]
    con.execute("DROP TABLE IF EXISTS eric_instruments")
    con.execute("CREATE TABLE eric_instruments AS SELECT * FROM inst_df")
    print("Written: eric_instruments")
    con.close()

    print("\n── Coverage by year ──")
    print(trust_annual.groupby("eric_year").agg(
        trusts        =(COL_TRUST_CODE,     "nunique"),
        backlog_cov   =("backlog_hs_per_m2", lambda x: x.notna().mean()),
        age_cov       =("pct_pre1985",        lambda x: x.notna().mean()),
        median_hs_m2  =("backlog_hs_per_m2", "median"),
        median_pre1985=("pct_pre1985",        "median"),
    ).round(3).to_string())

    print("\n── Instrument descriptives ──")
    print(trust_annual[["backlog_hs_per_m2","log_backlog_hs_per_m2",
                         "pct_pre1985","pct_pre1975"]].describe().round(3).to_string())

    print("\n── Instrument correlation ──")
    print(trust_annual[["log_backlog_hs_per_m2","pct_pre1985"]].corr().round(3).to_string())
    print("Target |corr| < 0.7")
    print("\nDone. Run 01_within_between.py next.")


if __name__ == "__main__":
    main()

