#!/usr/bin/env python3
"""
NHS Local Occupancy Safety Target Calculator
=============================================
Calculates a locally-calibrated bed occupancy safety ceiling for any NHS
acute trust using ODS code, drawing on trust-level proximity and historical
occupancy data from the Bevan Briefing NHS Bed Occupancy Analysis (2025).

METHODOLOGY SUMMARY
-------------------
Three factors determine a trust's local ceiling:

  1. Geographic buffer score  -- how much harm does rising occupancy cause,
     given the availability of alternatives? Estimated from a two-way FE
     interaction model (trust + time FE, clustered SE) on 3,551 trust-quarters
     2018-19 to 2024-25. Outcome: 12-hr waits per 1,000 A&E attendances.

  2. Demand unpredictability -- trusts with high occupancy volatility (seasonal
     swings) need more headroom below the ceiling. Computed from recent
     quarterly occupancy history.

  3. Emergency intensity      -- trusts with high A&E demand relative to bed
     base have less controllable occupancy. Computed from KH03 + A&E SitRep.
     If you have your own EDR (emergency admissions / occupied bed days)
     from SUS/ECDS, supply it via --edr for a more precise estimate.

USAGE
-----
  python occupancy_target_tool.py RRV
  python occupancy_target_tool.py REF --edr 0.52
  python occupancy_target_tool.py        (interactive prompt)

REQUIRED DATA FILES (relative to this script's location)
---------------------------------------------------------
  ../03-data-final/master-quarterly-trust.csv
  ../02-data-interim/geo/trust_proximity_features.csv

DEPENDENCIES
------------
  pip install pandas numpy
"""

import argparse
import sys
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent
MASTER_PATH = _BASE / "03-data-final" / "master-quarterly-trust.csv"
PROX_PATH   = _BASE / "02-data-interim" / "geo" / "trust_proximity_features.csv"

# ---------------------------------------------------------------------------
# Model coefficients (all-beds, ae_12h_per_1k_att, linear interaction model)
# Source: 05b-regressions_DEBUGGED.py, models_scaled_interactions.csv
# Two-way FE (trust + time), cluster-robust SE by trust, n=3,551
# ---------------------------------------------------------------------------
B_OCC_Z   = -0.603   # baseline occ effect (standardised)
B_NEAR_Z  =  1.627   # occ x nearest_trust_km_z
B_10KM_Z  = -4.283   # occ x trusts_within_10km_z  (buffering)
B_25KM_Z  =  4.162   # occ x trusts_within_25km_z  (amplifying)

# National proximity distributions (used to standardise trust values)
PROX_STATS = {
    "nearest_trust_km":   {"mean": 7.59,  "sd": 10.13},
    "trusts_within_10km": {"mean": 4.20,  "sd":  6.70},
    "trusts_within_25km": {"mean": 12.84, "sd": 15.72},
}

# Occupancy volatility distribution (recovery period 2022+, n=187 trusts)
OV_P50 = 0.0281
OV_P75 = 0.0417

# A&E intensity distribution (non-COVID, n=165 trusts)
# attendances per 100 available beds per quarter
AEI_P25 = 25.14
AEI_P75 = 38.08

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def standardise(value: float, mean: float, sd: float) -> float:
    return (value - mean) / sd if sd > 0 else 0.0


def compute_beta(nearest_km: float, within_10: int, within_25: int) -> float:
    """Geographic harm multiplier: predicted change in 12-hr waits
    per 1 SD rise in occupancy, given this trust's proximity profile."""
    z_near = standardise(nearest_km,  **PROX_STATS["nearest_trust_km"])
    z_10   = standardise(within_10,   **PROX_STATS["trusts_within_10km"])
    z_25   = standardise(within_25,   **PROX_STATS["trusts_within_25km"])
    return B_OCC_Z + B_NEAR_Z * z_near + B_10KM_Z * z_10 + B_25KM_Z * z_25


def geo_uplift(beta: float) -> tuple[int, str]:
    """Return (pp_uplift, label) based on geographic harm multiplier."""
    if beta < -3:
        return 8, "High geographic buffer"
    elif beta < -1:
        return 5, "Moderate geographic buffer"
    elif beta < 0:
        return 2, "Marginal geographic buffer"
    else:
        return 0, "No effective geographic buffer"


def ov_adjustment(ov: float) -> tuple[int, str]:
    """Return (pp_downward_adjustment, label) based on occupancy volatility."""
    if ov < 0.020:
        return 0, "Low seasonal swing"
    elif ov < 0.050:
        return 1, "Moderate seasonal swing"
    else:
        return 2, "High seasonal swing"


def edr_adjustment(edr: float | None, aei: float | None) -> tuple[int, str]:
    """Return (pp_downward_adjustment, label) based on EDR or AEI proxy."""
    if edr is not None:
        if edr < 0.35:
            return 0, f"Low emergency dependency (EDR={edr:.2f}, user-supplied)"
        elif edr < 0.50:
            return 1, f"Mixed emergency dependency (EDR={edr:.2f}, user-supplied)"
        else:
            return 2, f"High emergency dependency (EDR={edr:.2f}, user-supplied)"
    # Fall back to A&E intensity proxy
    if aei is None or np.isnan(aei):
        return 1, "Emergency dependency unknown (no A&E data) -- assuming moderate"
    if aei < AEI_P25:
        return 0, f"Low A&E intensity ({aei:.1f} att/100 beds/qtr) -- predominantly elective/specialist"
    elif aei < AEI_P75:
        return 1, f"Moderate A&E intensity ({aei:.1f} att/100 beds/qtr)"
    else:
        return 2, f"High A&E intensity ({aei:.1f} att/100 beds/qtr) -- emergency-dominant"


def sparkline(values: list[float], width: int = 28) -> str:
    """Return a simple ASCII bar chart of quarterly occupancy values."""
    chars = " _.-=+*#@"
    if not values:
        return ""
    lo, hi = 0.50, 1.0
    out = []
    for v in values[-width:]:
        if np.isnan(v):
            out.append("?")
        else:
            idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
            out.append(chars[max(0, min(len(chars) - 1, idx))])
    return "".join(out)


def fmt_pct(v: float) -> str:
    return f"{v*100:.1f}%"


def fmt_pp(v: int) -> str:
    return f"+{v} pp" if v > 0 else f"{v} pp"


def rule(char: str = "-", width: int = 70) -> str:
    return char * width


def wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=70, initial_indent=prefix,
                         subsequent_indent=prefix)


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------
def run(trust_code: str, edr: float | None = None) -> None:
    trust_code = trust_code.strip().upper()

    # -- Load data -----------------------------------------------------------
    if not MASTER_PATH.exists():
        sys.exit(f"ERROR: Cannot find master data at {MASTER_PATH}")
    if not PROX_PATH.exists():
        sys.exit(f"ERROR: Cannot find proximity data at {PROX_PATH}")

    master = pd.read_csv(MASTER_PATH)
    prox   = pd.read_csv(PROX_PATH)

    master["org_code"] = master["org_code"].str.strip().str.upper()
    prox["org_code"]   = prox["org_code"].str.strip().str.upper()

    # -- Trust lookup --------------------------------------------------------
    trust_rows = master[master["org_code"] == trust_code].copy()
    if trust_rows.empty:
        available = sorted(master["org_code"].unique())
        sys.exit(
            f"ERROR: Trust code '{trust_code}' not found in master data.\n"
            f"Available codes (sample): {available[:20]} ..."
        )

    trust_name = trust_rows["org_name"].dropna().iloc[0] \
        if "org_name" in trust_rows.columns else trust_code

    prox_rows = prox[prox["org_code"] == trust_code]
    has_prox  = not prox_rows.empty

    # -- Proximity -----------------------------------------------------------
    if has_prox:
        p           = prox_rows.iloc[0]
        nearest_km  = float(p["nearest_trust_km"])
        within_10   = int(p["trusts_within_10km"])
        within_25   = int(p["trusts_within_25km"])
        within_50   = int(p.get("trusts_within_50km", np.nan))
        beta        = compute_beta(nearest_km, within_10, within_25)
        z_near      = standardise(nearest_km, **PROX_STATS["nearest_trust_km"])
        z_10        = standardise(within_10,  **PROX_STATS["trusts_within_10km"])
        z_25        = standardise(within_25,  **PROX_STATS["trusts_within_25km"])
    else:
        print(f"WARNING: No proximity data for {trust_code}. "
              "Geographic adjustment will use national average (0 pp uplift).")
        nearest_km = within_10 = within_25 = within_50 = np.nan
        beta = B_OCC_Z   # baseline only
        z_near = z_10 = z_25 = 0.0

    geo_up, geo_label = geo_uplift(beta)

    # -- Occupancy history ---------------------------------------------------
    occ_col = "kh03_occ_all_rate"
    ga_col  = "kh03_occ_ga_rate"
    beds_col = "kh03_avail_all_avg_beds"

    for col in [occ_col, ga_col, beds_col]:
        trust_rows[col] = pd.to_numeric(trust_rows[col], errors="coerce")

    trust_rows["yr"] = trust_rows["period"].astype(str).str[:4].astype(int)
    non_covid = trust_rows[~trust_rows["yr"].isin([2020, 2021])]
    recent    = trust_rows[trust_rows["yr"] >= 2022]

    occ_all = trust_rows[occ_col].dropna()
    occ_nc  = non_covid[occ_col].dropna()
    occ_rec = recent[occ_col].dropna()

    mean_occ_all = occ_all.mean()
    mean_occ_nc  = occ_nc.mean()   if len(occ_nc)  else np.nan
    mean_occ_rec = occ_rec.mean()  if len(occ_rec) else np.nan

    pct_above_85 = (occ_all >= 0.85).mean() * 100
    pct_above_90 = (occ_all >= 0.90).mean() * 100
    pct_above_85_rec = (occ_rec >= 0.85).mean() * 100 if len(occ_rec) else np.nan
    pct_above_90_rec = (occ_rec >= 0.90).mean() * 100 if len(occ_rec) else np.nan

    q25 = occ_all.quantile(0.25)
    q75 = occ_all.quantile(0.75)
    p10 = occ_all.quantile(0.10)
    p90 = occ_all.quantile(0.90)

    # OV from recovery period (or full non-COVID if insufficient)
    ov_base = occ_rec if len(occ_rec) >= 6 else occ_nc
    ov_val  = (ov_base.std() / ov_base.mean()) if (len(ov_base) >= 4 and ov_base.mean() > 0) else np.nan
    ov_adj, ov_label = ov_adjustment(ov_val if not np.isnan(ov_val) else OV_P50)

    # Mean beds
    mean_beds_raw = trust_rows[beds_col].dropna().mean()
    mean_beds_est = mean_beds_raw * 100 if not np.isnan(mean_beds_raw) else np.nan

    # G&A occupancy
    ga_all  = trust_rows[ga_col].dropna()
    mean_ga = ga_all.mean()

    # A&E intensity proxy
    aei_col = "ae_type1_attendances"
    if aei_col in trust_rows.columns:
        trust_rows[aei_col] = pd.to_numeric(trust_rows[aei_col], errors="coerce")
        nc_aei = non_covid[[aei_col, beds_col]].dropna()
        if len(nc_aei) >= 4:
            aei = (nc_aei[aei_col] / (nc_aei[beds_col] * 100)).mean()
        else:
            aei = np.nan
    else:
        aei = np.nan

    edr_adj, edr_label = edr_adjustment(edr, aei if not np.isnan(aei) else None)

    # -- Outcome data (where available) -------------------------------------
    outcomes = {
        "A&E 4hr compliance":        "ae_type1_pct_4hr",
        "12-hr waits per 1k att":    "ae_12h_per_1k_att",
        "Cancelled ops per 100 beds":"canc_per_100_beds",
        "% cancelled not treated 28d":"canc_pct_not_treated_28_days",
        "NCTR mean (patients/day)":   "uec_uec_nctr_mean",
    }

    # -- Final target --------------------------------------------------------
    NATIONAL_FLOOR = 80
    local_ceiling  = NATIONAL_FLOOR + geo_up - ov_adj - edr_adj

    # Flag if structurally unachievable
    structurally_breached = (
        not np.isnan(mean_occ_rec) and
        mean_occ_rec > (local_ceiling / 100) + 0.05
    )

    # -- NCTR context --------------------------------------------------------
    nctr_col = "uec_uec_nctr_mean"
    if nctr_col in trust_rows.columns:
        trust_rows[nctr_col] = pd.to_numeric(trust_rows[nctr_col], errors="coerce")
        nctr_mean = trust_rows[nctr_col].dropna().mean()
    else:
        nctr_mean = np.nan

    # -- Output --------------------------------------------------------------
    W = 70

    print()
    print(rule("=", W))
    print(f"  NHS LOCAL OCCUPANCY TARGET CALCULATOR")
    print(f"  Bevan Briefing | NHS Bed Occupancy Analysis 2025")
    print(rule("=", W))

    # Trust header
    print()
    print(f"  Trust:     {trust_name}")
    print(f"  ODS code:  {trust_code}")
    if not np.isnan(mean_beds_est):
        print(f"  Bed base:  ~{mean_beds_est:.0f} beds (mean all available, KH03)")
    print()

    # --- SECTION 1: Geographic profile ---
    print(rule("-", W))
    print("  1. GEOGRAPHIC BUFFER ANALYSIS")
    print(rule("-", W))

    if has_prox:
        print(f"  Nearest acute trust:    {nearest_km:.1f} km  (national median: 4.3 km)")
        print(f"  Trusts within 10 km:    {within_10}  (national median: 2)")
        print(f"  Trusts within 25 km:    {within_25}  (national median: 5)")
        if not np.isnan(within_50):
            print(f"  Trusts within 50 km:    {within_50}")
        print()
        print(f"  Standardised position:")
        print(f"    z(nearest):   {z_near:+.2f}  {'(more isolated)' if z_near > 0 else '(closer than avg)'}")
        print(f"    z(within 10): {z_10:+.2f}  {'(more urban)' if z_10 > 0 else '(fewer close neighbours)'}")
        print(f"    z(within 25): {z_25:+.2f}  {'(denser region)' if z_25 > 0 else '(sparser region)'}")
        print()
        print(f"  Geographic harm multiplier (beta):  {beta:+.2f}")
        print(f"  Interpretation: for a 1 SD rise in occupancy, 12-hr waits")
        print(f"  change by {beta:+.2f} per 1,000 A&E attendances at this trust.")
        print()
        if beta < -3:
            print(wrap("A negative beta indicates that rising occupancy at this trust "
                       "is buffered by the density of nearby alternatives. Patients and "
                       "ambulances have real diversion options, dampening the "
                       "occupancy-to-harm conversion."))
        elif beta < 0:
            print(wrap("A moderately negative beta indicates partial geographic "
                       "buffering. Some diversion options exist but the benefit is "
                       "limited -- high occupancy at this trust will still cause "
                       "measurable harm if alternatives are simultaneously under pressure."))
        else:
            print(wrap("A positive beta indicates that this trust has no effective "
                       "geographic buffer. When occupancy rises, there is nowhere for "
                       "patients to go. Every percentage point above the safety "
                       "threshold directly translates into worse outcomes."))
    else:
        print("  No proximity data available for this trust.")
        print("  Geographic adjustment defaulting to 0 pp (national average).")

    print()
    print(f"  Geographic uplift applied:  {fmt_pp(geo_up)}  ({geo_label})")

    # --- SECTION 2: Occupancy history ---
    print()
    print(rule("-", W))
    print("  2. OCCUPANCY HISTORY")
    print(rule("-", W))
    print()

    spark_vals = trust_rows.sort_values("period")[occ_col].tolist()
    print(f"  Trend (2018-19 Q1 -> 2024-25 Q4):  [{sparkline(spark_vals)}]")
    print(f"  Scale: space=<55%  _=60%  .=65%  -=70%  ==75%  +=80%  *=85%  #=90%  @=95%+")
    print()
    print(f"  Full period     mean: {fmt_pct(mean_occ_all)}   "
          f"IQR: {fmt_pct(q25)}-{fmt_pct(q75)}   "
          f"p10-p90: {fmt_pct(p10)}-{fmt_pct(p90)}")
    if not np.isnan(mean_occ_nc):
        print(f"  Pre-COVID       mean: {fmt_pct(mean_occ_nc)}")
    if not np.isnan(mean_occ_rec):
        print(f"  Recovery (2022+) mean: {fmt_pct(mean_occ_rec)}")
    if not np.isnan(mean_ga):
        print(f"  G&A beds only   mean: {fmt_pct(mean_ga)}")
    print()
    print(f"  Quarters above 85% (full period): {pct_above_85:.0f}%  |  above 90%: {pct_above_90:.0f}%")
    if not np.isnan(pct_above_85_rec):
        print(f"  Quarters above 85% (since 2022):  {pct_above_85_rec:.0f}%  |  above 90%: {pct_above_90_rec:.0f}%")

    print()
    print(f"  Occupancy volatility (OV):  {ov_val:.4f}" if not np.isnan(ov_val) else "  Occupancy volatility (OV): insufficient data")
    print(f"  (National p25={OV_P50:.4f}, p75={OV_P75:.4f})")
    print()
    print(f"  OV adjustment applied:  -{ov_adj} pp  ({ov_label})")

    # --- SECTION 3: Emergency intensity ---
    print()
    print(rule("-", W))
    print("  3. DEMAND PROFILE")
    print(rule("-", W))
    print()
    if edr is not None:
        print(f"  EDR (user-supplied):  {edr:.2f}")
    elif not np.isnan(aei):
        print(f"  A&E intensity (auto): {aei:.1f} attendances per 100 available beds/qtr")
        print(f"  (National p25={AEI_P25:.1f}, p75={AEI_P75:.1f})")
        print()
        print(wrap("Note: A&E intensity is used as a proxy for emergency dependency "
                   "in the absence of SUS/ECDS admissions data. For a more precise "
                   "estimate, supply your own EDR (emergency admissions / total "
                   "occupied bed days) via --edr."))
    else:
        print("  No A&E data available. Emergency intensity assumed moderate.")
    print()
    print(f"  Emergency demand adjustment:  -{edr_adj} pp  ({edr_label})")

    # --- SECTION 4: Outcome context ---
    print()
    print(rule("-", W))
    print("  4. RECENT OUTCOME CONTEXT")
    print(rule("-", W))
    print()
    found_any = False
    for label, col in outcomes.items():
        if col in trust_rows.columns:
            v = pd.to_numeric(trust_rows[col], errors="coerce").dropna()
            if len(v) >= 2:
                val = v.mean()
                suffix = ""
                if col == "ae_type1_pct_4hr":
                    suffix = f" (national standard: 76%)"
                    val_str = f"{val*100:.1f}%"
                elif col == "ae_12h_per_1k_att":
                    val_str = f"{val:.1f}"
                elif col == "canc_per_100_beds":
                    val_str = f"{val:.1f}"
                elif col == "canc_pct_not_treated_28_days":
                    val_str = f"{val*100:.1f}%"
                else:
                    val_str = f"{val:.0f}"
                print(f"  {label:<32} {val_str}{suffix}")
                found_any = True
    if not found_any:
        print("  No outcome data available for this trust.")
    if not np.isnan(nctr_mean):
        print()
        print(wrap(f"NCTR context: {nctr_mean:.0f} mean delayed-transfer patients per day. "
                   "NCTR is the binding operational constraint for many high-occupancy trusts. "
                   "Reducing NCTR directly reduces occupancy without requiring additional beds."))

    # --- SECTION 5: Target ---
    print()
    print(rule("=", W))
    print("  LOCAL OCCUPANCY SAFETY CEILING")
    print(rule("=", W))
    print()
    print(f"  National floor                   80%")
    print(f"  + Geographic uplift          {fmt_pp(geo_up):>6}    ({geo_label})")
    print(f"  - OV adjustment              {fmt_pp(-ov_adj):>6}    ({ov_label})")
    print(f"  - Emergency demand adj.      {fmt_pp(-edr_adj):>6}    ({edr_label})")
    print(rule("-", W))
    print(f"  LOCAL CEILING:               {local_ceiling:>5}%")
    print()

    if structurally_breached:
        gap = mean_occ_rec * 100 - local_ceiling
        print(rule("!", W))
        print("  STRUCTURAL BREACH WARNING")
        print(rule("!", W))
        print()
        print(wrap(
            f"This trust's recent mean occupancy ({fmt_pct(mean_occ_rec)}) exceeds "
            f"the calculated local ceiling by {gap:.1f} percentage points. "
            f"The ceiling is structurally unachievable under current capacity and demand "
            f"conditions. This indicates a capacity gap, not a management failure."
        ))
        print()
        print(wrap(
            "Priority interventions: (1) NCTR reduction -- discharge pathway "
            "improvements are the fastest lever for reducing occupancy without new beds. "
            "(2) Demand management -- community alternatives, virtual wards, and "
            "anticipatory care can reduce avoidable admissions. "
            "(3) Capacity case -- if the gap exceeds 3 pp at the local ceiling, "
            "a formal bed base review is warranted."
        ))
    else:
        print(wrap(
            f"This trust's current occupancy is within or approaching the calculated "
            f"local ceiling. A target of {local_ceiling}% is operationally realistic "
            f"and clinically defensible given the trust's geographic position and "
            f"demand profile."
        ))

    # System-level reminder
    print()
    print(rule("-", W))
    print("  SYSTEM-LEVEL CHECK (apply separately)")
    print(rule("-", W))
    print()
    print(wrap(
        "If your ICB/ICS reports a mean acute occupancy above 85% across all "
        "trusts in the footprint, apply an additional -2 pp system stress "
        "penalty to the ceiling above. Individual trust ceilings provide no "
        "safety benefit when all regional alternatives are simultaneously full."
    ))
    print()
    print(wrap(
        "Portfolio rule: escalate to system level if three or more trusts in "
        "your footprint simultaneously exceed 90% in any given week."
    ))

    # Caveats
    print()
    print(rule("-", W))
    print("  IMPORTANT CAVEATS")
    print(rule("-", W))
    print()
    caveats = [
        ("Associations, not causes",
         "Coefficients are from a fixed-effects model (trust + time FE), not "
         "an experimentally identified causal estimate. Instrumental variables "
         "tested (estate condition, building age) were insufficiently strong "
         "for 2SLS. Treat coefficients as empirical descriptions conditional "
         "on time-invariant trust characteristics."),
        ("Proximity counts acute trusts only",
         "Mental health, community, and ambulance trusts are excluded. "
         "If nearby trusts are mainly non-acute, the geographic buffer may "
         "be overstated. Inspect the 10km/25km neighbour list manually."),
        ("Pre-COVID period showed no significant effects",
         "The occupancy-harm relationship is strongest in the post-2021 "
         "recovery period and may partially reflect simultaneous system-wide "
         "stress rather than isolated occupancy effects. Caution is warranted "
         "in interpreting the ceiling as a static long-run target."),
        ("EDR proxy",
         "A&E intensity (attendances per bed) is a rough proxy for emergency "
         "dependency. For greater precision, supply your own EDR from "
         "SUS/ECDS via --edr. Specialist/tertiary centres may have low AEI "
         "but high emergency complexity."),
    ]
    for i, (title, text) in enumerate(caveats, 1):
        print(f"  {i}. {title}")
        print(wrap(text, indent=5))
        print()

    print(rule("=", W))
    print("  Source: Bevan Briefing NHS Bed Occupancy Analysis 2025")
    print("  Data: KH03, NHS England A&E SitRep, UEC SitRep 2018-19 to 2024-25")
    print(rule("=", W))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate a locally-calibrated NHS occupancy safety ceiling."
    )
    parser.add_argument(
        "trust_code",
        nargs="?",
        help="ODS trust code (e.g. RRV, REF, RM1). "
             "Leave blank for interactive prompt.",
    )
    parser.add_argument(
        "--edr",
        type=float,
        default=None,
        metavar="0.00-1.00",
        help="Emergency dependency ratio (emergency admissions / occupied bed "
             "days) from SUS/ECDS. Optional -- if omitted, A&E intensity is "
             "used as a proxy.",
    )
    args = parser.parse_args()

    if args.trust_code:
        code = args.trust_code
    else:
        print("NHS Local Occupancy Target Calculator")
        print("--------------------------------------")
        # Show available codes
        if MASTER_PATH.exists():
            master = pd.read_csv(MASTER_PATH)
            master["org_code"] = master["org_code"].str.strip().str.upper()
            lookup = master[["org_code","org_name"]].drop_duplicates().dropna()
            lookup = lookup.sort_values("org_name")
            print(f"  {len(lookup)} trusts available in dataset.")
            print()
            q = input("  Search trust name (or press Enter to skip): ").strip()
            if q:
                matches = lookup[lookup["org_name"].str.upper().str.contains(q.upper())]
                if len(matches):
                    print()
                    for _, row in matches.iterrows():
                        print(f"    {row['org_code']:6}  {row['org_name']}")
                    print()
                else:
                    print("  No matches found.")
        code = input("  Enter ODS trust code: ").strip()
        if not code:
            sys.exit("No trust code provided.")
        if args.edr is None:
            edr_input = input("  Enter EDR (0.00-1.00) or press Enter to use auto-proxy: ").strip()
            if edr_input:
                try:
                    args.edr = float(edr_input)
                except ValueError:
                    print("  Invalid EDR -- using auto-proxy.")

    run(code, edr=args.edr)


if __name__ == "__main__":
    main()
