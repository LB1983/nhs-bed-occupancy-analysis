#!/usr/bin/env python
"""
85% Bed Occupancy Target Stress Test — Trust-quarter panel
=========================================================

Uses: 03-data-final/master-quarterly-trust.csv (MINIMAL MASTER)

What this script produces
-------------------------
1) Threshold models: outcome ~ I(occ >= t) + FE(trust, quarter)
2) Piecewise/spline models: allow slope to change around 80/85/90
3) Capacity interaction models: effect of high occupancy depends on bed base
   (the point of your argument: smaller bed base => lower 'safe' occupancy)
4) Outputs tidy CSVs for your write-up

Run (repo root)
---------------
python 04-analysis/05a-regressions_occupancy_target_stress_test.py

Dependencies
------------
pip install pandas numpy statsmodels linearmodels

Notes
-----
- Occupancy measures are in *rates* (0–1) and average bed base is avg beds.
- Outcomes are already in the master file (AE / cancelled ops / NCTR).
- We use two-way fixed effects (trust + time) with cluster-robust SE by trust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MASTER_PATH = Path("C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/03-data-final/master-quarterly-trust.csv")
OUT_DIR = Path("C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/05-outputs")

THRESHOLDS = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92]
SPLINE_KNOTS: Tuple[float, ...] = (0.80, 0.85, 0.90)

# Bed bases (four variants you asked for)
BED_BASES: Dict[str, Dict[str, str]] = {
    "all": {
        "occ": "kh03_occ_all_rate",
        "cap": "kh03_avail_all_avg_beds",
    },
    "ga": {
        "occ": "kh03_occ_ga_rate",
        "cap": "kh03_avail_ga_avg_beds",
    },
    "acute_proxy": {
        "occ": "kh03_occ_acute_rate",
        "cap": "kh03_avail_acute_avg_beds",
    },
    "adult_ga_proxy": {
        "occ": "kh03_occ_adult_ga_rate",
        "cap": "kh03_avail_ga_avg_beds",  # capacity proxy still G&A beds
    },
}

# Outcomes (edit names here if you add more datasets later)
OUTCOMES: Dict[str, str] = {
    "ae_type1_pct_4hr": "ae_type1_pct_4hr",
    "ae_waits_12hr_dta": "ae_waits_12hr_decision_to_admit",
    "canc_cancelled_ops": "canc_cancelled_ops",
    "canc_breaches_28d": "canc_breaches_28d",
    "canc_pct_not_treated_28d": "canc_pct_not_treated_28_days",
    "uec_nctr_mean": "uec_uec_nctr_mean",
    "uec_nctr_max": "uec_uec_nctr_max",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_time_index(period: pd.Series) -> pd.Series:
    """Map 'YYYY-YY Qn' to a sortable integer time index."""
    p = period.astype(str).str.strip()
    start_year = p.str.slice(0, 4).astype(int)
    q = p.str.extract(r"Q([1-4])")[0].astype(int)
    return start_year * 10 + q  # e.g. 20181 for 2018-19 Q1


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def spline_terms(x: pd.Series, knots: Tuple[float, ...]) -> pd.DataFrame:
    x = safe_numeric(x)
    out = {"occ": x}
    for k in knots:
        out[f"occ_gt_{int(k*100)}"] = (x - k).clip(lower=0)
    return pd.DataFrame(out)


def run_fe(df: pd.DataFrame, y_col: str, X: pd.DataFrame) -> PanelOLS:
    tmp = df[["org_code", "t_index", y_col]].join(X)
    tmp = tmp.dropna(subset=[y_col] + list(X.columns)).copy()
    tmp = tmp.set_index(["org_code", "t_index"]).sort_index()

    y = tmp[y_col]
    Xmat = tmp[X.columns]

    model = PanelOLS(y, Xmat, entity_effects=True, time_effects=True)
    res = model.fit(cov_type="clustered", cluster_entity=True)
    return res


def summarise_result(res, terms: List[str]) -> Dict[str, float]:
    d = {"nobs": int(res.nobs)}
    for t in terms:
        d[f"coef_{t}"] = float(res.params.get(t, np.nan))
        d[f"se_{t}"] = float(res.std_errors.get(t, np.nan))
        d[f"p_{t}"] = float(res.pvalues.get(t, np.nan))
    return d


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_PATH}")

    df = pd.read_csv(MASTER_PATH)
    df["org_code"] = df["org_code"].astype(str).str.strip().str.upper()
    df["period"] = df["period"].astype(str).str.strip()
    df["t_index"] = make_time_index(df["period"])

    # numeric casting — skip missing columns gracefully
    missing_outcomes = [y for y in OUTCOMES.values() if y not in df.columns]
    if missing_outcomes:
        print(f"  Skipping missing outcome columns: {missing_outcomes}")
    OUTCOMES_AVAIL = {k: v for k, v in OUTCOMES.items() if v in df.columns}
    for y in OUTCOMES_AVAIL.values():
        df[y] = safe_numeric(df[y])

    missing_beds = [c for bb in BED_BASES.values() for c in bb.values()
                    if c not in df.columns]
    if missing_beds:
        print(f"  Skipping missing bed-base columns: {missing_beds}")
    BED_BASES_AVAIL = {k: v for k, v in BED_BASES.items()
                       if all(c in df.columns for c in v.values())}
    for bb in BED_BASES_AVAIL.values():
        for c in bb.values():
            df[c] = safe_numeric(df[c])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # quick sanity dump
    sanity_cols = [
        "org_code", "org_name", "period",
        *[BED_BASES[k]["occ"] for k in BED_BASES],
        *[BED_BASES[k]["cap"] for k in BED_BASES],
        *list(OUTCOMES.values()),
    ]
    sanity_cols = [c for c in sanity_cols if c in df.columns]
    df[sanity_cols].head(500).to_csv(OUT_DIR / "sample_master_for_regression.csv", index=False)

    # ------------------------------------------------------------------
    # 1) Threshold models (per bed-base, per outcome)
    # ------------------------------------------------------------------
    thresh_rows = []
    for bed_base, spec in BED_BASES_AVAIL.items():
        occ = spec["occ"]
        for out_name, y_col in OUTCOMES_AVAIL.items():
            for t in THRESHOLDS:
                term = f"above_{int(t*100)}"
                X = pd.DataFrame({term: (df[occ] >= t).astype(float)})
                res = run_fe(df, y_col, X)
                row = {
                    "bed_base": bed_base,
                    "occ_col": occ,
                    "outcome": out_name,
                    "y_col": y_col,
                    "threshold": t,
                    **summarise_result(res, [term]),
                }
                thresh_rows.append(row)

    pd.DataFrame(thresh_rows).to_csv(OUT_DIR / "models_threshold.csv", index=False)

    # ------------------------------------------------------------------
    # 2) Spline (piecewise linear) models
    # ------------------------------------------------------------------
    spline_rows = []
    for bed_base, spec in BED_BASES_AVAIL.items():
        occ = spec["occ"]
        Xs = spline_terms(df[occ], knots=SPLINE_KNOTS)
        terms = list(Xs.columns)
        for out_name, y_col in OUTCOMES_AVAIL.items():
            res = run_fe(df, y_col, Xs)
            row = {
                "bed_base": bed_base,
                "occ_col": occ,
                "outcome": out_name,
                "y_col": y_col,
                "knots": ",".join(map(str, SPLINE_KNOTS)),
                **summarise_result(res, terms),
            }
            spline_rows.append(row)

    pd.DataFrame(spline_rows).to_csv(OUT_DIR / "models_spline.csv", index=False)

    # ------------------------------------------------------------------
    # 3) Capacity interaction models
    #     outcome ~ occ + cap + occ*cap  (FE)
    #
    # Interpretation:
    # - If beta(occ*cap) is negative for 'bad outcomes' (e.g. 12h waits),
    #   then high occupancy harms more when capacity is LOW (small bed base).
    #   (Because cap is larger => interaction dampens harm.)
    #
    # We also run a threshold*capacity model at 85% specifically.
    # ------------------------------------------------------------------
    inter_rows = []
    for bed_base, spec in BED_BASES_AVAIL.items():
        occ = spec["occ"]
        cap = spec["cap"]

        # scale cap for numerical stability
        cap_scaled = (df[cap] - df[cap].mean()) / df[cap].std(ddof=0)
        Xlin = pd.DataFrame({
            "occ": df[occ],
            "cap_z": cap_scaled,
            "occ_x_cap": df[occ] * cap_scaled,
        })
        for out_name, y_col in OUTCOMES_AVAIL.items():
            res = run_fe(df, y_col, Xlin)
            row = {
                "bed_base": bed_base,
                "occ_col": occ,
                "cap_col": cap,
                "outcome": out_name,
                "y_col": y_col,
                **summarise_result(res, ["occ", "cap_z", "occ_x_cap"]),
            }
            inter_rows.append(row)

        # Threshold*capacity at 85%
        thr = 0.85
        Xthr = pd.DataFrame({
            "above_85": (df[occ] >= thr).astype(float),
            "cap_z": cap_scaled,
            "above85_x_cap": (df[occ] >= thr).astype(float) * cap_scaled,
        })
        for out_name, y_col in OUTCOMES_AVAIL.items():
            res = run_fe(df, y_col, Xthr)
            row = {
                "bed_base": bed_base,
                "occ_col": occ,
                "cap_col": cap,
                "outcome": out_name,
                "y_col": y_col,
                "threshold": thr,
                **summarise_result(res, ["above_85", "cap_z", "above85_x_cap"]),
            }
            inter_rows.append(row)

    pd.DataFrame(inter_rows).to_csv(OUT_DIR / "models_capacity_interactions.csv", index=False)

    print("✅ Regression run complete.")
    print(f"Outputs: {OUT_DIR.resolve()}")
    print("Key files:")
    print(" - models_threshold.csv") 
    print(" - models_spline.csv") 
    print(" - models_capacity_interactions.csv") 
    print(" - sample_master_for_regression.csv")


if __name__ == "__main__":
    main()
