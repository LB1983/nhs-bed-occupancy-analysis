"""
02_covid_subperiods.py
======================
Run occupancy-performance models separately across three structural periods
to test whether the relationship is stable or COVID-distorted.

PERIODS:
    PRE-COVID  : up to 2019 Q4
                 Cleanest identification. Primary analysis period.
    COVID      : 2020 Q1 - 2021 Q2
                 Structural break. Elective suspension, A&E collapse for
                 non-occupancy reasons. Coefficients not causally interpreted.
    RECOVERY   : 2021 Q3 onwards
                 Post-COVID 'new normal'. Use as robustness check.

FOR EACH PERIOD x OUTCOME x BED TYPE:
    1. FE regression (trust + time FE, clustered SE)
    2. 85% threshold model
    3. Capacity interaction model

NOTE: COVID-period occupancy distribution is very different (mean ~70% vs
normal ~85%). Report COVID coefficients but do not interpret causally —
the period violates the identifying assumption that occupancy variation
is driven by demand/capacity balance rather than policy-mandated suspension.

OUTPUTS:
    covid_subperiod_results.csv
    covid_subperiod_plot.png
    period_occupancy_summary.csv

REQUIRES: pip install linearmodels statsmodels matplotlib --break-system-packages
"""

import warnings
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH    = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/nhs_occupancy.duckdb"
MAIN_TABLE = "merged_quarterly"
TRUST_ID   = "org_code"
QUARTER    = "quarter"

PERIODS = {
    "pre_covid" : ("Pre-COVID",  None,      "2019Q4"),
    "covid"     : ("COVID",      "2020Q1",  "2021Q2"),
    "recovery"  : ("Recovery",   "2021Q3",  None),
}

OCC_VARS = {
    "all_beds" : "kh03_occ_all_rate",
    "ga_beds"  : "kh03_occ_ga_rate",
}

OUTCOMES = {
    "ae_4hr_pct"    : ("ae_type1_pct_4hr",            "A&E 4-hr performance"),
    "ae_12h_per_1k" : ("ae_12h_per_1k_att",           "12-hr waits per 1k att"),
    "canc_per_100"  : ("canc_per_100_beds",            "Cancelled ops per 100 beds"),
    "canc_pct28"    : ("canc_pct_not_treated_28_days", "% cancelled not treated 28d"),
}

CONTROLS = ["kh03_avail_all_avg_beds"]

PERIOD_COLOURS = {
    "pre_covid" : "#1BA6A6",
    "covid"     : "#E24A3B",
    "recovery"  : "#F4A724",
}
BG     = "#F5F0E6"
NAVY   = "#0B2D39"


def load_data(con):
    all_cols = list(set(
        [TRUST_ID, QUARTER, "calendar_quarter"]
        + list(OCC_VARS.values())
        + [v[0] for v in OUTCOMES.values()]
        + CONTROLS
    ))
    df = con.execute(
        f"SELECT {', '.join(all_cols)} FROM {MAIN_TABLE}"
    ).df()
    df["quarter_pd"] = pd.PeriodIndex(df["calendar_quarter"], freq="Q")
    return df.sort_values([TRUST_ID, "quarter_pd"]).reset_index(drop=True)


def filter_period(df, start, end):
    d = df.copy()
    if start:
        d = d[d["quarter_pd"] >= pd.Period(start, freq="Q")]
    if end:
        d = d[d["quarter_pd"] <= pd.Period(end,   freq="Q")]
    return d


def run_fe_model(df, occ_col, outcome_col, controls, threshold=None,
                 capacity_col="kh03_avail_all_avg_beds"):
    """FE regression with optional threshold dummy or capacity interaction."""
    subset = [TRUST_ID, QUARTER, occ_col, outcome_col] + controls
    d = df[subset].dropna()
    if len(d) < 50:
        return None

    d = d.copy()
    d["occ_z"] = (d[occ_col] - d[occ_col].mean()) / d[occ_col].std()

    ctrl = (" + " + " + ".join(controls)) if controls else ""

    if threshold is not None:
        d["above_thresh"] = (d[occ_col] >= threshold).astype(float)
        formula = f"{outcome_col} ~ above_thresh{ctrl} + C({TRUST_ID}) + C({QUARTER})"
    else:
        formula = f"{outcome_col} ~ occ_z{ctrl} + C({TRUST_ID}) + C({QUARTER})"

    try:
        m = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[TRUST_ID]}
        )
    except Exception as e:
        return {"error": str(e)}

    key = "above_thresh" if threshold is not None else "occ_z"
    if key not in m.params:
        return None

    return {
        "coef"   : m.params[key],
        "se"     : m.bse[key],
        "pval"   : m.pvalues[key],
        "n_obs"  : len(d),
        "n_trust": d[TRUST_ID].nunique(),
        "r2"     : m.rsquared,
    }


def sig_stars(p):
    if pd.isna(p):  return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    if p < 0.10:    return "†"
    return ""


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    df  = load_data(con)
    con.close()

    print(f"Loaded {len(df):,} obs | {df[TRUST_ID].nunique()} trusts | "
          f"{df[QUARTER].nunique()} quarters")

    available_controls = [c for c in CONTROLS if c in df.columns]

    # ── Period occupancy summary ───────────────────────────────────────────────
    occ_summary_rows = []
    for pid, (plabel, pstart, pend) in PERIODS.items():
        d = filter_period(df, pstart, pend)
        for occ_label, occ_col in OCC_VARS.items():
            if occ_col not in d.columns:
                continue
            s = d[occ_col].dropna()
            occ_summary_rows.append({
                "period"      : pid,
                "period_label": plabel,
                "occ_var"     : occ_label,
                "n_obs"       : len(s),
                "mean"        : s.mean(),
                "median"      : s.median(),
                "sd"          : s.std(),
                "pct10"       : s.quantile(0.10),
                "pct90"       : s.quantile(0.90),
            })

    occ_summary = pd.DataFrame(occ_summary_rows)
    occ_summary.to_csv("period_occupancy_summary.csv", index=False)
    print("\n── Period occupancy summary ──")
    print(occ_summary[["period","occ_var","n_obs","mean","sd"]].round(3).to_string(index=False))

    # ── Main models ────────────────────────────────────────────────────────────
    all_results = []
    for pid, (plabel, pstart, pend) in PERIODS.items():
        d = filter_period(df, pstart, pend)
        print(f"\n── {plabel.upper()} (n={len(d):,}, "
              f"{d[TRUST_ID].nunique()} trusts) ──")

        for occ_label, occ_col in OCC_VARS.items():
            if occ_col not in d.columns:
                continue
            for out_label, (out_col, out_desc) in OUTCOMES.items():
                if out_col not in d.columns:
                    continue

                # Model 1: continuous FE
                r1 = run_fe_model(d, occ_col, out_col, available_controls)
                # Model 2: 85% threshold
                r2 = run_fe_model(d, occ_col, out_col, available_controls, threshold=85.0)

                for model_type, r in [("continuous", r1), ("threshold_85", r2)]:
                    if r and "error" not in r:
                        row = {
                            "period"      : pid,
                            "period_label": plabel,
                            "occ_var"     : occ_label,
                            "occ_col"     : occ_col,
                            "outcome"     : out_label,
                            "outcome_col" : out_col,
                            "outcome_desc": out_desc,
                            "model"       : model_type,
                            **r,
                        }
                        all_results.append(row)
                        if model_type == "continuous":
                            print(f"  {occ_label:8} {out_desc[:28]:28} "
                                  f"β={r['coef']:+.4f}{sig_stars(r['pval'])} "
                                  f"(SE={r['se']:.4f})")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("covid_subperiod_results.csv", index=False)
    print("\nSaved: covid_subperiod_results.csv")

    # ── Plot: coefficient comparison across periods ────────────────────────────
    cont = results_df[results_df["model"] == "continuous"].copy()
    combos = cont[["occ_var","outcome","outcome_desc"]].drop_duplicates()

    n_panels = len(combos)
    if n_panels == 0:
        print("No results to plot.")
        return

    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3.5))
    fig.patch.set_facecolor(BG)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (_, combo) in enumerate(combos.iterrows()):
        ax  = axes_flat[idx]
        ax.set_facecolor(BG)
        sub = cont[
            (cont["occ_var"] == combo["occ_var"]) &
            (cont["outcome"] == combo["outcome"])
        ]

        period_order = ["pre_covid","covid","recovery"]
        x_pos = np.arange(len(period_order))

        for xi, pid in enumerate(period_order):
            row = sub[sub["period"] == pid]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            color = PERIOD_COLOURS.get(pid, "#888888")
            ax.bar(xi, row["coef"], color=color, alpha=0.8, width=0.6)
            ax.errorbar(xi, row["coef"],
                        yerr=1.96*row["se"],
                        fmt="none", color=NAVY, capsize=4, linewidth=1.5)
            ax.text(xi, row["coef"] + (0.02 if row["coef"] >= 0 else -0.02),
                    sig_stars(row["pval"]), ha="center", va="bottom",
                    fontsize=10, color=NAVY)

        ax.axhline(0, color=NAVY, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Pre-COVID","COVID","Recovery"], fontsize=8)
        ax.set_title(f"{combo['outcome_desc']}\n({combo['occ_var']} beds)",
                     fontsize=9, fontweight="bold", color=NAVY)
        ax.tick_params(colors=NAVY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(NAVY)

    # Hide unused panels
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=PERIOD_COLOURS["pre_covid"], label="Pre-COVID"),
        mpatches.Patch(color=PERIOD_COLOURS["covid"],     label="COVID"),
        mpatches.Patch(color=PERIOD_COLOURS["recovery"],  label="Recovery"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, facecolor=BG, edgecolor=NAVY, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Occupancy-Performance Coefficients by Period\n"
                 "Trust + time FE, clustered SE. † p<0.10  * p<0.05  ** p<0.01  *** p<0.001",
                 fontsize=10, fontweight="bold", color=NAVY, y=1.01)
    plt.tight_layout()
    plt.savefig("covid_subperiod_plot.png", dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Saved: covid_subperiod_plot.png")
    print("\nDone. Run 03_iv_2sls.py next.")


if __name__ == "__main__":
    main()
