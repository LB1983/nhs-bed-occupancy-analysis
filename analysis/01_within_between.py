"""
01_within_between.py
====================
Mundlak-Chamberlain within-between decomposition of the occupancy-performance
relationship. Separates:

  WITHIN effect  : occupancy rising *within* a trust -> worse performance?
                   (the policy-relevant causal question)
  BETWEEN effect : trusts with structurally higher occupancy perform worse?
                   (cross-sectional confounding)

METHOD: Mundlak (1978) device.
  Augment FE model with trust-mean occupancy alongside within-deviation.
  y_it = alpha + B_within*(x_it - x_bar_i) + B_between*x_bar_i + FE + e_it

  B_within  = equivalent to FE (within-trust) estimate
  B_between = cross-sectional (between-trust) estimate

INTERPRETATION:
  B_within >> B_between, Hausman p < 0.05:
    Occupancy changes drive performance changes. Stronger causal story.
  B_between >> B_within, Hausman p < 0.05:
    High-occupancy trusts structurally different. Confounding likely.
    Prefer IV estimates as primary.
  Sign reversal between within/between:
    Classic Simpson's paradox. Report both; explain mechanism.

OUTPUTS:
  within_between_results.csv
  within_between_summary.png

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
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH    = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/nhs_occupancy.duckdb"
MAIN_TABLE = "merged_quarterly"
TRUST_ID   = "org_code"
QUARTER    = "quarter"

OCC_VARS = {
    "all_beds" : "kh03_occ_all_rate",
    "ga_beds"  : "kh03_occ_ga_rate",
}

OUTCOMES = {
    "ae_4hr_pct"    : ("ae_type1_pct_4hr",            "A&E 4-hr performance (%)"),
    "ae_12h_per_1k" : ("ae_12h_per_1k_att",           "12-hr waits per 1k attendances"),
    "canc_per_100"  : ("canc_per_100_beds",            "Cancelled ops per 100 beds"),
    "canc_pct28"    : ("canc_pct_not_treated_28_days", "% cancelled not treated 28d"),
}

CONTROLS  = ["kh03_avail_all_avg_beds"]

# Bevan Briefing palette
COL_WITHIN  = "#1BA6A6"
COL_BETWEEN = "#E24A3B"
COL_ZERO    = "#0B2D39"
BG          = "#F5F0E6"


def load_data(con):
    all_cols = list(set(
        [TRUST_ID, QUARTER, "calendar_quarter"]
        + list(OCC_VARS.values())
        + [v[0] for v in OUTCOMES.values()]
        + CONTROLS
    ))
    cols_sql = ", ".join(all_cols)
    df = con.execute(f"SELECT {cols_sql} FROM {MAIN_TABLE}").df()
    df["quarter_pd"] = pd.PeriodIndex(df["calendar_quarter"], freq="Q")
    return df.sort_values([TRUST_ID, "quarter_pd"]).reset_index(drop=True)


def mundlak_decomposition(df, occ_col, outcome_col, controls):
    subset = [TRUST_ID, occ_col, outcome_col] + controls
    d = df[subset].dropna()
    if len(d) < 100:
        return None

    d = d.copy()
    d["occ_mean"]   = d.groupby(TRUST_ID)[occ_col].transform("mean")
    d["occ_within"] = d[occ_col] - d["occ_mean"]
    d["occ_within_z"] = (d["occ_within"] - d["occ_within"].mean()) / d["occ_within"].std()
    d["occ_mean_z"]   = (d["occ_mean"]   - d["occ_mean"].mean())   / d["occ_mean"].std()

    ctrl = (" + " + " + ".join(controls)) if controls else ""
    formula = (f"{outcome_col} ~ occ_within_z + occ_mean_z"
               f"{ctrl} + C({TRUST_ID})")
    try:
        m = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[TRUST_ID]}
        )
    except Exception as e:
        return {"error": str(e)}

    def ex(name):
        if name not in m.params:
            return np.nan, np.nan, np.nan
        return m.params[name], m.bse[name], m.pvalues[name]

    cw, sw, pw = ex("occ_within_z")
    cb, sb, pb = ex("occ_mean_z")

    diff     = cw - cb
    se_diff  = np.sqrt(max(sw**2 - sb**2, 1e-12))
    h_stat   = diff / se_diff
    h_p      = 2 * (1 - stats.norm.cdf(abs(h_stat)))

    return {
        "occ_col"      : occ_col,
        "outcome_col"  : outcome_col,
        "n_obs"        : len(d),
        "n_trusts"     : d[TRUST_ID].nunique(),
        "coef_within"  : cw, "se_within"  : sw, "p_within"  : pw,
        "coef_between" : cb, "se_between" : sb, "p_between" : pb,
        "hausman_stat" : h_stat, "hausman_p" : h_p,
        "r2"           : m.rsquared,
    }


def sig_stars(p):
    if   pd.isna(p): return ""
    elif p < 0.001:  return "***"
    elif p < 0.01:   return "**"
    elif p < 0.05:   return "*"
    elif p < 0.10:   return "†"
    return ""


def print_table(results):
    print("\n" + "="*90)
    print("WITHIN-BETWEEN DECOMPOSITION (Mundlak device)")
    print("Standardised coefficients. SE clustered by trust.")
    print("† p<0.10  * p<0.05  ** p<0.01  *** p<0.001")
    print("-"*90)
    print(f"{'Outcome':<35} {'Beds':<8} {'Within β':>10} {'SE':>7} "
          f"{'Between β':>10} {'SE':>7} {'Hausman p':>10}")
    print("-"*90)
    for r in results:
        if not r or "error" in r:
            continue
        bl  = "All" if "all" in r["occ_col"] else "G&A"
        w   = f"{r['coef_within']:+.4f}{sig_stars(r['p_within'])}"
        b   = f"{r['coef_between']:+.4f}{sig_stars(r['p_between'])}"
        print(f"{r['outcome_col'][:34]:<35} {bl:<8} {w:>10} {r['se_within']:>7.4f} "
              f"{b:>10} {r['se_between']:>7.4f} {r['hausman_p']:>10.4f}")
    print("="*90)


def plot_coefficients(results, output_path="within_between_summary.png"):
    recs = [r for r in results if r and "error" not in r
            and not np.isnan(r.get("coef_within", np.nan))]
    if not recs:
        return

    n      = len(recs)
    y_pos  = np.arange(n)
    labels = [f"{r['outcome_col'].replace('_',' ')}\n"
              f"({'All' if 'all' in r['occ_col'] else 'G&A'} beds)"
              for r in recs]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.75)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for i, r in enumerate(recs):
        ax.errorbar(r["coef_within"],  y_pos[i]+0.15,
                    xerr=1.96*r["se_within"],  fmt="o",
                    color=COL_WITHIN,  markersize=7, capsize=4,
                    linewidth=1.5, label="Within-trust"  if i==0 else "")
        ax.errorbar(r["coef_between"], y_pos[i]-0.15,
                    xerr=1.96*r["se_between"], fmt="s",
                    color=COL_BETWEEN, markersize=7, capsize=4,
                    linewidth=1.5, label="Between-trust" if i==0 else "")

    ax.axvline(0, color=COL_ZERO, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Standardised coefficient (1 SD occupancy → outcome)", fontsize=10)
    ax.set_title("Within- vs Between-trust Estimates\n"
                 "Mundlak decomposition, trust & time FE, clustered SE",
                 fontsize=11, fontweight="bold", pad=12, color=COL_ZERO)
    handles = [mpatches.Patch(color=COL_WITHIN,  label="Within-trust (FE)"),
               mpatches.Patch(color=COL_BETWEEN, label="Between-trust (cross-sectional)")]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              facecolor=BG, edgecolor=COL_ZERO)
    for sp in ax.spines.values():
        sp.set_color(COL_ZERO)
    ax.tick_params(colors=COL_ZERO)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    df  = load_data(con)
    con.close()

    print(f"Loaded {len(df):,} trust-quarter observations")
    print(f"Trusts: {df[TRUST_ID].nunique()}  |  Quarters: {df[QUARTER].nunique()}")

    available_controls = [c for c in CONTROLS if c in df.columns]
    all_results = []

    for occ_label, occ_col in OCC_VARS.items():
        if occ_col not in df.columns:
            print(f"\nSkipping {occ_col} — not found in {MAIN_TABLE}")
            continue
        for out_label, (out_col, out_desc) in OUTCOMES.items():
            if out_col not in df.columns:
                print(f"  Skipping {out_col} — not found")
                continue
            r = mundlak_decomposition(df, occ_col, out_col, available_controls)
            if r and "error" not in r:
                all_results.append(r)
                print(f"  {occ_label:10} {out_desc[:30]:30} "
                      f"W={r['coef_within']:+.4f}{sig_stars(r['p_within'])}  "
                      f"B={r['coef_between']:+.4f}{sig_stars(r['p_between'])}  "
                      f"Hausman p={r['hausman_p']:.4f}")
            else:
                err = r.get("error","?") if r else "None returned"
                print(f"  ERR {occ_label} {out_label}: {err}")

    print_table(all_results)
    plot_coefficients(all_results)
    pd.DataFrame(all_results).to_csv("within_between_results.csv", index=False)
    print("Saved: within_between_results.csv")
    print("\nDone. Run 02_covid_subperiods.py next.")


if __name__ == "__main__":
    main()
