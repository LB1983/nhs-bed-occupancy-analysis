"""
03_iv_2sls.py
=============
Two-stage least squares (2SLS) estimation of the causal effect of bed
occupancy on NHS performance outcomes, using ERIC estate condition as
instruments.

INSTRUMENTS:
    Z1: log_backlog_hs_per_m2
        Log of (high + significant risk backlog £ / GIA m²).
        Estate at imminent risk of failure -> unplanned ward closures ->
        occupancy rises on remaining stock.

    Z2: pct_pre1985
        % of floor area built before 1985.
        Predetermined construction era -> higher depreciation rate ->
        more backlog generation. Exogenous to current management.

INSTRUMENT VALIDITY:
    Relevance: Tested via first-stage F-statistics (target >10 per instrument,
               Kleibergen-Paap rk F >10 for joint relevance).
    Exclusion: Estate condition affects A&E waits/cancellations ONLY through
               the occupancy channel, conditional on trust FE, time FE, and
               trust size. No direct clinical pathway from backlog to outcomes.
    Exogeneity: Building age is historically determined (pre-1985 stock
               reflects 1948-1985 construction decisions). Backlog is a slow-
               moving stock variable — current-year fluctuations in performance
               cannot cause backlog to appear or disappear.

DIAGNOSTICS REPORTED:
    First stage:
        F-stat per instrument (target > 10; Stock & Yogo 2005 critical values)
        Kleibergen-Paap rk F (joint; robust to heteroskedasticity/clustering)
        Partial R² per instrument
    Endogeneity:
        Durbin-Wu-Hausman test (H0: OLS consistent, i.e. no endogeneity)
        If p > 0.10: fail to reject, OLS preferred; report both
        If p < 0.05: reject, endogeneity confirmed, 2SLS preferred
    Overidentification (when both instruments used):
        Sargan-Hansen J test (H0: instruments valid)
        If p < 0.05: at least one instrument violates exclusion restriction

MODELS RUN:
    For each outcome × bed base:
        M1: OLS (FE, clustered SE) — benchmark
        M2: 2SLS with Z1 only
        M3: 2SLS with Z2 only
        M4: 2SLS with Z1 + Z2 (overidentified — enables Sargan-Hansen J test)

SAMPLE RESTRICTION:
    Pre-COVID only (up to 2019 Q4) for primary analysis.
    Full sample as robustness check.

OUTPUTS:
    iv_first_stage.csv    — first-stage coefficients and diagnostics
    iv_second_stage.csv   — 2SLS vs OLS comparison
    iv_diagnostics.csv    — all test statistics
    iv_summary_plot.png   — OLS vs 2SLS coefficient comparison

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

try:
    from linearmodels.iv import IV2SLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("WARNING: linearmodels not installed. Running manual 2SLS only.")
    print("Install: pip install linearmodels --break-system-packages")

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH         = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/nhs_occupancy.duckdb"
MAIN_TABLE      = "merged_quarterly"
ERIC_TABLE      = "eric_instruments"
TRUST_ID        = "org_code"
QUARTER         = "quarter"
PRE_COVID_END   = "2019Q4"

OCC_VARS = {
    "all_beds" : "kh03_occ_all_rate",
    "ga_beds"  : "kh03_occ_ga_rate",
}

OUTCOMES = {
    "ae_4hr_pct"    : ("ae_type1_pct_4hr",            "A&E 4-hr performance (%)"),
    "ae_12h_per_1k" : ("ae_12h_per_1k_att",           "12-hr waits per 1k att"),
    "canc_per_100"  : ("canc_per_100_beds",            "Cancelled ops per 100 beds"),
    "canc_pct28"    : ("canc_pct_not_treated_28_days", "% cancelled not treated 28d"),
}

INSTRUMENTS  = ["log_backlog_hs_per_m2", "pct_pre1985",
                "backlog_high_per_m2", "flu_severity_region"]
CONTROLS     = ["kh03_avail_all_avg_beds"]

COL_OLS  = "#0B2D39"
COL_2SLS = "#1BA6A6"
BG       = "#F5F0E6"
NAVY     = "#0B2D39"


def load_data(con):
    occ_cols  = list(OCC_VARS.values())
    out_cols  = [v[0] for v in OUTCOMES.values()]
    inst_cols = INSTRUMENTS
    ctrl_cols = CONTROLS

    # Merge occupancy/performance with ERIC instruments
    main_cols = ", ".join(
        [f"m.{TRUST_ID}", f"m.{QUARTER}", "m.calendar_quarter"]
        + [f"m.{c}" for c in occ_cols + out_cols + ctrl_cols if c]
    )
    # ERIC instruments — include backlog_high_per_m2 alongside original cols
    eric_fetch = list(set(
        [c for c in inst_cols if c != "flu_severity_region"]
        + ["backlog_high_per_m2"]
    ))
    inst_cols_sql = ", ".join([f"e.{c}" for c in eric_fetch])

    try:
        df = con.execute(f"""
            SELECT {main_cols}, {inst_cols_sql}
            FROM {MAIN_TABLE} m
            LEFT JOIN {ERIC_TABLE} e
              ON m.{TRUST_ID}        = e.{TRUST_ID}
             AND m.calendar_quarter  = TRIM(e.{QUARTER})
        """).df()
    except Exception:
        # Fallback if eric_instruments table doesn't exist yet
        print("WARNING: eric_instruments table not found. Run 00_clean_eric.py first.")
        df = con.execute(f"SELECT * FROM {MAIN_TABLE}").df()
        for inst in INSTRUMENTS:
            if inst not in df.columns:
                df[inst] = np.nan

    # Join flu instrument if available
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    if "flu_instrument" in tables:
        flu = con.execute("SELECT * FROM flu_instrument").df()
        df = df.merge(flu, on=[TRUST_ID, "calendar_quarter"], how="left")
        cov = df["flu_severity_region"].notna().mean()
        print(f"  flu_severity_region joined: {cov:.1%} coverage")
    else:
        df["flu_severity_region"] = np.nan
        print("  flu_instrument not found — run 00_build_flu_instrument.py to add it")

    df["quarter_pd"] = pd.PeriodIndex(df["calendar_quarter"], freq="Q")
    return df.sort_values([TRUST_ID, "quarter_pd"]).reset_index(drop=True)


def add_fe_dummies(df, trust_col, quarter_col):
    """Demean for within-trust, within-time FE (faster than dummy cols)."""
    d = df.copy()
    numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    # Trust demeaning
    d[numeric_cols] = d[numeric_cols] - d.groupby(trust_col)[numeric_cols].transform("mean")
    # Time demeaning
    d[numeric_cols] = d[numeric_cols] - d.groupby(quarter_col)[numeric_cols].transform("mean")
    return d


def run_ols(df, occ_col, outcome_col, controls):
    """OLS with trust and time FE, clustered SE. Returns coef, se, pval, n."""
    subset = [TRUST_ID, QUARTER, occ_col, outcome_col] + controls
    d = df[subset].dropna()
    if len(d) < 100:
        return None
    d = d.copy()
    d["occ_std"] = (d[occ_col] - d[occ_col].mean()) / d[occ_col].std()
    ctrl = (" + " + " + ".join(controls)) if controls else ""
    formula = f"{outcome_col} ~ occ_std{ctrl} + C({TRUST_ID}) + C({QUARTER})"
    try:
        m = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[TRUST_ID]}
        )
        return {
            "coef": m.params.get("occ_std", np.nan),
            "se"  : m.bse.get(  "occ_std", np.nan),
            "pval": m.pvalues.get("occ_std", np.nan),
            "n"   : len(d),
            "r2"  : m.rsquared,
        }
    except Exception as e:
        return {"error": str(e)}


def manual_2sls(df, occ_col, outcome_col, instruments, controls, trust_col, quarter_col):
    """
    Manual 2SLS via two OLS regressions.
    First stage: occ ~ instruments + controls + FE
    Second stage: outcome ~ occ_hat + controls + FE
    Returns first-stage diagnostics + second-stage coefficient.
    """
    avail_inst = [z for z in instruments if z in df.columns
                  and df[z].notna().sum() > 50]
    if not avail_inst:
        return None, None

    subset = [trust_col, quarter_col, occ_col, outcome_col] + controls + avail_inst
    d = df[subset].dropna()
    if len(d) < 100:
        return None, None

    d = d.copy()
    # Standardise occupancy for interpretable coefficients
    occ_mean = d[occ_col].mean()
    occ_sd   = d[occ_col].std()
    d["occ_std"] = (d[occ_col] - occ_mean) / occ_sd

    ctrl = (" + " + " + ".join(controls)) if controls else ""
    inst_str = " + ".join(avail_inst)

    # ── First stage ────────────────────────────────────────────────────────────
    fs_formula = f"occ_std ~ {inst_str}{ctrl} + C({trust_col}) + C({quarter_col})"
    try:
        fs = smf.ols(fs_formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[trust_col]}
        )
    except Exception as e:
        return {"error": f"First stage: {e}"}, None

    d["occ_hat"] = fs.fittedvalues

    # First-stage diagnostics
    fs_diagnostics = {}
    for inst in avail_inst:
        if inst in fs.params:
            t = fs.params[inst] / fs.bse[inst]
            fs_diagnostics[f"fs_coef_{inst}"]  = fs.params[inst]
            fs_diagnostics[f"fs_se_{inst}"]    = fs.bse[inst]
            fs_diagnostics[f"fs_pval_{inst}"]  = fs.pvalues[inst]
            fs_diagnostics[f"fs_tstat_{inst}"] = t
            fs_diagnostics[f"fs_f_{inst}"]     = t**2  # approx F per instrument

    # Joint F-stat (robust, approximate)
    inst_params = [fs.params.get(z, 0) for z in avail_inst]
    inst_vcov   = fs.cov_params().loc[
        [z for z in avail_inst if z in fs.cov_params().index],
        [z for z in avail_inst if z in fs.cov_params().index]
    ]
    try:
        k = len(inst_vcov)
        v = np.array(inst_params[:k])
        V = inst_vcov.values
        kp_f = float(v @ np.linalg.solve(V, v) / k)
    except Exception:
        kp_f = np.nan
    fs_diagnostics["kp_f_stat"] = kp_f
    fs_diagnostics["n_instruments"] = len(avail_inst)
    fs_diagnostics["fs_r2"]     = fs.rsquared
    fs_diagnostics["instruments_used"] = "|".join(avail_inst)

    # ── Second stage ───────────────────────────────────────────────────────────
    ss_formula = f"{outcome_col} ~ occ_hat{ctrl} + C({trust_col}) + C({quarter_col})"
    try:
        ss = smf.ols(ss_formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[trust_col]}
        )
    except Exception as e:
        return fs_diagnostics, {"error": f"Second stage: {e}"}

    coef_2sls = ss.params.get("occ_hat", np.nan)
    se_2sls   = ss.bse.get(  "occ_hat", np.nan)

    # ── Durbin-Wu-Hausman (endogeneity test) ───────────────────────────────────
    # Add first-stage residuals to second stage; test their significance
    d["fs_resid"] = fs.resid
    dwh_formula = (f"{outcome_col} ~ occ_std + fs_resid{ctrl}"
                   f" + C({trust_col}) + C({quarter_col})")
    try:
        dwh = smf.ols(dwh_formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[trust_col]}
        )
        dwh_coef  = dwh.params.get("fs_resid", np.nan)
        dwh_se    = dwh.bse.get(  "fs_resid", np.nan)
        dwh_pval  = dwh.pvalues.get("fs_resid", np.nan)
    except Exception:
        dwh_coef = dwh_se = dwh_pval = np.nan

    # ── Sargan-Hansen J (overidentification — only valid with 2 instruments) ──
    sargan_j = np.nan
    sargan_p = np.nan
    if len(avail_inst) > 1:
        d["ss_resid"] = d[outcome_col] - ss.predict(d)
        sargan_formula = (f"ss_resid ~ {inst_str}{ctrl}"
                          f" + C({trust_col}) + C({quarter_col})")
        try:
            sargan_m = smf.ols(sargan_formula, data=d).fit()
            sargan_j = sargan_m.rsquared * len(d)
            sargan_p = 1 - stats.chi2.cdf(sargan_j, df=len(avail_inst)-1)
        except Exception:
            pass

    ss_results = {
        "coef_2sls"  : coef_2sls,
        "se_2sls"    : se_2sls,
        "pval_2sls"  : 2 * (1 - stats.t.cdf(abs(coef_2sls / se_2sls), df=len(d)-1))
                       if not np.isnan(coef_2sls) else np.nan,
        "n"          : len(d),
        "n_trusts"   : d[trust_col].nunique(),
        "dwh_coef"   : dwh_coef,
        "dwh_se"     : dwh_se,
        "dwh_pval"   : dwh_pval,
        "sargan_j"   : sargan_j,
        "sargan_p"   : sargan_p,
    }
    return fs_diagnostics, ss_results


def sig_stars(p):
    if pd.isna(p):  return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    if p < 0.10:    return "†"
    return ""


def print_summary(all_ss, all_ols):
    print("\n" + "="*100)
    print("2SLS RESULTS — Pre-COVID sample, trust + time FE, clustered SE")
    print("Coefficients: standardised occupancy (1 SD increase -> outcome)")
    print("† p<0.10  * p<0.05  ** p<0.01  *** p<0.001")
    print("-"*100)
    print(f"{'Outcome':<35} {'Beds':<8} {'OLS β':>10} {'2SLS β':>10} "
          f"{'SE':>7} {'DWH p':>8} {'KP F':>8} {'J p':>8}")
    print("-"*100)
    for key, ss in all_ss.items():
        if not ss or "error" in ss:
            continue
        occ_lbl, out_col, inst_key = key
        ols_r = all_ols.get((occ_lbl, out_col))
        ols_b = f"{ols_r['coef']:+.4f}{sig_stars(ols_r['pval'])}" if ols_r else "—"
        ts_b  = f"{ss['coef_2sls']:+.4f}{sig_stars(ss.get('pval_2sls',np.nan))}"
        kp    = ss.get("kp_f", np.nan)
        kp_s  = f"{kp:.2f}" if not np.isnan(kp) else "—"
        jp    = ss.get("sargan_p", np.nan)
        jp_s  = f"{jp:.3f}" if not np.isnan(jp) else "—"
        dwh_s = f"{ss.get('dwh_pval',np.nan):.3f}" if not np.isnan(ss.get("dwh_pval",np.nan)) else "—"
        bl    = "All" if "all" in occ_lbl else "G&A"
        print(f"{out_col[:34]:<35} {bl:<8} {ols_b:>10} {ts_b:>10} "
              f"{ss['se_2sls']:>7.4f} {dwh_s:>8} {kp_s:>8} {jp_s:>8}")
    print("="*100)
    print("KP F < 10: weak instrument warning. DWH p < 0.05: endogeneity confirmed.")
    print("J p < 0.05: overidentification rejected (exclusion restriction concern).")


def plot_ols_vs_2sls(all_ss, all_ols, output_path="iv_summary_plot.png"):
    records = []
    for key, ss in all_ss.items():
        if not ss or "error" in ss or np.isnan(ss.get("coef_2sls", np.nan)):
            continue
        occ_lbl, out_col, inst_key = key
        if inst_key != "both":
            continue
        ols_r = all_ols.get((occ_lbl, out_col))
        if not ols_r:
            continue
        records.append({
            "label"     : f"{out_col.replace('_',' ')}\n({'All' if 'all' in occ_lbl else 'G&A'})",
            "ols_coef"  : ols_r["coef"],
            "ols_se"    : ols_r["se"],
            "sls_coef"  : ss["coef_2sls"],
            "sls_se"    : ss["se_2sls"],
            "weak_iv"   : ss.get("kp_f", 0) < 10,
        })
    if not records:
        return

    n     = len(records)
    y_pos = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.9)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for i, r in enumerate(records):
        ax.errorbar(r["ols_coef"], y_pos[i]+0.15,
                    xerr=1.96*r["ols_se"], fmt="o",
                    color=COL_OLS,  markersize=7, capsize=4, linewidth=1.5,
                    label="OLS (FE)"  if i==0 else "")
        color_2sls = "#888888" if r["weak_iv"] else COL_2SLS
        ax.errorbar(r["sls_coef"], y_pos[i]-0.15,
                    xerr=1.96*r["sls_se"], fmt="s",
                    color=color_2sls, markersize=7, capsize=4, linewidth=1.5,
                    label="2SLS (IV)" if i==0 else "")
        if r["weak_iv"]:
            ax.text(r["sls_coef"], y_pos[i]-0.15, " ⚠", fontsize=8,
                    color="#888888", va="center")

    ax.axvline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in records], fontsize=9)
    ax.set_xlabel("Standardised coefficient (1 SD occupancy)", fontsize=10)
    ax.set_title("OLS vs 2SLS Estimates\n"
                 "Pre-COVID sample, trust + time FE, clustered SE\n"
                 "⚠ = weak instrument (KP F < 10)",
                 fontsize=10, fontweight="bold", color=NAVY)
    handles = [mpatches.Patch(color=COL_OLS,  label="OLS (FE)"),
               mpatches.Patch(color=COL_2SLS, label="2SLS (IV — Z1+Z2)")]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              facecolor=BG, edgecolor=NAVY)
    for sp in ax.spines.values():
        sp.set_color(NAVY)
    ax.tick_params(colors=NAVY)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    df  = load_data(con)
    con.close()

    print(f"Loaded {len(df):,} obs | {df[TRUST_ID].nunique()} trusts")

    # Instrument coverage
    for inst in INSTRUMENTS:
        cov = df[inst].notna().mean() if inst in df.columns else 0
        print(f"  {inst}: {cov:.1%} coverage")

    # Pre-COVID sample for primary analysis
    df_pre = df[df["quarter_pd"] <= pd.Period(PRE_COVID_END, freq="Q")].copy()
    print(f"\nPre-COVID sample: {len(df_pre):,} obs | "
          f"{df_pre[TRUST_ID].nunique()} trusts")

    available_controls = [c for c in CONTROLS if c in df.columns]

    all_ols_results = {}
    all_fs_results  = {}
    all_ss_results  = {}
    all_diag_rows   = []

    instrument_sets = {
        "z1_backlog_hs"   : ["log_backlog_hs_per_m2"],
        "z2_pct1985"      : ["pct_pre1985"],
        "z3_backlog_high" : ["backlog_high_per_m2"],
        "z4_flu"          : ["flu_severity_region"],
        "eric_best"       : ["backlog_high_per_m2", "pct_pre1985"],
        "flu_eric"        : ["flu_severity_region", "backlog_high_per_m2"],
    }

    for occ_label, occ_col in OCC_VARS.items():
        if occ_col not in df.columns:
            continue
        for out_label, (out_col, out_desc) in OUTCOMES.items():
            if out_col not in df.columns:
                continue

            print(f"\n{occ_label.upper()} x {out_desc}")

            # OLS
            ols_r = run_ols(df_pre, occ_col, out_col, available_controls)
            all_ols_results[(occ_label, out_col)] = ols_r
            if ols_r and "error" not in ols_r:
                print(f"  OLS  β={ols_r['coef']:+.4f}{sig_stars(ols_r['pval'])} "
                      f"SE={ols_r['se']:.4f}  n={ols_r['n']:,}")

            # 2SLS variants
            for inst_label, inst_list in instrument_sets.items():
                avail = [z for z in inst_list
                         if z in df.columns and df_pre[z].notna().sum() > 50]
                if not avail:
                    continue

                fs_r, ss_r = manual_2sls(
                    df_pre, occ_col, out_col, avail,
                    available_controls, TRUST_ID, QUARTER
                )
                key = (occ_label, out_col, inst_label)
                all_fs_results[key] = fs_r
                all_ss_results[key] = ss_r

                if ss_r and "error" not in ss_r:
                    kp = fs_r.get("kp_f_stat", np.nan) if fs_r else np.nan
                    ss_r["kp_f"] = kp
                    print(f"  2SLS [{inst_label:7}]  "
                          f"β={ss_r['coef_2sls']:+.4f}"
                          f"{sig_stars(ss_r.get('pval_2sls',np.nan))}  "
                          f"SE={ss_r['se_2sls']:.4f}  "
                          f"KP-F={'—' if pd.isna(kp) else f'{kp:.2f}'}  "
                          f"DWH-p={ss_r.get('dwh_pval',np.nan):.3f}")

                    diag_row = {
                        "occ_var"      : occ_label,
                        "occ_col"      : occ_col,
                        "outcome"      : out_label,
                        "outcome_col"  : out_col,
                        "inst_set"     : inst_label,
                        "instruments"  : fs_r.get("instruments_used","") if fs_r else "",
                        "ols_coef"     : ols_r.get("coef", np.nan) if ols_r else np.nan,
                        "ols_se"       : ols_r.get("se",   np.nan) if ols_r else np.nan,
                        "ols_pval"     : ols_r.get("pval", np.nan) if ols_r else np.nan,
                        **{k:v for k,v in (fs_r or {}).items() if not k.startswith("_")},
                        **{f"ss_{k}":v for k,v in (ss_r or {}).items()},
                    }
                    all_diag_rows.append(diag_row)

    print_summary(all_ss_results, all_ols_results)
    plot_ols_vs_2sls(all_ss_results, all_ols_results)

    if all_diag_rows:
        diag_df = pd.DataFrame(all_diag_rows)
        diag_df.to_csv("iv_diagnostics.csv", index=False)
        print("Saved: iv_diagnostics.csv")

    # ── Robustness: full sample ────────────────────────────────────────────────
    print("\n── ROBUSTNESS: Full sample (including COVID) ─────────────────────")
    robust_rows = []
    for occ_label, occ_col in OCC_VARS.items():
        if occ_col not in df.columns:
            continue
        for out_label, (out_col, out_desc) in OUTCOMES.items():
            if out_col not in df.columns:
                continue
            avail = [z for z in INSTRUMENTS
                     if z in df.columns and df[z].notna().sum() > 50]
            if not avail:
                continue
            fs_r, ss_r = manual_2sls(
                df, occ_col, out_col, avail,
                available_controls, TRUST_ID, QUARTER
            )
            if ss_r and "error" not in ss_r:
                kp = fs_r.get("kp_f_stat", np.nan) if fs_r else np.nan
                ss_r["kp_f"] = kp
                print(f"  {occ_label:8} {out_desc[:28]:28}  "
                      f"β={ss_r['coef_2sls']:+.4f}"
                      f"{sig_stars(ss_r.get('pval_2sls',np.nan))}  "
                      f"KP-F={kp:.2f}")
                robust_rows.append({
                    "sample"      : "full",
                    "occ_var"     : occ_label,
                    "outcome_col" : out_col,
                    "coef_2sls"   : ss_r["coef_2sls"],
                    "se_2sls"     : ss_r["se_2sls"],
                    "kp_f"        : kp,
                    "dwh_pval"    : ss_r.get("dwh_pval", np.nan),
                })

    if robust_rows:
        pd.DataFrame(robust_rows).to_csv("iv_robustness_fullsample.csv", index=False)
        print("Saved: iv_robustness_fullsample.csv")

    print("\nDone.")
    print("\nKEY INTERPRETATION GUIDE:")
    print("  KP F > 10     : instruments are relevant (not weak)")
    print("  DWH p < 0.05  : endogeneity confirmed, 2SLS preferred over OLS")
    print("  DWH p > 0.10  : fail to reject, OLS consistent, report both")
    print("  Sargan J p < 0.05 : exclusion restriction may be violated")
    print("  |2SLS| > |OLS|    : attenuation bias in OLS (classical measurement error)")
    print("  |2SLS| < |OLS|    : OLS overstates effect (endogeneity bias confirmed)")


if __name__ == "__main__":
    main()
