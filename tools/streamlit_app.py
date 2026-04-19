"""
NHS Local Occupancy Target Calculator — Streamlit web app
==========================================================
Browser-based interface for occupancy_target_tool.py.

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Community Cloud:
    1. Push this file and the two data CSVs to GitHub
    2. Go to share.streamlit.io and connect the repo
    3. Set Main file path: 07-tools/streamlit_app.py

Data files needed (relative to this script):
    ../03-data-final/master-quarterly-trust.csv
    ../02-data-interim/geo/trust_proximity_features.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths — works both locally and on Streamlit Cloud
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent
MASTER_PATH = _BASE / "03-data-final" / "master-quarterly-trust.csv"
PROX_PATH   = _BASE / "02-data-interim" / "geo" / "trust_proximity_features.csv"

# ---------------------------------------------------------------------------
# Model coefficients (from models_scaled_interactions.csv)
# ---------------------------------------------------------------------------
B_OCC_Z  = -0.603
B_NEAR_Z =  1.627
B_10KM_Z = -4.283
B_25KM_Z =  4.162

PROX_STATS = {
    "nearest_trust_km":   {"mean": 7.59,  "sd": 10.13},
    "trusts_within_10km": {"mean": 4.20,  "sd":  6.70},
    "trusts_within_25km": {"mean": 12.84, "sd": 15.72},
}

OV_P25 = 0.0185; OV_P75 = 0.0417
AEI_P25 = 25.14; AEI_P75 = 38.08

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def standardise(v, mean, sd):
    return (v - mean) / sd if sd > 0 else 0.0

def compute_beta(nearest_km, within_10, within_25):
    z_near = standardise(nearest_km, **PROX_STATS["nearest_trust_km"])
    z_10   = standardise(within_10,  **PROX_STATS["trusts_within_10km"])
    z_25   = standardise(within_25,  **PROX_STATS["trusts_within_25km"])
    return B_OCC_Z + B_NEAR_Z * z_near + B_10KM_Z * z_10 + B_25KM_Z * z_25

def geo_uplift(beta):
    if beta < -3:   return 8, "High geographic buffer"
    elif beta < -1: return 5, "Moderate geographic buffer"
    elif beta < 0:  return 2, "Marginal geographic buffer"
    else:           return 0, "No effective geographic buffer"

def ov_adjustment(ov):
    if ov < 0.020:   return 0, "Low seasonal swing"
    elif ov < 0.050: return 1, "Moderate seasonal swing"
    else:            return 2, "High seasonal swing"

def edr_adjustment(edr, aei):
    if edr is not None:
        if edr < 0.35:   return 0, f"Low emergency dependency (EDR {edr:.2f}, user-supplied)"
        elif edr < 0.50: return 1, f"Mixed emergency dependency (EDR {edr:.2f}, user-supplied)"
        else:            return 2, f"High emergency dependency (EDR {edr:.2f}, user-supplied)"
    if aei is None or np.isnan(aei):
        return 1, "Emergency dependency unknown — assuming moderate"
    if aei < AEI_P25:   return 0, f"Low A&E intensity ({aei:.1f} att/100 beds/qtr)"
    elif aei < AEI_P75: return 1, f"Moderate A&E intensity ({aei:.1f} att/100 beds/qtr)"
    else:                return 2, f"High A&E intensity ({aei:.1f} att/100 beds/qtr)"

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    master = pd.read_csv(MASTER_PATH)
    prox   = pd.read_csv(PROX_PATH)
    master["org_code"] = master["org_code"].str.strip().str.upper()
    prox["org_code"]   = prox["org_code"].str.strip().str.upper()
    return master, prox

@st.cache_data
def build_trust_list(master):
    df = master[["org_code","org_name"]].drop_duplicates().dropna()
    return df.sort_values("org_name").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
BG    = "#F5F0E6"
NAVY  = "#0B2D39"
TEAL  = "#1BA6A6"
RED   = "#E24A3B"
AMBER = "#F4A724"

def plot_occupancy_history(occ_series, periods, ceiling, trust_name):
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    colours = []
    for v in occ_series:
        if v >= 0.90:    colours.append(RED)
        elif v >= 0.85:  colours.append(AMBER)
        else:            colours.append(TEAL)

    ax.bar(range(len(occ_series)), [v * 100 for v in occ_series],
           color=colours, width=0.8, alpha=0.85)

    ax.axhline(85, color=NAVY, linewidth=1.2, linestyle="--",
               alpha=0.6, label="National target (85%)")
    ax.axhline(ceiling, color=RED, linewidth=1.8, linestyle="-",
               alpha=0.9, label=f"Local ceiling ({ceiling}%)")

    # Label every 4th period
    tick_idx = list(range(0, len(periods), 4))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([periods[i] for i in tick_idx],
                       rotation=35, ha="right", fontsize=7.5, color=NAVY)

    ax.set_ylabel("Occupancy (%)", color=NAVY, fontsize=9)
    ax.set_ylim(50, 105)
    ax.tick_params(colors=NAVY, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(NAVY)

    patches = [
        mpatches.Patch(color=TEAL,  label="< 85%"),
        mpatches.Patch(color=AMBER, label="85–90%"),
        mpatches.Patch(color=RED,   label="> 90%"),
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0],[0], color=NAVY,  linestyle="--", label="National 85%"),
        plt.Line2D([0],[0], color=RED,   linestyle="-",  label=f"Local ceiling {ceiling}%"),
    ], fontsize=7.5, facecolor=BG, edgecolor=NAVY, loc="lower left", ncol=3)

    ax.set_title(f"Bed occupancy — {trust_name}", color=NAVY,
                 fontsize=10, fontweight="bold", pad=8)
    fig.tight_layout()
    return fig


def plot_gauge(ceiling, current_mean):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(75, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Zones
    ax.barh(0.5, 10, left=75, height=0.35, color=TEAL,  alpha=0.7)
    ax.barh(0.5, 5,  left=85, height=0.35, color=AMBER, alpha=0.7)
    ax.barh(0.5, 5,  left=90, height=0.35, color=RED,   alpha=0.7)

    # Ceiling marker
    ax.axvline(ceiling, color=RED, linewidth=2.5, ymin=0.25, ymax=0.9)
    ax.text(ceiling, 0.95, f"Local\nceiling\n{ceiling}%",
            ha="center", va="top", fontsize=8, color=RED, fontweight="bold")

    # National target
    ax.axvline(85, color=NAVY, linewidth=1.5, linestyle="--", ymin=0.25, ymax=0.9)
    ax.text(85, 0.05, "85%\nnational", ha="center", va="bottom",
            fontsize=7.5, color=NAVY, alpha=0.8)

    # Current mean arrow
    if not np.isnan(current_mean):
        cm = current_mean * 100
        ax.annotate("", xy=(cm, 0.5), xytext=(cm, 0.05),
                    arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
        ax.text(cm, 0.0, f"Current\nmean\n{cm:.1f}%",
                ha="center", va="top", fontsize=7.5, color=NAVY)

    ax.set_title("Target range", color=NAVY, fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="NHS Occupancy Target Calculator",
        page_icon="🏥",
        layout="wide",
    )

    # Header
    st.markdown(
        f"""
        <div style='background-color:{NAVY}; padding:1.2rem 1.5rem; border-radius:6px; margin-bottom:1rem'>
          <h2 style='color:white; margin:0; font-size:1.4rem'>
            NHS Local Bed Occupancy Target Calculator
          </h2>
          <p style='color:#aac8d0; margin:0.3rem 0 0 0; font-size:0.9rem'>
            Bevan Briefing · NHS Bed Occupancy Analysis 2025
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "The national 85% bed occupancy target applies equally to every NHS acute "
        "trust — from central London (49 acute neighbours within 25km) to Cornwall "
        "(none within 50km). This tool calculates a locally-calibrated ceiling for "
        "any trust based on its geographic position and operational profile, using "
        "a panel regression model on 3,500 trust-quarters (2018–25)."
    )

    # Load data
    try:
        master, prox = load_data()
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.stop()

    trust_list = build_trust_list(master)

    # ---------------------------------------------------------------------------
    # Sidebar inputs
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.header("Select trust")

        search = st.text_input("Search by name", placeholder="e.g. Cornwall, Manchester")
        if search:
            matches = trust_list[
                trust_list["org_name"].str.upper().str.contains(search.upper())
            ]
        else:
            matches = trust_list

        if len(matches) == 0:
            st.warning("No trusts match that search.")
            st.stop()

        options = {f"{row['org_name']} ({row['org_code']})": row["org_code"]
                   for _, row in matches.iterrows()}
        selected_label = st.selectbox("Trust", list(options.keys()))
        trust_code = options[selected_label]

        st.divider()
        st.header("Optional: supply your own EDR")
        st.caption(
            "Emergency Dependency Ratio = emergency admissions ÷ total occupied "
            "bed days, from SUS/ECDS. If left blank, A&E intensity is used as a proxy."
        )
        edr_input = st.number_input(
            "EDR (0.00 – 1.00)", min_value=0.0, max_value=1.0,
            value=0.0, step=0.01, format="%.2f"
        )
        edr = edr_input if edr_input > 0 else None

        st.divider()
        st.caption(
            "**Source**: KH03 bed returns, NHS England A&E & UEC SitRep 2018-19 to 2024-25. "
            "Methodology: two-way FE panel OLS (trust + time), cluster-robust SE. "
            "See [GitHub repo](https://github.com/bevan-briefing/nhs-bed-occupancy-analysis) "
            "for full code and data."
        )

    # ---------------------------------------------------------------------------
    # Compute
    # ---------------------------------------------------------------------------
    trust_rows = master[master["org_code"] == trust_code].copy()
    trust_name = trust_rows["org_name"].dropna().iloc[0] if len(trust_rows) else trust_code

    prox_rows  = prox[prox["org_code"] == trust_code]
    has_prox   = not prox_rows.empty

    if has_prox:
        p          = prox_rows.iloc[0]
        nearest_km = float(p["nearest_trust_km"])
        within_10  = int(p["trusts_within_10km"])
        within_25  = int(p["trusts_within_25km"])
        within_50  = int(p.get("trusts_within_50km", 0))
        beta       = compute_beta(nearest_km, within_10, within_25)
    else:
        nearest_km = within_10 = within_25 = within_50 = None
        beta = B_OCC_Z

    geo_up, geo_label = geo_uplift(beta)

    # Occupancy
    for col in ["kh03_occ_all_rate", "kh03_occ_ga_rate", "kh03_avail_all_avg_beds",
                "ae_type1_pct_4hr", "canc_pct_not_treated_28_days",
                "uec_uec_nctr_mean", "ae_type1_attendances"]:
        if col in trust_rows.columns:
            trust_rows[col] = pd.to_numeric(trust_rows[col], errors="coerce")

    trust_rows["yr"] = trust_rows["period"].astype(str).str[:4].astype(int)
    non_covid = trust_rows[~trust_rows["yr"].isin([2020, 2021])]
    recent    = trust_rows[trust_rows["yr"] >= 2022]

    occ_all = trust_rows["kh03_occ_all_rate"].dropna()
    occ_rec = recent["kh03_occ_all_rate"].dropna()

    mean_occ_all = occ_all.mean()
    mean_occ_rec = occ_rec.mean() if len(occ_rec) else np.nan
    pct85_all    = (occ_all >= 0.85).mean() * 100
    pct90_all    = (occ_all >= 0.90).mean() * 100
    pct85_rec    = (occ_rec >= 0.85).mean() * 100 if len(occ_rec) else np.nan
    pct90_rec    = (occ_rec >= 0.90).mean() * 100 if len(occ_rec) else np.nan

    ov_base = occ_rec if len(occ_rec) >= 6 else non_covid["kh03_occ_all_rate"].dropna()
    ov_val  = (ov_base.std() / ov_base.mean()) if (len(ov_base) >= 4 and ov_base.mean() > 0) else 0.03
    ov_adj, ov_label = ov_adjustment(ov_val)

    mean_beds = trust_rows["kh03_avail_all_avg_beds"].dropna().mean()
    mean_beds_est = mean_beds * 100 if not np.isnan(mean_beds) else None

    # AEI proxy
    nc_tmp = non_covid[["ae_type1_attendances","kh03_avail_all_avg_beds"]].dropna()
    aei = (nc_tmp["ae_type1_attendances"] / (nc_tmp["kh03_avail_all_avg_beds"] * 100)).mean() \
          if len(nc_tmp) >= 4 else None

    edr_adj, edr_label = edr_adjustment(edr, aei)

    FLOOR = 80
    ceiling = FLOOR + geo_up - ov_adj - edr_adj

    breach = (not np.isnan(mean_occ_rec)) and (mean_occ_rec > (ceiling / 100) + 0.05)

    # ---------------------------------------------------------------------------
    # Layout
    # ---------------------------------------------------------------------------
    st.subheader(trust_name)
    if mean_beds_est:
        st.caption(f"ODS code: {trust_code}  ·  Bed base: ~{mean_beds_est:.0f} beds")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Local ceiling", f"{ceiling}%",
                delta=f"{ceiling - 85:+d} pp vs national target",
                delta_color="normal")
    col2.metric("Mean occupancy (all time)", f"{mean_occ_all*100:.1f}%")
    col3.metric("Mean occupancy (since 2022)", f"{mean_occ_rec*100:.1f}%" if not np.isnan(mean_occ_rec) else "N/A")
    if breach:
        col4.metric("Status", "Structural breach",
                    delta=f"{mean_occ_rec*100 - ceiling:.1f} pp above ceiling",
                    delta_color="inverse")
    else:
        col4.metric("Status", "Within range")

    # Breach banner
    if breach:
        gap = mean_occ_rec * 100 - ceiling
        st.error(
            f"**Structural breach warning.** This trust's recent mean occupancy "
            f"({mean_occ_rec*100:.1f}%) exceeds the calculated local ceiling by "
            f"{gap:.1f} percentage points. This indicates a **capacity gap**, not a "
            f"management failure. See interventions below."
        )

    st.divider()

    # Two-column layout: charts left, calculation right
    left, right = st.columns([3, 2])

    with left:
        # Occupancy history chart
        hist = trust_rows.sort_values("period")[
            ["period","kh03_occ_all_rate"]].dropna()
        if len(hist) > 0:
            fig1 = plot_occupancy_history(
                hist["kh03_occ_all_rate"].tolist(),
                hist["period"].tolist(),
                ceiling, trust_name
            )
            st.pyplot(fig1, use_container_width=True)

        # Occupancy summary table
        st.markdown("**Occupancy statistics**")
        summary_data = {
            "Period": ["All time", "Pre-COVID", "Recovery (2022+)"],
            "Mean occ.": [
                f"{mean_occ_all*100:.1f}%",
                f"{non_covid['kh03_occ_all_rate'].dropna().mean()*100:.1f}%" if len(non_covid) else "—",
                f"{mean_occ_rec*100:.1f}%" if not np.isnan(mean_occ_rec) else "—",
            ],
            "% above 85%": [
                f"{pct85_all:.0f}%",
                f"{(non_covid['kh03_occ_all_rate'].dropna()>=0.85).mean()*100:.0f}%" if len(non_covid) else "—",
                f"{pct85_rec:.0f}%" if not np.isnan(pct85_rec) else "—",
            ],
            "% above 90%": [
                f"{pct90_all:.0f}%",
                f"{(non_covid['kh03_occ_all_rate'].dropna()>=0.90).mean()*100:.0f}%" if len(non_covid) else "—",
                f"{pct90_rec:.0f}%" if not np.isnan(pct90_rec) else "—",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    with right:
        # Gauge
        fig2 = plot_gauge(ceiling, mean_occ_rec if not np.isnan(mean_occ_rec) else mean_occ_all)
        st.pyplot(fig2, use_container_width=True)

        # Calculation breakdown
        st.markdown("**How the ceiling is calculated**")
        calc_data = {
            "Component": [
                "National floor",
                "Geographic uplift",
                "Volatility adjustment",
                "Emergency demand adjustment",
                "**Local ceiling**",
            ],
            "Value": [
                "80%",
                f"+{geo_up} pp",
                f"−{ov_adj} pp",
                f"−{edr_adj} pp",
                f"**{ceiling}%**",
            ],
            "Reason": [
                "Minimum safety margin",
                geo_label,
                ov_label,
                edr_label,
                "",
            ],
        }
        st.dataframe(pd.DataFrame(calc_data), hide_index=True, use_container_width=True)

        # Proximity profile
        if has_prox:
            st.markdown("**Geographic profile**")
            prox_data = {
                "Measure": ["Nearest trust", "Trusts within 10km", "Trusts within 25km", "Trusts within 50km", "Harm multiplier (β)"],
                "This trust": [
                    f"{nearest_km:.1f} km",
                    str(within_10),
                    str(within_25),
                    str(within_50),
                    f"{beta:+.2f}",
                ],
                "National median": ["4.3 km", "2", "5", "—", "—"],
            }
            st.dataframe(pd.DataFrame(prox_data), hide_index=True, use_container_width=True)

            if beta >= 0:
                st.warning(
                    f"β = {beta:+.2f}: **no geographic buffer**. Rising occupancy at this trust "
                    "directly increases patient harm — there are no nearby alternatives to absorb overflow."
                )
            elif beta < -3:
                st.success(
                    f"β = {beta:+.2f}: **high geographic buffer**. Dense nearby alternatives mean "
                    "rising occupancy is partially absorbed by the surrounding system."
                )
            else:
                st.info(
                    f"β = {beta:+.2f}: **moderate buffer**. Some diversion options exist "
                    "but benefit is limited under simultaneous regional pressure."
                )

    # Outcomes
    st.divider()
    st.markdown("**Outcome data (mean, all quarters)**")
    out_cols = st.columns(4)
    outcomes = [
        ("A&E 4hr compliance", "ae_type1_pct_4hr", True, 0.76, "higher"),
        ("% cancelled not treated 28d", "canc_pct_not_treated_28_days", True, None, "lower"),
        ("NCTR (mean patients/day)", "uec_uec_nctr_mean", False, None, "lower"),
        ("G&A occupancy", "kh03_occ_ga_rate", True, None, "lower"),
    ]
    for i, (label, col, is_pct, benchmark, direction) in enumerate(outcomes):
        if col in trust_rows.columns:
            v = trust_rows[col].dropna()
            if len(v):
                val = v.mean()
                display = f"{val*100:.1f}%" if is_pct else f"{val:.0f}"
                if benchmark and is_pct:
                    delta_val = (val - benchmark) * 100
                    out_cols[i].metric(label, display,
                                       delta=f"{delta_val:+.1f} pp vs standard",
                                       delta_color="normal" if direction=="higher" else "inverse")
                else:
                    out_cols[i].metric(label, display)

    # Interventions (only if breached)
    if breach:
        st.divider()
        st.markdown("### Recommended interventions")
        gap = mean_occ_rec * 100 - ceiling
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. NCTR reduction** (fastest lever)")
            st.markdown(
                "Discharge pathway improvements — seven-day discharge teams, enhanced community "
                "step-down capacity, virtual ward expansion. Each NCTR patient freed is one "
                "bed returned without capital expenditure."
            )
        with c2:
            st.markdown("**2. Demand management**")
            st.markdown(
                "Community alternatives, anticipatory care, and admission avoidance schemes "
                "can reduce avoidable emergency pressure. Particularly important for isolated "
                "trusts where diversion is not an option."
            )
        with c3:
            st.markdown("**3. Formal bed base review**")
            if gap > 3:
                st.markdown(
                    f"The gap of **{gap:.1f} pp** between current mean occupancy and the "
                    f"local ceiling ({ceiling}%) is large enough to warrant a formal capacity "
                    "case. Operational interventions alone are unlikely to close this gap sustainably."
                )
            else:
                st.markdown(
                    "The gap is small enough that operational improvements may be sufficient. "
                    "Monitor quarterly — if the gap persists beyond 12 months, initiate a "
                    "capacity review."
                )

    # System check
    st.divider()
    with st.expander("System-level check (apply separately)"):
        st.markdown(
            "An individual trust's ceiling is only valid if the surrounding health economy has slack. "
            "If your ICB reports a **mean acute occupancy above 85%** across all trusts in the footprint, "
            "apply an additional **−2 pp system stress penalty**.\n\n"
            "**Portfolio rule**: escalate to system level if three or more trusts in your footprint "
            "simultaneously exceed 90% in any given week.\n\n"
            "This tool cannot calculate the ICB-level check automatically — it requires real-time "
            "SitRep data from NHS England."
        )

    # Caveats
    with st.expander("Methodology and limitations"):
        st.markdown(
            """
**Model**: Two-way fixed-effects panel OLS (trust + time effects), cluster-robust standard errors
by trust. N ≈ 3,551 trust-quarters, 140 trusts, 2018-19 Q1 to 2024-25 Q4.
Outcome: 12-hour waits per 1,000 A&E attendances. Coefficients from `models_scaled_interactions.csv`.

**Associations, not causes**: Instrumental variables tested (ERIC estate condition, regional flu
severity) were insufficiently strong (Kleibergen-Paap F < 2 vs threshold of 10). Estimates
describe within-trust associations conditional on time-invariant trust characteristics.

**Proximity counts all NHS trust types**: Mental health, community, and ambulance trusts are
included in the neighbour count. Non-acute neighbours provide no real diversion capacity —
the geographic buffer may be overstated for trusts whose nearby neighbours are non-acute.

**Pre-COVID caution**: In the pre-COVID period (2018-19 to 2019-20) the occupancy-harm
relationship is weaker and less consistently significant. Results are strongest in the
post-2021 recovery period, which may reflect a system under particular stress.

**EDR proxy**: A&E intensity (type 1 attendances per available bed) is a rough proxy for
emergency dependency. Supply your own EDR from SUS/ECDS for greater precision.

**Source code and data**: [github.com/bevan-briefing/nhs-bed-occupancy-analysis](https://github.com/bevan-briefing/nhs-bed-occupancy-analysis)

**Citation**: Bevan Briefing (2025). *NHS Bed Occupancy: The Case for Local Targets.*
            """
        )


if __name__ == "__main__":
    main()
