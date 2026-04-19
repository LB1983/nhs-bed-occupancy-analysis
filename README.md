# NHS Bed Occupancy Analysis

Replication code and results for *The 85% Myth*, published in
[The Bevan Briefing](https://thebevanbriefing.com).

## Summary

Analysis of approximately 5,400 trust-quarters covering ~140 NHS acute
trusts in England from 2018-19 to 2024-25. Key findings:

- No threshold effect at 85% — or at any level from 80% to 92% — for
  A&E performance, 12-hour waits, or cancelled operations
- Occupancy risk is mediated by bed base size and geographic position,
  not by crossing any universal percentage
- Trusts within 10km competitor rings are substantially buffered against
  occupancy pressure (p < 10^-10 for 12-hour waits)
- For isolated trusts (e.g. Royal Cornwall), the 85% target is
  structurally unachievable and represents surveillance without agency

## Structure

    analysis/
      cleaning/           Data cleaning pipeline
        00_clean_eric.py          ERIC estate data (backlog, age profiles)
        load_master.py            Load panel into DuckDB
        01a-clean-kh03.py         KH03 bed occupancy
        02a-clean-ae.py           A&E SitRep
        03a-clean-canc.py         Cancelled operations
      04-merge-quarterly-datasets.py   Panel assembly
      05a-regressions_*.py        Threshold and spline models
      05b-regressions_DEBUGGED.py Full interaction model (geographic terms)
      01_within_between.py        Mundlak within-between decomposition
      02_covid_subperiods.py      Structural period analysis
      03_iv_2sls.py               IV/2SLS estimation (estate instruments)

    tools/
      occupancy_target_tool.py          Trust-specific ceiling calculator
      01-build_proximity_features.py    Compute proximity features from ODS
      streamlit_app.py                  Interactive web version

    outputs/
      models_threshold.csv              Threshold models (80-92%)
      models_spline.csv                 Piecewise linear spline models
      models_capacity_interactions.csv  Occupancy x bed base interactions
      models_scaled_interactions.csv    Full geographic interaction model
      models_scaled_threshold.csv       Threshold x geography interactions
      models_size_quintiles.csv         Effects by trust size quintile
      regression_results.csv            Primary results table

    data/
      master-quarterly-trust.csv        Trust-quarter panel (all sources)

## Quickstart

    pip install pandas numpy statsmodels linearmodels duckdb matplotlib openpyxl

Run in order:

    python analysis/cleaning/00_clean_eric.py
    python analysis/cleaning/load_master.py
    python analysis/05b-regressions_DEBUGGED.py
    python analysis/01_within_between.py
    python analysis/02_covid_subperiods.py
    python analysis/03_iv_2sls.py

## Local occupancy ceiling calculator

    python tools/occupancy_target_tool.py REF
    python tools/occupancy_target_tool.py RRV --edr 0.38
    python tools/occupancy_target_tool.py        # interactive prompt

## Data sources

All source data is publicly available from NHS England:

- KH03 bed availability and occupancy
- A&E Attendances and Emergency Admissions SitRep
- Urgent and Emergency Care daily SitRep
- Cancelled Operations quarterly return
- ERIC NHS estates and facilities return

Trust geographic proximity features computed from ODS postcode data.

## Citation

    Bevan Briefing (2025). The 85% Myth: What the Data Actually Says
    About NHS Bed Occupancy. https://thebevanbriefing.com

## Licence

Code: MIT. Data: Open Government Licence v3.0.
