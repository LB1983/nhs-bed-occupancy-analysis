import pandas as pd
import duckdb
import numpy as np

db_path  = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/nhs_occupancy.duckdb"
csv_path = "C:/Users/laure/OneDrive/Documents/BevanBriefing/nhs-bed-occupancy-analysis/03-data-final/master-quarterly-trust.csv"

df = pd.read_csv(csv_path, encoding="latin1")
print("Loaded rows:", len(df))

df.rename(columns={"period": "quarter"}, inplace=True)

df["ae_12h_per_1k_att"] = df["ae_waits_12hr_decision_to_admit"] / df["ae_type1_attendances"] * 1000
df["canc_per_100_beds"] = df["canc_cancelled_ops"].div(
    df["kh03_avail_all_avg_beds"].where(df["kh03_avail_all_avg_beds"] >= 1)
) * 100

print("Quarter range:", df["quarter"].min(), "to", df["quarter"].max())
print("Trusts:", df["org_code"].nunique())
print("ae_12h coverage:", round(df["ae_12h_per_1k_att"].notna().mean() * 100, 1), "%")
print("canc_per_100 coverage:", round(df["canc_per_100_beds"].notna().mean() * 100, 1), "%")

# Convert NHS financial year quarters (e.g. "2018-19 Q1") to calendar quarters
# (e.g. "2018Q2") to match eric_instruments format
def nhs_to_cal_quarter(q):
    year = int(q[:4])
    qnum = int(q[-1])
    if qnum == 4:
        return f"{year + 1}Q1"
    return f"{year}Q{qnum + 1}"

df["calendar_quarter"] = df["quarter"].map(nhs_to_cal_quarter)

con = duckdb.connect(db_path)
con.execute("DROP TABLE IF EXISTS merged_quarterly")
con.execute("CREATE TABLE merged_quarterly AS SELECT * FROM df")
print("Written merged_quarterly:", len(df), "rows")

joined = con.execute("""
    SELECT COUNT(*) as n, COUNT(e.backlog_hs_per_m2) as n_with_eric
    FROM merged_quarterly m
    LEFT JOIN eric_instruments e
      ON m.org_code        = e.org_code
     AND m.calendar_quarter = TRIM(e.quarter)
""").df()
print("Join check - total rows:", joined["n"].iloc[0], "with ERIC:", joined["n_with_eric"].iloc[0])
con.close()
print("Done. Ready to run 01_within_between.py")