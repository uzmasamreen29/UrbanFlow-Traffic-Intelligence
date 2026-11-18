import sqlite3
import pandas as pd

conn = sqlite3.connect("data/cleaned/urbanflow.db")

with open("sql/urbanflow_analysis.sql", "r", encoding="utf-8") as f:
    raw_sql = f.read()

# Strip comment lines so queries execute cleanly
clean_lines = [line for line in raw_sql.splitlines() if not line.strip().startswith("--")]
clean_sql = "\n".join(clean_lines)

queries = [q.strip() for q in clean_sql.split(";") if q.strip()]

query_names = [
    "Q0_Audit_Summary",
    "Q1_Top_Congestion_Hotspots",
    "Q2_Hourly_Traffic_Profile",
    "Q3_Weekday_vs_Weekend",
    "Q4_Directional_Bottlenecks",
    "Q5_Rolling_3Hr_Peak_Windows",
    "Q6_ML_High_Risk_Intersections",
    "Q7_ML_Model_Accuracy_By_Intersection"
]

for name, query in zip(query_names, queries):
    print(f"\n==================== {name} ====================")
    df_res = pd.read_sql_query(query, conn)
    print(df_res.to_string(index=False))
    df_res.to_csv(f"data/cleaned/{name}.csv", index=False)

conn.close()
print("\nDone! All 8 SQL query results saved to data/cleaned/.")