import pandas as pd
import numpy as np
import sqlite3
import os

os.makedirs("data/cleaned", exist_ok=True)
os.makedirs("sql", exist_ok=True)

# 1. Load Raw CSV
print("Loading data/raw/austin_radar_traffic.csv...")
df = pd.read_csv("data/raw/austin_radar_traffic.csv")
df.columns = df.columns.str.strip().str.lower()

print(f"Raw Rows: {len(df):,} | Columns: {list(df.columns)}")

# 2. Clean Timestamps & Drop Invalid Sensor Records
df["curdatetime"] = pd.to_datetime(df["curdatetime"], errors="coerce")
df = df.dropna(subset=["curdatetime", "volume", "speed", "intname", "int_id"])

# Filter out physical sensor glitches (negative volume, 0 speed, or >100 mph)
df = df[(df["volume"] >= 0) & (df["speed"] > 0) & (df["speed"] <= 100)].copy()

# 3. Feature Engineering (Time Dimensions & Congestion Metrics)
df["date"] = df["curdatetime"].dt.date.astype(str)
df["hour"] = df["curdatetime"].dt.hour
df["day_name"] = df["curdatetime"].dt.day_name()
df["is_weekend"] = np.where(df["curdatetime"].dt.dayofweek >= 5, "Weekend", "Weekday")

def classify_time_slot(h):
    if 6 <= h <= 9:
        return "1. Morning Peak (06-09)"
    elif 10 <= h <= 15:
        return "2. Midday (10-15)"
    elif 16 <= h <= 19:
        return "3. Evening Peak (16-19)"
    else:
        return "4. Night / Off-Peak (20-05)"

df["time_period"] = df["hour"].apply(classify_time_slot)

# 85th Percentile Free-Flow Speed per Intersection
free_flow = df.groupby("int_id")["speed"].quantile(0.85).rename("free_flow_speed")
df = df.merge(free_flow, on="int_id", how="left")

# Speed Drop % & Congestion Severity Classification
df["speed_drop_pct"] = (((df["free_flow_speed"] - df["speed"]) / df["free_flow_speed"]) * 100).clip(lower=0).round(2)
df["free_flow_speed"] = df["free_flow_speed"].round(2)

df["congestion_level"] = pd.cut(
    df["speed_drop_pct"],
    bins=[-1, 15, 35, 100],
    labels=["Free Flow", "Moderate Congestion", "Severe Congestion"]
)

# Save cleaned CSV for Power BI
cleaned_path = "data/cleaned/fact_traffic_cleaned.csv"
df.to_csv(cleaned_path, index=False)
print(f"Cleaned Rows Saved: {len(df):,} -> {cleaned_path}\n")

# 4. Run SQL Analytical Queries via SQLite & Print Results
conn = sqlite3.connect("data/cleaned/urbanflow.db")
df.to_sql("fact_traffic_cleaned", conn, if_exists="replace", index=False)

queries = {
    "Q1_Top_Congestion_Hotspots": """
        SELECT 
            int_id,
            intname AS intersection_name,
            COUNT(*) AS intervals,
            SUM(volume) AS total_vehicles,
            ROUND(AVG(speed), 1) AS avg_speed_mph,
            ROUND(AVG(free_flow_speed), 1) AS free_flow_mph,
            ROUND(AVG(speed_drop_pct), 1) AS avg_speed_drop_pct
        FROM fact_traffic_cleaned
        GROUP BY int_id, intname
        HAVING COUNT(*) >= 100
        ORDER BY avg_speed_drop_pct DESC, total_vehicles DESC
        LIMIT 10;
    """,
    "Q2_Hourly_Peak_Profile": """
        SELECT 
            hour,
            SUM(volume) AS total_volume,
            ROUND(AVG(speed), 1) AS avg_speed_mph,
            ROUND(AVG(occupancy), 1) AS avg_occupancy_pct
        FROM fact_traffic_cleaned
        GROUP BY hour
        ORDER BY hour;
    """,
    "Q3_Weekday_vs_Weekend": """
        SELECT 
            is_weekend,
            time_period,
            SUM(volume) AS total_volume,
            ROUND(AVG(speed), 1) AS avg_speed_mph,
            ROUND(AVG(speed_drop_pct), 1) AS avg_speed_drop_pct,
            ROUND(100.0 * SUM(CASE WHEN congestion_level = 'Severe Congestion' THEN 1 ELSE 0 END) / COUNT(*), 2) AS severe_share_pct
        FROM fact_traffic_cleaned
        GROUP BY is_weekend, time_period
        ORDER BY is_weekend, time_period;
    """
}

for name, sql in queries.items():
    print(f"=== {name} ===")
    res = pd.read_sql_query(sql, conn)
    print(res.to_string(index=False))
    print()
    res.to_csv(f"data/cleaned/{name}.csv", index=False)

conn.close()
print("All SQL summary tables exported to data/cleaned/!")