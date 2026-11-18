import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# -------------------------------------------------------------------------
# 0. Setup Output Directories (Zero modification to existing files)
# -------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("dashboard/ml_evaluation", exist_ok=True)

print("1. Loading cleaned traffic telemetry...")
df = pd.read_csv("data/cleaned/fact_traffic_cleaned.csv")
df["curdatetime"] = pd.to_datetime(df["curdatetime"], utc=True)

# -------------------------------------------------------------------------
# 1. Aggregate 15-Minute Lane Telemetry to Hourly Intersection Level
# -------------------------------------------------------------------------
df["hour_ts"] = df["curdatetime"].dt.floor("h")

hourly = (
    df.groupby(["int_id", "intname", "hour_ts"], as_index=False)
    .agg(
        hourly_volume=("volume", "sum"),
        avg_speed=("speed", "mean"),
        avg_occupancy=("occupancy", "mean"),
        free_flow_speed=("free_flow_speed", "mean"),
        speed_drop_pct=("speed_drop_pct", "mean"),
        active_lanes=("detid", "nunique"),
    )
    .sort_values(["int_id", "hour_ts"])
    .reset_index(drop=True)
)

# Calendar & Cyclical Time Features (known at current hour t)
hourly["hour"] = hourly["hour_ts"].dt.hour
hourly["day_of_week"] = hourly["hour_ts"].dt.dayofweek
hourly["is_weekend_flag"] = (hourly["day_of_week"] >= 5).astype(int)
hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24.0)
hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24.0)

# -------------------------------------------------------------------------
# 2. Historical Feature Engineering (Strictly Past/Current t, t-1, t-2, t-3)
# -------------------------------------------------------------------------
grp = hourly.groupby("int_id")

hourly["lag_1h_volume"] = grp["hourly_volume"].shift(1)
hourly["lag_2h_volume"] = grp["hourly_volume"].shift(2)
hourly["lag_1h_speed"] = grp["avg_speed"].shift(1)
hourly["lag_1h_speed_drop"] = grp["speed_drop_pct"].shift(1)
hourly["lag_2h_speed_drop"] = grp["speed_drop_pct"].shift(2)

# Rolling 3-hour mean using only current & past observations (t, t-1, t-2)
hourly["rolling_3h_volume"] = grp["hourly_volume"].transform(
    lambda s: s.rolling(window=3, min_periods=1).mean()
)
hourly["rolling_3h_speed_drop"] = grp["speed_drop_pct"].transform(
    lambda s: s.rolling(window=3, min_periods=1).mean()
)

# -------------------------------------------------------------------------
# 3. Target Creation: Next-Hour (t + 1h) Congestion & Risk Class
# -------------------------------------------------------------------------
hourly["next_hour_ts"] = grp["hour_ts"].shift(-1)
hourly["target_next_hour_speed_drop"] = grp["speed_drop_pct"].shift(-1)
hourly["target_next_hour_volume"] = grp["hourly_volume"].shift(-1)

# Keep only rows where next_hour_ts is strictly +1 hour (no missing-hour jumps)
hourly["hour_diff"] = (hourly["next_hour_ts"] - hourly["hour_ts"]).dt.total_seconds() / 3600.0
model_df = hourly[hourly["hour_diff"] == 1.0].dropna().copy()

def map_risk_label(drop_pct):
    if drop_pct <= 15.0:
        return "Low Risk"
    elif drop_pct <= 35.0:
        return "Medium Risk"
    else:
        return "High Risk"

model_df["current_risk_level"] = model_df["speed_drop_pct"].apply(map_risk_label)
model_df["target_next_hour_risk"] = model_df["target_next_hour_speed_drop"].apply(map_risk_label)

print(f"   Hourly modeling dataset ready: {len(model_df):,} consecutive intersection-hours")
print("\n   Target Next-Hour Risk Class Distribution:")
print(model_df["target_next_hour_risk"].value_counts().to_string())

# -------------------------------------------------------------------------
# 4. Chronological Train-Test Split (80% Past / 20% Future — Zero Leakage)
# -------------------------------------------------------------------------
model_df = model_df.sort_values(["hour_ts", "int_id"]).reset_index(drop=True)
split_idx = int(len(model_df) * 0.80)
split_timestamp = model_df.loc[split_idx, "hour_ts"]

train_df = model_df[model_df["hour_ts"] < split_timestamp].copy()
test_df = model_df[model_df["hour_ts"] >= split_timestamp].copy()

feature_cols = [
    "int_id",
    "hourly_volume",
    "avg_speed",
    "avg_occupancy",
    "free_flow_speed",
    "speed_drop_pct",
    "active_lanes",
    "hour",
    "day_of_week",
    "is_weekend_flag",
    "hour_sin",
    "hour_cos",
    "lag_1h_volume",
    "lag_2h_volume",
    "lag_1h_speed",
    "lag_1h_speed_drop",
    "lag_2h_speed_drop",
    "rolling_3h_volume",
    "rolling_3h_speed_drop",
]

X_train, X_test = train_df[feature_cols], test_df[feature_cols]
y_reg_train, y_reg_test = (
    train_df["target_next_hour_speed_drop"],
    test_df["target_next_hour_speed_drop"],
)
y_clf_train, y_clf_test = (
    train_df["target_next_hour_risk"],
    test_df["target_next_hour_risk"],
)

print(f"\n2. Chronological Split Cutoff: {split_timestamp}")
print(f"   Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

# -------------------------------------------------------------------------
# 5. Train Feature A (Random Forest Regressor) & Feature B (RF Classifier)
# -------------------------------------------------------------------------
print("\n3. Training Random Forest Regressor (Next-Hour Speed Drop %)...")
rf_reg = RandomForestRegressor(
    n_estimators=150, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf_reg.fit(X_train, y_reg_train)
reg_preds = np.clip(rf_reg.predict(X_test), 0, 100)

mae = mean_absolute_error(y_reg_test, reg_preds)
rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
r2 = r2_score(y_reg_test, reg_preds)

print("4. Training Random Forest Classifier (Next-Hour Risk Class)...")
rf_clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_clf.fit(X_train, y_clf_train)
clf_preds = rf_clf.predict(X_test)
clf_probs = rf_clf.predict_proba(X_test)
high_risk_idx = list(rf_clf.classes_).index("High Risk")
high_risk_prob = clf_probs[:, high_risk_idx]

acc = accuracy_score(y_clf_test, clf_preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_clf_test, clf_preds, average="weighted", zero_division=0
)

print("\n=== MODEL EVALUATION RESULTS (CHRONOLOGICAL TEST SET) ===")
print(f"Regressor  -> MAE: {mae:.2f}% | RMSE: {rmse:.2f}% | R2 Score: {r2:.4f}")
print(
    f"Classifier -> Accuracy: {acc*100:.2f}% | Weighted Precision: {prec*100:.2f}% | "
    f"Weighted Recall: {rec*100:.2f}% | Weighted F1: {f1*100:.2f}%"
)
print("\nClassification Report:\n", classification_report(y_clf_test, clf_preds, zero_division=0))

# -------------------------------------------------------------------------
# 6. Save Trained Models via Joblib
# -------------------------------------------------------------------------
joblib.dump(rf_reg, "models/rf_congestion_regressor.joblib")
joblib.dump(rf_clf, "models/rf_risk_classifier.joblib")
print("Saved trained models to models/rf_congestion_regressor.joblib and models/rf_risk_classifier.joblib")

# -------------------------------------------------------------------------
# 7. Generate Evaluation Charts (Matplotlib / Seaborn)
# -------------------------------------------------------------------------
# Chart 1: Confusion Matrix
labels_order = ["Low Risk", "Medium Risk", "High Risk"]
cm = confusion_matrix(y_clf_test, clf_preds, labels=labels_order)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order
)
plt.title("Next-Hour Congestion Risk — Confusion Matrix")
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")
plt.tight_layout()
plt.savefig("dashboard/ml_evaluation/confusion_matrix.png", dpi=150)
plt.close()

# Chart 2: Actual vs Predicted Speed Drop %
sample_plot = pd.DataFrame(
    {"Actual": y_reg_test.values, "Predicted": reg_preds}
).sample(n=min(1500, len(y_reg_test)), random_state=42)
plt.figure(figsize=(6, 5))
plt.scatter(sample_plot["Actual"], sample_plot["Predicted"], alpha=0.35, color="#1f77b4", s=15)
plt.plot([0, 80], [0, 80], "r--", linewidth=1.5, label="Ideal Prediction")
plt.title(f"Actual vs. Predicted Next-Hour Speed Drop % (R² = {r2:.3f})")
plt.xlabel("Actual Next-Hour Speed Drop %")
plt.ylabel("Predicted Next-Hour Speed Drop %")
plt.legend()
plt.tight_layout()
plt.savefig("dashboard/ml_evaluation/actual_vs_predicted.png", dpi=150)
plt.close()

# Chart 3: Top 10 Feature Importances
importances = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(10)
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#2ca02c")
plt.title("Top 10 Predictive Features — Congestion Risk Classifier")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("dashboard/ml_evaluation/feature_importance.png", dpi=150)
plt.close()

# -------------------------------------------------------------------------
# 8. Export Predictions & Metrics for Power BI & SQLite Integration
# -------------------------------------------------------------------------
out_df = test_df[
    [
        "int_id",
        "intname",
        "hour_ts",
        "next_hour_ts",
        "hourly_volume",
        "avg_speed",
        "speed_drop_pct",
        "current_risk_level",
        "target_next_hour_speed_drop",
        "target_next_hour_risk",
    ]
].copy()

out_df["hour_ts"] = out_df["hour_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
out_df["next_hour_ts"] = out_df["next_hour_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
out_df["predicted_next_hour_speed_drop"] = np.round(reg_preds, 2)
out_df["predicted_next_hour_risk"] = clf_preds
out_df["high_risk_probability_pct"] = np.round(high_risk_prob * 100, 1)
out_df["prediction_correct"] = (
    out_df["target_next_hour_risk"] == out_df["predicted_next_hour_risk"]
).astype(int)

out_df.to_csv("data/cleaned/ml_next_hour_predictions.csv", index=False)

metrics_df = pd.DataFrame(
    [
        {"model": "Random Forest Regressor", "metric": "MAE (Speed Drop %)", "value": round(mae, 2)},
        {"model": "Random Forest Regressor", "metric": "RMSE (Speed Drop %)", "value": round(rmse, 2)},
        {"model": "Random Forest Regressor", "metric": "R2 Score", "value": round(r2, 4)},
        {"model": "Random Forest Classifier", "metric": "Test Accuracy %", "value": round(acc * 100, 2)},
        {"model": "Random Forest Classifier", "metric": "Weighted Precision %", "value": round(prec * 100, 2)},
        {"model": "Random Forest Classifier", "metric": "Weighted Recall %", "value": round(rec * 100, 2)},
        {"model": "Random Forest Classifier", "metric": "Weighted F1 Score %", "value": round(f1 * 100, 2)},
    ]
)
metrics_df.to_csv("data/cleaned/ml_model_metrics.csv", index=False)

# Sync into existing SQLite database
conn = sqlite3.connect("data/cleaned/urbanflow.db")
out_df.to_sql("ml_next_hour_predictions", conn, if_exists="replace", index=False)
metrics_df.to_sql("ml_model_metrics", conn, if_exists="replace", index=False)
conn.close()

print("\n=== SAMPLE NEXT-HOUR PREDICTIONS ===")
print(
    out_df[
        [
            "intname",
            "next_hour_ts",
            "target_next_hour_speed_drop",
            "predicted_next_hour_speed_drop",
            "target_next_hour_risk",
            "predicted_next_hour_risk",
            "high_risk_probability_pct",
        ]
    ]
    .head(8)
    .to_string(index=False)
)
print("\nStage 2 Complete! Exported predictions to data/cleaned/ml_next_hour_predictions.csv and urbanflow.db.")