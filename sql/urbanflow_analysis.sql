-- ============================================================================
-- Project: UrbanFlow – Smart City Traffic Pattern & Congestion Intelligence
-- Dataset: City of Austin Radar Traffic Counts (Cleaned Fact Table)
-- Author: [Your Name]
-- Description: Analytical SQL queries identifying congestion hotspots,
--              peak hours, weekday vs. weekend shifts, directional bottlenecks,
--              and rolling peak traffic windows.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 0. SCHEMA & DATA QUALITY CHECK
-- Verify total cleaned records, distinct intersections, and date coverage
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_cleaned_records,
    COUNT(DISTINCT int_id) AS total_intersections,
    MIN(date) AS observation_start_date,
    MAX(date) AS observation_end_date,
    SUM(volume) AS total_vehicles_observed
FROM fact_traffic_cleaned;


-- ----------------------------------------------------------------------------
-- 1. TOP 10 CONGESTION HOTSPOTS (SEVERITY + VOLUME RANKING)
-- Identifies intersections experiencing the steepest speed drops relative to
-- their own 85th-percentile free-flow speed (minimum 100 intervals for validity)
-- ----------------------------------------------------------------------------
SELECT 
    int_id,
    intname AS intersection_name,
    COUNT(*) AS active_intervals,
    SUM(volume) AS total_vehicles,
    ROUND(AVG(speed), 1) AS avg_actual_speed_mph,
    ROUND(AVG(free_flow_speed), 1) AS free_flow_speed_mph,
    ROUND(AVG(speed_drop_pct), 1) AS avg_speed_drop_pct,
    ROUND(AVG(occupancy), 1) AS avg_lane_occupancy_pct
FROM fact_traffic_cleaned
GROUP BY int_id, intname
HAVING COUNT(*) >= 100
ORDER BY avg_speed_drop_pct DESC, total_vehicles DESC
LIMIT 10;


-- ----------------------------------------------------------------------------
-- 2. HOURLY TRAFFIC PROFILE & CONGESTION CURVE (00:00 TO 23:00)
-- Quantifies the inverse relationship between vehicle volume and average speed
-- across the 24-hour cycle to pinpoint rush-hour peaks
-- ----------------------------------------------------------------------------
SELECT 
    hour,
    SUM(volume) AS total_volume,
    ROUND(AVG(volume), 1) AS avg_15min_volume,
    ROUND(AVG(speed), 1) AS avg_speed_mph,
    ROUND(AVG(occupancy), 1) AS avg_occupancy_pct,
    ROUND(AVG(speed_drop_pct), 1) AS avg_speed_drop_pct
FROM fact_traffic_cleaned
GROUP BY hour
ORDER BY hour ASC;


-- ----------------------------------------------------------------------------
-- 3. WEEKDAY VS. WEEKEND CONGESTION SHIFT BY TIME SLOT
-- Uses conditional aggregation (CASE WHEN) to calculate the exact percentage
-- of severe congestion intervals across commuter vs. leisure periods
-- ----------------------------------------------------------------------------
SELECT 
    is_weekend,
    time_period,
    COUNT(*) AS total_intervals,
    SUM(volume) AS total_volume,
    ROUND(AVG(speed), 1) AS avg_speed_mph,
    ROUND(AVG(speed_drop_pct), 1) AS avg_speed_drop_pct,
    ROUND(
        100.0 * SUM(CASE WHEN congestion_level = 'Severe Congestion' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS severe_congestion_share_pct
FROM fact_traffic_cleaned
GROUP BY is_weekend, time_period
ORDER BY is_weekend ASC, time_period ASC;


-- ----------------------------------------------------------------------------
-- 4. DIRECTIONAL LANE BOTTLENECK ANALYSIS
-- Isolates specific travel directions (NB, SB, EB, WB) and lanes (inside/outside)
-- to determine whether congestion is asymmetric (e.g., inbound morning rush)
-- ----------------------------------------------------------------------------
SELECT 
    intname AS intersection_name,
    detname AS lane_direction,
    SUM(volume) AS total_vehicles,
    ROUND(AVG(speed), 1) AS avg_speed_mph,
    ROUND(AVG(speed_drop_pct), 1) AS avg_speed_drop_pct
FROM fact_traffic_cleaned
GROUP BY intname, detname
HAVING COUNT(*) >= 100
ORDER BY avg_speed_drop_pct DESC, total_vehicles DESC
LIMIT 15;


-- ----------------------------------------------------------------------------
-- 5. ROLLING 3-HOUR PEAK DEMAND WINDOW (CTE + WINDOW FUNCTION)
-- Aggregates 15-minute intervals into hourly totals and applies a sliding
-- 3-hour window function to detect sustained multi-hour traffic surges
-- ----------------------------------------------------------------------------
WITH hourly_traffic AS (
    SELECT 
        intname AS intersection_name,
        date,
        hour,
        SUM(volume) AS hourly_volume,
        ROUND(AVG(speed), 1) AS hourly_avg_speed
    FROM fact_traffic_cleaned
    GROUP BY intname, date, hour
)
SELECT 
    intersection_name,
    date,
    hour,
    hourly_volume,
    hourly_avg_speed,
    ROUND(
        AVG(hourly_volume) OVER (
            PARTITION BY intersection_name, date 
            ORDER BY hour 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ),
        0
    ) AS rolling_3hr_avg_volume,
    RANK() OVER (
        PARTITION BY intersection_name 
        ORDER BY hourly_volume DESC
    ) AS peak_hour_rank
FROM hourly_traffic
ORDER BY hourly_volume DESC
LIMIT 20;
-- Q6: Top AI-Predicted High & Medium Risk Congestion Intersections
SELECT 
    int_id,
    intname AS intersection_name,
    COUNT(*) AS forecasted_hours,
    ROUND(AVG(target_next_hour_speed_drop), 2) AS avg_actual_speed_drop_pct,
    ROUND(AVG(predicted_next_hour_speed_drop), 2) AS avg_predicted_speed_drop_pct,
    SUM(CASE WHEN predicted_next_hour_risk = 'High Risk' THEN 1 ELSE 0 END) AS predicted_high_risk_hours,
    SUM(CASE WHEN predicted_next_hour_risk = 'Medium Risk' THEN 1 ELSE 0 END) AS predicted_medium_risk_hours,
    ROUND(MAX(high_risk_probability_pct), 1) AS peak_high_risk_prob_pct
FROM ml_next_hour_predictions
GROUP BY int_id, intname
ORDER BY predicted_high_risk_hours DESC, avg_predicted_speed_drop_pct DESC
LIMIT 15;

-- Q7: ML Model Classification Accuracy & Forecast Error by Intersection
SELECT 
    intname AS intersection_name,
    COUNT(*) AS evaluated_hours,
    ROUND(AVG(ABS(target_next_hour_speed_drop - predicted_next_hour_speed_drop)), 2) AS intersection_mae_pct,
    ROUND(100.0 * SUM(prediction_correct) / COUNT(*), 2) AS risk_classification_accuracy_pct
FROM ml_next_hour_predictions
GROUP BY intname
ORDER BY evaluated_hours DESC;
