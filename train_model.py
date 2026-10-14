"""
First AIOps model: load telemetry + fault labels, train an Isolation
Forest anomaly detector, and report how well it catches the faults you
actually injected.

Run from inside the aiops-env virtual environment:
    python3 train_model.py
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------
telemetry = pd.read_csv("telemetry.csv", parse_dates=["timestamp"])
faults = pd.read_csv("faults_log.csv", parse_dates=["start_time", "end_time"])

print(f"Loaded {len(telemetry)} telemetry rows across "
      f"{telemetry['router'].nunique()} routers.")
print(f"Loaded {len(faults)} labeled fault windows.")

# ---------------------------------------------------------------------
# 2. Label each telemetry row: normal, or which fault it falls inside
# ---------------------------------------------------------------------
telemetry["label"] = "normal"

for _, fault in faults.iterrows():
    in_window = (
        (telemetry["timestamp"] >= fault["start_time"])
        & (telemetry["timestamp"] <= fault["end_time"])
    )
    telemetry.loc[in_window, "label"] = fault["fault_type"]

print("\nRows per label:")
print(telemetry["label"].value_counts())

# ---------------------------------------------------------------------
# 3. Build features per router+interface over time
#    (rate of change is more informative than raw cumulative counters,
#    since FRR reports totals since boot, not per-interval deltas)
# ---------------------------------------------------------------------
telemetry = telemetry.sort_values(["router", "interface", "timestamp"])

feature_cols = ["rx_packets", "rx_bytes", "rx_errors", "rx_dropped",
                 "tx_packets", "tx_bytes", "tx_errors", "tx_dropped"]

for col in feature_cols:
    telemetry[f"{col}_delta"] = (
        telemetry.groupby(["router", "interface"])[col].diff().fillna(0)
    )

delta_cols = [f"{c}_delta" for c in feature_cols]

# ---------------------------------------------------------------------
# 4. Train Isolation Forest on the deltas
#    contamination = rough guess at what fraction of rows are anomalous
# ---------------------------------------------------------------------
X = telemetry[delta_cols]

fault_fraction = (telemetry["label"] != "normal").mean()
contamination = max(min(fault_fraction, 0.5), 0.01)  # keep it sane

model = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
)
telemetry["anomaly_score"] = model.fit_predict(X)
# fit_predict returns -1 for anomaly, 1 for normal
telemetry["predicted"] = telemetry["anomaly_score"].map(
    {-1: "anomaly", 1: "normal"}
)

# ---------------------------------------------------------------------
# 5. Evaluate: how many real faults did it catch?
# ---------------------------------------------------------------------
actual_fault_rows = telemetry[telemetry["label"] != "normal"]
actual_normal_rows = telemetry[telemetry["label"] == "normal"]

caught = (actual_fault_rows["predicted"] == "anomaly").sum()
missed = (actual_fault_rows["predicted"] == "normal").sum()
false_alarms = (actual_normal_rows["predicted"] == "anomaly").sum()

print("\n--- Results ---")
print(f"Fault rows caught as anomalies:  {caught} / {len(actual_fault_rows)}")
print(f"Fault rows missed:               {missed} / {len(actual_fault_rows)}")
print(f"Normal rows flagged (false alarms): {false_alarms} / {len(actual_normal_rows)}")

print("\nDetection rate by fault type:")
for fault_type in faults["fault_type"].unique():
    rows = telemetry[telemetry["label"] == fault_type]
    if len(rows) == 0:
        print(f"  {fault_type}: no telemetry rows found in this window")
        continue
    hit_rate = (rows["predicted"] == "anomaly").mean()
    print(f"  {fault_type}: {hit_rate:.0%} of rows flagged "
          f"({len(rows)} rows total)")

# ---------------------------------------------------------------------
# 6. Save the labeled + scored data for later use (dashboard, review)
# ---------------------------------------------------------------------
telemetry.to_csv("telemetry_scored.csv", index=False)
print("\nSaved full results to telemetry_scored.csv")
