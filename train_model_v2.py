"""
AIOps Model V2
- Uses time-normalized traffic rates
- Trains Isolation Forest on normal data only
- Scores all telemetry
- Produces a real continuous anomaly score
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
telemetry = pd.read_csv("telemetry.csv", parse_dates=["timestamp"])
faults = pd.read_csv(
    "faults_log.csv",
    parse_dates=["start_time", "end_time"]
)

print(f"Loaded {len(telemetry)} telemetry rows across "
      f"{telemetry['router'].nunique()} routers.")
print(f"Loaded {len(faults)} labeled fault windows.")

# ---------------------------------------------------------------------
# 2. Label telemetry rows
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
# 3. Sort telemetry
# ---------------------------------------------------------------------
telemetry = telemetry.sort_values(
    ["router", "interface", "timestamp"]
).reset_index(drop=True)

# ---------------------------------------------------------------------
# 4. Calculate elapsed time between samples
# ---------------------------------------------------------------------
telemetry["dt_seconds"] = (
    telemetry
    .groupby(["router", "interface"])["timestamp"]
    .diff()
    .dt.total_seconds()
)

# Prevent division by zero
telemetry["dt_seconds"] = telemetry["dt_seconds"].fillna(1)
telemetry.loc[telemetry["dt_seconds"] <= 0, "dt_seconds"] = 1

# ---------------------------------------------------------------------
# 5. Calculate counter deltas
# ---------------------------------------------------------------------
counter_cols = [
    "rx_packets",
    "rx_bytes",
    "rx_errors",
    "rx_dropped",
    "tx_packets",
    "tx_bytes",
    "tx_errors",
    "tx_dropped",
]

for col in counter_cols:
    telemetry[f"{col}_delta"] = (
        telemetry
        .groupby(["router", "interface"])[col]
        .diff()
    )

# First sample for each interface has no previous sample
delta_cols = []

for col in counter_cols:
    delta_col = f"{col}_delta"
    telemetry[delta_col] = telemetry[delta_col].fillna(0)

    # Counter resets can produce negative values.
    # Treat them as zero for rate calculations.
    telemetry[delta_col] = telemetry[delta_col].clip(lower=0)

    delta_cols.append(delta_col)

# ---------------------------------------------------------------------
# 6. Convert counter deltas into rates per second
# ---------------------------------------------------------------------
rate_cols = []

for delta_col in delta_cols:
    rate_col = delta_col.replace("_delta", "_rate")

    telemetry[rate_col] = (
        telemetry[delta_col] / telemetry["dt_seconds"]
    )

    rate_cols.append(rate_col)

# ---------------------------------------------------------------------
# 7. Add useful drop/error ratios
# ---------------------------------------------------------------------
telemetry["rx_drop_rate"] = (
    telemetry["rx_dropped_rate"] /
    (telemetry["rx_packets_rate"] + 1)
)

telemetry["tx_drop_rate"] = (
    telemetry["tx_dropped_rate"] /
    (telemetry["tx_packets_rate"] + 1)
)

telemetry["rx_error_rate"] = (
    telemetry["rx_errors_rate"] /
    (telemetry["rx_packets_rate"] + 1)
)

telemetry["tx_error_rate"] = (
    telemetry["tx_errors_rate"] /
    (telemetry["tx_packets_rate"] + 1)
)

# ---------------------------------------------------------------------
# 8. Select ML features
# ---------------------------------------------------------------------
feature_cols = rate_cols + [
    "rx_drop_rate",
    "tx_drop_rate",
    "rx_error_rate",
    "tx_error_rate",
]

X = telemetry[feature_cols]

# ---------------------------------------------------------------------
# 9. Train ONLY on normal telemetry
# ---------------------------------------------------------------------
normal_mask = telemetry["label"] == "normal"

X_normal = X[normal_mask]

print(f"\nTraining rows (normal only): {len(X_normal)}")
print(f"Features used: {len(feature_cols)}")

# ---------------------------------------------------------------------
# 10. Train Isolation Forest
# ---------------------------------------------------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)

model.fit(X_normal)

# ---------------------------------------------------------------------
# 11. Score every telemetry row
# ---------------------------------------------------------------------
telemetry["anomaly_score"] = model.decision_function(X)

# Lower score = more anomalous
telemetry["prediction"] = model.predict(X)

telemetry["predicted"] = telemetry["prediction"].map({
    -1: "anomaly",
     1: "normal"
})

# ---------------------------------------------------------------------
# 12. Evaluate
# ---------------------------------------------------------------------
actual_fault_rows = telemetry[
    telemetry["label"] != "normal"
]

actual_normal_rows = telemetry[
    telemetry["label"] == "normal"
]

caught = (
    actual_fault_rows["predicted"] == "anomaly"
).sum()

missed = (
    actual_fault_rows["predicted"] == "normal"
).sum()

false_alarms = (
    actual_normal_rows["predicted"] == "anomaly"
).sum()

print("\n--- V2 Results ---")

print(
    f"Fault rows caught as anomalies:  "
    f"{caught} / {len(actual_fault_rows)}"
)

print(
    f"Fault rows missed:               "
    f"{missed} / {len(actual_fault_rows)}"
)

print(
    f"Normal rows flagged (false alarms): "
    f"{false_alarms} / {len(actual_normal_rows)}"
)

# ---------------------------------------------------------------------
# 13. Detection rate by fault type
# ---------------------------------------------------------------------
print("\nDetection rate by fault type:")

for fault_type in faults["fault_type"].unique():

    rows = telemetry[
        telemetry["label"] == fault_type
    ]

    if len(rows) == 0:
        print(
            f"  {fault_type}: "
            "no telemetry rows found in this window"
        )
        continue

    hit_rate = (
        rows["predicted"] == "anomaly"
    ).mean()

    print(
        f"  {fault_type}: "
        f"{hit_rate:.0%} of rows flagged "
        f"({len(rows)} rows total)"
    )

# ---------------------------------------------------------------------
# 14. Save results
# ---------------------------------------------------------------------
telemetry.to_csv(
    "telemetry_scored_v2.csv",
    index=False
)

print("\nSaved results to telemetry_scored_v2.csv")
