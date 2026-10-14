import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

TELEMETRY_FILE = "telemetry_enhanced_v2.csv"
RTT_FILE = "rtt_telemetry_v2.csv"
LINK_FILE = "link_state_telemetry_v2.csv"
FAULTS_FILE = "faults_log_enhanced_v2.csv"
OUTPUT_FILE = "telemetry_scored_enhanced_v2.csv"

telemetry = pd.read_csv(TELEMETRY_FILE)
rtt = pd.read_csv(RTT_FILE)
links = pd.read_csv(LINK_FILE)
faults = pd.read_csv(FAULTS_FILE)

for frame, column in [
    (telemetry, "timestamp"),
    (rtt, "timestamp"),
    (links, "timestamp"),
    (faults, "start_time"),
    (faults, "end_time"),
]:
    frame[column] = pd.to_datetime(frame[column], utc=True)

telemetry["label"] = "normal"
for fault in faults.itertuples(index=False):
    in_window = (
        (telemetry["timestamp"] >= fault.start_time)
        & (telemetry["timestamp"] <= fault.end_time)
    )
    telemetry.loc[in_window, "label"] = fault.fault_type

keys = ["router", "interface"]
telemetry = telemetry.sort_values(keys + ["timestamp"]).reset_index(drop=True)

dt = telemetry.groupby(keys)["timestamp"].diff().dt.total_seconds()
telemetry["dt_seconds"] = dt.where(dt > 0, 1).fillna(1)

counter_cols = [
    "rx_packets", "rx_bytes", "rx_errors", "rx_dropped",
    "tx_packets", "tx_bytes", "tx_errors", "tx_dropped",
]
rate_cols = []

for col in counter_cols:
    delta = telemetry.groupby(keys)[col].diff().fillna(0).clip(lower=0)
    rate_col = f"{col}_rate"
    telemetry[rate_col] = delta / telemetry["dt_seconds"]
    rate_cols.append(rate_col)

telemetry["rx_drop_rate"] = telemetry["rx_dropped_rate"] / (telemetry["rx_packets_rate"] + 1)
telemetry["tx_drop_rate"] = telemetry["tx_dropped_rate"] / (telemetry["tx_packets_rate"] + 1)
telemetry["rx_error_rate"] = telemetry["rx_errors_rate"] / (telemetry["rx_packets_rate"] + 1)
telemetry["tx_error_rate"] = telemetry["tx_errors_rate"] / (telemetry["tx_packets_rate"] + 1)

rtt["rtt_ms"] = pd.to_numeric(rtt["rtt_ms"], errors="coerce")
rtt["rtt_timeout"] = rtt["status"].eq("timeout").astype(int)
rtt["rtt_observed"] = 1
rtt_features = rtt[["timestamp", "rtt_ms", "rtt_timeout", "rtt_observed"]]
telemetry = pd.merge_asof(
    telemetry.sort_values("timestamp"),
    rtt_features.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(seconds=8),
)

telemetry["rtt_missing"] = telemetry["rtt_observed"].isna().astype(int)
telemetry["rtt_timeout"] = telemetry["rtt_timeout"].fillna(0)
telemetry["rtt_ms"] = telemetry["rtt_ms"].fillna(0)

links["link_up"] = links["oper_state"].str.upper().eq("UP").astype(int)
links["link_observed"] = 1
link_features = links[["timestamp", "router", "interface", "link_up", "link_observed"]]
telemetry = pd.merge_asof(
    telemetry.sort_values(["timestamp", "router", "interface"]),
    link_features.sort_values(["timestamp", "router", "interface"]),
    on="timestamp",
    by=["router", "interface"],
    direction="nearest",
    tolerance=pd.Timedelta(seconds=8),
)

telemetry["link_missing"] = telemetry["link_observed"].isna().astype(int)
telemetry["link_up"] = telemetry["link_up"].fillna(1)

feature_cols = rate_cols + [
    "rx_drop_rate", "tx_drop_rate", "rx_error_rate", "tx_error_rate",
    "rtt_ms", "rtt_timeout", "rtt_missing", "link_up", "link_missing",
]
X = telemetry[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

normal_mask = telemetry["label"].eq("normal")
print(f"Loaded {len(telemetry)} telemetry rows across {telemetry['router'].nunique()} routers.")
print(f"Loaded {len(faults)} labeled fault windows.")
print(f"Training rows (normal only): {normal_mask.sum()}")
print(f"Features used: {len(feature_cols)}")
print(f"RTT samples matched: {(1 - telemetry['rtt_missing'].mean()):.1%}")
print(f"Link-state samples matched: {(1 - telemetry['link_missing'].mean()):.1%}")

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)
model.fit(X.loc[normal_mask])

telemetry["anomaly_score"] = model.decision_function(X)
telemetry["prediction"] = model.predict(X)
telemetry["predicted"] = telemetry["prediction"].map({
    -1: "anomaly",
     1: "normal",
})

fault_rows = telemetry[telemetry["label"] != "normal"]
normal_rows = telemetry[telemetry["label"] == "normal"]
caught = fault_rows["predicted"].eq("anomaly").sum()
false_alarms = normal_rows["predicted"].eq("anomaly").sum()

print("\n--- Enhanced V2 Results ---")
print(f"Fault rows caught as anomalies: {caught} / {len(fault_rows)}")
print(f"Fault rows missed: {len(fault_rows) - caught} / {len(fault_rows)}")
print(f"Normal rows flagged (false alarms): {false_alarms} / {len(normal_rows)}")
print("\nDetection rate by fault type:")

for fault_type in faults["fault_type"].unique():
    rows = telemetry[telemetry["label"] == fault_type]
    if len(rows):
        rate = rows["predicted"].eq("anomaly").mean()
        print(f"  {fault_type}: {rate:.0%} ({len(rows)} rows)")

telemetry.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved results to {OUTPUT_FILE}")
