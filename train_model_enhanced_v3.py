import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

telemetry = pd.read_csv("telemetry_enhanced_v2.csv")
rtt = pd.read_csv("rtt_telemetry_v2.csv")
links = pd.read_csv("link_state_telemetry_v2.csv")
faults = pd.read_csv("faults_log_enhanced_v2.csv")

for frame, column in [
    (telemetry, "timestamp"),
    (rtt, "timestamp"),
    (links, "timestamp"),
    (faults, "start_time"),
    (faults, "end_time"),
]:
    frame[column] = pd.to_datetime(frame[column], utc=True)

telemetry["label"] = "normal"
telemetry["in_fault_window"] = False

targets = {
    "link_down": {("r1", "eth2"), ("r2", "eth1")},
    "high_latency": {("r3", "eth2"), ("r4", "eth3")},
    "packet_loss": {("r3", "eth2"), ("r4", "eth3")},
    "congestion": {
        ("r2", "eth2"), ("r4", "eth2"),
        ("r3", "eth2"), ("r4", "eth3"),
    },
}

for fault in faults.itertuples(index=False):
    time_mask = (
        (telemetry["timestamp"] >= fault.start_time)
        & (telemetry["timestamp"] <= fault.end_time)
    )
    telemetry.loc[time_mask, "in_fault_window"] = True

    target_mask = pd.Series(False, index=telemetry.index)
    for router, interface in targets[fault.fault_type]:
        target_mask |= (
            telemetry["router"].eq(router)
            & telemetry["interface"].eq(interface)
        )

    telemetry.loc[time_mask & target_mask, "label"] = fault.fault_type

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
]
X = telemetry[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

train_mask = ~telemetry["in_fault_window"]
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)
model.fit(X.loc[train_mask])

telemetry["anomaly_score"] = model.decision_function(X)
telemetry["iforest_prediction"] = pd.Series(model.predict(X), index=telemetry.index).map({
    -1: "anomaly",
     1: "normal",
})

telemetry["latency_rule"] = telemetry["rtt_ms"] >= 100
telemetry["link_down_rule"] = telemetry["link_up"].eq(0)
telemetry["hybrid_prediction"] = np.where(
    telemetry["latency_rule"] | telemetry["link_down_rule"],
    "anomaly",
    telemetry["iforest_prediction"],
)

fault_rows = telemetry[telemetry["label"] != "normal"]
normal_rows = telemetry[~telemetry["in_fault_window"]]

print(f"Loaded {len(telemetry)} rows across {telemetry['router'].nunique()} routers.")
print(f"Training rows outside fault windows: {train_mask.sum()}")
print(f"Fault rows on affected interfaces: {len(fault_rows)}")
print(f"Matched RTT samples: {(1 - telemetry['rtt_missing'].mean()):.1%}")
print(f"Matched link-state samples: {(1 - telemetry['link_missing'].mean()):.1%}")

for prediction_col, title in [
    ("iforest_prediction", "Isolation Forest only"),
    ("hybrid_prediction", "Hybrid (Isolation Forest + RTT/link rules)"),
]:
    caught = fault_rows[prediction_col].eq("anomaly").sum()
    false_alarms = normal_rows[prediction_col].eq("anomaly").sum()

    print(f"\n--- {title} ---")
    print(f"Affected fault rows detected: {caught} / {len(fault_rows)}")
    print(f"Normal rows flagged: {false_alarms} / {len(normal_rows)}")

    for fault_type in faults["fault_type"].unique():
        rows = telemetry[telemetry["label"] == fault_type]
        if len(rows):
            rate = rows[prediction_col].eq("anomaly").mean()
            print(f"  {fault_type}: {rate:.0%} ({len(rows)} affected-interface rows)")

telemetry.to_csv("telemetry_scored_enhanced_v3.csv", index=False)
print("\nSaved results to telemetry_scored_enhanced_v3.csv")
