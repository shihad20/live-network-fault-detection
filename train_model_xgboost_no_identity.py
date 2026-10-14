from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(".")
EXPECTED_FAULTS = {"link_down", "high_latency", "packet_loss", "congestion"}

TARGETS = {
    "link_down": {("r1", "eth2"), ("r2", "eth1")},
    "high_latency": {("r3", "eth2"), ("r4", "eth3")},
    "packet_loss": {("r3", "eth2"), ("r4", "eth3")},
    "congestion": {
        ("r2", "eth2"), ("r4", "eth2"),
        ("r3", "eth2"), ("r4", "eth3"),
    },
}

REQUIRED = [
    "telemetry.csv", "rtt.csv", "link_state.csv",
    "throughput.csv", "faults.csv",
]


def usable_run(path):
    if not all((path / name).is_file() for name in REQUIRED):
        return False
    faults = pd.read_csv(path / "faults.csv")
    throughput = pd.read_csv(path / "throughput.csv")
    telemetry = pd.read_csv(path / "telemetry.csv")
    ok_samples = throughput["status"].eq("ok").sum()
    return (
        len(faults) == 4
        and set(faults["fault_type"]) == EXPECTED_FAULTS
        and ok_samples >= 70
        and len(telemetry) > 0
    )


runs = sorted(
    (p for p in (ROOT / "dataset_runs").glob("run_*") if usable_run(p)),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)[:30]

if len(runs) != 30:
    raise SystemExit(
        f"Expected 30 usable runs with four fault labels and throughput data; "
        f"found {len(runs)}. No model was trained."
    )

frames = []
for run in runs:
    data = pd.read_csv(run / "telemetry.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["run_id"] = run.name
    data["label"] = "normal"

    faults = pd.read_csv(run / "faults.csv")
    faults["start_time"] = pd.to_datetime(faults["start_time"], utc=True)
    faults["end_time"] = pd.to_datetime(faults["end_time"], utc=True)

    for fault in faults.itertuples(index=False):
        in_window = data["timestamp"].between(fault.start_time, fault.end_time)
        for router, interface in TARGETS[fault.fault_type]:
            affected = (
                in_window
                & data["router"].eq(router)
                & data["interface"].eq(interface)
            )
            data.loc[affected, "label"] = fault.fault_type

    frames.append(data)

data = pd.concat(frames, ignore_index=True)
keys = ["run_id", "router", "interface"]
data = data.sort_values(keys + ["timestamp"]).reset_index(drop=True)

counter_cols = [
    "rx_packets", "rx_bytes", "rx_errors", "rx_dropped",
    "tx_packets", "tx_bytes", "tx_errors", "tx_dropped",
]
groups = data.groupby(keys, sort=False)
dt = groups["timestamp"].diff().dt.total_seconds()
data["dt_seconds"] = dt.where(dt > 0, np.nan).fillna(1)

rate_cols = []
for col in counter_cols:
    delta = data.groupby(keys, sort=False)[col].diff().fillna(0).clip(lower=0)
    name = f"{col}_rate"
    data[name] = delta / data["dt_seconds"]
    rate_cols.append(name)

for direction in ("rx", "tx"):
    data[f"{direction}_drop_ratio"] = (
        data[f"{direction}_dropped_rate"] / (data[f"{direction}_packets_rate"] + 1)
    )
    data[f"{direction}_error_ratio"] = (
        data[f"{direction}_errors_rate"] / (data[f"{direction}_packets_rate"] + 1)
    )

rtt_parts = []
link_parts = []
throughput_parts = []

for run in runs:
    run_id = run.name

    rtt = pd.read_csv(run / "rtt.csv")
    rtt["timestamp"] = pd.to_datetime(rtt["timestamp"], utc=True)
    rtt["run_id"] = run_id
    rtt["rtt_avg_ms"] = pd.to_numeric(rtt["rtt_avg_ms"], errors="coerce")
    rtt["packet_loss_pct"] = pd.to_numeric(rtt["packet_loss_pct"], errors="coerce")
    rtt_parts.append(
        rtt[["run_id", "timestamp", "rtt_avg_ms", "packet_loss_pct"]]
    )

    links = pd.read_csv(run / "link_state.csv")
    links["timestamp"] = pd.to_datetime(links["timestamp"], utc=True)
    links["run_id"] = run_id
    links["link_up"] = links["oper_state"].str.upper().eq("UP").astype(int)
    link_parts.append(
        links[["run_id", "router", "interface", "timestamp", "link_up"]]
    )

    throughput = pd.read_csv(run / "throughput.csv")
    throughput["timestamp"] = pd.to_datetime(throughput["timestamp"], utc=True)
    throughput["run_id"] = run_id
    throughput["throughput_mbps"] = pd.to_numeric(
        throughput["throughput_mbps"], errors="coerce"
    )
    throughput_parts.append(
        throughput[["run_id", "timestamp", "throughput_mbps"]]
    )

rtt_all = pd.concat(rtt_parts, ignore_index=True).sort_values("timestamp")
data = pd.merge_asof(
    data.sort_values("timestamp"),
    rtt_all,
    on="timestamp",
    by="run_id",
    direction="nearest",
    tolerance=pd.Timedelta(seconds=8),
)
data["rtt_missing"] = data["rtt_avg_ms"].isna().astype(int)
data["rtt_avg_ms"] = data["rtt_avg_ms"].fillna(0)
data["packet_loss_pct"] = data["packet_loss_pct"].fillna(0)

link_all = pd.concat(link_parts, ignore_index=True).sort_values("timestamp")
data = pd.merge_asof(
    data.sort_values("timestamp"),
    link_all,
    on="timestamp",
    by=["run_id", "router", "interface"],
    direction="nearest",
    tolerance=pd.Timedelta(seconds=8),
)
data["link_missing"] = data["link_up"].isna().astype(int)
data["link_up"] = data["link_up"].fillna(1)

throughput_all = pd.concat(throughput_parts, ignore_index=True).sort_values("timestamp")
data = pd.merge_asof(
    data.sort_values("timestamp"),
    throughput_all,
    on="timestamp",
    by="run_id",
    direction="nearest",
    tolerance=pd.Timedelta(seconds=8),
)
data["throughput_missing"] = data["throughput_mbps"].isna().astype(int)
data["throughput_mbps"] = data["throughput_mbps"].fillna(0)

numeric_cols = rate_cols + [
    "rx_drop_ratio", "tx_drop_ratio", "rx_error_ratio", "tx_error_ratio",
    "rtt_avg_ms", "packet_loss_pct", "rtt_missing",
    "link_up", "link_missing", "throughput_mbps", "throughput_missing",
]
numeric = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
X = numeric.astype(float)

encoder = LabelEncoder()
y = encoder.fit_transform(data["label"])
groups = data["run_id"].to_numpy()

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
if set(groups[train_idx]) & set(groups[test_idx]):
    raise SystemExit("Run leakage detected; no model was trained.")

weights = compute_sample_weight("balanced", y[train_idx])
model = XGBClassifier(
    n_estimators=350,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    device="cpu",
    n_jobs=4,
    random_state=42,
)
model.fit(X.iloc[train_idx], y[train_idx], sample_weight=weights)

pred = model.predict(X.iloc[test_idx]).astype(int)
normal_id = int(encoder.transform(["normal"])[0])
report = classification_report(
    y[test_idx],
    pred,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0,
    output_dict=True,
)
matrix = confusion_matrix(
    y[test_idx], pred, labels=np.arange(len(encoder.classes_))
)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
outdir = ROOT / "model_runs" / f"xgb_30run_{stamp}"
outdir.mkdir(parents=True, exist_ok=False)
model.save_model(outdir / "model.json")

predictions = data.iloc[test_idx][
    ["run_id", "timestamp", "router", "interface", "label"]
].copy()
predictions["predicted"] = encoder.inverse_transform(pred)
predictions.to_csv(outdir / "holdout_predictions.csv", index=False)

metrics = {
    "runs_used": [p.name for p in reversed(runs)],
    "train_run_count": len(set(groups[train_idx])),
    "test_run_count": len(set(groups[test_idx])),
    "train_rows": int(len(train_idx)),
    "test_rows": int(len(test_idx)),
    "accuracy": float(accuracy_score(y[test_idx], pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
    "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
    "normal_false_alarm_rate": float(
        ((y[test_idx] == normal_id) & (pred != normal_id)).sum()
        / max(1, (y[test_idx] == normal_id).sum())
    ),
    "class_report": report,
    "confusion_matrix": matrix.tolist(),
    "class_names": encoder.classes_.tolist(),
    "feature_names": X.columns.tolist(),
}
(outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

print(f"Usable runs: {len(runs)}")
print(f"Training runs: {metrics['train_run_count']}; held-out test runs: {metrics['test_run_count']}")
print(f"Rows: {len(data)} total; {len(train_idx)} train; {len(test_idx)} test")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
print(f"Macro F1: {metrics['macro_f1']:.3f}")
print(f"Normal false-alarm rate: {metrics['normal_false_alarm_rate']:.1%}")
print("\nPer-class results:")
for name, values in report.items():
    if isinstance(values, dict) and "f1-score" in values:
        print(
            f"  {name}: precision={values['precision']:.3f}, "
            f"recall={values['recall']:.3f}, F1={values['f1-score']:.3f}"
        )
print(f"\nSaved new model and evaluation files in: {outdir}")
