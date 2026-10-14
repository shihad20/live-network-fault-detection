from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path.home() / "aiops-project"
SYNTHETIC = ROOT / "synthetic_aiops_data" / "synthetic_labeled_dataset.csv"
EXPECTED_FAULTS = {"link_down", "high_latency", "packet_loss", "congestion"}
FEATURES = [
    "rx_packets_rate", "rx_bytes_rate", "rx_errors_rate", "rx_dropped_rate",
    "tx_packets_rate", "tx_bytes_rate", "tx_errors_rate", "tx_dropped_rate",
    "rx_drop_ratio", "tx_drop_ratio", "rx_error_ratio", "tx_error_ratio",
    "rtt_avg_ms", "packet_loss_pct", "throughput_mbps", "link_up",
]
LINK_ENDPOINTS = {
    "r1-r2": {("r1", "eth2"), ("r2", "eth1")},
    "r1-r3": {("r1", "eth3"), ("r3", "eth1")},
    "r2-r4": {("r2", "eth2"), ("r4", "eth2")},
    "r3-r4": {("r3", "eth2"), ("r4", "eth3")},
}
ALL_CONGESTION_ENDPOINTS = set().union(*LINK_ENDPOINTS.values())


if not SYNTHETIC.is_file():
    raise SystemExit(f"Synthetic labeled CSV not found: {SYNTHETIC}")

synthetic = pd.read_csv(SYNTHETIC)
missing = sorted(set(FEATURES + ["run_id", "label"]) - set(synthetic.columns))
if missing:
    raise SystemExit(f"Synthetic CSV is missing required columns: {missing}")

run_dirs = []
for path in (ROOT / "dataset_runs").glob("run_*"):
    required = ["telemetry.csv", "rtt.csv", "link_state.csv", "throughput.csv", "faults.csv"]
    if not all((path / name).is_file() for name in required):
        continue
    faults = pd.read_csv(path / "faults.csv")
    throughput = pd.read_csv(path / "throughput.csv")
    if (
        len(faults) == 4
        and set(faults["fault_type"]) == EXPECTED_FAULTS
        and throughput["status"].eq("ok").sum() >= 70
    ):
        run_dirs.append(path)
run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[:30]
if not run_dirs:
    raise SystemExit("No complete real runs with throughput samples were found.")

frames = []
rtt_frames, link_frames, throughput_frames = [], [], []
for run in run_dirs:
    rid = run.name
    data = pd.read_csv(run / "telemetry.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["run_id"] = rid
    data["label"] = "normal"

    faults = pd.read_csv(run / "faults.csv")
    faults["start_time"] = pd.to_datetime(faults["start_time"], utc=True)
    faults["end_time"] = pd.to_datetime(faults["end_time"], utc=True)
    for fault in faults.itertuples(index=False):
        affected = set(ALL_CONGESTION_ENDPOINTS) if fault.fault_type == "congestion" else LINK_ENDPOINTS.get(fault.location, set())
        window = data["timestamp"].between(fault.start_time, fault.end_time)
        for router, interface in affected:
            mask = window & data["router"].eq(router) & data["interface"].eq(interface)
            data.loc[mask, "label"] = fault.fault_type
    frames.append(data)

    rtt = pd.read_csv(run / "rtt.csv")
    rtt["timestamp"] = pd.to_datetime(rtt["timestamp"], utc=True)
    rtt["run_id"] = rid
    rtt["rtt_avg_ms"] = pd.to_numeric(rtt["rtt_avg_ms"], errors="coerce")
    rtt["packet_loss_pct"] = pd.to_numeric(rtt["packet_loss_pct"], errors="coerce")
    rtt_frames.append(rtt[["run_id", "timestamp", "rtt_avg_ms", "packet_loss_pct"]])

    links = pd.read_csv(run / "link_state.csv")
    links["timestamp"] = pd.to_datetime(links["timestamp"], utc=True)
    links["run_id"] = rid
    links["link_up"] = links["oper_state"].str.upper().eq("UP").astype(int)
    link_frames.append(links[["run_id", "router", "interface", "timestamp", "link_up"]])

    throughput = pd.read_csv(run / "throughput.csv")
    throughput["timestamp"] = pd.to_datetime(throughput["timestamp"], utc=True)
    throughput["run_id"] = rid
    throughput["throughput_mbps"] = pd.to_numeric(throughput["throughput_mbps"], errors="coerce")
    throughput_frames.append(throughput[["run_id", "timestamp", "throughput_mbps"]])

real = pd.concat(frames, ignore_index=True)
groups = ["run_id", "router", "interface"]
real = real.sort_values(groups + ["timestamp"]).reset_index(drop=True)
dt = real.groupby(groups, sort=False)["timestamp"].diff().dt.total_seconds()
real["dt_seconds"] = dt.where(dt > 0, np.nan).fillna(1)
counter_cols = [
    "rx_packets", "rx_bytes", "rx_errors", "rx_dropped",
    "tx_packets", "tx_bytes", "tx_errors", "tx_dropped",
]
for col in counter_cols:
    delta = real.groupby(groups, sort=False)[col].diff().fillna(0).clip(lower=0)
    real[f"{col}_rate"] = delta / real["dt_seconds"]
for direction in ("rx", "tx"):
    real[f"{direction}_drop_ratio"] = real[f"{direction}_dropped_rate"] / (real[f"{direction}_packets_rate"] + 1)
    real[f"{direction}_error_ratio"] = real[f"{direction}_errors_rate"] / (real[f"{direction}_packets_rate"] + 1)

rtt_all = pd.concat(rtt_frames, ignore_index=True).sort_values("timestamp")
real = pd.merge_asof(
    real.sort_values("timestamp"), rtt_all, on="timestamp", by="run_id",
    direction="nearest", tolerance=pd.Timedelta(seconds=8),
)
link_all = pd.concat(link_frames, ignore_index=True).sort_values("timestamp")
real = pd.merge_asof(
    real.sort_values("timestamp"), link_all, on="timestamp",
    by=["run_id", "router", "interface"], direction="nearest",
    tolerance=pd.Timedelta(seconds=8),
)
throughput_all = pd.concat(throughput_frames, ignore_index=True).sort_values("timestamp")
real = pd.merge_asof(
    real.sort_values("timestamp"), throughput_all, on="timestamp", by="run_id",
    direction="nearest", tolerance=pd.Timedelta(seconds=8),
)
real["rtt_avg_ms"] = real["rtt_avg_ms"].fillna(0)
real["packet_loss_pct"] = real["packet_loss_pct"].fillna(0)
real["link_up"] = real["link_up"].fillna(1)
real["throughput_mbps"] = real["throughput_mbps"].fillna(0)

for frame in (synthetic, real):
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

encoder = LabelEncoder()
encoder.fit(synthetic["label"])
unknown = sorted(set(real["label"]) - set(encoder.classes_))
if unknown:
    raise SystemExit(f"Real data contains labels absent from synthetic training: {unknown}")
y_train = encoder.transform(synthetic["label"])
y_test = encoder.transform(real["label"])
weights = compute_sample_weight("balanced", y_train)

model = XGBClassifier(
    n_estimators=350, max_depth=5, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=2,
    objective="multi:softprob", eval_metric="mlogloss",
    tree_method="hist", device="cpu", n_jobs=4, random_state=42,
)
model.fit(synthetic[FEATURES], y_train, sample_weight=weights)
pred = model.predict(real[FEATURES]).astype(int)
normal_id = int(encoder.transform(["normal"])[0])
report = classification_report(
    y_test, pred, labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_, zero_division=0, output_dict=True,
)
matrix = confusion_matrix(y_test, pred, labels=np.arange(len(encoder.classes_)))

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
outdir = ROOT / "model_runs" / f"xgb_synthetic_to_real_{stamp}"
outdir.mkdir(parents=True, exist_ok=False)
model.save_model(outdir / "model.json")
predictions = real[["run_id", "timestamp", "router", "interface", "label"]].copy()
predictions["predicted"] = encoder.inverse_transform(pred)
predictions.to_csv(outdir / "real_run_predictions.csv", index=False)
metrics = {
    "synthetic_training_rows": int(len(synthetic)),
    "synthetic_training_runs": int(synthetic["run_id"].nunique()),
    "real_test_rows": int(len(real)),
    "real_test_runs": [p.name for p in reversed(run_dirs)],
    "feature_columns": FEATURES,
    "accuracy": float(accuracy_score(y_test, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    "macro_f1": float(f1_score(y_test, pred, average="macro")),
    "normal_false_alarm_rate": float(
        ((y_test == normal_id) & (pred != normal_id)).sum() / max(1, (y_test == normal_id).sum())
    ),
    "class_report": report,
    "confusion_matrix": matrix.tolist(),
    "class_names": encoder.classes_.tolist(),
}
(outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

print(f"Synthetic training: {len(synthetic)} rows across {synthetic['run_id'].nunique()} runs")
print(f"Real evaluation: {len(real)} rows across {len(run_dirs)} runs")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
print(f"Macro F1: {metrics['macro_f1']:.3f}")
print(f"Normal false-alarm rate: {metrics['normal_false_alarm_rate']:.1%}")
print("Per-class precision / recall / F1:")
for name, values in report.items():
    if isinstance(values, dict) and "f1-score" in values:
        print(f"  {name}: {values['precision']:.3f} / {values['recall']:.3f} / {values['f1-score']:.3f}")
print(f"Saved separate synthetic-to-real evaluation in: {outdir}")
