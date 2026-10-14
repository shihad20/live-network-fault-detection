#!/usr/bin/env python3
"""Evaluate a saved real-run model on one separately collected fault profile."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

from live_detector import ROOT, apply_domain_overrides, build_features, read_csv


MODEL_DIR = ROOT / "model_runs" / "xgb_35run_20261013T082155Z"
TARGETS = {
    "link_down": {("r1", "eth2"), ("r2", "eth1")},
    "high_latency": {("r3", "eth2"), ("r4", "eth3")},
    "packet_loss": {("r3", "eth2"), ("r4", "eth3")},
    "congestion": {
        ("r2", "eth2"), ("r4", "eth2"),
        ("r3", "eth2"), ("r4", "eth3"),
    },
}
LINK_ENDPOINTS = {
    "r1-r2": {("r1", "eth2"), ("r2", "eth1")},
    "r1-r3": {("r1", "eth3"), ("r3", "eth1")},
    "r2-r4": {("r2", "eth2"), ("r4", "eth2")},
    "r3-r4": {("r3", "eth2"), ("r4", "eth3")},
}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory collected with a fault profile held out from training")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--fault-class", choices=["link_down", "high_latency", "packet_loss", "congestion"],
        help="Evaluate this event as a binary class against all other rows; useful when later fault injections were bypassed by routing.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    required = ["telemetry.csv", "rtt.csv", "link_state.csv", "throughput.csv", "faults.csv"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Run is missing required files: {', '.join(missing)}")

    metadata = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))
    model = XGBClassifier()
    model.load_model(args.model_dir / "model.json")
    feature_names = metadata["feature_names"]
    classes = metadata["class_names"]
    class_to_id = {name: i for i, name in enumerate(classes)}

    telemetry = read_csv(run_dir / "telemetry.csv", ["timestamp", "router", "interface"])
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], utc=True, errors="coerce")
    telemetry = telemetry.dropna(subset=["timestamp"])
    telemetry["label"] = "normal"
    faults = pd.read_csv(run_dir / "faults.csv")
    for fault in faults.itertuples(index=False):
        if fault.fault_type not in TARGETS:
            raise SystemExit(f"Unknown fault type in {run_dir / 'faults.csv'}: {fault.fault_type}")
        start = pd.to_datetime(fault.start_time, utc=True)
        end = pd.to_datetime(fault.end_time, utc=True)
        within = telemetry["timestamp"].between(start, end)
        targets = LINK_ENDPOINTS.get(str(fault.location), TARGETS[fault.fault_type])
        for router, interface in targets:
            telemetry.loc[
                within & telemetry["router"].eq(router) & telemetry["interface"].eq(interface),
                "label",
            ] = fault.fault_type

    telemetry = telemetry.sort_values(["router", "interface", "timestamp"]).reset_index(drop=True)
    features = build_features(run_dir, feature_names).reset_index(drop=True)
    if len(features) != len(telemetry):
        raise SystemExit(f"Feature/label row mismatch: {len(features)} features, {len(telemetry)} labels")
    unknown = sorted(set(telemetry["label"]) - set(classes))
    if unknown:
        raise SystemExit(f"Evaluation labels are absent from the model: {unknown}")
    y_true = telemetry["label"].map(class_to_id).to_numpy()
    probabilities = model.predict_proba(features)
    predicted_ids, confidence = apply_domain_overrides(probabilities, features, classes)
    predicted = [classes[i] for i in predicted_ids]
    normal_id = class_to_id["normal"]
    if args.fault_class:
        actual_positive = telemetry["label"].eq(args.fault_class).to_numpy()
        predicted_positive = np.asarray(predicted) == args.fault_class
        report = classification_report(
            actual_positive, predicted_positive, labels=[False, True],
            target_names=["other", args.fault_class], zero_division=0,
            output_dict=True,
        )
        matrix = confusion_matrix(
            actual_positive, predicted_positive, labels=[False, True]
        )
        metrics = {
            "evaluation_run": run_dir.name,
            "model_dir": str(args.model_dir),
            "evaluation_scope": args.fault_class,
            "held_out_profile": faults.to_dict(orient="records"),
            "test_rows": int(len(telemetry)),
            "accuracy": float(accuracy_score(actual_positive, predicted_positive)),
            "balanced_accuracy": float(balanced_accuracy_score(actual_positive, predicted_positive)),
            "macro_f1": float(f1_score(actual_positive, predicted_positive, average="macro")),
            "non_target_false_alarm_rate": float(
                ((~actual_positive) & predicted_positive).sum()
                / max(1, (~actual_positive).sum())
            ),
            "positive_samples": int(actual_positive.sum()),
            "class_names": ["other", args.fault_class],
            "class_report": report,
            "confusion_matrix": matrix.tolist(),
        }
    else:
        report = classification_report(
            y_true, predicted_ids, labels=np.arange(len(classes)),
            target_names=classes, zero_division=0, output_dict=True,
        )
        matrix = confusion_matrix(y_true, predicted_ids, labels=np.arange(len(classes)))
        metrics = {
            "evaluation_run": run_dir.name,
            "model_dir": str(args.model_dir),
            "evaluation_scope": "all_classes",
            "held_out_profile": faults.to_dict(orient="records"),
            "test_rows": int(len(telemetry)),
            "accuracy": float(accuracy_score(y_true, predicted_ids)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_ids)),
            "macro_f1": float(f1_score(y_true, predicted_ids, average="macro")),
            "normal_false_alarm_rate": float(
                ((y_true == normal_id) & (predicted_ids != normal_id)).sum()
                / max(1, (y_true == normal_id).sum())
            ),
            "class_names": classes,
            "class_report": report,
            "confusion_matrix": matrix.tolist(),
        }
    predictions = telemetry[["timestamp", "router", "interface", "label"]].copy()
    predictions["predicted"] = predicted
    predictions["confidence"] = confidence
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "model_runs" / f"xgb_holdout_profile_{run_dir.name}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(out / "holdout_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scope = f"{args.fault_class} vs. other" if args.fault_class else "all classes"
    print(f"Held-out run: {run_dir.name} ({len(telemetry)} rows; scope={scope})")
    print(f"Accuracy={metrics['accuracy']:.3f} | balanced_accuracy={metrics['balanced_accuracy']:.3f} | macro_F1={metrics['macro_f1']:.3f}")
    if args.fault_class:
        entry = report[args.fault_class]
        print(f"{args.fault_class}: precision={entry['precision']:.3f}, recall={entry['recall']:.3f}, F1={entry['f1-score']:.3f}, support={int(entry['support'])}")
        print(f"Non-target false-alarm rate={metrics['non_target_false_alarm_rate']:.1%}")
    else:
        print(f"Normal false-alarm rate={metrics['normal_false_alarm_rate']:.1%}")
    for name in ([] if args.fault_class else classes):
        entry = report[name]
        print(f"{name}: precision={entry['precision']:.3f}, recall={entry['recall']:.3f}, F1={entry['f1-score']:.3f}, support={int(entry['support'])}")
    print(f"Saved metrics and predictions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
