#!/usr/bin/env python3
"""Predict network faults from the lab's continuously appended telemetry CSVs.

Run beside the telemetry, RTT, link-state, and throughput collectors. The
detector rebuilds features from the current CSV contents and emits only rows
newer than the previous poll. It uses the feature schema from a saved
train_model_xgboost_runs.py run.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model_runs" / "xgb_35run_20261013T082155Z"
COUNTERS = [
    "rx_packets", "rx_bytes", "rx_errors", "rx_dropped",
    "tx_packets", "tx_bytes", "tx_errors", "tx_dropped",
]
REQUIRED = ["timestamp", "router", "interface", *COUNTERS]


def read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=required)
    try:
        frame = pd.read_csv(path, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=required)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def build_features(run_dir: Path, feature_names: list[str]) -> pd.DataFrame:
    telemetry = read_csv(run_dir / "telemetry.csv", REQUIRED)
    if telemetry.empty:
        return pd.DataFrame(columns=feature_names)
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], utc=True, errors="coerce")
    telemetry = telemetry.dropna(subset=["timestamp"])
    for col in COUNTERS:
        telemetry[col] = pd.to_numeric(telemetry[col], errors="coerce").fillna(0)
    keys = ["router", "interface"]
    telemetry = telemetry.sort_values(keys + ["timestamp"]).reset_index(drop=True)
    grouped = telemetry.groupby(keys, sort=False)
    dt = grouped["timestamp"].diff().dt.total_seconds()
    telemetry["dt_seconds"] = dt.where(dt > 0, np.nan).fillna(1)
    for col in COUNTERS:
        delta = telemetry.groupby(keys, sort=False)[col].diff().fillna(0).clip(lower=0)
        telemetry[f"{col}_rate"] = delta / telemetry["dt_seconds"]
    for direction in ("rx", "tx"):
        telemetry[f"{direction}_drop_ratio"] = telemetry[f"{direction}_dropped_rate"] / (telemetry[f"{direction}_packets_rate"] + 1)
        telemetry[f"{direction}_error_ratio"] = telemetry[f"{direction}_errors_rate"] / (telemetry[f"{direction}_packets_rate"] + 1)

    telemetry["run_id"] = run_dir.name
    rtt = read_csv(run_dir / "rtt.csv", ["timestamp", "rtt_avg_ms", "packet_loss_pct"])
    if not rtt.empty:
        rtt["timestamp"] = pd.to_datetime(rtt["timestamp"], utc=True, errors="coerce")
        rtt["run_id"] = run_dir.name
        for col in ("rtt_avg_ms", "packet_loss_pct"):
            rtt[col] = pd.to_numeric(rtt[col], errors="coerce")
        rtt = rtt.dropna(subset=["timestamp"]).sort_values("timestamp")
        telemetry = pd.merge_asof(telemetry.sort_values("timestamp"), rtt, on="timestamp", by="run_id", direction="nearest", tolerance=pd.Timedelta(seconds=8))
    else:
        telemetry["rtt_avg_ms"] = np.nan
        telemetry["packet_loss_pct"] = np.nan
    telemetry["rtt_missing"] = telemetry["rtt_avg_ms"].isna().astype(int)
    telemetry[["rtt_avg_ms", "packet_loss_pct"]] = telemetry[["rtt_avg_ms", "packet_loss_pct"]].fillna(0)

    links = read_csv(run_dir / "link_state.csv", ["timestamp", "router", "interface", "oper_state"])
    if not links.empty:
        links["timestamp"] = pd.to_datetime(links["timestamp"], utc=True, errors="coerce")
        links["link_up"] = links["oper_state"].astype(str).str.upper().eq("UP").astype(int)
        links = links.dropna(subset=["timestamp"]).sort_values("timestamp")
        telemetry = pd.merge_asof(telemetry.sort_values("timestamp"), links[["timestamp", "router", "interface", "link_up"]], on="timestamp", by=["router", "interface"], direction="nearest", tolerance=pd.Timedelta(seconds=8))
    else:
        telemetry["link_up"] = np.nan
    telemetry["link_missing"] = telemetry["link_up"].isna().astype(int)
    telemetry["link_up"] = telemetry["link_up"].fillna(1)

    throughput = read_csv(run_dir / "throughput.csv", ["timestamp", "throughput_mbps"])
    if not throughput.empty:
        throughput["timestamp"] = pd.to_datetime(throughput["timestamp"], utc=True, errors="coerce")
        throughput["run_id"] = run_dir.name
        throughput["throughput_mbps"] = pd.to_numeric(throughput["throughput_mbps"], errors="coerce")
        if "status" in throughput:
            throughput.loc[throughput["status"].ne("ok"), "throughput_mbps"] = np.nan
        throughput = throughput.dropna(subset=["timestamp"]).sort_values("timestamp")
        telemetry = pd.merge_asof(telemetry.sort_values("timestamp"), throughput[["timestamp", "run_id", "throughput_mbps"]], on="timestamp", by="run_id", direction="nearest", tolerance=pd.Timedelta(seconds=8))
    else:
        telemetry["throughput_mbps"] = np.nan
    telemetry["throughput_missing"] = telemetry["throughput_mbps"].isna().astype(int)
    telemetry["throughput_mbps"] = telemetry["throughput_mbps"].fillna(0)

    telemetry = telemetry.sort_values(keys + ["timestamp"]).reset_index(drop=True)
    categories = pd.get_dummies(telemetry[["router", "interface"]], dtype=int)
    base = telemetry.drop(columns=["timestamp", "run_id", "router", "interface", *COUNTERS, "dt_seconds"], errors="ignore")
    features = pd.concat([base, categories], axis=1).reindex(columns=feature_names, fill_value=0)
    features = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    features.index = telemetry.index
    return features


def apply_domain_overrides(
    probabilities: np.ndarray,
    features: pd.DataFrame,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the definitive interface-state signal after model scoring.

    An interface observed DOWN is a link-down fault regardless of its location
    in the topology. This keeps the classifier from memorizing only the links
    present in its training runs.
    """
    predicted_ids = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    if "link_down" in classes and "link_up" in features:
        down_id = classes.index("link_down")
        down_mask = pd.to_numeric(features["link_up"], errors="coerce").fillna(1).to_numpy() == 0
        predicted_ids[down_mask] = down_id
        confidence[down_mask] = 1.0
    return predicted_ids, confidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing live telemetry.csv, rtt.csv, link_state.csv, throughput.csv")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL, help="Saved model run directory with model.json and metrics.json")
    parser.add_argument("--interval", type=float, default=5, help="Polling period in seconds (default: 5)")
    parser.add_argument("--once", action="store_true", help="Score current rows once, useful for replaying a completed run")
    parser.add_argument("--output", type=Path, help="Append prediction rows to this CSV")
    parser.add_argument("--alert-threshold", type=float, default=0.60, help="Print alert only for fault classes at or above this probability")
    args = parser.parse_args()
    metrics = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))
    feature_names = metrics["feature_names"]
    classes = metrics["class_names"]
    model = XGBClassifier()
    model.load_model(args.model_dir / "model.json")
    out = args.output or (args.run_dir / "live_predictions.csv")
    last_key = None
    print(f"Live detector ready | model={args.model_dir.name} | run={args.run_dir} | interval={args.interval:g}s", flush=True)
    while True:
        features = build_features(args.run_dir, feature_names)
        telemetry = read_csv(args.run_dir / "telemetry.csv", REQUIRED)
        if not telemetry.empty and len(features) == len(telemetry):
            telemetry = telemetry.copy().sort_values(["router", "interface", "timestamp"]).reset_index(drop=True)
            features = features.reset_index(drop=True)
            telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], utc=True, errors="coerce")
            keys = pd.MultiIndex.from_frame(telemetry[["timestamp", "router", "interface"]].astype({"router": str, "interface": str}))
            new_mask = np.ones(len(keys), dtype=bool) if last_key is None else ~keys.isin(last_key)
            if new_mask.any():
                ids = np.flatnonzero(new_mask)
                selected = features.iloc[ids]
                probabilities = model.predict_proba(selected)
                pred_idx, confidence = apply_domain_overrides(probabilities, selected, classes)
                result = telemetry.iloc[ids][["timestamp", "router", "interface"]].copy()
                result["predicted"] = [classes[i] for i in pred_idx]
                result["confidence"] = confidence
                result["alert"] = [c != "normal" and p >= args.alert_threshold for c, p in zip(result["predicted"], result["confidence"])]
                result["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
                out.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(out, mode="a", header=not out.exists(), index=False)
                for row in result.itertuples(index=False):
                    kind = "ALERT" if row.alert else "prediction"
                    print(f"{kind} {row.timestamp.isoformat()} {row.router}/{row.interface} => {row.predicted} ({row.confidence:.1%})", flush=True)
                last_key = keys
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
