#!/bin/bash

DEST="10.0.2.10"
OUTFILE="rtt_telemetry_v3.csv"
COUNT=5
INTERVAL=5

if [ ! -e "$OUTFILE" ]; then
  echo "timestamp,source,destination,rtt_avg_ms,rtt_min_ms,rtt_max_ms,packet_loss_pct,status" > "$OUTFILE"
fi

echo "Recording 5-ping RTT and loss samples in $OUTFILE. Stop with Ctrl+C."

while true; do
  timestamp=$(date -u +%FT%TZ)
  result=$(docker exec clab-aiops-diamond-h1 ping -n -i 0.2 -c "$COUNT" -W 1 "$DEST" 2>&1 || true)

  loss=$(printf '%s\n' "$result" | sed -nE 's/.*([0-9]+([.][0-9]+)?)% packet loss.*/\1/p' | tail -n 1)
  rtt=$(printf '%s\n' "$result" | sed -nE 's/.* = ([0-9.]+)\/([0-9.]+)\/([0-9.]+)\/.*/\1,\2,\3/p' | tail -n 1)

  if [ -n "$rtt" ]; then
    IFS=, read -r rtt_min rtt_avg rtt_max <<< "$rtt"
  else
    rtt_min=""
    rtt_avg=""
    rtt_max=""
  fi

  if [ -z "$loss" ]; then
    loss=""
    status="unknown"
  elif awk -v loss="$loss" 'BEGIN { exit !(loss == 0) }'; then
    status="ok"
  elif awk -v loss="$loss" 'BEGIN { exit !(loss >= 100) }'; then
    status="timeout"
  else
    status="packet_loss"
  fi

  echo "$timestamp,h1,$DEST,$rtt_avg,$rtt_min,$rtt_max,$loss,$status" >> "$OUTFILE"
  sleep "$INTERVAL"
done
