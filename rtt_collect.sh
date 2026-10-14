#!/bin/bash

DEST="10.0.2.10"
OUTFILE="rtt_telemetry.csv"
INTERVAL=5

if [ ! -e "$OUTFILE" ]; then
  echo "timestamp,source,destination,rtt_ms,status" > "$OUTFILE"
fi

echo "Recording RTT from h1 to h2 in $OUTFILE every ${INTERVAL}s. Stop with Ctrl+C."

while true; do
  timestamp=$(date -u +%FT%TZ)
  result=$(docker exec clab-aiops-diamond-h1 ping -n -c 1 -W 3 "$DEST" 2>&1 || true)
  rtt=$(printf '%s\n' "$result" | sed -n 's/.*time=\([0-9.]*\) ms.*/\1/p' | head -n 1)

  if [ -n "$rtt" ]; then
    status="ok"
  else
    rtt=""
    status="timeout"
  fi

  echo "$timestamp,h1,$DEST,$rtt,$status" >> "$OUTFILE"
  sleep "$INTERVAL"
done
