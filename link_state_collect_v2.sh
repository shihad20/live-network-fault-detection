#!/bin/bash

OUTFILE="link_state_telemetry_v2.csv"
INTERVAL=5

declare -A IFACES
IFACES[r1]="eth1 eth2 eth3"
IFACES[r2]="eth1 eth2"
IFACES[r3]="eth1 eth2"
IFACES[r4]="eth1 eth2 eth3"

if [ ! -e "$OUTFILE" ]; then
  echo "timestamp,router,interface,flags,oper_state,carrier" > "$OUTFILE"
fi

echo "Recording router interface states in $OUTFILE every ${INTERVAL}s. Stop with Ctrl+C."

while true; do
  timestamp=$(date -u +%FT%TZ)

  for router in r1 r2 r3 r4; do
    for iface in ${IFACES[$router]}; do
      line=$(docker exec "clab-aiops-diamond-$router" ip -o link show dev "$iface" 2>/dev/null)
      flags=$(printf '%s\n' "$line" | sed -n 's/.*<\([^>]*\)>.*/\1/p')
      flags=${flags//,/|}
      oper_state=$(printf '%s\n' "$line" | awk '{for (i=1; i<=NF; i++) if ($i=="state") {print $(i+1); exit}}')
      carrier=$(docker exec "clab-aiops-diamond-$router" cat "/sys/class/net/$iface/carrier" 2>/dev/null)

      echo "$timestamp,$router,$iface,${flags:-UNKNOWN},${oper_state:-UNKNOWN},${carrier:-UNKNOWN}" >> "$OUTFILE"
    done
  done

  sleep "$INTERVAL"
done
