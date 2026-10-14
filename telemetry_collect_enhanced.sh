#!/bin/bash
# Collects per-interface telemetry (packets, bytes, errors) from all 4
# routers every 5 seconds and writes it to telemetry.csv with a UTC
# timestamp on every row. Run this alongside baseline traffic and fault
# injection so the two datasets line up in time.
#
# Stop it anytime with Ctrl+C - it just stops appending, nothing else
# in the lab is affected.

OUTFILE="telemetry_enhanced.csv"
INTERVAL=5

declare -A IFACES
IFACES[r1]="eth1 eth2 eth3"
IFACES[r2]="eth1 eth2"
IFACES[r3]="eth1 eth2"
IFACES[r4]="eth1 eth2 eth3"

echo "timestamp,router,interface,rx_packets,rx_bytes,rx_errors,rx_dropped,tx_packets,tx_bytes,tx_errors,tx_dropped" > "$OUTFILE"
echo "Collecting telemetry every ${INTERVAL}s into $OUTFILE. Press Ctrl+C to stop."

while true; do
  ts=$(date -u +%FT%TZ)

  for router in r1 r2 r3 r4; do
    for iface in ${IFACES[$router]}; do
      # Pull stats straight from the kernel via `ip -s link` instead of
      # vtysh - this FRR build doesn't print packet/byte counters for
      # veth-type interfaces through `show interface`, but the kernel
      # always has the real numbers regardless of interface type.
      out=$(docker exec "clab-aiops-diamond-$router" ip -s link show "$iface" 2>/dev/null)

      rx_line=$(echo "$out" | awk '/RX:/{getline; print}')
      tx_line=$(echo "$out" | awk '/TX:/{getline; print}')

      rx_bytes=$(echo "$rx_line" | awk '{print $1}')
      rx_packets=$(echo "$rx_line" | awk '{print $2}')
      rx_errors=$(echo "$rx_line" | awk '{print $3}')
      rx_dropped=$(echo "$rx_line" | awk '{print $4}')
      tx_bytes=$(echo "$tx_line" | awk '{print $1}')
      tx_packets=$(echo "$tx_line" | awk '{print $2}')
      tx_errors=$(echo "$tx_line" | awk '{print $3}')
      tx_dropped=$(echo "$tx_line" | awk '{print $4}')

      echo "$ts,$router,$iface,${rx_packets:-0},${rx_bytes:-0},${rx_errors:-0},${rx_dropped:-0},${tx_packets:-0},${tx_bytes:-0},${tx_errors:-0},${tx_dropped:-0}" >> "$OUTFILE"
    done
  done

  sleep "$INTERVAL"
done
