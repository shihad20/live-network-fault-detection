#!/bin/bash

LOGFILE="faults_log_enhanced_v2.csv"
LATENCY_MS="${LATENCY_MS:-200}"
PACKET_LOSS_PCT="${PACKET_LOSS_PCT:-5}"
CONGESTION_RATE="${CONGESTION_RATE:-5mbit}"
LINK_DOWN_NODE="${LINK_DOWN_NODE:-r1}"
LINK_DOWN_INTERFACE="${LINK_DOWN_INTERFACE:-eth2}"
LINK_DOWN_LOCATION="${LINK_DOWN_LOCATION:-r1-r2}"

if [[ ! "$LATENCY_MS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LATENCY_MS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$PACKET_LOSS_PCT" =~ ^([1-9]|[1-9][0-9]|100)$ ]]; then
  echo "PACKET_LOSS_PCT must be an integer from 1 to 100." >&2
  exit 2
fi
if [[ ! "$CONGESTION_RATE" =~ ^[1-9][0-9]*([kKmMgG]bit)?$ ]]; then
  echo "CONGESTION_RATE must be a positive rate such as 5mbit or 3000kbit." >&2
  exit 2
fi

echo "start_time,end_time,fault_type,location,severity" > "$LOGFILE"

node_exec() {
  docker exec "clab-aiops-diamond-$1" "${@:2}"
}

cleanup() {
  node_exec "$LINK_DOWN_NODE" ip link set "$LINK_DOWN_INTERFACE" up >/dev/null 2>&1 || true
  node_exec r2 tc qdisc del dev eth2 root >/dev/null 2>&1 || true
  node_exec r3 tc qdisc del dev eth2 root >/dev/null 2>&1 || true
}

trap cleanup EXIT
trap 'exit 130' INT TERM

log_fault() {
  echo "$1,$2,$3,$4,$5" >> "$LOGFILE"
}

run_link_down() {
  local start end
  start=$(date -u +%FT%TZ)
  node_exec "$LINK_DOWN_NODE" ip link set "$LINK_DOWN_INTERFACE" down
  sleep 60
  node_exec "$LINK_DOWN_NODE" ip link set "$LINK_DOWN_INTERFACE" up
  end=$(date -u +%FT%TZ)
  log_fault "$start" "$end" "link_down" "$LINK_DOWN_LOCATION" "interface-down"
}

run_latency() {
  local start end
  start=$(date -u +%FT%TZ)
  node_exec r3 tc qdisc add dev eth2 root netem delay "${LATENCY_MS}ms"
  sleep 60
  node_exec r3 tc qdisc del dev eth2 root
  end=$(date -u +%FT%TZ)
  log_fault "$start" "$end" "high_latency" "r3-r4" "${LATENCY_MS}ms"
}

run_packet_loss() {
  local start end
  start=$(date -u +%FT%TZ)
  node_exec r3 tc qdisc add dev eth2 root netem loss "${PACKET_LOSS_PCT}%"
  sleep 60
  node_exec r3 tc qdisc del dev eth2 root
  end=$(date -u +%FT%TZ)
  log_fault "$start" "$end" "packet_loss" "r3-r4" "${PACKET_LOSS_PCT}%"
}

run_congestion() {
  local start end
  start=$(date -u +%FT%TZ)
  node_exec r2 tc qdisc add dev eth2 root tbf rate "$CONGESTION_RATE" burst 32000 latency 400ms
  node_exec r3 tc qdisc add dev eth2 root tbf rate "$CONGESTION_RATE" burst 32000 latency 400ms
  sleep 60
  node_exec r2 tc qdisc del dev eth2 root
  node_exec r3 tc qdisc del dev eth2 root
  end=$(date -u +%FT%TZ)
  log_fault "$start" "$end" "congestion" "r2 and r3 (both diamond paths throttled)" "$CONGESTION_RATE"
}

echo "Running enhanced v2 fault sequence. Each fault lasts ~60s with 30s gaps."
run_link_down
sleep 30
run_latency
sleep 30
run_packet_loss
sleep 30
run_congestion

echo "Done. Labels saved to $LOGFILE"
