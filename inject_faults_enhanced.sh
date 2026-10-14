#!/bin/bash
# Fault injection script for the AIOps diamond lab.
# Run from the host machine using: sudo docker exec clab-aiops-diamond-<node> <cmd>
# Logs every fault with start/end time and label -> faults_log.csv (ground truth for evaluation)

LOGFILE="faults_log_enhanced.csv"
echo "start_time,end_time,fault_type,location" > "$LOGFILE"

log_fault() {
  local start=$1 end=$2 type=$3 loc=$4
  echo "$start,$end,$type,$loc" >> "$LOGFILE"
}

node_exec() {
  docker exec "clab-aiops-diamond-$1" "${@:2}"
}

# --- Fault 1: Link down (R1-R2) ---
run_link_down() {
  local start end
  start=$(date -u +%FT%TZ)
  node_exec r1 ip link set eth2 down
  sleep 60
  node_exec r1 ip link set eth2 up
  end=$(date -u +%FT%TZ)
  log_fault "$start" "$end" "link_down" "r1-r2"
}

# --- Fault 2: High latency (R2-R4) ---
run_latency() {
  local start end
  start=$(date -u +%FT%TZ)
  node_exec r2 tc qdisc add dev eth2 root netem delay 200ms
  sleep 60
  node_exec r2 tc qdisc del dev eth2 root netem
  end=$(date -u +%FT%TZ)
  log_fault "$start" "$end" "high_latency" "r2-r4"
}

# --- Fault 3: Packet loss (R3-R4) ---
run_packet_loss() {
  local start end
  start=$(date -u +%FT%TZ)
  node_exec r3 tc qdisc add dev eth2 root netem loss 5%
  sleep 60
  node_exec r3 tc qdisc del dev eth2 root netem
  end=$(date -u +%FT%TZ)
  log_fault "$start" "$end" "packet_loss" "r3-r4"
}

# --- Fault 4: Congestion (throttle both diamond paths via tc) ---
# Note: this deliberately does NOT start a second iperf3 test. If you
# already have an iperf3 flow running (h1 -> h2) during fault injection,
# a second iperf3 test would collide with it ("server is busy") since
# iperf3 only handles one test at a time per server. Instead, this
# throttles both equal-cost paths (r2 and r3's link toward r4) down to
# a low rate with `tc tbf`, creating a real bandwidth bottleneck against
# whatever traffic is already flowing - which is what congestion
# actually looks like on a real network, regardless of which path OSPF
# happens to pick for any given flow.
run_congestion() {
  local start end
  start=$(date -u +%FT%TZ)
  node_exec r2 tc qdisc add dev eth2 root tbf rate 5mbit burst 32000 latency 400ms
  node_exec r3 tc qdisc add dev eth2 root tbf rate 5mbit burst 32000 latency 400ms
  sleep 60
  node_exec r2 tc qdisc del dev eth2 root tbf
  node_exec r3 tc qdisc del dev eth2 root tbf
  end=$(date -u +%FT%TZ)
  log_fault "$start" "$end" "congestion" "r2 and r3 (both diamond paths throttled)"
}

echo "Running fault sequence. Each fault runs for ~60s with a gap between."
run_link_down
sleep 30
run_latency
sleep 30
run_packet_loss
sleep 30
run_congestion

echo "Done. Ground-truth labels written to $LOGFILE"
