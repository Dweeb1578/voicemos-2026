#!/usr/bin/env bash
# Push the built SOMOS scoring kernels to Kaggle, respecting the platform's
# concurrency caps, and print one line per kernel that reaches a terminal
# state.  Reads enable_gpu from each kernel-metadata.json rather than hardcoding
# a runner list, so it stays correct if a runner moves between CPU and GPU.
#
# Usage: bash scripts/somos_launch_queue.sh <kernel-root> [gpu-slots] [cpu-slots]
#   kernel-root  directory built by scripts.somos_kaggle_orchestrate
#
# Kaggle allows at most two concurrent batch GPU sessions.  A push rejected for
# that reason is backpressure, not an error, and is retried on the next cycle.
#
# Rerunning is safe: a kernel that already completed is skipped, and one that is
# still running is adopted rather than pushed again.
set -u

ROOT="${1:?usage: somos_launch_queue.sh <kernel-root> [gpu-slots] [cpu-slots]}"
GPU_SLOTS="${2:-2}"
CPU_SLOTS="${3:-3}"
POLL_SECONDS=60
MAX_CYCLES=1200

command -v kaggle >/dev/null || { echo "kaggle CLI not found"; exit 1; }

slug_of() { grep -o '"id"[^,]*' "$1/kernel-metadata.json" | head -1 | cut -d'"' -f4; }
count_words() { set -- $1; echo $#; }

gpu_pending=""; cpu_pending=""
gpu_running=""; cpu_running=""
failures=""; completed=0

# Classify every kernel by device and by what Kaggle already thinks of it.
# Progress lines go to stdout while the lists accumulate in globals; a helper
# must never return a list through stdout, or a status line gets parsed back
# as a directory path on the next cycle.
for meta in "$ROOT"/*/part-*/kernel-metadata.json; do
  [ -f "$meta" ] || continue
  dir="$(dirname "$meta")"
  slug="$(slug_of "$dir")"
  status="$(kaggle kernels status "$slug" 2>&1)"
  case "$status" in
    *COMPLETE*)
      echo "SKIP  $slug already complete"; completed=$((completed + 1)); continue ;;
    *RUNNING*|*QUEUED*)
      echo "ADOPT $slug already running"
      if grep -q '"enable_gpu": true' "$meta"; then
        gpu_running="$gpu_running $dir"
      else
        cpu_running="$cpu_running $dir"
      fi
      continue ;;
  esac
  if grep -q '"enable_gpu": true' "$meta"; then
    gpu_pending="$gpu_pending $dir"
  else
    cpu_pending="$cpu_pending $dir"
  fi
done
echo "queued: $(count_words "$gpu_pending") GPU, $(count_words "$cpu_pending") CPU; adopted: $(count_words "$gpu_running$cpu_running")"

# Sets HARVEST_RESULT to the still-running subset of "$1".
harvest() {
  local still="" dir slug status
  for dir in $1; do
    slug="$(slug_of "$dir")"
    status="$(kaggle kernels status "$slug" 2>&1)"
    case "$status" in
      *COMPLETE*) echo "DONE  $slug"; completed=$((completed + 1)) ;;
      *ERROR*)    echo "FAIL  $slug"; failures="$failures $slug" ;;
      *CANCEL*)   echo "FAIL  $slug cancelled"; failures="$failures $slug" ;;
      *) still="$still $dir" ;;
    esac
  done
  HARVEST_RESULT="$still"
}

# Sets FILL_RUNNING and FILL_PENDING from running="$1" pending="$2" slots="$3".
fill() {
  local running="$1" pending="$2" slots="$3" next rest out
  while [ "$(count_words "$running")" -lt "$slots" ] && [ -n "$(echo $pending)" ]; do
    set -- $pending
    next="$1"; shift; rest="$*"
    out="$(kaggle kernels push -p "$next" 2>&1 | tail -1)"
    case "$out" in
      *successfully*)
        echo "PUSH  $(slug_of "$next")"
        running="$running $next"; pending="$rest" ;;
      *"Maximum batch"*) break ;;
      *) echo "RETRY $(slug_of "$next"): $out"; break ;;
    esac
  done
  FILL_RUNNING="$running"; FILL_PENDING="$pending"
}

cycle=0
while [ $cycle -lt $MAX_CYCLES ]; do
  harvest "$gpu_running"; gpu_running="$HARVEST_RESULT"
  harvest "$cpu_running"; cpu_running="$HARVEST_RESULT"

  fill "$gpu_running" "$gpu_pending" "$GPU_SLOTS"
  gpu_running="$FILL_RUNNING"; gpu_pending="$FILL_PENDING"
  fill "$cpu_running" "$cpu_pending" "$CPU_SLOTS"
  cpu_running="$FILL_RUNNING"; cpu_pending="$FILL_PENDING"

  [ -z "$(echo $gpu_running$gpu_pending$cpu_running$cpu_pending)" ] && break
  cycle=$((cycle + 1))
  sleep "$POLL_SECONDS"
done

if [ -n "$(echo $failures)" ]; then
  echo "QUEUE DONE: $completed complete, failures:$failures"
else
  echo "QUEUE DONE: $completed complete, no failures"
fi
