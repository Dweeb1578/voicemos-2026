#!/usr/bin/env bash
# Push the built SOMOS scoring kernels to Kaggle, respecting the platform's
# concurrency caps, and report one line per kernel that reaches a terminal
# state.  Reads enable_gpu from each kernel-metadata.json rather than hardcoding
# a runner list, so it stays correct if a runner moves between CPU and GPU.
#
# Usage: bash scripts/somos_launch_queue.sh <kernel-root> [gpu-slots] [cpu-slots]
#   kernel-root  directory built by scripts.somos_kaggle_orchestrate
#
# Kaggle allows at most two concurrent batch GPU sessions.  A push rejected for
# that reason is backpressure, not an error, and is retried on the next cycle.
set -u

ROOT="${1:?usage: somos_launch_queue.sh <kernel-root> [gpu-slots] [cpu-slots]}"
GPU_SLOTS="${2:-2}"
CPU_SLOTS="${3:-3}"
POLL_SECONDS=60
MAX_CYCLES=1200

command -v kaggle >/dev/null || { echo "kaggle CLI not found"; exit 1; }

# Collect kernel directories, partitioned by the device their metadata requests.
gpu_pending=""
cpu_pending=""
for meta in "$ROOT"/*/part-*/kernel-metadata.json; do
  [ -f "$meta" ] || continue
  dir="$(dirname "$meta")"
  if grep -q '"enable_gpu": true' "$meta"; then
    gpu_pending="$gpu_pending $dir"
  else
    cpu_pending="$cpu_pending $dir"
  fi
done

# Read the id without depending on a Python interpreter being on PATH.
slug_of() { grep -o '"id"[^,]*' "$1/kernel-metadata.json" | head -1 | cut -d'"' -f4; }
count_words() { set -- $1; echo $#; }

# A kernel that already completed is not pushed again, so an interrupted run can
# be resumed by rerunning this script.
filter_incomplete() {
  local out=""
  for dir in $1; do
    local slug; slug="$(slug_of "$dir")"
    local status; status="$(kaggle kernels status "$slug" 2>&1)"
    case "$status" in
      *COMPLETE*) echo "SKIP $slug already complete" >&2 ;;
      *) out="$out $dir" ;;
    esac
  done
  echo "$out"
}

gpu_pending="$(filter_incomplete "$gpu_pending")"
cpu_pending="$(filter_incomplete "$cpu_pending")"
echo "queued: $(count_words "$gpu_pending") GPU, $(count_words "$cpu_pending") CPU"

gpu_running=""
cpu_running=""
failures=""
completed=0

harvest() {
  local running="$1" still=""
  for dir in $running; do
    local slug; slug="$(slug_of "$dir")"
    local status; status="$(kaggle kernels status "$slug" 2>&1)"
    case "$status" in
      *COMPLETE*)  echo "DONE  $slug"; completed=$((completed + 1)) ;;
      *ERROR*)     echo "FAIL  $slug"; failures="$failures $slug" ;;
      *CANCEL*)    echo "FAIL  $slug cancelled"; failures="$failures $slug" ;;
      *) still="$still $dir" ;;
    esac
  done
  echo "$still"
}

fill() {
  local running="$1" pending="$2" slots="$3" launched=""
  while [ "$(count_words "$running")" -lt "$slots" ] && [ -n "$(echo $pending)" ]; do
    set -- $pending
    local next="$1"; shift; local rest="$*"
    local out; out="$(kaggle kernels push -p "$next" 2>&1 | tail -1)"
    case "$out" in
      *successfully*)
        echo "PUSH  $(slug_of "$next")" >&2
        running="$running $next"; pending="$rest" ;;
      *"Maximum batch"*)
        break ;;                       # slot not free yet; retry next cycle
      *)
        echo "PUSHFAIL $(slug_of "$next"): $out" >&2
        break ;;                       # transient network error; retry later
    esac
  done
  # Return both lists through a delimiter the paths never contain.
  echo "$running|$pending"
}

cycle=0
while [ $cycle -lt $MAX_CYCLES ]; do
  gpu_running="$(harvest "$gpu_running")"
  cpu_running="$(harvest "$cpu_running")"

  pair="$(fill "$gpu_running" "$gpu_pending" "$GPU_SLOTS")"
  gpu_running="${pair%%|*}"; gpu_pending="${pair#*|}"
  pair="$(fill "$cpu_running" "$cpu_pending" "$CPU_SLOTS")"
  cpu_running="${pair%%|*}"; cpu_pending="${pair#*|}"

  if [ -z "$(echo $gpu_running$gpu_pending$cpu_running$cpu_pending)" ]; then
    break
  fi
  cycle=$((cycle + 1))
  sleep "$POLL_SECONDS"
done

if [ -n "$(echo $failures)" ]; then
  echo "QUEUE DONE: $completed complete, failures:$failures"
else
  echo "QUEUE DONE: $completed complete, no failures"
fi
