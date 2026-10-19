#!/usr/bin/env bash
# Usage:
#   train_entrypoint.sh /data/lora_runs/<id>/train.yaml   run one config and exit
#   train_entrypoint.sh                                   watch /data/lora_runs for "queued" markers
set -uo pipefail

RUNS_DIR="${LORA_RUNS_DIR:-/data/lora_runs}"

run_one() {
  local config="$1"
  local run_dir
  run_dir="$(dirname "$config")"
  rm -f "$run_dir/queued"
  touch "$run_dir/running"
  echo "[trainer] start $config" | tee -a "$run_dir/train.log"
  if (cd /opt/ai-toolkit && python run.py "$config") >>"$run_dir/train.log" 2>&1; then
    rm -f "$run_dir/running"
    touch "$run_dir/done"
    echo "[trainer] done" >>"$run_dir/train.log"
    return 0
  fi
  rm -f "$run_dir/running"
  touch "$run_dir/failed"
  echo "[trainer] failed" >>"$run_dir/train.log"
  return 1
}

if [[ $# -ge 1 ]]; then
  run_one "$1"
  exit $?
fi

echo "[trainer] watching $RUNS_DIR"
while true; do
  for marker in "$RUNS_DIR"/*/queued; do
    [[ -e "$marker" ]] || continue
    run_one "$(dirname "$marker")/train.yaml" || true
  done
  sleep 10
done
