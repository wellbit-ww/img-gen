#!/usr/bin/env bash
set -euo pipefail

CONFIG="${TRAIN_CONFIG_PATH:-}"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "TRAIN_CONFIG_PATH must point to train.yaml (got: $CONFIG)" >&2
  exit 1
fi

cd /opt/ai-toolkit
exec python run.py "$CONFIG"
