#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PYTHON_BIN=${PYTHON_BIN:-python3}
DATASET=${DATASET:-dental_v1}
MODEL=${MODEL:-qwen3_8b}
PEFT=${PEFT:-lora}
INFRA=${INFRA:-local}
TRACKING=${TRACKING:-mlflow_local}
DATASET_PATH=${DATASET_PATH:-}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

if [[ -z "$DATASET_PATH" ]]; then
  echo "Set DATASET_PATH to a local training JSONL before launching."
  exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset file not found: $DATASET_PATH"
  exit 1
fi

overrides=(
  "dataset=$DATASET"
  "model=$MODEL"
  "peft=$PEFT"
  "infra=$INFRA"
  "tracking=$TRACKING"
  "dataset.path=$DATASET_PATH"
)

if [[ -n "$EXTRA_OVERRIDES" ]]; then
  read -r -a extra <<< "$EXTRA_OVERRIDES"
  overrides+=("${extra[@]}")
fi

"$PYTHON_BIN" -m velodoc_train.cli "${overrides[@]}"
