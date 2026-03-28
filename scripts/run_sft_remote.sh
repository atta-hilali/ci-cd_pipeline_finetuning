#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
DATASET=${DATASET:-dental_v1}
MODEL=${MODEL:-qwen3_8b}
PEFT=${PEFT:-dora}
INFRA=${INFRA:-dgx_spark}
TRACKING=${TRACKING:-mlflow_dgx}
DATASET_PATH=${DATASET_PATH:-}
EVAL_PATH=${EVAL_PATH:-}
VALIDATION_STRATEGY=${VALIDATION_STRATEGY:-auto}
TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-$REPO_ROOT/outputs}
LOG_ROOT=${LOG_ROOT:-$TRAIN_OUTPUT_ROOT/launcher_logs}
INSTALL_DEPS=${INSTALL_DEPS:-0}
BACKGROUND=${BACKGROUND:-1}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

if [[ -z "$DATASET_PATH" ]]; then
  echo "Set DATASET_PATH to the Linux path of your training JSONL before launching."
  exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset file not found: $DATASET_PATH"
  exit 1
fi

if [[ -n "$EVAL_PATH" && ! -f "$EVAL_PATH" ]]; then
  echo "Validation file not found: $EVAL_PATH"
  exit 1
fi

if [[ "$TRACKING" == "mlflow_dgx" && -z "${MLFLOW_TRACKING_URI:-}" ]]; then
  echo "TRACKING=mlflow_dgx requires MLFLOW_TRACKING_URI to be exported."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available; cannot detect GPU capacity."
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
  NPROC_PER_NODE=${NPROC_PER_NODE:-${#visible_devices[@]}}
else
  detected_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  NPROC_PER_NODE=${NPROC_PER_NODE:-$detected_gpus}
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is not available. Install the project environment first."
  exit 1
fi

mkdir -p "$TRAIN_OUTPUT_ROOT" "$LOG_ROOT"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -e '.[deepspeed]'
fi

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$LOG_ROOT/${timestamp}_${MODEL}_${PEFT}_${INFRA}.log"

overrides=(
  "dataset=$DATASET"
  "model=$MODEL"
  "peft=$PEFT"
  "infra=$INFRA"
  "tracking=$TRACKING"
  "dataset.path=$DATASET_PATH"
  "dataset.validation.strategy=$VALIDATION_STRATEGY"
  "run.output_dir=${TRAIN_OUTPUT_ROOT}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}_\${run.name}"
)

if [[ -n "$EVAL_PATH" ]]; then
  overrides+=(
    "dataset.eval_path=$EVAL_PATH"
    "dataset.validation.strategy=file"
  )
fi

if [[ -n "$EXTRA_OVERRIDES" ]]; then
  read -r -a extra <<< "$EXTRA_OVERRIDES"
  overrides+=("${extra[@]}")
fi

cmd=(
  torchrun
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  -m
  velodoc_train.cli
  "${overrides[@]}"
)

echo "Launching $MODEL with $PEFT on $NPROC_PER_NODE visible GPU(s)."
echo "Logs: $log_file"
printf 'Command: '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$BACKGROUND" == "1" ]]; then
  nohup "${cmd[@]}" >"$log_file" 2>&1 &
  pid=$!
  echo "Started background job with PID $pid"
  echo "Follow logs with: tail -f $log_file"
  echo "Stop with: kill $pid"
else
  "${cmd[@]}" 2>&1 | tee "$log_file"
fi
