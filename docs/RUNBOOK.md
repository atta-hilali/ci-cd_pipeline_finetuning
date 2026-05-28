# VeloDoc Fine-Tuning Runbook

## Local Smoke Test

Install the project once:

```bash
python -m pip install -e .
```

Run a CPU smoke test against a small JSONL file:

```bash
DATASET_PATH=/path/to/train.jsonl \
EXTRA_OVERRIDES="stage.trainer.epochs=0.01 stage.trainer.save_steps=5 stage.trainer.eval_steps=5" \
bash scripts/run_sft_local.sh
```

The JSONL rows must contain `id`, `source`, `prompt`, and `completion`.

## DGX Spark Launch

On the DGX Spark machine:

```bash
cd /path/to/VeloDoc_pipeline
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[deepspeed]'
```

Export the required runtime values:

```bash
export DATASET_PATH=/data/velodoc/unified_dental_data_full.jsonl
export MLFLOW_TRACKING_URI=http://DGX_SPARK_IP:5000
export HF_TOKEN=...
```

Launch Gemma 4 E2B instruction tuning:

```bash
bash scripts/run_sft_remote.sh
```

Useful overrides:

```bash
MODEL=gemma4_e4b_it bash scripts/run_sft_remote.sh
MODEL=qwen3_8b PEFT=lora bash scripts/run_sft_remote.sh
EXTRA_OVERRIDES="stage.trainer.lr=1e-5 dataset.processing.max_seq_len=4096" bash scripts/run_sft_remote.sh
```

## Security Toggles

`trust_remote_code=true` is blocked unless `VELODOC_ALLOW_REMOTE_CODE=1` is set. Only set it after reviewing the model repository code.

Hub upload is opt-in. Set `HF_PUSH_TO_HUB=1` to upload checkpoints. Uploads are private by default; set `HF_PRIVATE_REPO=0` only when you intentionally want a public repository.

## Gemma 4 Notes

The default remote model is `configs/model/gemma4_e2b_it.yaml`, using `google/gemma-4-E2B-it`. Larger Gemma 4 variants can be added as separate model configs with the same fields, then selected with `MODEL=<config_name>`.

## Merge PEFT Adapter For Serving

Fine-tuning saves a PEFT adapter, not a full standalone model. For serving with vLLM or a model server, merge the adapter into the base model first:

```bash
python scripts/merge_peft_adapter.py \
  --base-model google/gemma-4-E2B-it \
  --adapter-dir outputs/2026-05-26/22-05-30_gemma4-e2b-it-sft-v1.3-dental-gcc-dora/checkpoints/checkpoint-3037 \
  --output-dir outputs/merged/gemma4_e2b_it_ft_merged \
  --dtype bf16 \
  --max-shard-size 4GB
```

Then serve the merged model:

```bash
vllm serve outputs/merged/gemma4_e2b_it_ft_merged \
  --host 0.0.0.0 \
  --port 8081 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```
