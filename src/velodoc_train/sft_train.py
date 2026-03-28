import os
import json
import inspect
import mlflow
from typing import Optional
from omegaconf import DictConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import HfApi
from velodoc_train.data.jsonl_loader import load_jsonl, split_train_eval
from velodoc_train.data.registry import build_dataset_manifest, save_manifest
from velodoc_train.models.build import load_tokenizer, load_causal_lm
from velodoc_train.models.peft import apply_peft
from velodoc_train.training.distributed import is_main_process
from velodoc_train.tracking.summaries import write_summary
from velodoc_train.utils.git import get_git_commit

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None


def _maybe_write_deepspeed_config(cfg: DictConfig, out_dir: str) -> Optional[str]:
    if str(cfg.infra.distributed).lower() != "deepspeed":
        return None
    ds = {
        "train_micro_batch_size_per_gpu": int(cfg.stage.trainer.micro_batch_size),
        "gradient_accumulation_steps": int(cfg.stage.trainer.grad_accum),
        "zero_optimization": {"stage": int(cfg.infra.deepspeed.zero_stage)},
        "bf16": {"enabled": str(cfg.infra.precision).lower() == "bf16"},
        "fp16": {"enabled": str(cfg.infra.precision).lower() == "fp16"},
        "gradient_clipping": float(cfg.stage.trainer.max_grad_norm),
        "wall_clock_breakdown": False,
    }
    p = os.path.join(out_dir, "deepspeed_config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2)
    return p

def _push_best_to_hf(cfg: DictConfig, best_dir: str):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return
    prefix = os.getenv("HF_REPO_PREFIX", "VelodocAI")
    repo_id = f"{prefix}/{cfg.run.name}"
    api = HfApi(token=hf_token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=best_dir,
        commit_message=f"Upload best checkpoint for {cfg.run.name}",
    )
    mlflow.log_param("hf_repo_id", repo_id)


def _build_training_args(cfg: DictConfig, out_dir: str, ds_path: Optional[str], has_eval: bool):
    args_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    params = inspect.signature(args_cls.__init__).parameters
    args_kwargs = {
        "output_dir": os.path.join(out_dir, "checkpoints"),
        "num_train_epochs": float(cfg.stage.trainer.epochs),
        "per_device_train_batch_size": int(cfg.stage.trainer.micro_batch_size),
        "per_device_eval_batch_size": int(cfg.stage.trainer.micro_batch_size),
        "gradient_accumulation_steps": int(cfg.stage.trainer.grad_accum),
        "learning_rate": float(cfg.stage.trainer.lr),
        "warmup_ratio": float(cfg.stage.trainer.warmup_ratio),
        "weight_decay": float(cfg.stage.trainer.weight_decay),
        "max_grad_norm": float(cfg.stage.trainer.max_grad_norm),
        "logging_steps": int(cfg.stage.trainer.logging_steps),
        "eval_steps": int(cfg.stage.trainer.eval_steps) if has_eval else None,
        "save_strategy": "steps",
        "save_steps": int(cfg.stage.trainer.save_steps),
        "save_total_limit": int(cfg.artifact_policy.keep_last_n),
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": str(cfg.stage.trainer.metric_for_best),
        "bf16": str(cfg.infra.precision).lower() == "bf16",
        "fp16": str(cfg.infra.precision).lower() == "fp16",
        "tf32": bool(cfg.infra.tf32),
        "report_to": [],
        "deepspeed": ds_path,
    }

    eval_strategy_key = "evaluation_strategy" if "evaluation_strategy" in params else "eval_strategy"
    args_kwargs[eval_strategy_key] = "steps" if has_eval else "no"

    if "dataloader_num_workers" in params:
        args_kwargs["dataloader_num_workers"] = int(cfg.infra.num_workers)
    if "use_cpu" in params:
        args_kwargs["use_cpu"] = str(cfg.infra.device).lower() == "cpu"
    if "gradient_checkpointing" in params:
        gc_enabled = bool(getattr(getattr(cfg.infra, "deepspeed", {}), "gradient_checkpointing", False))
        args_kwargs["gradient_checkpointing"] = gc_enabled
    if "max_length" in params:
        args_kwargs["max_length"] = int(cfg.dataset.processing.max_seq_len)
    if "packing" in params:
        args_kwargs["packing"] = bool(cfg.dataset.processing.packing)

    return args_cls(**args_kwargs)


def _build_sft_trainer(model, tok, ds_train, ds_eval, args, max_seq_len: int):
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters

    def format_fn(ex):
        return f"{ex['prompt']}\n\nAssistant: {ex['completion']}"

    trainer_kwargs = {
        "model": model,
        "train_dataset": ds_train,
        "eval_dataset": ds_eval,
        "formatting_func": format_fn,
        "args": args,
    }
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tok
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tok
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = max_seq_len

    return SFTTrainer(**trainer_kwargs)


def _load_sft_datasets(cfg: DictConfig):
    train_path = str(cfg.dataset.path)
    raw_eval_path = str(cfg.dataset.eval_path) if cfg.dataset.eval_path else None
    validation_cfg = cfg.dataset.validation if "validation" in cfg.dataset else {}
    strategy = str(getattr(validation_cfg, "strategy", "auto")).lower()
    split_ratio = float(getattr(validation_cfg, "split_ratio", 0.1))
    split_seed = int(getattr(validation_cfg, "seed", cfg.run.seed))

    ds_train = load_jsonl(train_path, limit=None)
    eval_path = raw_eval_path if raw_eval_path and os.path.isfile(raw_eval_path) else None

    if strategy == "file":
        if eval_path is None:
            raise FileNotFoundError(f"Validation file not found: {raw_eval_path}")
        ds_eval = load_jsonl(eval_path, limit=None)
        return ds_train, ds_eval, eval_path, "file", None

    if strategy == "split":
        ds_train, ds_eval = split_train_eval(ds_train, eval_ratio=split_ratio, seed=split_seed)
        return ds_train, ds_eval, None, "split", {
            "split_ratio": split_ratio,
            "split_seed": split_seed,
            "missing_eval_path": raw_eval_path,
        }

    if eval_path is not None:
        ds_eval = load_jsonl(eval_path, limit=None)
        return ds_train, ds_eval, eval_path, "file", None

    print(
        f"Validation file not found ({raw_eval_path}); "
        f"creating a validation split from train data with ratio={split_ratio} and seed={split_seed}."
    )
    ds_train, ds_eval = split_train_eval(ds_train, eval_ratio=split_ratio, seed=split_seed)
    return ds_train, ds_eval, None, "split", {
        "split_ratio": split_ratio,
        "split_seed": split_seed,
        "missing_eval_path": raw_eval_path,
    }

def run_sft(cfg: DictConfig, out_dir: str) -> None:
    process_is_main = is_main_process()

    # Dataset + manifest
    ds_train, ds_eval, eval_path, eval_source, split_meta = _load_sft_datasets(cfg)

    if process_is_main:
        manifest = build_dataset_manifest(str(cfg.dataset.path), eval_path)
        manifest["train_num_rows"] = len(ds_train)
        manifest["eval_num_rows"] = len(ds_eval) if ds_eval is not None else 0
        manifest["eval_source"] = eval_source
        if split_meta is not None:
            manifest.update(split_meta)
        man_path = save_manifest(manifest, out_dir)
        mlflow.log_artifact(man_path, artifact_path="data")
        mlflow.log_params({
            "dataset.train_sha256": manifest["train_sha256"],
            "dataset.eval_sha256": manifest["eval_sha256"] or "",
            "dataset.eval_source": eval_source,
            "git.commit": get_git_commit(),
        })
        if split_meta is not None:
            mlflow.log_params({
                "dataset.eval_split_ratio": split_meta["split_ratio"],
                "dataset.eval_split_seed": split_meta["split_seed"],
            })

    # Model/tokenizer
    tok = load_tokenizer(str(cfg.model.hf_id), trust_remote_code=bool(cfg.model.trust_remote_code))
    model = load_causal_lm(str(cfg.model.hf_id), dtype=str(cfg.model.dtype), trust_remote_code=bool(cfg.model.trust_remote_code))
    model = apply_peft(model, {
        "method": str(cfg.peft.method),
        "r": int(cfg.peft.r),
        "alpha": int(cfg.peft.alpha),
        "dropout": float(cfg.peft.dropout),
        "target_modules": cfg.peft.target_modules,
    })

    # Training args
    ds_path = _maybe_write_deepspeed_config(cfg, out_dir)
    args = _build_training_args(cfg, out_dir, ds_path, has_eval=ds_eval is not None)
    trainer = _build_sft_trainer(
        model=model,
        tok=tok,
        ds_train=ds_train,
        ds_eval=ds_eval,
        args=args,
        max_seq_len=int(cfg.dataset.processing.max_seq_len),
    )

    # Train
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer_is_main = trainer.is_world_process_zero()
    if trainer_is_main:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

    # Evaluate
    if ds_eval is not None:
        eval_metrics = trainer.evaluate()
        if trainer_is_main:
            for k, v in eval_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"eval.{k}", float(v))

    # Save best checkpoint folder (HF push uses it)
    best_dir = args.output_dir  # Trainer stores best model in output_dir when load_best_model_at_end
    # Persist tokenizer too
    if trainer_is_main and cfg.artifact_policy.save_tokenizer:
        tok_dir = os.path.join(out_dir, "tokenizer")
        tok.save_pretrained(tok_dir)
        mlflow.log_artifacts(tok_dir, artifact_path="tokenizer")

    # Log checkpoint dir as artifact (may be large)
    if trainer_is_main:
        mlflow.log_artifacts(args.output_dir, artifact_path="checkpoints")

    # Summary
    if trainer_is_main:
        summary_path = write_summary(out_dir, {
            "run_name": str(cfg.run.name),
            "model": str(cfg.model.hf_id),
            "dataset_version": str(cfg.dataset.version),
            "peft": str(cfg.peft.method),
            "metrics": {**metrics},
        })
        mlflow.log_artifact(summary_path, artifact_path="summaries")

    # Push best to HF (optional; requires HF_TOKEN)
    if trainer_is_main:
        _push_best_to_hf(cfg, best_dir)
