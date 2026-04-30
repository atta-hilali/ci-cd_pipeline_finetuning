import os
from typing import Iterable


def _require_keys(section, keys: Iterable[str], section_name: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"Missing required config keys in {section_name}: {', '.join(missing)}")


def validate_config(cfg) -> None:
    _require_keys(cfg, ("dataset", "model", "stage", "peft", "infra", "tracking", "run"), "root")
    _require_keys(cfg.dataset, ("path", "processing", "validation"), "dataset")
    _require_keys(cfg.model, ("name", "hf_id", "dtype", "trust_remote_code"), "model")
    _require_keys(cfg.stage, ("name", "trainer"), "stage")
    _require_keys(cfg.peft, ("method", "r", "alpha", "dropout", "target_modules"), "peft")

    train_path = str(cfg.dataset.path)
    if not train_path:
        raise ValueError("dataset.path must be set to a JSONL training file.")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training dataset file not found: {train_path}")

    eval_path = str(cfg.dataset.eval_path) if cfg.dataset.get("eval_path") else ""
    validation_strategy = str(cfg.dataset.validation.strategy).lower()
    if validation_strategy == "file" and not eval_path:
        raise ValueError("dataset.validation.strategy=file requires dataset.eval_path.")
    if eval_path and not os.path.isfile(eval_path):
        raise FileNotFoundError(f"Validation dataset file not found: {eval_path}")

    if int(cfg.dataset.processing.max_seq_len) <= 0:
        raise ValueError("dataset.processing.max_seq_len must be greater than 0.")

    if str(cfg.model.dtype).lower() not in {"bf16", "fp16", "fp32"}:
        raise ValueError("model.dtype must be one of: bf16, fp16, fp32.")
    if str(cfg.infra.precision).lower() not in {"bf16", "fp16", "fp32"}:
        raise ValueError("infra.precision must be one of: bf16, fp16, fp32.")

    trainer = cfg.stage.trainer
    if int(trainer.micro_batch_size) < 1:
        raise ValueError("stage.trainer.micro_batch_size must be at least 1.")
    if int(trainer.grad_accum) < 1:
        raise ValueError("stage.trainer.grad_accum must be at least 1.")
    if float(trainer.lr) <= 0:
        raise ValueError("stage.trainer.lr must be greater than 0.")

    save_steps = int(trainer.save_steps)
    eval_steps = int(trainer.eval_steps)
    if save_steps < 1 or eval_steps < 1:
        raise ValueError("stage.trainer.save_steps and stage.trainer.eval_steps must be at least 1.")
    if save_steps % eval_steps != 0:
        raise ValueError(
            "stage.trainer.save_steps must be a multiple of stage.trainer.eval_steps "
            "when best-checkpoint loading is enabled."
        )

    limits = getattr(cfg.infra, "limits", {})
    max_train_steps = getattr(limits, "max_train_steps", None)
    if max_train_steps is not None and int(max_train_steps) < 1:
        raise ValueError("infra.limits.max_train_steps must be at least 1 when set.")
    max_eval_samples = getattr(limits, "max_eval_samples", None)
    if max_eval_samples is not None and int(max_eval_samples) < 1:
        raise ValueError("infra.limits.max_eval_samples must be at least 1 when set.")
