import json
import os
from typing import Optional, Tuple
from datasets import Dataset

REQUIRED_FIELDS = ("id", "source", "prompt", "completion")
TEXT_FIELDS = ("source", "prompt", "completion")


def load_jsonl(path: str, limit: Optional[int] = None) -> Dataset:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i + 1} of {path}: {exc.msg}") from exc
            for k in REQUIRED_FIELDS:
                if k not in obj:
                    raise ValueError(f"Missing field '{k}' on line {i + 1} of {path}")
            for k in TEXT_FIELDS:
                if not isinstance(obj[k], str) or not obj[k].strip():
                    raise ValueError(f"Field '{k}' must be a non-empty string on line {i + 1} of {path}")
            rows.append({k: str(obj[k]) for k in REQUIRED_FIELDS})
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return Dataset.from_list(rows)


def split_train_eval(ds: Dataset, eval_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be between 0 and 1, got {eval_ratio}")
    if len(ds) < 2:
        raise ValueError("Need at least 2 rows to create a train/validation split.")

    split = ds.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
    return split["train"], split["test"]
