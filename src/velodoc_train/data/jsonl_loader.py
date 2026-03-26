import json
from typing import Optional, Tuple
from datasets import Dataset

REQUIRED_FIELDS = ("id", "source", "prompt", "completion")


def load_jsonl(path: str, limit: Optional[int] = None) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            for k in REQUIRED_FIELDS:
                if k not in obj:
                    raise ValueError(f"Missing field '{k}' in row {i}")
            rows.append({k: obj[k] for k in REQUIRED_FIELDS})
    return Dataset.from_list(rows)


def split_train_eval(ds: Dataset, eval_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be between 0 and 1, got {eval_ratio}")
    if len(ds) < 2:
        raise ValueError("Need at least 2 rows to create a train/validation split.")

    split = ds.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
    return split["train"], split["test"]
