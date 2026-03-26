import os, json
from typing import Optional
from velodoc_train.utils.hashing import sha256_file


def build_dataset_manifest(train_path: str, eval_path: Optional[str]) -> dict:
    manifest = {
        "train_path": train_path,
        "eval_path": eval_path,
        "train_sha256": sha256_file(train_path),
        "eval_sha256": sha256_file(eval_path) if eval_path else None,
    }
    return manifest

def save_manifest(manifest: dict, out_dir: str) -> str:
    p = os.path.join(out_dir, "dataset_manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return p
