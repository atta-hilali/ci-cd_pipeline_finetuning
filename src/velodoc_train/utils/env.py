import os

def ensure_dirs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)