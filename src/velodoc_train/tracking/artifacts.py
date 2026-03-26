import os
from omegaconf import OmegaConf

def save_config_snapshot(cfg, out_dir: str) -> str:
    p = os.path.join(out_dir, "config_resolved.yaml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    return p