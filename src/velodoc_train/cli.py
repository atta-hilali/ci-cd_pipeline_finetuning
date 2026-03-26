import os
import hydra
from omegaconf import DictConfig
from velodoc_train.utils.seed import set_global_seed

from velodoc_train.tracking.mlflow_utils import setup_mlflow
from velodoc_train.tracking.artifacts import save_config_snapshot
from velodoc_train.utils.env import ensure_dirs


@hydra.main(config_path="../../configs", config_name="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    # Resolve & create output dir
    out_dir = cfg.run.output_dir
    ensure_dirs(out_dir)

    # Seed
    set_global_seed(int(cfg.run.seed))

    # Tracking
    mlflow_run = setup_mlflow(cfg)

    # Snapshot config (traceability)
    if cfg.artifact_policy.save_config_snapshot:
        save_config_snapshot(cfg, out_dir)
        # log as artifact
        import mlflow
        mlflow.log_artifact(os.path.join(out_dir, "config_resolved.yaml"), artifact_path="config")

    # Dispatch stage
    stage = str(cfg.stage.name).lower()
    if stage == "sft":
        from velodoc_train.sft_train import run_sft

        run_sft(cfg, out_dir)
    elif stage == "dpo":
        raise NotImplementedError("DPO stage is configured but no DPO trainer module exists in this repository yet.")
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Finish
    if mlflow_run is not None:
        import mlflow
        mlflow.end_run()

if __name__ == "__main__":
    main()
