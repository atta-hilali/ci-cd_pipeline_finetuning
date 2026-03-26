import os
import mlflow

def setup_mlflow(cfg):
    tracking_backend = str(cfg.tracking.backend).lower()
    if tracking_backend != "mlflow":
        return None

    tracking_uri = str(cfg.tracking.tracking_uri)
    # allow override from env (useful for Colab -> DGX)
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        tracking_uri = env_uri

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(cfg.tracking.experiment_name))

    run_name = str(cfg.run.name)
    run = mlflow.start_run(run_name=run_name)

    # Log core params
    flat = {
        "project.name": str(cfg.project.name),
        "dataset.version": str(cfg.dataset.version),
        "model.hf_id": str(cfg.model.hf_id),
        "stage": str(cfg.stage.name),
        "peft.method": str(cfg.peft.method),
        "seed": int(cfg.run.seed),
    }
    mlflow.log_params(flat)
    return run