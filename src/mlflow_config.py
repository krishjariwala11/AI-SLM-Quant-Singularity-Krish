"""
MLflow Configuration — Experiment tracking setup.
Must be initialized before the first training run.
"""
import mlflow, os, json, logging
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "nifty-signal-pod"
TRACKING_URI = str(Path(__file__).parent.parent / "mlruns")


def init_mlflow():
    """Initialize MLflow with local file store."""
    os.makedirs(TRACKING_URI, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{TRACKING_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow initialized: {TRACKING_URI}")
    return mlflow


def log_training_run(params: dict, metrics: dict, artifacts: list = None, tags: dict = None):
    """Log a training run with parameters, metrics, and optional artifacts."""
    init_mlflow()
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        if artifacts:
            for art_path in artifacts:
                if os.path.exists(art_path):
                    mlflow.log_artifact(art_path)
        logger.info(f"Logged run: {run.info.run_id}")
        return run.info.run_id


def log_eval_results(results: dict, run_name: str = "evaluation"):
    """Log evaluation results as an MLflow run."""
    init_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        # Flatten and log metrics
        flat = {}
        if "overall_directional_accuracy" in results:
            flat["overall_accuracy"] = results["overall_directional_accuracy"]
        if "schema_pass_rate" in results:
            flat["schema_pass_rate"] = results["schema_pass_rate"]
        if "conviction_validity" in results:
            flat["conviction_validity"] = results["conviction_validity"].get("rate", 0)
        orch = results.get("orchestrator_rates", {})
        for k in ["suppression_rate", "downgrade_rate", "parse_failure_rate", "pass_rate"]:
            if k in orch:
                flat[f"orch_{k}"] = orch[k]
        vix = results.get("vix_regime", {})
        for k in ["high_vix_accuracy", "low_vix_accuracy", "accuracy_gap"]:
            if k in vix:
                flat[f"vix_{k}"] = vix[k]
        ci = results.get("confidence_intervals", {}).get("directional_accuracy", {})
        for k in ["ci_95_lower", "ci_95_upper"]:
            if k in ci:
                flat[k] = ci[k]

        mlflow.log_metrics(flat)
        # Log full results as artifact
        results_path = Path(TRACKING_URI) / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))
        mlflow.set_tag("eval_type", run_name)
        logger.info(f"Logged eval run: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_mlflow()
    print(f"MLflow tracking URI: {TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
