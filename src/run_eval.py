"""
Run Evaluation — Walk-forward evaluation pipeline.
Loads eval data (days 31-60), runs through orchestrator, computes all metrics.
"""
import json, logging, sys, os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval_suite import (
    get_eval_dates, run_full_evaluation, print_eval_report,
    WALK_FORWARD_BLOCKS
)
from src.orchestrator import Orchestrator
from src.signal_pod import SignalPod, parse_model_output, generate_signal_id
from src.mlflow_config import log_eval_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent
MARKET_STATES_PATH = DATA_DIR / "market_states.parquet"
MODEL_PATH = DATA_DIR / "lora_adapter"  # Path to LoRA weights


def load_eval_data():
    """Load evaluation data (days 31-60) from market_states.parquet."""
    df = pd.read_parquet(MARKET_STATES_PATH)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["ts"].dt.date
    dates = sorted(df["date"].unique())

    eval_dates = dates[30:]  # Days 31-60
    eval_df = df[df["date"].isin(eval_dates)].copy()

    logger.info(f"Eval data: {len(eval_df)} rows, dates {eval_dates[0]} to {eval_dates[-1]}")
    return eval_df, dates


def run_walk_forward_eval(model_path: str = None, use_rag: bool = False):
    """Run walk-forward evaluation across days 31-60."""
    eval_df, all_dates = load_eval_data()
    date_info = get_eval_dates(str(MARKET_STATES_PATH))

    # Initialize pod and orchestrator
    pod = None
    if model_path and Path(model_path).exists():
        pod = SignalPod(model_path=model_path, use_rag=use_rag)
        pod.load_model()

    orchestrator = Orchestrator(pod)

    predictions = []
    actuals = []
    vix_values = []
    eval_dates_list = []

    for _, row in eval_df.iterrows():
        market_state = {
            "nifty_spot": row["nifty_spot"],
            "atm_iv": row["atm_iv"],
            "iv_skew_25d": row["iv_skew_25d"],
            "pcr": row["pcr"],
            "adx_14": row["adx_14"],
            "realized_vol_5d": row["realized_vol_5d"],
            "vix_india": row["vix_india"],
            "dte_nearest": int(row["dte_nearest"]),
            "moneyness_band": row["moneyness_band"],
        }
        timestamp = row["timestamp"]
        actual_label = row["label"]

        if pod is not None:
            # Run through full orchestrator (model inference)
            rag_ctx = None
            if use_rag:
                from src.rag_experiment import get_rag_context
                rag_ctx = get_rag_context(market_state, k=3)
            output = orchestrator.process(market_state, timestamp, rag_context=rag_ctx)
        else:
            # No model loaded — use orchestrator with mock pod output for testing
            mock_signal = {
                "direction": "NEUTRAL",
                "conviction": 0.5,
                "horizon": "intraday",
                "signal_id": generate_signal_id(market_state, timestamp),
                "generated_at": timestamp,
            }
            output = orchestrator.process_pod_output(market_state, timestamp, mock_signal)

        predictions.append(output)
        actuals.append(actual_label)
        vix_values.append(row["vix_india"])
        eval_dates_list.append(row["date"])

    # Run full evaluation
    results = run_full_evaluation(
        predictions=predictions,
        actuals=actuals,
        orch_logs=orchestrator.get_decision_log(),
        vix_values=vix_values,
        dates=eval_dates_list,
        date_map=date_info["date_map"],
        blocks=date_info["blocks"],
    )

    # Print report
    print_eval_report(results)

    # Log to MLflow
    run_name = f"eval_{'rag' if use_rag else 'no_rag'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        log_eval_results(results, run_name=run_name)
        logger.info(f"Results logged to MLflow as '{run_name}'")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    # Save results to file
    results_path = DATA_DIR / "data" / f"eval_results_{'rag' if use_rag else 'no_rag'}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run walk-forward evaluation")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to LoRA adapter weights")
    parser.add_argument("--rag", action="store_true", help="Enable RAG context")
    args = parser.parse_args()

    model_path = args.model_path or str(MODEL_PATH)
    results = run_walk_forward_eval(model_path=model_path, use_rag=args.rag)
