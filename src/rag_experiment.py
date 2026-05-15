"""
RAG Experiment — Ablation study: pod with vs without retrieval-augmented context.
Uses the provided retrieve.py function (unmodified).
"""
import json, logging, sys
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Add parent dir so we can import retrieve
sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieve import retrieve


def format_rag_context(episodes: List[Dict]) -> str:
    """Format retrieved episodes into a context string for the prompt."""
    lines = []
    for i, ep in enumerate(episodes, 1):
        ms = ep.get("market_state", {})
        lines.append(
            f"Episode {i} | Regime: {ep.get('regime', 'unknown')} | "
            f"ADX {ms.get('adx_14', 0):.1f}, VIX {ms.get('vix_india', 0):.1f}, "
            f"PCR {ms.get('pcr', 0):.2f}, DTE {ms.get('dte_nearest', 0)}. "
            f"{ep.get('regime_description', '')} "
            f"Result: {ep.get('outcome', 'unknown')}. "
            f"{ep.get('outcome_description', '')}"
        )
    return "\n".join(lines)


def get_rag_context(market_state: Dict, k: int = 3) -> str:
    """Retrieve similar episodes and format as context string."""
    episodes = retrieve(market_state, k=k)
    return format_rag_context(episodes)


def run_ablation(market_states: List[Dict], timestamps: List[str],
                 actuals: List[str], signal_pod, orchestrator_cls):
    """
    Run full ablation: generate signals with and without RAG context.
    Returns results for both conditions.
    """
    from src.orchestrator import Orchestrator

    results = {"with_rag": [], "without_rag": [],
               "rag_logs": [], "no_rag_logs": []}

    # Condition A: Without RAG
    logger.info("Running Condition A: Without RAG")
    orch_no_rag = Orchestrator(signal_pod)
    for ms, ts, actual in zip(market_states, timestamps, actuals):
        output = orch_no_rag.process(ms, ts, rag_context=None)
        results["without_rag"].append(output)
    results["no_rag_logs"] = orch_no_rag.get_decision_log()

    # Condition B: With RAG
    logger.info("Running Condition B: With RAG")
    orch_rag = Orchestrator(signal_pod)
    for ms, ts, actual in zip(market_states, timestamps, actuals):
        rag_ctx = get_rag_context(ms, k=3)
        output = orch_rag.process(ms, ts, rag_context=rag_ctx)
        results["with_rag"].append(output)
    results["rag_logs"] = orch_rag.get_decision_log()

    return results


def analyze_conviction_changes(with_rag: List[Dict], without_rag: List[Dict],
                                market_states: List[Dict]):
    """Analyze how RAG context changed conviction scores."""
    changes = []
    for i, (wr, nr, ms) in enumerate(zip(with_rag, without_rag, market_states)):
        conv_rag = wr.get("conviction", 0.0)
        conv_no_rag = nr.get("conviction", 0.0)
        dir_rag = wr.get("direction", "NEUTRAL")
        dir_no_rag = nr.get("direction", "NEUTRAL")
        delta = conv_rag - conv_no_rag

        if abs(delta) > 0.01 or dir_rag != dir_no_rag:
            changes.append({
                "index": i,
                "conv_no_rag": conv_no_rag,
                "conv_rag": conv_rag,
                "conv_delta": delta,
                "dir_no_rag": dir_no_rag,
                "dir_rag": dir_rag,
                "direction_changed": dir_rag != dir_no_rag,
                "adx": ms.get("adx_14", 0),
                "vix": ms.get("vix_india", 0),
            })

    summary = {
        "total_samples": len(with_rag),
        "conviction_changed": len([c for c in changes if abs(c["conv_delta"]) > 0.01]),
        "direction_changed": len([c for c in changes if c["direction_changed"]]),
        "mean_conv_delta": (sum(c["conv_delta"] for c in changes) / len(changes)
                            if changes else 0.0),
        "changes": changes[:20],  # Cap for readability
    }
    return summary


if __name__ == "__main__":
    from src.run_eval import load_eval_data
    from src.signal_pod import SignalPod
    from src.orchestrator import Orchestrator
    
    # Load data
    eval_df, _ = load_eval_data()
    market_states = []
    timestamps = []
    actuals = []
    for _, row in eval_df.iterrows():
        market_state = {
            "nifty_spot": row["nifty_spot"], "atm_iv": row["atm_iv"], "iv_skew_25d": row["iv_skew_25d"],
            "pcr": row["pcr"], "adx_14": row["adx_14"], "realized_vol_5d": row["realized_vol_5d"],
            "vix_india": row["vix_india"], "dte_nearest": int(row["dte_nearest"]), "moneyness_band": row["moneyness_band"]
        }
        market_states.append(market_state)
        timestamps.append(row["timestamp"])
        actuals.append(row["label"])

    pod = SignalPod(model_path="lora_adapter/final", use_rag=True)
    pod.load_model()
    
    results = run_ablation(market_states, timestamps, actuals, pod, Orchestrator)
    summary = analyze_conviction_changes(results["with_rag"], results["without_rag"], market_states)
    
    print("\n=== RAG Ablation Results ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Conviction changed in {summary['conviction_changed']} samples")
    print(f"Direction changed in {summary['direction_changed']} samples")
    print(f"Mean conviction delta: {summary['mean_conv_delta']:.4f}")
    
    print("\nTop Changes:")
    for change in summary['changes']:
        print(f"Index {change['index']}: ADX {change['adx']:.1f}, VIX {change['vix']:.1f}")
        print(f"  No RAG: {change['dir_no_rag']} (conv: {change['conv_no_rag']:.2f})")
        print(f"  With RAG: {change['dir_rag']} (conv: {change['conv_rag']:.2f})")
        print(f"  Delta: {change['conv_delta']:+.2f}")
