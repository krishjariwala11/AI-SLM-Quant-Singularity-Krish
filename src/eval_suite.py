"""
Eval Suite — Pre-committed evaluation metrics and thresholds.
MUST be committed before the first training run.
Walk-forward evaluation: Days 31-60, six 5-day rolling blocks.
"""
import json, logging, numpy as np, pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# PRE-COMMITTED THRESHOLDS
THRESHOLDS = {
    "directional_accuracy_pass": 0.38,
    "directional_accuracy_fail": 0.33,
    "schema_pass_rate_pass": 0.95,
    "schema_pass_rate_fail": 0.90,
    "conviction_validity_pass": 1.00,
    "conviction_validity_fail": 0.98,
    "suppression_rate_min": 0.15,
    "suppression_rate_max": 0.25,
    "downgrade_rate_max_pass": 0.30,
    "downgrade_rate_max_fail": 0.50,
    "parse_failure_rate_pass": 0.05,
    "parse_failure_rate_fail": 0.10,
    "vix_regime_gap_pass": 0.15,
    "vix_regime_gap_fail": 0.25,
}

CONVICTION_BINS = [
    (0.0, 0.4, "very_low"), (0.4, 0.5, "low"),
    (0.5, 0.6, "moderate"), (0.6, 0.8, "high"), (0.8, 1.0, "very_high"),
]

WALK_FORWARD_BLOCKS = [
    ("block_1", 31, 35), ("block_2", 36, 40), ("block_3", 41, 45),
    ("block_4", 46, 50), ("block_5", 51, 55), ("block_6", 56, 60),
]


def get_eval_dates(market_states_path: str) -> Dict:
    df = pd.read_parquet(market_states_path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["ts"].dt.date
    dates = sorted(df["date"].unique())
    date_map = {i + 1: d for i, d in enumerate(dates)}
    blocks = {}
    for name, s, e in WALK_FORWARD_BLOCKS:
        blocks[name] = [date_map[d] for d in range(s, e + 1) if d in date_map]
    return {"train_dates": dates[:30], "eval_dates": dates[30:],
            "date_map": date_map, "blocks": blocks}


def compute_directional_accuracy(predictions, actuals):
    if not predictions: return 0.0
    return sum(1 for p, a in zip(predictions, actuals)
               if p.get("direction") == a) / len(actuals)


def compute_schema_pass_rate(predictions):
    if not predictions: return 0.0
    req = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
    valid = 0
    for p in predictions:
        try:
            if not req.issubset(p.keys()): continue
            if p["direction"] not in {"CE", "PE", "NEUTRAL"}: continue
            if p["horizon"] not in {"intraday", "next_session"}: continue
            c = p["conviction"]
            if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0): continue
            valid += 1
        except: continue
    return valid / len(predictions)


def compute_conviction_validity(predictions):
    if not predictions: return 1.0, []
    invalid = []
    for i, p in enumerate(predictions):
        c = p.get("conviction")
        if c is None or isinstance(c, str) or not isinstance(c, (int, float)):
            invalid.append(i)
        elif not (0.0 <= c <= 1.0):
            invalid.append(i)
    return 1.0 - len(invalid) / len(predictions), invalid


def compute_conviction_calibration(predictions, actuals):
    bins_result = {}
    for low, high, label in CONVICTION_BINS:
        bp, ba = [], []
        for p, a in zip(predictions, actuals):
            c = p.get("conviction", 0.0)
            if isinstance(c, (int, float)) and low <= c < high:
                bp.append(p); ba.append(a)
        acc = compute_directional_accuracy(bp, ba) if bp else None
        bins_result[label] = {"accuracy": acc, "count": len(bp), "range": (low, high)}
    accs = [bins_result[l]["accuracy"] for _, _, l in CONVICTION_BINS
            if bins_result[l]["accuracy"] is not None]
    mono = all(accs[i] <= accs[i+1] for i in range(len(accs)-1)) if len(accs) >= 2 else False
    return {"bins": bins_result, "is_monotonic": mono, "accuracies_sequence": accs}


def compute_orchestrator_rates(logs):
    if not logs: return {"suppression_rate": 0, "downgrade_rate": 0,
                         "parse_failure_rate": 0, "pass_rate": 0}
    t = len(logs)
    return {
        "suppression_rate": sum(1 for l in logs if l.get("reason_code") == "ADX_BELOW_THRESHOLD") / t,
        "downgrade_rate": sum(1 for l in logs if l.get("reason_code") == "LOW_CONVICTION") / t,
        "parse_failure_rate": sum(1 for l in logs if l.get("reason_code") == "PARSE_FAILURE") / t,
        "pass_rate": sum(1 for l in logs if l.get("reason_code") == "PASS") / t,
        "counts": {"total": t,
                   "suppressed": sum(1 for l in logs if l.get("reason_code") == "ADX_BELOW_THRESHOLD"),
                   "parse_failures": sum(1 for l in logs if l.get("reason_code") == "PARSE_FAILURE"),
                   "downgrades": sum(1 for l in logs if l.get("reason_code") == "LOW_CONVICTION"),
                   "passed": sum(1 for l in logs if l.get("reason_code") == "PASS")}
    }


def compute_vix_regime_metrics(predictions, actuals, vix_values,
                                vix_high=None, vix_low=None):
    v = np.array(vix_values)
    if vix_high is None: vix_high = np.percentile(v, 75)
    if vix_low is None: vix_low = np.percentile(v, 25)
    hp, ha, lp, la = [], [], [], []
    for p, a, vx in zip(predictions, actuals, vix_values):
        if vx >= vix_high: hp.append(p); ha.append(a)
        elif vx <= vix_low: lp.append(p); la.append(a)
    h_acc = compute_directional_accuracy(hp, ha)
    l_acc = compute_directional_accuracy(lp, la)
    return {"high_vix_accuracy": h_acc, "low_vix_accuracy": l_acc,
            "high_vix_count": len(hp), "low_vix_count": len(lp),
            "accuracy_gap": abs(h_acc - l_acc),
            "vix_threshold_high": vix_high, "vix_threshold_low": vix_low}


def bootstrap_ci(values, n=1000, confidence=0.95):
    if not values: return 0.0, 0.0, 0.0
    v = np.array(values)
    boots = [np.mean(np.random.choice(v, len(v), replace=True)) for _ in range(n)]
    a = (1 - confidence) / 2
    return float(np.mean(v)), float(np.percentile(boots, 100*a)), float(np.percentile(boots, 100*(1-a)))


def run_full_evaluation(predictions, actuals, orch_logs, vix_values, dates, date_map, blocks):
    results = {"timestamp": datetime.now().isoformat(), "total_predictions": len(predictions),
               "thresholds": THRESHOLDS}
    results["overall_directional_accuracy"] = compute_directional_accuracy(predictions, actuals)

    # Per-block accuracy
    br = {}
    for bn, bd in blocks.items():
        bp = [(p, a) for p, a, d in zip(predictions, actuals, dates) if d in bd]
        if bp:
            bpreds, bacts = zip(*bp)
            br[bn] = {"accuracy": compute_directional_accuracy(list(bpreds), list(bacts)),
                      "count": len(bp)}
    results["per_block_accuracy"] = br
    results["schema_pass_rate"] = compute_schema_pass_rate(predictions)
    cv, inv = compute_conviction_validity(predictions)
    results["conviction_validity"] = {"rate": cv, "invalid_count": len(inv)}
    results["conviction_calibration"] = compute_conviction_calibration(predictions, actuals)
    results["orchestrator_rates"] = compute_orchestrator_rates(orch_logs)

    # Per-block orchestrator rates
    bor = {}
    for bn, bd in blocks.items():
        bl = [l for l, d in zip(orch_logs, dates) if d in bd]
        if bl: bor[bn] = compute_orchestrator_rates(bl)
    results["per_block_orchestrator_rates"] = bor
    results["vix_regime"] = compute_vix_regime_metrics(predictions, actuals, vix_values)

    # Bootstrap CI
    acc_ind = [1.0 if p.get("direction") == a else 0.0 for p, a in zip(predictions, actuals)]
    m, lo, hi = bootstrap_ci(acc_ind)
    results["confidence_intervals"] = {"directional_accuracy": {"mean": m, "ci_95_lower": lo, "ci_95_upper": hi}}
    results["assessment"] = assess_thresholds(results)
    return results


def assess_thresholds(results):
    a = {}
    acc = results.get("overall_directional_accuracy", 0)
    a["directional_accuracy"] = {"value": acc, "status": "PASS" if acc >= THRESHOLDS["directional_accuracy_pass"]
                                  else "FAIL" if acc < THRESHOLDS["directional_accuracy_fail"] else "MARGINAL"}
    spr = results.get("schema_pass_rate", 0)
    a["schema_pass_rate"] = {"value": spr, "status": "PASS" if spr >= THRESHOLDS["schema_pass_rate_pass"]
                              else "FAIL" if spr < THRESHOLDS["schema_pass_rate_fail"] else "MARGINAL"}
    cv = results.get("conviction_validity", {}).get("rate", 0)
    a["conviction_validity"] = {"value": cv, "status": "PASS" if cv >= THRESHOLDS["conviction_validity_pass"]
                                 else "FAIL" if cv < THRESHOLDS["conviction_validity_fail"] else "MARGINAL"}
    pfr = results.get("orchestrator_rates", {}).get("parse_failure_rate", 0)
    a["parse_failure_rate"] = {"value": pfr, "status": "PASS" if pfr <= THRESHOLDS["parse_failure_rate_pass"]
                                else "FAIL" if pfr > THRESHOLDS["parse_failure_rate_fail"] else "MARGINAL"}
    vg = results.get("vix_regime", {}).get("accuracy_gap", 0)
    a["vix_regime_gap"] = {"value": vg, "status": "PASS" if vg <= THRESHOLDS["vix_regime_gap_pass"]
                            else "FAIL" if vg > THRESHOLDS["vix_regime_gap_fail"] else "MARGINAL"}
    mono = results.get("conviction_calibration", {}).get("is_monotonic", False)
    a["conviction_calibration"] = {"value": mono, "status": "PASS" if mono else "MARGINAL"}
    statuses = [v["status"] for v in a.values()]
    a["overall"] = "FAIL" if "FAIL" in statuses else "PASS" if all(s == "PASS" for s in statuses) else "MARGINAL"
    return a


def print_eval_report(results):
    print("\n" + "=" * 70)
    print("SIGNAL POD EVALUATION REPORT")
    print("=" * 70)
    print(f"\nTotal predictions: {results['total_predictions']}")
    print(f"Overall accuracy: {results['overall_directional_accuracy']:.3f}")
    ci = results["confidence_intervals"]["directional_accuracy"]
    print(f"  95% CI: [{ci['ci_95_lower']:.3f}, {ci['ci_95_upper']:.3f}]")
    print("\n--- Per-Block Accuracy ---")
    for b, d in results.get("per_block_accuracy", {}).items():
        print(f"  {b}: {d['accuracy']:.3f} (n={d['count']})")
    print(f"\nSchema pass rate: {results['schema_pass_rate']:.3f}")
    print(f"Conviction validity: {results['conviction_validity']['rate']:.3f}")
    print("\n--- Conviction Calibration ---")
    for l, bd in results.get("conviction_calibration", {}).get("bins", {}).items():
        if bd["count"] > 0:
            print(f"  {l}: acc={bd['accuracy']:.3f}, n={bd['count']}")
    print(f"\n--- Orchestrator Rates ---")
    o = results.get("orchestrator_rates", {})
    print(f"  Suppression: {o.get('suppression_rate', 0):.3f}")
    print(f"  Downgrade: {o.get('downgrade_rate', 0):.3f}")
    print(f"  Parse fail: {o.get('parse_failure_rate', 0):.3f}")
    print(f"\n--- VIX Regime ---")
    v = results.get("vix_regime", {})
    print(f"  High VIX acc: {v.get('high_vix_accuracy', 0):.3f} (n={v.get('high_vix_count', 0)})")
    print(f"  Low VIX acc: {v.get('low_vix_accuracy', 0):.3f} (n={v.get('low_vix_count', 0)})")
    print(f"\n--- Assessment ---")
    for m, d in results.get("assessment", {}).items():
        if m == "overall": continue
        s = "✓" if d["status"] == "PASS" else "✗" if d["status"] == "FAIL" else "~"
        print(f"  {s} {m}: {d['value']} [{d['status']}]")
    print(f"\n  OVERALL: {results['assessment'].get('overall', 'N/A')}")


if __name__ == "__main__":
    print("Pre-committed thresholds:")
    print(json.dumps(THRESHOLDS, indent=2))
