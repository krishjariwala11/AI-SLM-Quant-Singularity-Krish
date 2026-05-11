"""
Data Preparation — Audit and clean finetune_instructions.jsonl.
Filters out corrupted examples and produces a clean training set.
"""
import json, logging, os
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path(__file__).parent.parent
RAW_PATH = DATA_DIR / "finetune_instructions.jsonl"
CLEAN_PATH = DATA_DIR / "data" / "train_clean.jsonl"

REQUIRED_OUTPUT_KEYS = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS = {"intraday", "next_session"}


def load_raw_data(path: str = None) -> List[Dict]:
    p = Path(path) if path else RAW_PATH
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def audit_example(idx: int, example: Dict) -> Tuple[bool, str]:
    """Audit a single training example. Returns (is_valid, reason)."""
    # Check top-level keys
    for key in ("instruction", "input", "output"):
        if key not in example:
            return False, f"missing_top_level_key:{key}"

    # Parse output JSON
    try:
        out = json.loads(example["output"])
    except json.JSONDecodeError as e:
        return False, f"output_json_parse_error:{e}"

    # Parse input JSON
    try:
        inp = json.loads(example["input"])
    except json.JSONDecodeError as e:
        return False, f"input_json_parse_error:{e}"

    # Check required output keys
    missing = REQUIRED_OUTPUT_KEYS - set(out.keys())
    if missing:
        return False, f"missing_output_keys:{missing}"

    # Check direction value
    if out["direction"] not in VALID_DIRECTIONS:
        return False, f"invalid_direction:{out['direction']}"

    # Check horizon value
    if out["horizon"] not in VALID_HORIZONS:
        return False, f"invalid_horizon:{out['horizon']}"

    # Check conviction type and range — THIS IS THE KEY FILTER
    conv = out["conviction"]
    if isinstance(conv, str):
        return False, f"conviction_is_string:'{conv}'"
    if not isinstance(conv, (int, float)):
        return False, f"conviction_wrong_type:{type(conv).__name__}"
    if not (0.0 <= conv <= 1.0):
        return False, f"conviction_out_of_range:{conv}"

    return True, "clean"


def prepare_clean_dataset(raw_path: str = None, clean_path: str = None):
    """Load raw data, audit each example, save clean subset."""
    raw = load_raw_data(raw_path)
    logger.info(f"Loaded {len(raw)} raw examples")

    clean = []
    rejected = []
    for i, ex in enumerate(raw):
        is_valid, reason = audit_example(i, ex)
        if is_valid:
            clean.append(ex)
        else:
            rejected.append({"index": i, "reason": reason})
            logger.warning(f"Rejected example {i}: {reason}")

    logger.info(f"Clean: {len(clean)}, Rejected: {len(rejected)}")

    # Log rejection summary
    from collections import Counter
    reasons = Counter(r["reason"].split(":")[0] for r in rejected)
    for reason, count in reasons.most_common():
        logger.info(f"  Rejection reason '{reason}': {count}")

    # Save clean dataset
    out_path = Path(clean_path) if clean_path else CLEAN_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(clean)} clean examples to {out_path}")

    # Save audit report
    report = {
        "total_raw": len(raw),
        "total_clean": len(clean),
        "total_rejected": len(rejected),
        "rejection_details": rejected,
        "rejection_summary": dict(reasons),
    }
    report_path = out_path.parent / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Audit report saved to {report_path}")

    return clean, rejected


if __name__ == "__main__":
    clean, rejected = prepare_clean_dataset()
    print(f"\nResult: {len(clean)} clean, {len(rejected)} rejected")
