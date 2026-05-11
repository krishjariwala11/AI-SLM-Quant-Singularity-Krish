"""
Orchestrator — Deterministic wrapper around the Signal Pod.
Applies three sequential suppression rules before any signal reaches downstream.

Rules (in order):
1. ADX < 20 → suppress entirely, return NEUTRAL (don't call model)
2. Parse failure → return NEUTRAL, log raw output
3. Conviction < 0.40 → downgrade direction to NEUTRAL

Every decision is logged with a reason code and triggering values.
"""
import json, logging, uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

ADX_THRESHOLD = 20.0
CONVICTION_THRESHOLD = 0.40

# Reason codes
REASON_ADX_BELOW = "ADX_BELOW_THRESHOLD"
REASON_PARSE_FAIL = "PARSE_FAILURE"
REASON_LOW_CONVICTION = "LOW_CONVICTION"
REASON_PASS = "PASS"


def make_neutral_signal(market_state: Dict, timestamp: str, signal_id: str = None) -> Dict:
    """Create a NEUTRAL signal with conviction 0.0."""
    if not signal_id:
        ns = uuid.UUID("12345678-1234-5678-1234-567812345678")
        key = json.dumps(market_state, sort_keys=True) + timestamp
        signal_id = str(uuid.uuid5(ns, key))
    return {
        "direction": "NEUTRAL",
        "conviction": 0.0,
        "horizon": "intraday",
        "signal_id": signal_id,
        "generated_at": timestamp,
    }


def make_decision_log(
    signal_id: str,
    timestamp: str,
    reason_code: str,
    details: Dict,
    pod_output: Optional[Dict],
    orchestrator_output: Dict,
) -> Dict:
    """Create a structured decision log entry."""
    return {
        "signal_id": signal_id,
        "timestamp": timestamp,
        "reason_code": reason_code,
        "details": details,
        "pod_output": pod_output,
        "orchestrator_output": orchestrator_output,
    }


class Orchestrator:
    """
    Wraps the Signal Pod and applies deterministic suppression rules.
    Downstream pipeline reads ONLY orchestrator output, never raw pod signal.
    """

    def __init__(self, signal_pod=None):
        """
        Args:
            signal_pod: A SignalPod instance with a .generate() method.
                       If None, orchestrator can still be used with process_pod_output().
        """
        self.signal_pod = signal_pod
        self.decision_log: List[Dict] = []

    def process(self, market_state: Dict, timestamp: str,
                rag_context: str = None) -> Dict:
        """
        Full orchestration pipeline: regime check → model call → validation.
        Returns the final orchestrator output (never raw pod signal).
        """
        adx = market_state.get("adx_14", 0.0)

        # RULE 1: ADX < 20 → suppress entirely, do NOT call model
        if adx < ADX_THRESHOLD:
            output = make_neutral_signal(market_state, timestamp)
            log_entry = make_decision_log(
                signal_id=output["signal_id"],
                timestamp=timestamp,
                reason_code=REASON_ADX_BELOW,
                details={"adx_value": adx, "threshold": ADX_THRESHOLD},
                pod_output=None,  # Model was never called
                orchestrator_output=output,
            )
            self.decision_log.append(log_entry)
            logger.info(f"[{REASON_ADX_BELOW}] ADX={adx:.1f} < {ADX_THRESHOLD}. Suppressed.")
            return output

        # Call the signal pod
        pod_output = None
        if self.signal_pod is not None:
            pod_output = self.signal_pod.generate(market_state, timestamp, rag_context)
        else:
            logger.error("No signal pod configured")
            pod_output = {"_fallback": True}

        # RULE 2: Parse failure check
        if pod_output.get("_fallback", False):
            output = make_neutral_signal(market_state, timestamp)
            log_entry = make_decision_log(
                signal_id=output["signal_id"],
                timestamp=timestamp,
                reason_code=REASON_PARSE_FAIL,
                details={"raw_output": pod_output.get("_raw_output", "unknown")},
                pod_output=pod_output,
                orchestrator_output=output,
            )
            self.decision_log.append(log_entry)
            logger.warning(f"[{REASON_PARSE_FAIL}] Model output failed to parse.")
            return output

        # Validate schema before Rule 3
        if not self._validate_schema(pod_output):
            output = make_neutral_signal(market_state, timestamp)
            log_entry = make_decision_log(
                signal_id=output["signal_id"],
                timestamp=timestamp,
                reason_code=REASON_PARSE_FAIL,
                details={"reason": "schema_validation_failed", "pod_output": pod_output},
                pod_output=pod_output,
                orchestrator_output=output,
            )
            self.decision_log.append(log_entry)
            logger.warning(f"[{REASON_PARSE_FAIL}] Schema validation failed.")
            return output

        # RULE 3: Conviction < 0.40 → downgrade direction to NEUTRAL
        conviction = pod_output.get("conviction", 0.0)
        if conviction < CONVICTION_THRESHOLD:
            output = pod_output.copy()
            original_direction = output["direction"]
            output["direction"] = "NEUTRAL"
            log_entry = make_decision_log(
                signal_id=output["signal_id"],
                timestamp=timestamp,
                reason_code=REASON_LOW_CONVICTION,
                details={
                    "conviction": conviction,
                    "threshold": CONVICTION_THRESHOLD,
                    "original_direction": original_direction,
                },
                pod_output=pod_output,
                orchestrator_output=output,
            )
            self.decision_log.append(log_entry)
            logger.info(f"[{REASON_LOW_CONVICTION}] conv={conviction:.2f} < {CONVICTION_THRESHOLD}. "
                        f"Downgraded {original_direction} → NEUTRAL.")
            return output

        # All checks passed
        output = pod_output.copy()
        log_entry = make_decision_log(
            signal_id=output["signal_id"],
            timestamp=timestamp,
            reason_code=REASON_PASS,
            details={"adx": adx, "conviction": conviction},
            pod_output=pod_output,
            orchestrator_output=output,
        )
        self.decision_log.append(log_entry)
        logger.debug(f"[{REASON_PASS}] Signal passed all checks.")
        return output

    def process_pod_output(self, market_state: Dict, timestamp: str,
                           pod_output: Dict) -> Dict:
        """
        Process a pre-computed pod output through Rules 2 and 3 only.
        Useful for evaluation when model outputs are pre-generated.
        Rule 1 (ADX) must be checked separately before calling this.
        """
        adx = market_state.get("adx_14", 0.0)

        # Rule 1: ADX check
        if adx < ADX_THRESHOLD:
            output = make_neutral_signal(market_state, timestamp)
            log_entry = make_decision_log(
                output["signal_id"], timestamp, REASON_ADX_BELOW,
                {"adx_value": adx, "threshold": ADX_THRESHOLD},
                None, output)
            self.decision_log.append(log_entry)
            return output

        # Rule 2: Parse failure
        if pod_output.get("_fallback", False) or not self._validate_schema(pod_output):
            output = make_neutral_signal(market_state, timestamp)
            log_entry = make_decision_log(
                output["signal_id"], timestamp, REASON_PARSE_FAIL,
                {"raw_output": str(pod_output)[:500]},
                pod_output, output)
            self.decision_log.append(log_entry)
            return output

        # Rule 3: Low conviction
        conv = pod_output.get("conviction", 0.0)
        if conv < CONVICTION_THRESHOLD:
            output = pod_output.copy()
            orig_dir = output["direction"]
            output["direction"] = "NEUTRAL"
            log_entry = make_decision_log(
                output["signal_id"], timestamp, REASON_LOW_CONVICTION,
                {"conviction": conv, "threshold": CONVICTION_THRESHOLD,
                 "original_direction": orig_dir},
                pod_output, output)
            self.decision_log.append(log_entry)
            return output

        # Pass
        output = pod_output.copy()
        log_entry = make_decision_log(
            output["signal_id"], timestamp, REASON_PASS,
            {"adx": adx, "conviction": conv},
            pod_output, output)
        self.decision_log.append(log_entry)
        return output

    def _validate_schema(self, signal: Dict) -> bool:
        """Validate that a signal has correct schema."""
        required = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
        if not required.issubset(signal.keys()):
            return False
        if signal["direction"] not in {"CE", "PE", "NEUTRAL"}:
            return False
        if signal["horizon"] not in {"intraday", "next_session"}:
            return False
        c = signal["conviction"]
        if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0):
            return False
        return True

    def get_decision_log(self) -> List[Dict]:
        return self.decision_log

    def clear_log(self):
        self.decision_log = []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orch = Orchestrator()

    # Test Rule 1: ADX < 20
    ms1 = {"adx_14": 14.0, "vix_india": 28.0, "nifty_spot": 22000}
    r1 = orch.process_pod_output(ms1, "2024-11-15T09:30:00+05:30",
                                  {"direction": "CE", "conviction": 0.7,
                                   "horizon": "intraday", "signal_id": "test1",
                                   "generated_at": "2024-11-15T09:30:00+05:30"})
    print(f"Rule 1 test: {r1['direction']} (expected NEUTRAL)")

    # Test Rule 3: Low conviction
    ms2 = {"adx_14": 25.0, "vix_india": 15.0, "nifty_spot": 22000}
    r2 = orch.process_pod_output(ms2, "2024-11-15T10:00:00+05:30",
                                  {"direction": "CE", "conviction": 0.35,
                                   "horizon": "intraday", "signal_id": "test2",
                                   "generated_at": "2024-11-15T10:00:00+05:30"})
    print(f"Rule 3 test: {r2['direction']} (expected NEUTRAL)")

    # Test pass-through
    ms3 = {"adx_14": 30.0, "vix_india": 15.0, "nifty_spot": 22000}
    r3 = orch.process_pod_output(ms3, "2024-11-15T10:30:00+05:30",
                                  {"direction": "PE", "conviction": 0.65,
                                   "horizon": "intraday", "signal_id": "test3",
                                   "generated_at": "2024-11-15T10:30:00+05:30"})
    print(f"Pass test: {r3['direction']} (expected PE)")

    print(f"\nDecision log ({len(orch.decision_log)} entries):")
    for log in orch.decision_log:
        print(f"  {log['reason_code']}: {log['details']}")
