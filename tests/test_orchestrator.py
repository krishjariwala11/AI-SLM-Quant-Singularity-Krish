"""
Unit tests for the Orchestrator.
Tests all 3 suppression rules, edge cases, and logging.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import Orchestrator, REASON_ADX_BELOW, REASON_PARSE_FAIL, REASON_LOW_CONVICTION, REASON_PASS


def make_signal(direction="CE", conviction=0.6, horizon="intraday",
                signal_id="test-id", generated_at="2024-11-15T09:30:00+05:30"):
    return {"direction": direction, "conviction": conviction, "horizon": horizon,
            "signal_id": signal_id, "generated_at": generated_at}


def make_market_state(adx=25.0, vix=15.0, spot=22000):
    return {"adx_14": adx, "vix_india": vix, "nifty_spot": spot,
            "atm_iv": 12.0, "iv_skew_25d": 1.5, "pcr": 1.0,
            "realized_vol_5d": 12.0, "dte_nearest": 2, "moneyness_band": "ATM"}


def test_rule1_adx_below_threshold():
    """ADX < 20 → NEUTRAL, conviction 0.0, model NOT called."""
    orch = Orchestrator()
    ms = make_market_state(adx=14.0)
    sig = make_signal(direction="CE", conviction=0.8)
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", sig)

    assert result["direction"] == "NEUTRAL", f"Expected NEUTRAL, got {result['direction']}"
    assert result["conviction"] == 0.0, f"Expected 0.0, got {result['conviction']}"
    assert orch.decision_log[-1]["reason_code"] == REASON_ADX_BELOW
    assert orch.decision_log[-1]["pod_output"] is None  # Model was not called
    print("  ✓ Rule 1: ADX < 20 suppression works")


def test_rule1_adx_exactly_20():
    """ADX == 20 should NOT trigger suppression (threshold is strictly <20)."""
    orch = Orchestrator()
    ms = make_market_state(adx=20.0)
    sig = make_signal(direction="CE", conviction=0.6)
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", sig)

    assert result["direction"] == "CE", f"Expected CE, got {result['direction']}"
    assert orch.decision_log[-1]["reason_code"] == REASON_PASS
    print("  ✓ Rule 1: ADX == 20 passes through correctly")


def test_rule2_parse_failure():
    """Parse failure → NEUTRAL, conviction 0.0, raw output logged."""
    orch = Orchestrator()
    ms = make_market_state(adx=25.0)
    bad_signal = {"_fallback": True, "_raw_output": "garbled text here"}
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", bad_signal)

    assert result["direction"] == "NEUTRAL"
    assert result["conviction"] == 0.0
    assert orch.decision_log[-1]["reason_code"] == REASON_PARSE_FAIL
    print("  ✓ Rule 2: Parse failure handled correctly")


def test_rule2_schema_invalid():
    """Invalid schema (missing keys) → treated as parse failure."""
    orch = Orchestrator()
    ms = make_market_state(adx=25.0)
    bad_signal = {"direction": "CE"}  # Missing required keys
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", bad_signal)

    assert result["direction"] == "NEUTRAL"
    assert orch.decision_log[-1]["reason_code"] == REASON_PARSE_FAIL
    print("  ✓ Rule 2: Schema validation catches missing keys")


def test_rule3_low_conviction():
    """Conviction < 0.40 → direction downgraded to NEUTRAL, conviction preserved."""
    orch = Orchestrator()
    ms = make_market_state(adx=25.0)
    sig = make_signal(direction="PE", conviction=0.35)
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", sig)

    assert result["direction"] == "NEUTRAL", f"Expected NEUTRAL, got {result['direction']}"
    assert result["conviction"] == 0.35  # Conviction preserved, only direction changed
    assert orch.decision_log[-1]["reason_code"] == REASON_LOW_CONVICTION
    details = orch.decision_log[-1]["details"]
    assert details["original_direction"] == "PE"
    print("  ✓ Rule 3: Low conviction downgrade works")


def test_rule3_conviction_exactly_040():
    """Conviction == 0.40 should NOT trigger downgrade (threshold is <0.40)."""
    orch = Orchestrator()
    ms = make_market_state(adx=25.0)
    sig = make_signal(direction="CE", conviction=0.40)
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", sig)

    assert result["direction"] == "CE"
    assert orch.decision_log[-1]["reason_code"] == REASON_PASS
    print("  ✓ Rule 3: Conviction == 0.40 passes through")


def test_pass_through():
    """Valid signal with ADX >= 20 and conviction >= 0.40 passes through."""
    orch = Orchestrator()
    ms = make_market_state(adx=30.0)
    sig = make_signal(direction="PE", conviction=0.65)
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", sig)

    assert result["direction"] == "PE"
    assert result["conviction"] == 0.65
    assert orch.decision_log[-1]["reason_code"] == REASON_PASS
    print("  ✓ Pass-through: Valid signal preserved")


def test_rule_priority():
    """Rule 1 (ADX) takes priority over Rule 3 (conviction)."""
    orch = Orchestrator()
    ms = make_market_state(adx=14.0)
    sig = make_signal(direction="CE", conviction=0.35)  # Would also trigger Rule 3
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", sig)

    assert orch.decision_log[-1]["reason_code"] == REASON_ADX_BELOW  # Rule 1 fires first
    print("  ✓ Rule priority: ADX check fires before conviction check")


def test_decision_log_completeness():
    """Every call produces a log entry with required fields."""
    orch = Orchestrator()
    ms = make_market_state(adx=25.0)
    sig = make_signal()
    orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", sig)

    log = orch.decision_log[-1]
    required = {"signal_id", "timestamp", "reason_code", "details",
                "pod_output", "orchestrator_output"}
    assert required.issubset(log.keys()), f"Missing log fields: {required - set(log.keys())}"
    print("  ✓ Decision log: All required fields present")


def test_neutral_signal_schema():
    """Suppressed signals still have valid schema."""
    orch = Orchestrator()
    ms = make_market_state(adx=10.0)
    sig = make_signal()
    result = orch.process_pod_output(ms, "2024-11-15T09:30:00+05:30", sig)

    required = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
    assert required.issubset(result.keys())
    assert result["direction"] in {"CE", "PE", "NEUTRAL"}
    assert isinstance(result["conviction"], (int, float))
    assert 0.0 <= result["conviction"] <= 1.0
    print("  ✓ Neutral signal has valid schema")


def test_scenario_expiry_thursday():
    """Section 5 scenario: VIX 3σ spike + ADX=14 on expiry Thursday."""
    orch = Orchestrator()
    ms = make_market_state(adx=14.0, vix=30.0)  # VIX spike, low ADX
    sig = make_signal(direction="CE", conviction=0.85)
    result = orch.process_pod_output(ms, "2024-11-21T09:30:00+05:30", sig)

    # ADX=14 < 20, so Rule 1 fires — model is never called
    assert result["direction"] == "NEUTRAL"
    assert result["conviction"] == 0.0
    assert orch.decision_log[-1]["reason_code"] == REASON_ADX_BELOW
    print("  ✓ Expiry Thursday scenario: ADX<20 suppresses despite high VIX")


if __name__ == "__main__":
    print("Running Orchestrator Tests...")
    test_rule1_adx_below_threshold()
    test_rule1_adx_exactly_20()
    test_rule2_parse_failure()
    test_rule2_schema_invalid()
    test_rule3_low_conviction()
    test_rule3_conviction_exactly_040()
    test_pass_through()
    test_rule_priority()
    test_decision_log_completeness()
    test_neutral_signal_schema()
    test_scenario_expiry_thursday()
    print("\nAll 11 tests passed! ✓")
