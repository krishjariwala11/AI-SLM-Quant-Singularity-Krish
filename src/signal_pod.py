"""
Signal Pod — Model inference with structured JSON output and fallback.
Loads the LoRA fine-tuned TinyLlama model and generates trading signals.
"""
import json, logging, uuid, re
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Default fallback signal — returned on any failure
FALLBACK_SIGNAL = {
    "direction": "NEUTRAL",
    "conviction": 0.0,
    "horizon": "intraday",
    "signal_id": "",
    "generated_at": "",
}

SYSTEM_PROMPT = (
    "You are a NIFTY 50 options trading signal generator. "
    "Analyze the market state and return ONLY valid JSON. "
    'Schema: {"direction": "CE"|"PE"|"NEUTRAL", "conviction": float 0.0-1.0, '
    '"horizon": "intraday"|"next_session", "signal_id": string, "generated_at": string}'
)

REQUIRED_KEYS = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS = {"intraday", "next_session"}


def format_prompt(market_state: Dict, rag_context: str = None) -> str:
    """Build the instruction prompt for the model."""
    ms_json = json.dumps(market_state, ensure_ascii=False)

    if rag_context:
        user_msg = (
            f"Historical Context:\n{rag_context}\n\n"
            f"Current Market State: {ms_json}\n\n"
            "Generate a trading signal based on the current state and historical context."
        )
    else:
        user_msg = f"Market State: {ms_json}"

    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n</s>\n"
        f"<|user|>\n{user_msg}\n</s>\n"
        f"<|assistant|>\n"
    )
    return prompt


def generate_signal_id(market_state: Dict, timestamp: str) -> str:
    """Generate a deterministic UUID5 signal ID from market state."""
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    key = json.dumps(market_state, sort_keys=True) + timestamp
    return str(uuid.uuid5(namespace, key))


def parse_model_output(raw_output: str, market_state: Dict, timestamp: str) -> Dict:
    """
    Parse raw model output into a valid signal dict.
    Returns fallback signal if parsing fails.
    """
    # Try to extract JSON from the output
    try:
        # First, try direct parse
        signal = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        # Try to find JSON in the output using regex
        json_match = re.search(r'\{[^{}]*\}', raw_output)
        if json_match:
            try:
                signal = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.error(f"JSON parse failed. Raw output: {raw_output[:500]}")
                return _make_fallback(market_state, timestamp, raw_output)
        else:
            logger.error(f"No JSON found. Raw output: {raw_output[:500]}")
            return _make_fallback(market_state, timestamp, raw_output)

    # Validate schema
    if not REQUIRED_KEYS.issubset(signal.keys()):
        missing = REQUIRED_KEYS - set(signal.keys())
        logger.error(f"Missing keys: {missing}")
        return _make_fallback(market_state, timestamp, raw_output)

    if signal["direction"] not in VALID_DIRECTIONS:
        logger.error(f"Invalid direction: {signal['direction']}")
        return _make_fallback(market_state, timestamp, raw_output)

    if signal["horizon"] not in VALID_HORIZONS:
        logger.error(f"Invalid horizon: {signal['horizon']}")
        return _make_fallback(market_state, timestamp, raw_output)

    # Validate conviction is a float in [0, 1]
    conv = signal["conviction"]
    if isinstance(conv, str):
        try:
            conv = float(conv)
        except ValueError:
            logger.error(f"Conviction not numeric: {conv}")
            return _make_fallback(market_state, timestamp, raw_output)
    if not isinstance(conv, (int, float)) or not (0.0 <= conv <= 1.0):
        logger.error(f"Conviction out of range: {conv}")
        return _make_fallback(market_state, timestamp, raw_output)
    signal["conviction"] = float(conv)

    # Ensure signal_id and generated_at
    if not signal.get("signal_id"):
        signal["signal_id"] = generate_signal_id(market_state, timestamp)
    if not signal.get("generated_at"):
        signal["generated_at"] = timestamp

    return signal


def _make_fallback(market_state: Dict, timestamp: str, raw_output: str = "") -> Dict:
    """Create a NEUTRAL fallback signal and log the failure."""
    fallback = FALLBACK_SIGNAL.copy()
    fallback["signal_id"] = generate_signal_id(market_state, timestamp)
    fallback["generated_at"] = timestamp
    fallback["_fallback"] = True
    fallback["_raw_output"] = raw_output[:500]
    return fallback


class SignalPod:
    """Signal pod wrapping a fine-tuned TinyLlama model."""

    def __init__(self, model_path: str = None, use_rag: bool = False):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.use_rag = use_rag
        self._loaded = False

    def load_model(self):
        """Load the fine-tuned model with 4-bit quantization for CPU inference."""
        if self._loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            logger.info(f"Loading model from {self.model_path}")

            # Load base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                device_map="cpu",
                torch_dtype="auto",
            )

            # Load LoRA adapter if path provided
            if self.model_path and Path(self.model_path).exists():
                self.model = PeftModel.from_pretrained(self.model, self.model_path)
                logger.info("LoRA adapter loaded successfully")

            self.model.eval()
            self._loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise

    def generate(self, market_state: Dict, timestamp: str,
                 rag_context: str = None) -> Dict:
        """Generate a trading signal for the given market state."""
        if not self._loaded:
            self.load_model()

        prompt = format_prompt(market_state, rag_context)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=1024)
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the generated part
            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            logger.debug(f"Raw output: {raw_output}")

            return parse_model_output(raw_output, market_state, timestamp)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return _make_fallback(market_state, timestamp, str(e))


if __name__ == "__main__":
    # Test prompt formatting
    test_state = {
        "nifty_spot": 22859.61, "atm_iv": 13.41, "iv_skew_25d": 3.88,
        "pcr": 1.13, "adx_14": 29.35, "realized_vol_5d": 13.61,
        "vix_india": 14.08, "dte_nearest": 2, "moneyness_band": "ATM"
    }
    print("=== Prompt (no RAG) ===")
    print(format_prompt(test_state))
    print("\n=== Prompt (with RAG) ===")
    rag = ("Episode 1 | trending_low_vol | ADX 31.6, VIX 12.5. Result: CE\n"
           "Episode 2 | mean_reverting | ADX 18.2, VIX 22.1. Result: NEUTRAL")
    print(format_prompt(test_state, rag))
