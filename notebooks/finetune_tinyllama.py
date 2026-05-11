"""
TinyLlama LoRA Fine-tuning Script for Kaggle T4 GPU.
Run this on Kaggle with GPU T4 accelerator enabled.

Usage on Kaggle:
1. Upload slm_intern_data/ as a Kaggle dataset
2. Create a new notebook with T4 GPU
3. Copy this script and run
"""
import json, os, logging, time
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
# On Kaggle, update this to your dataset path
DATA_DIR = Path("/kaggle/input/slm-intern-data")  # Kaggle dataset path
OUTPUT_DIR = Path("/kaggle/working/lora_adapter")
MLFLOW_DIR = Path("/kaggle/working/mlruns")

# If running locally for testing, uncomment:
# DATA_DIR = Path(".")
# OUTPUT_DIR = Path("./lora_adapter")
# MLFLOW_DIR = Path("./mlruns")

CLEAN_DATA_PATH = DATA_DIR / "data" / "train_clean.jsonl"
# If clean data not pre-built, fall back to raw + filter
RAW_DATA_PATH = DATA_DIR / "finetune_instructions.jsonl"

# Model config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_R = 8          # Rank: balanced for 255 examples
LORA_ALPHA = 16     # 2x rank (standard scaling)
LORA_DROPOUT = 0.1  # Regularization for small dataset
TARGET_MODULES = ["q_proj", "v_proj"]

# Training config
NUM_EPOCHS = 4
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.1

SYSTEM_PROMPT = (
    "You are a NIFTY 50 options trading signal generator. "
    "Analyze the market state and return ONLY valid JSON. "
    'Schema: {"direction": "CE"|"PE"|"NEUTRAL", "conviction": float 0.0-1.0, '
    '"horizon": "intraday"|"next_session", "signal_id": string, "generated_at": string}'
)


def install_dependencies():
    """Install required packages on Kaggle."""
    os.system("pip install -q peft bitsandbytes accelerate transformers datasets mlflow trl")


def load_and_filter_data():
    """Load clean training data (or filter raw data if clean not available)."""
    if CLEAN_DATA_PATH.exists():
        logger.info(f"Loading pre-cleaned data from {CLEAN_DATA_PATH}")
        with open(CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        logger.info(f"Clean data not found, filtering raw data from {RAW_DATA_PATH}")
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f]

        data = []
        for i, ex in enumerate(raw):
            try:
                out = json.loads(ex["output"])
                conv = out.get("conviction")
                if isinstance(conv, str):
                    logger.warning(f"Skipping line {i}: string conviction '{conv}'")
                    continue
                if not isinstance(conv, (int, float)) or not (0.0 <= conv <= 1.0):
                    logger.warning(f"Skipping line {i}: invalid conviction {conv}")
                    continue
                data.append(ex)
            except Exception as e:
                logger.warning(f"Skipping line {i}: {e}")

    logger.info(f"Loaded {len(data)} training examples")
    return data


def format_training_example(example):
    """Format a training example into the chat template."""
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n</s>\n"
        f"<|user|>\nMarket State: {input_text}\n</s>\n"
        f"<|assistant|>\n{output_text}\n</s>"
    )
    return {"text": text}


def main():
    start_time = time.time()
    install_dependencies()

    import torch
    import mlflow
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    # Initialize MLflow
    os.makedirs(str(MLFLOW_DIR), exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
    mlflow.set_experiment("nifty-signal-pod")

    # Load data
    data = load_and_filter_data()
    formatted = [format_training_example(ex) for ex in data]
    dataset = Dataset.from_list(formatted)
    logger.info(f"Dataset size: {len(dataset)}")

    # Quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    logger.info(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Training arguments
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",  # We use MLflow manually
        seed=42,
    )

    # MLflow tracking
    with mlflow.start_run(run_name="tinyllama_lora_finetune") as run:
        # Log params
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": str(TARGET_MODULES),
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "train_examples": len(dataset),
            "trainable_params": trainable,
            "total_params": total,
        })

        # Train
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        logger.info("Starting training...")
        train_result = trainer.train()

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime_seconds": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        })

        # Save LoRA adapter
        adapter_path = OUTPUT_DIR / "final"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        logger.info(f"LoRA adapter saved to {adapter_path}")

        # Log adapter as artifact
        mlflow.log_artifacts(str(adapter_path), "lora_adapter")

        elapsed = time.time() - start_time
        mlflow.log_metric("total_time_seconds", elapsed)
        logger.info(f"Training complete in {elapsed/60:.1f} minutes")
        logger.info(f"MLflow run ID: {run.info.run_id}")

    # Quick inference test
    logger.info("Running quick inference test...")
    test_state = '{"nifty_spot": 22859.61, "atm_iv": 13.41, "iv_skew_25d": 3.88, "pcr": 1.13, "adx_14": 29.35, "realized_vol_5d": 13.61, "vix_india": 14.08, "dte_nearest": 2, "moneyness_band": "ATM"}'
    test_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n</s>\n<|user|>\nMarket State: {test_state}\n</s>\n<|assistant|>\n"

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
    gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.info(f"Test output: {gen}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print(f"Adapter saved to: {adapter_path}")
    print(f"Download the 'final' folder for local inference")
    print("=" * 50)


if __name__ == "__main__":
    main()
