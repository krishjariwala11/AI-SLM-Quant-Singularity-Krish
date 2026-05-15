# NIFTY Options Signal Pod — AI-SLM Screening

## Overview

An AI-supervised signal pod for NIFTY 50 options: a LoRA fine-tuned TinyLlama-1.1B model that generates structured JSON trading signals, wrapped by a deterministic orchestrator with suppression rules, evaluated via walk-forward methodology with RAG ablation study.

## Architecture

```
Market State → Orchestrator → [ADX Check] → Signal Pod (± RAG) → [Parse Check] → [Conviction Check] → Final Output
```

## Project Structure

```
slm_intern_data/
├── data/                      # Processed data
│   └── train_clean.jsonl      # Cleaned training set (255 examples)
├── notebooks/
│   └── finetune_tinyllama.py  # Kaggle training script
├── src/
│   ├── data_prep.py           # Data audit & cleaning
│   ├── eval_suite.py          # Pre-committed eval metrics & thresholds
│   ├── signal_pod.py          # Model inference with JSON fallback
│   ├── orchestrator.py        # 3-rule deterministic wrapper
│   ├── rag_experiment.py      # RAG prompt template & ablation
│   ├── run_eval.py            # Walk-forward evaluation pipeline
│   └── mlflow_config.py       # MLflow tracking setup
├── tests/
│   └── test_orchestrator.py   # Unit tests for orchestrator
├── report/
│   └── report.md              # Written report (convert to PDF)
├── mlruns/                    # MLflow artifacts
├── finetune_instructions.jsonl # Raw training data (DO NOT USE DIRECTLY)
├── market_states.parquet       # 60-day market data
├── rag_corpus.jsonl            # RAG retrieval corpus
├── retrieve.py                 # Provided retrieval function (DO NOT MODIFY)
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Prepare clean training data
python src/data_prep.py

# 2. Run baseline evaluation (Takes ~1 hour on CPU)
python src/run_eval.py --model-path lora_adapter/final

# 3. Run evaluation with RAG context (Takes ~1 hour on CPU)
python src/run_eval.py --model-path lora_adapter/final --rag

# 4. Analyze RAG ablation changes in conviction
python src/rag_experiment.py

# 5. Populate report.md with evaluation results automatically
python src/generate_report.py
```

## Kaggle Notebook

**URL**: https://www.kaggle.com/code/krishjariwala11/model-train (for training the model with CPU)
https://www.kaggle.com/code/krishjariwala11/mainpy (for exporting lora_adapter and mlruns)

## Key Design Decisions

- **Model**: TinyLlama-1.1B (not Phi-2) — fits comfortably on T4 with room for experimentation
- **Data Cleaning**: Dropped 45/300 corrupted examples with string conviction values
- **LoRA Config**: r=8, alpha=16, dropout=0.1, targeting q_proj and v_proj
- **Evaluation**: Walk-forward only, 5-day rolling blocks, no k-fold
- **Safety**: 3-rule deterministic orchestrator with JSON fallback

## Final Deliverables Checklist

1. **Written Report**: `report/report.md` (and exported PDF).
2. **Kaggle Notebook**: URL in this README.
3. **LoRA Weights**: `lora_adapter/final/`.
4. **Clean Training Data**: `data/train_clean.jsonl` (generated via `src/data_prep.py`).
5. **Evaluation Results**: `data/eval_results_no_rag.json` and `data/eval_results_rag.json`.
6. **MLflow Artifacts**: Local `mlruns/` directory included in repo.
7. **Codebase**: Orchestrator, Signal Pod, and Eval Suite in `src/`.
