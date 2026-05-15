"""Data audit script for finetune_instructions.jsonl and market_states.parquet"""
import json
import collections
import pandas as pd

# ============================================================
# 1. FINETUNE INSTRUCTIONS AUDIT
# ============================================================
print("=" * 60)
print("FINETUNE INSTRUCTIONS AUDIT")
print("=" * 60)

lines = open("finetune_instructions.jsonl", "r", encoding="utf-8").readlines()
data = [json.loads(l) for l in lines]
print(f"Total examples: {len(data)}")

# Check keys in each example
print("\n--- KEY STRUCTURE ---")
key_sets = collections.Counter(tuple(sorted(d.keys())) for d in data)
for ks, cnt in key_sets.items():
    print(f"  {ks}: {cnt}")

# Check instruction variations
print("\n--- INSTRUCTION VARIATIONS ---")
instr_set = set(d["instruction"] for d in data)
print(f"  Unique instructions: {len(instr_set)}")
for i, ins in enumerate(instr_set):
    print(f"  [{i}] {ins[:200]}...")

# Parse outputs and check schema
print("\n--- OUTPUT SCHEMA ANALYSIS ---")
directions = collections.Counter()
horizons = collections.Counter()
bad_json = 0
bad_schema_list = []
conviction_issues = []
adx_below_20_nonneutral = []
conv_values = []
high_conviction_neutral = []
low_adx_directional = []
duplicate_signals = []
seen_ids = set()

for i, d in enumerate(data):
    try:
        out = json.loads(d["output"])
    except Exception as e:
        bad_json += 1
        print(f"  BAD JSON at line {i}: {str(e)} | raw: {d['output'][:100]}")
        continue

    try:
        inp = json.loads(d["input"])
    except Exception as e:
        print(f"  BAD INPUT JSON at line {i}: {str(e)}")
        continue

    # Check direction
    dir_val = out.get("direction", "MISSING")
    directions[dir_val] += 1
    horizons[out.get("horizon", "MISSING")] += 1

    # Check conviction range and type
    conv = out.get("conviction", None)
    if conv is not None:
        if isinstance(conv, str):
            conviction_issues.append((i, f"STRING: '{conv}'"))
            try:
                conv = float(conv)
            except:
                conv = None
        if conv is not None:
            conv_values.append(conv)
            if not (0.0 <= conv <= 1.0):
                conviction_issues.append((i, f"OUT_OF_RANGE: {conv}"))

    # Check ADX < 20 with non-NEUTRAL direction (orchestrator would suppress these)
    adx = inp.get("adx_14", 99)
    if adx < 20 and dir_val != "NEUTRAL":
        adx_below_20_nonneutral.append((i, adx, dir_val, conv))

    # Check high conviction on NEUTRAL
    if dir_val == "NEUTRAL" and conv is not None and conv > 0.7:
        high_conviction_neutral.append((i, conv))

    # Check required keys
    required = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
    missing = required - set(out.keys())
    if missing:
        bad_schema_list.append((i, missing))

    # Check for valid direction values
    if dir_val not in ("CE", "PE", "NEUTRAL"):
        print(f"  INVALID DIRECTION at line {i}: {dir_val}")

    # Check for valid horizon values
    hor = out.get("horizon", "")
    if hor not in ("intraday", "next_session"):
        print(f"  INVALID HORIZON at line {i}: {hor}")

    # Check duplicate signal IDs
    sid = out.get("signal_id", "")
    if sid in seen_ids:
        duplicate_signals.append((i, sid))
    seen_ids.add(sid)

print(f"\n  Bad JSON count: {bad_json}")
print(f"  Bad schema count: {len(bad_schema_list)}")
if bad_schema_list:
    for idx, missing in bad_schema_list[:10]:
        print(f"    Line {idx}: missing keys {missing}")

print(f"\n  Directions: {dict(directions)}")
print(f"  Horizons: {dict(horizons)}")

if conv_values:
    print(f"\n  Conviction range: [{min(conv_values):.3f}, {max(conv_values):.3f}]")
    print(f"  Conviction mean: {sum(conv_values)/len(conv_values):.3f}")

print(f"\n  Conviction out-of-range issues: {len(conviction_issues)}")
for idx, cv in conviction_issues[:10]:
    print(f"    Line {idx}: conviction={cv}")

print(f"\n  ADX<20 but non-NEUTRAL (training conflict): {len(adx_below_20_nonneutral)}")
for idx, adx, d, c in adx_below_20_nonneutral[:10]:
    print(f"    Line {idx}: ADX={adx}, dir={d}, conv={c}")

print(f"\n  High conviction NEUTRAL (>0.7): {len(high_conviction_neutral)}")
for idx, cv in high_conviction_neutral[:10]:
    print(f"    Line {idx}: conviction={cv}")

print(f"\n  Duplicate signal IDs: {len(duplicate_signals)}")
for idx, sid in duplicate_signals[:5]:
    print(f"    Line {idx}: {sid}")

# Check if training samples align with market_states timestamps
print("\n--- TIMESTAMP ALIGNMENT WITH MARKET_STATES ---")
df = pd.read_parquet("market_states.parquet")
ms_timestamps = set(df["timestamp"].values)
train_timestamps = []
for d in data:
    try:
        out = json.loads(d["output"])
        train_timestamps.append(out.get("generated_at", ""))
    except:
        pass

aligned = sum(1 for t in train_timestamps if t in ms_timestamps)
print(f"  Training examples with matching market_states timestamp: {aligned}/{len(train_timestamps)}")

# Check if training data input matches market_states values
print("\n--- INPUT VALUE CROSS-CHECK ---")
mismatches = 0
for i, d in enumerate(data):
    try:
        inp = json.loads(d["input"])
        out = json.loads(d["output"])
        ts = out.get("generated_at", "")
        row = df[df["timestamp"] == ts]
        if len(row) > 0:
            row = row.iloc[0]
            for col in ["nifty_spot", "atm_iv", "adx_14", "vix_india", "pcr"]:
                if col in inp and col in row.index:
                    if abs(inp[col] - row[col]) > 0.01:
                        mismatches += 1
                        if mismatches <= 5:
                            print(f"    Line {i}: {col} mismatch - input={inp[col]}, parquet={row[col]}")
    except:
        pass
print(f"  Total value mismatches: {mismatches}")

# Check label alignment
print("\n--- LABEL ALIGNMENT ---")
label_mismatches = 0
for i, d in enumerate(data):
    try:
        inp = json.loads(d["input"])
        out = json.loads(d["output"])
        ts = out.get("generated_at", "")
        row = df[df["timestamp"] == ts]
        if len(row) > 0:
            parquet_label = row.iloc[0]["label"]
            json_dir = out["direction"]
            if parquet_label != json_dir:
                label_mismatches += 1
                if label_mismatches <= 10:
                    print(f"    Line {i}: instruction dir={json_dir}, parquet label={parquet_label}")
    except:
        pass
print(f"  Total label mismatches: {label_mismatches}")

# ============================================================
# 2. MARKET STATES ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("MARKET STATES ANALYSIS")
print("=" * 60)

df["ts"] = pd.to_datetime(df["timestamp"])
df["date"] = df["ts"].dt.date
dates = sorted(df["date"].unique())
print(f"Total unique dates: {len(dates)}")
print(f"Date range: {dates[0]} to {dates[-1]}")

print("\nRows per date (first 5):")
for d in dates[:5]:
    n = len(df[df["date"] == d])
    print(f"  {d}: {n} rows")
print("  ...")
print("Rows per date (last 5):")
for d in dates[-5:]:
    n = len(df[df["date"] == d])
    print(f"  {d}: {n} rows")

# Training vs eval split
train_dates = dates[:30]
eval_dates = dates[30:]
print(f"\nTraining dates (days 1-30): {train_dates[0]} to {train_dates[-1]}")
print(f"Eval dates (days 31-60): {eval_dates[0]} to {eval_dates[-1]}")

train_df = df[df["date"].isin(train_dates)]
eval_df = df[df["date"].isin(eval_dates)]
print(f"Training rows: {len(train_df)}")
print(f"Eval rows: {len(eval_df)}")

print("\nLabel distribution:")
print(df["label"].value_counts().to_string())

print("\nLabel distribution (train):")
print(train_df["label"].value_counts().to_string())

print("\nLabel distribution (eval):")
print(eval_df["label"].value_counts().to_string())

print("\nVIX stats (full):")
print(df["vix_india"].describe().to_string())

print("\nADX stats (full):")
print(df["adx_14"].describe().to_string())

# VIX regime analysis
print("\nVIX regime breakdown (eval):")
high_vix = eval_df[eval_df["vix_india"] > eval_df["vix_india"].quantile(0.75)]
low_vix = eval_df[eval_df["vix_india"] <= eval_df["vix_india"].quantile(0.25)]
print(f"  High VIX (>75th pctile): {len(high_vix)} rows, VIX range: {high_vix['vix_india'].min():.1f}-{high_vix['vix_india'].max():.1f}")
print(f"  Low VIX (<=25th pctile): {len(low_vix)} rows, VIX range: {low_vix['vix_india'].min():.1f}-{low_vix['vix_india'].max():.1f}")

# ADX < 20 in eval set
adx_low_eval = eval_df[eval_df["adx_14"] < 20]
print(f"\nADX < 20 in eval set: {len(adx_low_eval)} rows ({100*len(adx_low_eval)/len(eval_df):.1f}%)")

# Moneyness bands
print("\nMoneyness bands:")
print(df["moneyness_band"].value_counts().to_string())

# Check for any NaN/null values
print("\nNull values per column:")
print(df.isnull().sum().to_string())

print("\n\nDONE.")
