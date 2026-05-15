"""Deep audit of conviction string issues and other subtle problems."""
import json
import collections

lines = open("finetune_instructions.jsonl", "r", encoding="utf-8").readlines()
data = [json.loads(l) for l in lines]

print("=" * 60)
print("DEEP CONVICTION AUDIT")
print("=" * 60)

string_convictions = []
numeric_convictions = []
for i, d in enumerate(data):
    out = json.loads(d["output"])
    conv = out.get("conviction")
    if isinstance(conv, str):
        string_convictions.append((i, conv, out.get("direction"), out.get("horizon")))
    else:
        numeric_convictions.append((i, conv, out.get("direction"), out.get("horizon")))

print(f"\nNumeric convictions: {len(numeric_convictions)}")
print(f"String convictions: {len(string_convictions)}")
print(f"\nAll string conviction values:")
str_vals = collections.Counter(cv for _, cv, _, _ in string_convictions)
for val, cnt in str_vals.most_common():
    print(f"  '{val}': {cnt}")

print(f"\nString conviction line numbers: {[i for i, _, _, _ in string_convictions]}")
print(f"  Contiguous block? Indices {string_convictions[0][0]} to {string_convictions[-1][0]}")

# Show full examples of corrupted rows
print("\n--- SAMPLE CORRUPTED EXAMPLES ---")
for idx, conv, dirn, hor in string_convictions[:5]:
    d = data[idx]
    print(f"\n  Line {idx}:")
    inp = json.loads(d["input"])
    out = json.loads(d["output"])
    print(f"    Input: spot={inp['nifty_spot']}, ADX={inp['adx_14']}, VIX={inp['vix_india']}")
    print(f"    Output: dir={dirn}, conviction='{conv}', horizon={hor}")
    print(f"    signal_id: {out.get('signal_id', 'N/A')}")

# Check if corrupted examples have any pattern in their directions
print("\n--- DIRECTION DISTRIBUTION IN CORRUPTED VS CLEAN ---")
corrupt_dirs = collections.Counter(d for _, _, d, _ in string_convictions)
clean_dirs = collections.Counter(d for _, _, d, _ in numeric_convictions)
print(f"  Corrupted: {dict(corrupt_dirs)}")
print(f"  Clean: {dict(clean_dirs)}")

# Check horizon distribution
corrupt_hor = collections.Counter(h for _, _, _, h in string_convictions)
clean_hor = collections.Counter(h for _, _, _, h in numeric_convictions)
print(f"\n  Corrupted horizons: {dict(corrupt_hor)}")
print(f"  Clean horizons: {dict(clean_hor)}")

# Check if any other output fields have issues in clean data
print("\n--- ADDITIONAL CHECKS ON CLEAN DATA ---")
# Conviction distribution
import statistics
clean_convs = [c for _, c, _, _ in numeric_convictions]
print(f"  Clean conviction stats:")
print(f"    Mean: {statistics.mean(clean_convs):.3f}")
print(f"    Median: {statistics.median(clean_convs):.3f}")
print(f"    Stdev: {statistics.stdev(clean_convs):.3f}")
print(f"    Min: {min(clean_convs):.3f}, Max: {max(clean_convs):.3f}")

# Conviction by direction
print(f"\n  Clean conviction by direction:")
for d in ['CE', 'PE', 'NEUTRAL']:
    vals = [c for _, c, dr, _ in numeric_convictions if dr == d]
    if vals:
        print(f"    {d}: mean={statistics.mean(vals):.3f}, min={min(vals):.3f}, max={max(vals):.3f}, n={len(vals)}")

# Check for low conviction with directional signals
low_conv_directional = [(i, c, d) for i, c, d, _ in numeric_convictions if c < 0.40 and d != 'NEUTRAL']
print(f"\n  Low conviction (<0.40) directional signals: {len(low_conv_directional)}")
for idx, c, d in low_conv_directional[:5]:
    print(f"    Line {idx}: conv={c}, dir={d}")

# Check for NEUTRAL with high conviction
neutral_high = [(i, c) for i, c, d, _ in numeric_convictions if d == 'NEUTRAL' and c > 0.5]
print(f"\n  NEUTRAL with conviction > 0.5: {len(neutral_high)}")

# Check timestamp ordering
print("\n--- TIMESTAMP ORDERING ---")
timestamps = []
for d in data:
    out = json.loads(d["output"])
    timestamps.append(out.get("generated_at", ""))

sorted_ts = sorted(timestamps)
is_sorted = timestamps == sorted_ts
print(f"  Are examples in chronological order? {is_sorted}")

# Check how many unique timestamps
unique_ts = len(set(timestamps))
print(f"  Unique timestamps: {unique_ts} (out of {len(timestamps)})")

# Check for future timestamps (eval period leaking into training data)
import pandas as pd
df = pd.read_parquet("market_states.parquet")
df["ts"] = pd.to_datetime(df["timestamp"])
df["date"] = df["ts"].dt.date
dates = sorted(df["date"].unique())
train_cutoff = dates[29]  # Last training date

print(f"\n  Training cutoff date: {train_cutoff}")
future_leaks = 0
for i, d in enumerate(data):
    out = json.loads(d["output"])
    ts = pd.to_datetime(out["generated_at"]).date()
    if ts > train_cutoff:
        future_leaks += 1
        if future_leaks <= 3:
            print(f"    LEAK: Line {i} has timestamp {ts} (after cutoff)")
print(f"  Examples from eval period: {future_leaks}")

# Check moneyness band distribution in train data vs instructions
print("\n--- MONEYNESS BAND COVERAGE ---")
instr_bands = collections.Counter()
for d in data:
    inp = json.loads(d["input"])
    instr_bands[inp.get("moneyness_band", "MISSING")] += 1
print(f"  In instructions: {dict(instr_bands)}")
print(f"  In market_states (train): {dict(df[df['date'] <= train_cutoff]['moneyness_band'].value_counts())}")

print("\n\nDONE.")
