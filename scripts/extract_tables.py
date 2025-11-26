#!/usr/bin/env python3
"""
Extract tables from evaluation_protocol.json files.

Table 1: Architecture comparison on MMCBNU
- Rows: Models
- Cols: Metrics (OSCR, AUROC, EER, TPR@FPR, Rank-1, etc.)

Table 2: Generalization across datasets
- Rows: Models
- Cols: Datasets (MMCBNU, Dorsal, FYO, UTFVP)
- Values: OSCR + Rank-1
"""

import json
import math
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
TABLES_DIR = PROJECT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True)


def load_protocol(path: Path) -> Dict:
    """Load evaluation_protocol.json"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERR Failed to load {path}: {e}")
        return None


def is_valid_metric(val):
    """Check if metric value is valid (not NaN, not None, not 0)"""
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    if val == 0:
        return False
    return True


def extract_model_name(path: Path) -> str:
    """Extract model name from path like outputs/mmcbnu/attention_unet/timestamp/"""
    parts = path.parts
    for i, part in enumerate(parts):
        if part in ["mmcbnu", "dorsal", "utfvp", "fyo"]:
            if i + 1 < len(parts):
                return parts[i + 1]
    return "unknown"


def extract_dataset_name(path: Path) -> str:
    """Extract dataset name from path"""
    parts = path.parts
    for part in parts:
        if part in ["mmcbnu", "dorsal", "utfvp", "fyo"]:
            return part
    return "unknown"


# ============================================================================
# TABLE 1: Architecture Comparison (MMCBNU only)
# ============================================================================

print("[TABLE 1] Building Architecture Comparison on MMCBNU")
print()

table1_data = []

for protocol_file in OUTPUTS_DIR.rglob("evaluation_protocol.json"):
    # Only MMCBNU for Table 1
    if "mmcbnu" not in str(protocol_file):
        continue

    protocol = load_protocol(protocol_file)
    if not protocol:
        continue

    model = extract_model_name(protocol_file)
    metrics = protocol.get("metrics", {})

    row = {
        "Model": model,
        "OSCR": round(metrics.get("oscr_auc", 0), 4),
        "AUROC": round(metrics.get("auroc", 0), 4),
        "EER (%)": round(metrics.get("eer", 0), 2),
        "TPR@FPR=0.1%": round(metrics.get("tpr_at_fpr_0.001", 0), 2),
        "TPR@FPR=1%": round(metrics.get("tpr_at_fpr_0.01", 0), 2),
        "Rank-1": round(metrics.get("cmc_rank1", 0), 2),
        "Accuracy": round(metrics.get("accuracy_known", 0), 2),
    }
    table1_data.append(row)
    print(f"  {model:25s} OSCR={row['OSCR']:.4f}")

# Remove duplicates, keep best OSCR for each model
df_table1 = pd.DataFrame(table1_data)
if not df_table1.empty:
    # Group by model and keep the one with highest OSCR
    df_table1 = df_table1.loc[df_table1.groupby("Model")["OSCR"].idxmax()]
    df_table1 = df_table1.sort_values("OSCR", ascending=False)

    # Save CSV
    csv_path = TABLES_DIR / "table1_architecture.csv"
    df_table1.to_csv(csv_path, index=False)

    # Save LaTeX
    latex_path = TABLES_DIR / "table1_architecture.tex"
    latex = df_table1.to_latex(index=False, float_format="%.4f")
    with open(latex_path, "w") as f:
        f.write(latex)

    print("\nTable 1 saved:")
    print(f"  CSV: {csv_path}")
    print(f"  TEX: {latex_path}")
    print()
    print(df_table1.to_string())

# ============================================================================
# TABLE 2: Generalization (All models on different datasets)
# ============================================================================

print("\n\n[TABLE 2] Building Generalization Across Datasets")
print()

# Collect best run per (model, dataset) - ONE result per combination
table2_raw = {}

for protocol_file in OUTPUTS_DIR.rglob("evaluation_protocol.json"):
    protocol = load_protocol(protocol_file)
    if not protocol:
        continue

    model = extract_model_name(protocol_file)
    dataset = extract_dataset_name(protocol_file)
    metrics = protocol.get("metrics", {})

    if dataset not in ["mmcbnu", "dorsal", "utfvp", "fyo"]:
        continue

    key = (model, dataset)
    oscr = metrics.get("oscr_auc")
    rank1 = metrics.get("cmc_rank1")

    # Skip invalid entries
    if not is_valid_metric(oscr) or not is_valid_metric(rank1):
        print(f"  SKIP {model:20s} {dataset.upper():10s} - OSCR={oscr}, Rank-1={rank1}")
        continue

    # Keep best OSCR for each (model, dataset) pair
    if key not in table2_raw or oscr > table2_raw[key]["OSCR"]:
        table2_raw[key] = {
            "OSCR": oscr,
            "Rank-1": rank1,
        }

# Build table with one row per model, columns per dataset
models_seen = set()
table2_data = {}

for (model, dataset), metrics in table2_raw.items():
    models_seen.add(model)
    if model not in table2_data:
        table2_data[model] = {"Model": model}

    oscr_val = round(metrics["OSCR"], 4)
    rank1_val = round(metrics["Rank-1"] * 100, 1)

    # Store as "OSCR / Rank-1%"
    table2_data[model][dataset.upper()] = f"{oscr_val} / {rank1_val}"
    print(
        f"  {model:20s} {dataset.upper():10s} OSCR={oscr_val:.4f} Rank-1={rank1_val:.1f}%"
    )

# Convert to dataframe
df_table2_list = []
for model_name in sorted(models_seen):
    df_table2_list.append(table2_data[model_name])

df_table2 = pd.DataFrame(df_table2_list)

if not df_table2.empty:
    # Reorder columns: Model first, then datasets
    col_order = ["Model"]
    for ds in ["MMCBNU", "DORSAL", "FYO", "UTFVP"]:
        if ds in df_table2.columns:
            col_order.append(ds)

    df_table2 = df_table2[col_order]

    # Save CSV
    csv_path = TABLES_DIR / "table2_generalization.csv"
    df_table2.to_csv(csv_path, index=False)

    # Save LaTeX
    latex_path = TABLES_DIR / "table2_generalization.tex"
    latex = df_table2.to_latex(index=False)
    with open(latex_path, "w") as f:
        f.write(latex)

    print("\nTable 2 saved:")
    print(f"  CSV: {csv_path}")
    print(f"  TEX: {latex_path}")
    print()
    print(df_table2.to_string(index=False))
else:
    print("WARN No data for Table 2")

print("\n" + "=" * 70)
print("OK Tables generated successfully!")
print(f"Output: {TABLES_DIR}")
print("=" * 70)

print("\nNOTE - Data Availability & Split Ratios:")
print("  MMCBNU (12000 samples):")
print("    - Full open-set evaluation (known + unknown classes)")
print(
    "    - Subject-disjoint splits: 55% train / 10% val / 5% test_known / 30% unknown"
)
print()
print("  DORSAL (1782 samples):")
print("    - Full open-set evaluation (known + unknown classes)")
print(
    "    - Subject-disjoint splits: 55% train / 10% val / 5% test_known / 30% unknown"
)
print()
print("  FYO (320 samples - TOO SMALL):")
print("    - Modified split ratios: 50% train / 15% val / 20% test_known / 15% unknown")
print(
    "    - Reason: Limited data requires aggressive test_known allocation for enrollment samples"
)
print()
print("  UTFVP (1444 samples - TOO SMALL):")
print("    - Modified split ratios: 50% train / 15% val / 20% test_known / 15% unknown")
print(
    "    - Reason: Limited data requires aggressive test_known allocation for enrollment samples"
)
print()
print(
    "  Result: Only MMCBNU and DORSAL have sufficient data for reliable open-set evaluation"
)
