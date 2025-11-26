#!/bin/bash

# ============================================================================
# Paper Experiments Runner - Open-Set Recognition
# Runs all experiments needed for the paper and collects metrics
# ============================================================================

# Don't exit on error - continue with next experiment
set +e
set -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Create output directories
mkdir -p tables
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EPOCHS=100

echo "🚀 Starting Paper Experiments"
echo "📂 Project: $PROJECT_DIR"
echo "⏱️  Epochs per training: $EPOCHS"
echo "📊 Results will be saved to: tables/"
echo "📝 Logs: logs/experiments_${TIMESTAMP}.log"
echo "============================================================================"

# Track results
declare -A RESULTS

# ============================================================================
# A. ARCHITECTURE COMPARISON (Tabela 2 - Backbone Impact)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 SECTION A: Architecture Comparison on MMCBNU"
echo "════════════════════════════════════════════════════════════════════════════"

ARCH_CONFIGS=(
    "basic|Basic CNN"
    "unet|U-Net"
    "attention_unet|Attention U-Net"
    "resnet50_cbam_center|ResNet50+CBAM+Center"
)

for config in "${ARCH_CONFIGS[@]}"; do
    IFS='|' read -r model_name model_label <<< "$config"

    echo ""
    echo "🔄 Training: $model_label (model=$model_name, data=mmcbnu)"

    LOG_FILE="logs/arch_${model_name}_${TIMESTAMP}.log"

    uv run src/train.py \
        data=mmcbnu \
        model="$model_name" \
        trainer.epochs=$EPOCHS \
        loader.num_workers=0 \
        < /dev/null 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Completed: $model_label"
        RESULTS["arch_${model_name}"]="success"
    else
        echo "❌ Failed: $model_label (exit code: $EXIT_CODE)"
        RESULTS["arch_${model_name}"]="failed"
    fi
done

# ============================================================================
# B. GENERALIZATION (Tabela 3 - Cross-Dataset Performance)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 SECTION B: Generalization - Best Model on Different Datasets"
echo "════════════════════════════════════════════════════════════════════════════"

DATASETS=("mmcbnu" "dorsal" "utfvp" "fyo")
BEST_MODEL="resnet50_cbam_center"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "🔄 Training: model=$BEST_MODEL, data=$dataset"

    LOG_FILE="logs/generalize_${dataset}_${TIMESTAMP}.log"

    timeout 3600 uv run src/train.py \
        data="$dataset" \
        model="$BEST_MODEL" \
        trainer.epochs=$EPOCHS \
        loader.num_workers=0 \
        < /dev/null 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Completed: $dataset"
        RESULTS["gen_${dataset}"]="success"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "⏱️  Timeout: $dataset"
        RESULTS["gen_${dataset}"]="timeout"
    else
        echo "❌ Failed: $dataset (exit code: $EXIT_CODE)"
        RESULTS["gen_${dataset}"]="failed"
    fi
done

# ============================================================================
# C. ABLATION STUDIES (Tabela 1/4 - Loss & Embedding Dim)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 SECTION C: Ablation Studies - Loss Functions & Embedding Dimensions"
echo "════════════════════════════════════════════════════════════════════════════"

# Loss functions
echo ""
echo "--- Loss Function Comparison ---"
LOSSES=(
    "triplet|Triplet Loss"
    "contrastive|Contrastive Loss"
    "triplet_center|Triplet+Center (default)"
)

for loss_config in "${LOSSES[@]}"; do
    IFS='|' read -r loss_name loss_label <<< "$loss_config"

    echo ""
    echo "🔄 Training with $loss_label"

    LOG_FILE="logs/ablation_loss_${loss_name}_${TIMESTAMP}.log"

    uv run src/train.py \
        data=mmcbnu \
        model=resnet50_cbam_center \
        model.loss.name="$loss_name" \
        trainer.epochs=$EPOCHS \
        loader.num_workers=0 \
        < /dev/null 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Completed: $loss_label"
        RESULTS["ablation_loss_${loss_name}"]="success"
    else
        echo "❌ Failed: $loss_label (exit code: $EXIT_CODE)"
        RESULTS["ablation_loss_${loss_name}"]="failed"
    fi
done

# Embedding dimensions
echo ""
echo "--- Embedding Dimension Comparison ---"
DIMS=("128" "256" "384" "512")

for dim in "${DIMS[@]}"; do
    echo ""
    echo "🔄 Training with embedding_dim=$dim"

    LOG_FILE="logs/ablation_dim_${dim}_${TIMESTAMP}.log"

    uv run src/train.py \
        data=mmcbnu \
        model=resnet50_cbam_center \
        model.embedding_dim="$dim" \
        trainer.epochs=$EPOCHS \
        loader.num_workers=0 \
        < /dev/null 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Completed: embedding_dim=$dim"
        RESULTS["ablation_dim_${dim}"]="success"
    else
        echo "❌ Failed: embedding_dim=$dim (exit code: $EXIT_CODE)"
        RESULTS["ablation_dim_${dim}"]="failed"
    fi
done

# ============================================================================
# D. K-NN DECISION RULE TEST (Tabela 4)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 SECTION D: K-NN Decision Rule Comparison"
echo "════════════════════════════════════════════════════════════════════════════"

# 1. Train model ONCE for K=1 (default)
echo ""
echo "🔄 Training base model for K-NN experiments (K=1)"
LOG_FILE="logs/knn_base_train_${TIMESTAMP}.log"

# Use a specific output dir to easily find the checkpoint
KNN_OUTPUT_DIR="outputs/knn_experiment/${TIMESTAMP}"

uv run src/train.py \
    data=mmcbnu \
    model=resnet50_cbam_center \
    trainer.epochs=$EPOCHS \
    k_neighbors=1 \
    hydra.run.dir="$KNN_OUTPUT_DIR" \
    loader.num_workers=0 \
    < /dev/null 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Base model training completed"
    RESULTS["knn_1"]="success"
    
    # Find the checkpoint (train.py creates nested structure)
    CHECKPOINT_PATH=$(find "$KNN_OUTPUT_DIR" -name "checkpoint_best.pt" | head -n 1)
    
    if [ -n "$CHECKPOINT_PATH" ]; then
        echo "📍 Checkpoint found: $CHECKPOINT_PATH"
        
        # 2. Evaluate for K=3 and K=5 using the trained model
        K_VALUES=("3" "5")
        
        for k in "${K_VALUES[@]}"; do
            echo ""
            echo "🔄 Evaluating with k=$k neighbors (using pre-trained model)"
            
            LOG_FILE="logs/ablation_knn_${k}_${TIMESTAMP}.log"
            
            uv run src/train.py \
                data=mmcbnu \
                model=resnet50_cbam_center \
                trainer.epochs=0 \
                k_neighbors="$k" \
                model.checkpoint_path="$CHECKPOINT_PATH" \
                loader.num_workers=0 \
                < /dev/null 2>&1 | tee "$LOG_FILE"
                
            EXIT_CODE=${PIPESTATUS[0]}
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ Completed: k=$k"
                RESULTS["knn_${k}"]="success"
            else
                echo "❌ Failed: k=$k (exit code: $EXIT_CODE)"
                RESULTS["knn_${k}"]="failed"
            fi
        done
    else
        echo "❌ Checkpoint not found in $KNN_OUTPUT_DIR"
        RESULTS["knn_base"]="failed_no_checkpoint"
    fi
else
    echo "❌ Base model training failed (exit code: $EXIT_CODE)"
    RESULTS["knn_1"]="failed"
fi

# ============================================================================
# COLLECT METRICS
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 Collecting metrics from all experiments..."
echo "════════════════════════════════════════════════════════════════════════════"

uv run python << 'PYTHON_SCRIPT'
import json
import torch
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_DIR = Path(".")
OUTPUTS_DIR = PROJECT_DIR / "outputs"
TABLES_DIR = PROJECT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True)

print("🔍 Scanning for experiment results...")

# Find all prototypes.pt files
results = []

for prototypes_file in OUTPUTS_DIR.rglob("prototypes.pt"):
    try:
        # Load metrics from prototypes.pt
        data = torch.load(prototypes_file, map_location='cpu', weights_only=False)
        metrics = data.get('metrics', {})

        # Extract path info
        parts = prototypes_file.parts
        dataset = "unknown"
        model = "unknown"
        
        # Try to infer dataset and model from path
        for part in parts:
            if part in ['mmcbnu', 'dorsal', 'utfvp', 'fyo']:
                dataset = part
            if part in ['basic', 'unet', 'attention_unet', 'resnet50_cbam_center', 'resnet50_cbam', 'simple_cnn']:
                model = part

        # Try to load protocol for detailed config
        protocol_path = prototypes_file.parent / "evaluation_protocol.json"
        loss_name = "unknown"
        embedding_dim = "unknown"
        k_neighbors = 1
        
        if protocol_path.exists():
            with open(protocol_path, 'r') as f:
                protocol = json.load(f)
                
                # Extract config details
                if 'training' in protocol:
                    loss_name = protocol['training'].get('loss', 'unknown')
                
                if 'model' in protocol:
                    embedding_dim = protocol['model'].get('embedding_dim', 'unknown')
                    # Fallback if model name not found in path
                    if model == "unknown":
                        model = protocol['model'].get('name', 'unknown')
                
                if 'dataset' in protocol:
                    # Fallback if dataset not found in path
                    if dataset == "unknown":
                        dataset = protocol['dataset'].get('name', 'unknown')
                        
                if 'inference' in protocol:
                    k_neighbors = protocol['inference'].get('k', 1)

        # Timestamp from path (usually the parent folder name)
        timestamp = prototypes_file.parent.name

        result = {
            'dataset': dataset,
            'model': model,
            'loss': loss_name,
            'embedding_dim': embedding_dim,
            'k': k_neighbors,
            'timestamp': timestamp,
            'path': str(prototypes_file),
            'auroc': metrics.get('auroc', 'N/A'),
            'oscr': metrics.get('oscr_auc', metrics.get('oscr', 'N/A')),
            'eer': metrics.get('eer', 'N/A'),
            'rank1': metrics.get('cmc_rank1', 'N/A'),
            'accuracy': metrics.get('known_accuracy', 'N/A'),
            'registration_time_ms': metrics.get('registration_time_ms', 'N/A'),
            'query_time_per_sample_ms': metrics.get('avg_query_time_per_sample_ms', 'N/A'),
        }
        results.append(result)
        print(f"✅ Found: {dataset:10s} | {model:20s} | Loss: {loss_name:15s} | Dim: {str(embedding_dim):5s} | K: {str(k_neighbors):3s} | OSCR: {result['oscr']}")

    except Exception as e:
        print(f"⚠️  Error loading {prototypes_file}: {e}")

# Save all results
if results:
    df_all = pd.DataFrame(results)

    # Save raw results
    raw_json_path = TABLES_DIR / "all_results.json"
    with open(raw_json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Saved all results to: {raw_json_path}")

    # 1. Table 2: Architecture Comparison (MMCBNU, default loss/dim/k)
    # Filter for default params: loss=triplet_center (or similar), dim=256 (default?), k=1
    # Actually, for architecture comparison, we just want to compare models on mmcbnu.
    # We should pick the 'standard' run for each model.
    # Assuming the architecture loop runs with defaults.
    df_arch = df_all[
        (df_all['dataset'] == 'mmcbnu') & 
        (df_all['k'] == 1)
    ].copy()
    
    # If we have multiple runs (e.g. from ablation), we need to filter for the "main" configuration.
    # The main config usually has embedding_dim=256 and loss=triplet_center (or whatever default is).
    # But simpler: just group by model and take the latest timestamp or best score?
    # For now, let's just dump what we have, maybe filtering by loss if possible.
    
    if not df_arch.empty:
        # Sort by OSCR descending
        try:
            df_arch['oscr'] = pd.to_numeric(df_arch['oscr'], errors='coerce')
            df_arch = df_arch.sort_values('oscr', ascending=False)
        except:
            pass
            
        table2 = df_arch[['model', 'loss', 'embedding_dim', 'oscr', 'auroc', 'eer', 'rank1', 'accuracy', 'query_time_per_sample_ms']]
        table2_path = TABLES_DIR / "table2_architecture_comparison.csv"
        table2.to_csv(table2_path, index=False)
        print(f"📊 Table 2 (Architecture): {table2_path}")

    # 2. Table 3: Generalization (Best Model on all datasets)
    df_gen = df_all[
        (df_all['model'] == 'resnet50_cbam_center') &
        (df_all['k'] == 1) &
        (df_all['loss'].isin(['triplet_center', 'triplet_center_loss', 'unknown'])) # Default loss
    ].copy()
    
    if not df_gen.empty:
        table3 = df_gen[['dataset', 'oscr', 'auroc', 'eer', 'rank1', 'accuracy']]
        table3_path = TABLES_DIR / "table3_generalization.csv"
        table3.to_csv(table3_path, index=False)
        print(f"📊 Table 3 (Generalization): {table3_path}")

    # 3. Table 4a: Ablation - Loss Functions
    df_loss = df_all[
        (df_all['dataset'] == 'mmcbnu') &
        (df_all['model'] == 'resnet50_cbam_center') &
        (df_all['k'] == 1)
    ].copy()
    
    if not df_loss.empty:
        # We might have duplicates due to dim ablation (which also uses resnet50_cbam_center).
        # We want to see variation in 'loss'.
        # Filter for default dim (256) to isolate loss impact
        df_loss_filtered = df_loss[df_loss['embedding_dim'].astype(str) == '256']
        
        if not df_loss_filtered.empty:
            table4a = df_loss_filtered[['loss', 'oscr', 'auroc', 'eer', 'rank1']]
            table4a_path = TABLES_DIR / "table4a_ablation_loss.csv"
            table4a.to_csv(table4a_path, index=False)
            print(f"📊 Table 4a (Ablation Loss): {table4a_path}")

    # 4. Table 4b: Ablation - Embedding Dimensions
    df_dim = df_all[
        (df_all['dataset'] == 'mmcbnu') &
        (df_all['model'] == 'resnet50_cbam_center') &
        (df_all['k'] == 1)
    ].copy()
    
    if not df_dim.empty:
        # Filter for default loss to isolate dim impact
        # Default loss is usually triplet_center
        df_dim_filtered = df_dim[df_dim['loss'].isin(['triplet_center', 'triplet_center_loss'])]
        
        if not df_dim_filtered.empty:
            table4b = df_dim_filtered[['embedding_dim', 'oscr', 'auroc', 'eer', 'rank1']]
            table4b_path = TABLES_DIR / "table4b_ablation_dim.csv"
            table4b.to_csv(table4b_path, index=False)
            print(f"📊 Table 4b (Ablation Dim): {table4b_path}")

    # 5. Table 4c: K-NN
    df_knn = df_all[
        (df_all['dataset'] == 'mmcbnu') &
        (df_all['model'] == 'resnet50_cbam_center')
    ].copy()
    
    if not df_knn.empty:
        # Filter for default loss and dim
        df_knn_filtered = df_knn[
            (df_knn['loss'].isin(['triplet_center', 'triplet_center_loss'])) &
            (df_knn['embedding_dim'].astype(str) == '256')
        ]
        
        if not df_knn_filtered.empty:
            table4c = df_knn_filtered[['k', 'oscr', 'auroc', 'eer', 'rank1']]
            table4c_path = TABLES_DIR / "table4c_knn.csv"
            table4c.to_csv(table4c_path, index=False)
            print(f"📊 Table 4c (K-NN): {table4c_path}")

    print(f"\n✅ All tables saved to: {TABLES_DIR}")
else:
    print("⚠️  No results found. Make sure experiments completed successfully.")

PYTHON_SCRIPT

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "✨ EXPERIMENTS COMPLETED"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results Summary:"
for key in "${!RESULTS[@]}"; do
    echo "  $key: ${RESULTS[$key]}"
done
echo ""
echo "📂 Output locations:"
echo "  - Logs: logs/experiments_${TIMESTAMP}.log"
echo "  - Tables: tables/"
echo "  - Detailed metrics: tables/all_results.json"
echo ""
echo "✅ Run 'python3 scripts/generate_paper_figures.py' to create visualizations"
echo "════════════════════════════════════════════════════════════════════════════"
