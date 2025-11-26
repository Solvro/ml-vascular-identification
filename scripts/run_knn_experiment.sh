#!/bin/bash

# ============================================================================
# K-NN Experiment Runner (Fix for Section D)
# ============================================================================

set -e
set -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EPOCHS=100
KNN_OUTPUT_DIR="outputs/knn_experiment/${TIMESTAMP}"

echo "🚀 Starting K-NN Experiments (Fix)"
echo "📂 Output Dir: $KNN_OUTPUT_DIR"

# 1. Check for existing checkpoint or Train
echo "🔍 Searching for latest checkpoint in outputs/mmcbnu/resnet50_cbam/..."
LATEST_DIR=$(ls -td outputs/mmcbnu/resnet50_cbam/*/ 2>/dev/null | head -n 1)
CHECKPOINT_PATH="${LATEST_DIR}checkpoint_best.pt"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "✅ Found existing checkpoint: $CHECKPOINT_PATH"
    echo "⏭️  Skipping K=1 training and using existing model."
else
    echo "⚠️  No checkpoint found. Starting training for K=1..."
    LOG_FILE="logs/knn_base_train_${TIMESTAMP}.log"

    uv run src/train.py \
        data=mmcbnu \
        model=resnet50_cbam_center \
        trainer.epochs=$EPOCHS \
        k_neighbors=1 \
        loader.num_workers=0 \
        < /dev/null 2>&1 | tee "$LOG_FILE"

    # Find the new checkpoint
    LATEST_DIR=$(ls -td outputs/mmcbnu/resnet50_cbam/*/ | head -n 1)
    CHECKPOINT_PATH="${LATEST_DIR}checkpoint_best.pt"

    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "❌ Training failed or checkpoint not saved at $CHECKPOINT_PATH! Aborting."
        exit 1
    fi
    echo "✅ New checkpoint created: $CHECKPOINT_PATH"
fi

# 3. Evaluate for K=3 and K=5
K_VALUES=("3" "5")

for k in "${K_VALUES[@]}"; do
    echo ""
    echo "🔄 Evaluating with k=$k neighbors..."
    
    LOG_FILE="logs/ablation_knn_${k}_${TIMESTAMP}.log"
    
    # Use ++k_neighbors to ensure override
    uv run src/train.py \
        data=mmcbnu \
        model=resnet50_cbam_center \
        trainer.epochs=0 \
        ++k_neighbors="$k" \
        +model.checkpoint_path="$CHECKPOINT_PATH" \
        loader.num_workers=0 \
        < /dev/null 2>&1 | tee "$LOG_FILE"
        
    echo "✅ Completed k=$k"
done

echo ""
echo "✨ K-NN Experiments Completed."
echo "📊 Collecting K-NN metrics..."

uv run python << 'PYTHON_SCRIPT'
import json
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(".")
OUTPUTS_DIR = PROJECT_DIR / "outputs"
TABLES_DIR = PROJECT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True)

results = []
print(f"🔍 Scanning {OUTPUTS_DIR} for K-NN results...")

for protocol_file in OUTPUTS_DIR.rglob("evaluation_protocol.json"):
    try:
        with open(protocol_file, 'r') as f:
            p = json.load(f)
        
        dataset = p.get('dataset', {}).get('name')
        model = p.get('model', {}).get('name')
        k = p.get('inference', {}).get('k', 1)
        
        # Filter for relevant experiments (mmcbnu + resnet50 variants)
        if dataset == 'mmcbnu' and 'resnet50' in str(model):
            metrics = p.get('metrics', {})
            result = {
                'k': k,
                'oscr': metrics.get('oscr_auc', metrics.get('oscr', 0)),
                'auroc': metrics.get('auroc', 0),
                'eer': metrics.get('eer', 0),
                'rank1': metrics.get('cmc_rank1', 0),
                'timestamp': p.get('timestamp'),
                'path': str(protocol_file)
            }
            results.append(result)
            
    except Exception as e:
        pass

if results:
    df = pd.DataFrame(results)
    # Sort by timestamp descending to get latest runs first
    df = df.sort_values('timestamp', ascending=False)
    
    # Take the latest run for each K
    df_latest = df.groupby('k').first().reset_index()
    df_latest = df_latest.sort_values('k')
    
    # Save JSON
    json_path = TABLES_DIR / "knn_results.json"
    df_latest.to_json(json_path, orient='records', indent=2)
    print(f"💾 Saved K-NN results to {json_path}")
    
    # Save CSV
    csv_path = TABLES_DIR / "table4c_knn.csv"
    cols = ['k', 'oscr', 'auroc', 'eer', 'rank1']
    df_latest[cols].to_csv(csv_path, index=False)
    print(f"📊 Saved CSV table to {csv_path}")
    print(df_latest[cols].to_string(index=False))
else:
    print("⚠️ No K-NN results found.")

PYTHON_SCRIPT
