#!/bin/bash

# Script to run all model + dataset combinations
# Runs training for each combination regardless of how long it takes

# Removed set -e to allow script to continue after errors

PROJECT_DIR="/home/licho/.programming/Projects/open_set"
cd "$PROJECT_DIR"

# Get all model configs (without .yaml extension)
MODELS=()
for model_file in config/model/*.yaml; do
    model_name=$(basename "$model_file" .yaml)
    MODELS+=("$model_name")
done

# Get all dataset configs (without .yaml extension)
DATASETS=()
for data_file in config/data/*.yaml; do
    data_name=$(basename "$data_file" .yaml)
    DATASETS+=("$data_name")
done

# Counter for tracking progress
TOTAL=$((${#MODELS[@]} * ${#DATASETS[@]}))
CURRENT=0

echo "🚀 Starting training runs for all combinations"
echo "📊 Models found: ${MODELS[@]}"
echo "📊 Datasets found: ${DATASETS[@]}"
echo "📊 Total combinations: $TOTAL"
echo "================================"

# Iterate through all combinations
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "[$CURRENT/$TOTAL] 🔄 Running: MODEL=$MODEL, DATASET=$DATASET"
        echo "================================"
        
        # Run training with uv - use pipefail to capture exit code correctly
        set +e  # Disable exit on error for this command
        (set -o pipefail; uv run src/train.py \
            model="$MODEL" \
            data="$DATASET" \
            2>&1 | tee -a "training_${MODEL}_${DATASET}.log")
        EXIT_CODE=$?
        set -e  # Re-enable for safety (though we removed global set -e)
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Successfully completed: $MODEL on $DATASET"
        else
            echo "❌ Failed: $MODEL on $DATASET (exit code: $EXIT_CODE, see training_${MODEL}_${DATASET}.log)"
        fi
        
        echo "================================"
    done
done

echo ""
echo "✨ All training runs completed!"
echo "📂 Check individual logs: training_*.log for details"
