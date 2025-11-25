# Paper Experiments Guide

## Overview

This guide explains how to run all experiments needed for the Open-Set Recognition paper and generate publication-ready tables and figures.

## Quick Start

```bash
# Run all experiments (takes several hours)
bash scripts/run_paper_experiments.sh

# Generate figures and tables from results
python3 scripts/generate_paper_figures.py
```

## Experiments Breakdown

### A. Architecture Comparison (Table 2)
Tests different CNN architectures on MMCBNU dataset:
- Basic CNN
- U-Net
- Attention U-Net
- ResNet50 + CBAM + Center Loss (proposed)

**Metrics compared:** OSCR, AUROC, EER, Rank-1, Accuracy, Inference Time

### B. Generalization (Table 3)
Tests the best model (ResNet50+CBAM+Center) on 4 different datasets:
- MMCBNU (finger vein - small dataset)
- Dorsal (hand vein imaging - full images)
- UTFVP (finger vein - multi-session)
- FYO (multi-body-part - dorsal/palm/wrist)

**Metrics compared:** OSCR, AUROC, EER, Rank-1, Accuracy

### C. Ablation Studies (Table 1/4)

#### Loss Functions
- Triplet Loss (baseline)
- Contrastive Loss
- Triplet + Center Loss (proposed)

#### Embedding Dimensions
- 128D
- 256D
- 384D (default)
- 512D

### D. Decision Rule Comparison (Table 4)
Tests k-NN decision rules (1-NN, 3-NN, 5-NN) on best model.

## Output Structure

```
tables/
├── table2_architecture.csv          # Architecture comparison (CSV)
├── table2_architecture.tex          # Architecture comparison (LaTeX)
├── table3_generalization.csv        # Generalization (CSV)
├── table3_generalization.tex        # Generalization (LaTeX)
├── all_results.json                 # Raw metrics from all experiments
└── figures/
    ├── figure_architecture_comparison.pdf
    ├── figure_generalization.pdf
    └── figure_cmc_curves.pdf

logs/
└── experiments_[TIMESTAMP].log      # Detailed training logs
```

## Key Metrics Explained

### OSCR (Open-Set Classification Rate)
- **What:** Correctly classifies known samples AND rejects unknown samples
- **Range:** 0-1 (higher is better)
- **Why:** Core metric for open-set recognition

### AUROC (Area Under ROC)
- **What:** Discriminates between known and unknown classes
- **Range:** 0-1 (higher is better, 0.5 = random)
- **Why:** Shows how well model separates known/unknown

### EER (Equal Error Rate)
- **What:** Where False Acceptance Rate = False Rejection Rate
- **Range:** 0-100% (lower is better)
- **Why:** Standard verification metric

### Rank-1 (CMC Rank-1)
- **What:** Accuracy when retrieving most similar known sample
- **Range:** 0-1 (higher is better)
- **Why:** Closed-set identification accuracy

## Viewing Results

### CSV Tables (for spreadsheets)
```bash
cat tables/table2_architecture.csv
```

### LaTeX Tables (for papers)
```bash
cat tables/table2_architecture.tex
```

### All Raw Metrics (JSON)
```bash
python3 -m json.tool tables/all_results.json
```

### Python Loading
```python
import pandas as pd
import torch

# Load CSV results
df = pd.read_csv('tables/table2_architecture.csv')

# Load raw metrics
import json
with open('tables/all_results.json') as f:
    results = json.load(f)

# Load individual experiment
data = torch.load('outputs/mmcbnu/resnet50_cbam_center/20251125_030103/prototypes.pt')
metrics = data['metrics']
print(f"OSCR: {metrics['oscr_auc']:.4f}")
```

## Troubleshooting

### Script won't run
```bash
# Make executable
chmod +x scripts/run_paper_experiments.sh
chmod +x scripts/generate_paper_figures.py

# Run directly with bash
bash scripts/run_paper_experiments.sh
```

### Memory errors during evaluation
Edit `src/train.py` and change `num_workers` to 0 in evaluation loaders

### Missing results
- Check `logs/experiments_*.log` for training errors
- Ensure `config/data/*.yaml` files exist
- Verify `config/model/*.yaml` files exist

### Figures not generating
```bash
# Install missing dependencies
pip install seaborn scikit-learn matplotlib pandas

# Run individually
python3 scripts/generate_paper_figures.py
```

## Customization

### Run single experiment
```bash
uv run src/train.py data=mmcbnu model=resnet50_cbam_center trainer.epochs=3
```

### Change hyperparameters
```bash
# Modify embedding dimension
uv run src/train.py data=mmcbnu model=resnet50_cbam_center model.embedding_dim=512

# Change loss function
uv run src/train.py data=mmcbnu model=resnet50_cbam_center model.loss.name=contrastive

# Adjust training epochs
uv run src/train.py data=mmcbnu model=resnet50_cbam_center trainer.epochs=10
```

## Citation

If you use these experiments, please cite:

## Support

For issues or questions:
1. Check training logs in `logs/`
2. Verify config files in `config/`
3. Run individual experiments with `trainer.epochs=1` for testing
4. Check GPU/CPU memory availability
