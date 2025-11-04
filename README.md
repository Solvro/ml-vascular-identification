# Vascular Identification with OpenSet Recognition

Deep learning-based vascular biometric identification system using metric learning and OpenSet recognition. The system can identify known subjects while rejecting unknown subjects using dorsal hand vein or finger vein patterns.

## Features

- **OpenSet Recognition**: Distinguish between known and unknown subjects
- **Metric Learning**: Triplet/Contrastive loss for discriminative embeddings
- **Subject-Disjoint Splits**: Proper evaluation with non-overlapping subjects
- **Session-Based Protocol**: Separate enrollment and testing samples
- **Comprehensive Metrics**: CMC curves, OSCR, EER, AUROC, TPR@FPR
- **k-NN Support**: Configurable k-nearest neighbor decision making
- **Automatic Threshold Optimization**: Find optimal threshold for OpenSet detection

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)
- UV package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/Solvro/ml-vascular-identification.git
cd ml-vascular-identification

# Install dependencies using UV
uv sync

# Or using pip
pip install -e .
```

## Project Structure

```
ml-vascular-identification/
├── src/
│   ├── train.py              # Main training script
│   ├── utils.py              # Utilities (plotting, protocol saving)
│   ├── data/                 # Data loading and preprocessing
│   │   ├── mmcbnu.py        # MMCBNU dataset
│   │   ├── dorsal.py        # Dorsal hand vein dataset
│   │   ├── data_loaders.py  # OpenSet data loaders
│   │   ├── splits.py        # Subject-disjoint splitting
│   │   └── transforms.py    # Image augmentations
│   └── models/              # Model architectures and losses
│       ├── base.py          # Base classes and factories
│       ├── metrics.py       # Evaluation metrics
│       ├── losses.py        # Triplet/Contrastive losses
│       └── basic/
│           └── basic_cnn.py # CNN architecture
├── config/                   # Hydra configuration files
│   ├── train.yaml           # Main config
│   ├── data/                # Dataset configs
│   ├── model/               # Model configs
│   └── trainer/             # Training configs
├── tests/                    # Unit tests
├── examples/                 # Usage examples
└── data/                    # Dataset storage (not in repo)
    └── mmcbnu/              # MMCBNU dataset
```

## Quick Start

### 1. Basic Training

Train a model on MMCBNU dataset with default settings:

```bash
uv run src/train.py data=mmcbnu
```

### 2. Custom Configuration

Train with custom parameters:

```bash
uv run src/train.py \
    data=mmcbnu \
    model.embedding_dim=512 \
    model.loss.margin=0.5 \
    loader.sampler.P=16 \
    loader.sampler.K=4 \
    trainer.epochs=100 \
    k_neighbors=1 \
    threshold_metric=oscr
```

### 3. Evaluation Only

Skip training and evaluate existing model:

```bash
uv run src/train.py data=mmcbnu trainer.epochs=0
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Key parameters:

### Dataset Configuration (`config/data/`)

```yaml
# config/data/mmcbnu.yaml
name: mmcbnu
mode: openset
known_ratio: 0.7        # 70% patients as known classes
val_ratio: 0.15         # 15% of known for validation
subject_disjoint: true  # Enforce subject-disjoint splits
enrollment_samples: 7   # Samples for building prototypes
test_samples: 3         # Samples for testing
```

### Model Configuration (`config/model/`)

```yaml
# config/model/basic.yaml
name: simple_cnn
embedding_dim: 384      # Embedding dimensionality
dropout: 0.2            # Dropout rate
use_attention: false    # Use attention mechanism

loss:
  name: triplet
  margin: 0.5           # Triplet loss margin

optimizer:
  lr: 0.0001           # Learning rate
  weight_decay: 0.0005 # L2 regularization
```

### Training Configuration (`config/trainer/`)

```yaml
# config/trainer/default.yaml
epochs: 100
log_interval: 10

early_stopping:
  enabled: true
  patience: 15          # Stop after 15 epochs without improvement
  min_delta: 0.0001    # Minimum change to qualify as improvement
```

### Sampling Configuration (`config/loader/`)

```yaml
# config/loader/default.yaml
batch_size: 32
num_workers: 4

sampler:
  P: 16  # Classes per batch
  K: 4   # Samples per class
  # Effective batch size = P × K = 64
```

## Usage Examples

### Python API

```python
from data import create_openset_data_loaders
from models import create_model, create_loss
import torch

# Create OpenSet data loaders
loaders, info = create_openset_data_loaders(
    dataset_name='mmcbnu',
    img_size=224,
    known_ratio=0.7,
    val_ratio=0.15,
    P=16, K=4,
    seed=42
)

# Access loaders
train_loader = loaders['train']
val_loader = loaders['val_known']
enrollment_loader = loaders['test_known_enrollment']  # For prototypes
query_loader = loaders['test_known_query']            # For testing
unknown_loader = loaders['test_unknown']

# Create model
model = create_model('simple_cnn', embedding_dim=256)

# Create loss
criterion = create_loss('triplet', margin=0.3)

# Training loop
for epoch in range(epochs):
    for images, labels, metadata in train_loader:
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        # ... backward pass
```

### OpenSet Evaluation

```python
from train import compute_prototypes, evaluate_openset

# 1. Compute prototypes from enrollment samples
prototypes, class_to_idx = compute_prototypes(
    model, 
    enrollment_loader, 
    num_classes=420, 
    embedding_dim=256, 
    device='cuda'
)

# 2. Evaluate on known and unknown samples
metrics = evaluate_openset(
    model,
    query_loader,           # Known class queries
    unknown_loader,         # Unknown class samples
    prototypes,
    device='cuda',
    class_to_idx=class_to_idx,
    threshold=0.9,         # OpenSet threshold
    k=1                    # k-NN (1 for nearest neighbor)
)

# 3. Check results
print(f"CMC Rank-1: {metrics['cmc_rank1']:.2%}")
print(f"OSCR: {metrics['oscr']:.2%}")
print(f"EER: {metrics['eer']:.2f}%")
print(f"AUROC: {metrics['auroc']:.4f}")
```

## Metrics Explained

### Identification Metrics
- **CMC Rank-1/5/10**: Cumulative Match Characteristic - probability correct match is in top-k
- **Known Accuracy**: % of known samples correctly identified

### OpenSet Metrics
- **OSCR**: Open-Set Classification Rate - correct classification rate vs FPR
- **AUROC**: Area Under ROC - discrimination between known/unknown
- **EER**: Equal Error Rate - where FAR = FRR
- **TPR@FPR**: True Positive Rate at specific False Positive Rate (0.1%, 1%, 10%)

### Rejection Metrics
- **Unknown Rejection**: % of unknown samples correctly rejected

## Dataset Format

### MMCBNU (Finger Vein)

```
data/mmcbnu/
├── Captured images/
│   ├── 001/                    # Patient 001
│   │   ├── L_index_01.bmp     # Left index finger, sample 1
│   │   ├── L_index_02.bmp
│   │   └── ...
│   └── 100/                    # Patient 100
└── ROIs/                       # Region of Interest (alternative)
    └── ...
```

Each patient has 6 fingers × 10 samples = 60 images.

### Dorsal Hand Vein

```
data/dorsal/
├── patient_001/
│   ├── sample_01.png
│   ├── sample_02.png
│   └── ...
└── patient_100/
```

## Training Outputs

After training, the following files are generated:

```
best_model_mmcbnu.pt              # Best model checkpoint
best_model_mmcbnu_prototypes.pt   # Prototypes and evaluation metrics
evaluation_protocol_mmcbnu.json   # Detailed evaluation results
det_curve_mmcbnu.png             # Detection Error Tradeoff curve
roc_curve_mmcbnu.png             # ROC curve
```

### Evaluation Protocol JSON

```json
{
  "timestamp": "2025-11-03T12:00:00",
  "dataset": {
    "name": "mmcbnu",
    "known_classes": 420,
    "unknown_classes": 180
  },
  "metrics": {
    "cmc_rank1": 0.9929,
    "cmc_rank5": 0.9976,
    "oscr": 0.9853,
    "auroc": 0.9902,
    "eer": 3.40,
    "known_accuracy": 0.9579,
    "unknown_rejection": 0.9789
  }
}
```

## Testing

Run the test suite:

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/test_train.py

# With coverage
uv run pytest --cov=src --cov-report=html
```

## Advanced Usage

### Custom Dataset

To add a new dataset:

1. Create dataset class in `src/data/your_dataset.py`:

```python
from data.base import VascularDataset

class YourDataset(VascularDataset):
    def scan_dataset(self):
        # Scan and return list of sample dicts
        pass
```

2. Add configuration in `config/data/your_dataset.yaml`

3. Register in `src/data/__init__.py`

### Custom Model

To add a new model architecture:

1. Create model in `src/models/your_model/`:

```python
from models.base import BaseEmbeddingModel

class YourModel(BaseEmbeddingModel):
    def __init__(self, embedding_dim=256):
        super().__init__()
        # Define architecture
        
    def forward(self, x):
        # Forward pass
        return embeddings
```

2. Register in `src/models/__init__.py`:

```python
MODEL_REGISTRY['your_model'] = YourModel
```

## Performance Tips

1. **Batch Size**: Use P×K = 64 for good GPU utilization
2. **Learning Rate**: Start with 1e-4, reduce if unstable
3. **Margin**: 0.3-0.5 works well for triplet loss
4. **Embedding Dim**: 256-512 is sufficient
5. **Early Stopping**: Patience of 10-15 epochs prevents overfitting

## Results

### MMCBNU Dataset (100 patients, 600 finger classes)

| Metric | Value |
|--------|-------|
| CMC Rank-1 | 99.29% |
| CMC Rank-5 | 99.76% |
| OSCR | 98.53% |
| AUROC | 99.02% |
| EER | 3.40% |
| Known Accuracy | 95.79% |
| Unknown Rejection | 97.89% |

**Configuration**: 384-dim embeddings, triplet loss (margin=0.5), P=16, K=4, 52 epochs

## Troubleshooting

### Low Accuracy
- Check data augmentation is not too aggressive
- Verify prototypes computed from enrollment samples
- Ensure subject-disjoint splits are correct
- Try increasing embedding dimension

### Memory Issues
- Reduce batch size (decrease P or K)
- Use smaller images (e.g., 128x128 instead of 224x224)
- Enable gradient checkpointing
- Reduce number of workers

### Training Instability
- Decrease learning rate
- Add learning rate warmup
- Check for NaN values in loss
- Verify data normalization
