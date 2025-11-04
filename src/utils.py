"""
Utility functions for visualization and protocol logging.

Provides functions for:
- Plotting DET and ROC curves
- Saving evaluation protocols to JSON
- Timing measurements for inference
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import det_curve, roc_curve


def plot_det_curve(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    save_path: str,
    title: str = "DET Curve"
) -> None:
    """
    Plot and save Detection Error Tradeoff (DET) curve.
    
    DET curve plots False Rejection Rate (FRR) vs False Acceptance Rate (FAR)
    on a log scale, commonly used in biometric verification.
    
    Args:
        genuine_scores: Similarity scores for genuine pairs
        impostor_scores: Similarity scores for impostor pairs
        save_path: Path to save the plot
        title: Plot title
    """
    # Combine scores and labels
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    # Compute DET curve
    fpr, fnr, thresholds = det_curve(labels, scores)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr * 100, fnr * 100, linewidth=2, label='DET Curve')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Acceptance Rate (%)', fontsize=12)
    plt.ylabel('False Rejection Rate (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved DET curve to {save_path}")


def plot_roc_curve(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    save_path: str,
    title: str = "ROC Curve",
    auroc: Optional[float] = None
) -> None:
    """
    Plot and save Receiver Operating Characteristic (ROC) curve.
    
    ROC curve plots True Positive Rate (TPR) vs False Positive Rate (FPR),
    with Area Under Curve (AUC/AUROC) as performance metric.
    
    Args:
        genuine_scores: Similarity scores for genuine pairs
        impostor_scores: Similarity scores for impostor pairs
        save_path: Path to save the plot
        title: Plot title
        auroc: Optional AUROC value to display in legend
    """
    # Combine scores and labels
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    label_text = f'ROC Curve (AUC={auroc:.4f})' if auroc is not None else 'ROC Curve'
    plt.plot(fpr, tpr, linewidth=2, label=label_text)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='lower right')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved ROC curve to {save_path}")


def _convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable types to native Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def save_evaluation_protocol(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save evaluation protocol to JSON file.
    
    Protocol includes:
    - Dataset configuration (known/unknown splits, sizes)
    - Model hyperparameters
    - All evaluation metrics
    - Timestamp and reproducibility info
    
    Args:
        metrics: Dictionary of evaluation metrics
        config: Dictionary of configuration parameters
        save_path: Path to save JSON file
    """
    # Convert metrics to JSON-serializable format
    metrics_serializable = _convert_to_json_serializable(metrics)
    config_serializable = _convert_to_json_serializable(config)
    
    protocol = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'name': config_serializable.get('dataset_name', 'unknown'),
            'total_classes': config_serializable.get('total_classes', 0),
            'known_classes': config_serializable.get('known_classes', 0),
            'unknown_classes': config_serializable.get('unknown_classes', 0),
            'subject_disjoint': config_serializable.get('subject_disjoint', True),
        },
        'splits': {
            'train_size': config_serializable.get('train_size', 0),
            'val_size': config_serializable.get('val_size', 0),
            'test_known_size': config_serializable.get('test_known_size', 0),
            'test_unknown_size': config_serializable.get('test_unknown_size', 0),
            'enrollment_size': config_serializable.get('enrollment_size', 0),
        },
        'model': {
            'name': config_serializable.get('model_name', 'unknown'),
            'embedding_dim': config_serializable.get('embedding_dim', 256),
            'parameters': config_serializable.get('total_parameters', 0),
        },
        'training': {
            'loss': config_serializable.get('loss_name', 'unknown'),
            'margin': config_serializable.get('margin', 0.3),
            'optimizer': config_serializable.get('optimizer', 'AdamW'),
            'learning_rate': config_serializable.get('learning_rate', 0.0003),
            'epochs': config_serializable.get('epochs', 0),
            'best_epoch': config_serializable.get('best_epoch', 0),
            'best_val_loss': config_serializable.get('best_val_loss', 0.0),
            'P': config_serializable.get('P', 8),
            'K': config_serializable.get('K', 4),
        },
        'inference': {
            'k': config_serializable.get('k', 1),
            'threshold': config_serializable.get('threshold', 0.5),
            'threshold_optimization': config_serializable.get('threshold_optimization', 'manual'),
        },
        'metrics': metrics_serializable,
    }
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(protocol, f, indent=2)
    
    print(f"📋 Saved evaluation protocol to {save_path}")


class InferenceTimer:
    """Context manager for timing inference operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.perf_counter() - self.start_time
        print(f"⏱️  {self.name}: {self.elapsed_time*1000:.2f} ms")
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_time * 1000 if self.elapsed_time else 0.0
