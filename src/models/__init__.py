"""
Models package for vascular identification with OpenSet recognition support.

This package contains model architectures and loss functions for embedding learning
in vascular biometric identification tasks, optimized for OpenSet recognition.

Key components:
- BaseEmbeddingModel: Abstract base for embedding models with L2-normalization
- TripletLoss/ContrastiveLoss: Metric learning losses for embedding training
- CosineClassifier: Cosine similarity head with temperature for inference
- OpenSet metrics: EER, OSCR, AUROC, TPR@FPR, CMC curves
"""
from .base import (
    BaseEmbeddingModel,
    BaseLoss,
    CosineClassifier,
    create_loss,
    create_model,
)
from .basic import SimpleEmbeddingModel
from .losses import ContrastiveLoss, TripletLoss

# Import metrics utilities
from .metrics import (
    compute_auroc,
    compute_cmc_curve,
    compute_eer,
    compute_frr_at_far,
    compute_openset_metrics,
    compute_oscr,
    compute_tpr_at_fpr,
)
from .unet import AttentionUNet, UNetEmbedding
from .visual_trasformer import DeiTEmbedding, VisionTransformerEmbedding

__all__ = [
    # Base classes
    "BaseEmbeddingModel",
    "BaseLoss",
    "CosineClassifier",
    # Factory functions
    "create_model",
    "create_loss",
    # Basic CNN models
    "SimpleEmbeddingModel",
    # U-Net models
    "UNetEmbedding",
    "AttentionUNet",
    # Vision Transformer models
    "VisionTransformerEmbedding",
    "DeiTEmbedding",
    # Loss functions
    "TripletLoss",
    "ContrastiveLoss",
    # OpenSet metrics
    "compute_eer",
    "compute_frr_at_far",
    "compute_tpr_at_fpr",
    "compute_auroc",
    "compute_oscr",
    "compute_openset_metrics",
    "compute_cmc_curve",
]
