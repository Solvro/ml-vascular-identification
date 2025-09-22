"""
Models package for vascular identification.

This package contains model architectures and loss functions for embedding learning
in vascular biometric identification tasks.
"""
from .base import BaseEmbeddingModel, BaseLoss, create_loss, create_model
from .basic import SimpleEmbeddingModel
from .losses import ContrastiveLoss, TripletLoss
from .unet import AttentionUNet, UNetEmbedding
from .visual_trasformer import DeiTEmbedding, VisionTransformerEmbedding

__all__ = [
    # Base classes
    "BaseEmbeddingModel",
    "BaseLoss",
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
]
