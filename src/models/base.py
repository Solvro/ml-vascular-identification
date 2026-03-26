"""
Base classes for models and losses.

This module provides abstract base classes that define common interfaces
for embedding models and loss functions used in the vascular identification system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseEmbeddingModel(nn.Module, ABC):
    """Abstract base class for embedding models.

    Defines the interface that all embedding models should implement.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns normalized embeddings.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        pass

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.__class__.__name__,
            "embedding_dim": self.embedding_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


class BaseLoss(nn.Module, ABC):
    """Abstract base class for loss functions.

    Defines the interface for loss functions used in embedding learning.
    """

    @abstractmethod
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss from embeddings and labels.

        Args:
            embeddings: Embedding vectors of shape (batch_size, embedding_dim).
            labels: Target labels of shape (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        pass

    def get_loss_info(self) -> Dict[str, Any]:
        """Get loss function information."""
        return {
            "loss_name": self.__class__.__name__,
            "parameters": {
                k: v
                for k, v in self.__dict__.items()
                if not k.startswith("_") and not callable(v)
            },
        }


def create_model(model_name: str, **kwargs) -> BaseEmbeddingModel:
    """Factory function to create models by name.

    Args:
        model_name: Name of the model to create.
        **kwargs: Model-specific parameters.

    Returns:
        Model instance.

    Raises:
        ValueError: If model_name is not recognized.
    """
    # Import here to avoid circular imports
    from .basic import SimpleEmbeddingModel
    from .resnet_cbam import ResNet50CBAM, ResNetCBAM
    from .unet import AttentionUNet, UNetEmbedding

    models = {
        # Basic CNN models
        "simple_cnn": SimpleEmbeddingModel,
        "simple_embedding": SimpleEmbeddingModel,  # alias
        # U-Net models
        "unet_embedding": UNetEmbedding,
        "attention_unet": AttentionUNet,
        # ResNet with CBAM models
        "resnet_cbam": ResNetCBAM,
        "resnet50_cbam": ResNet50CBAM,
    }

    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return models[model_name](**kwargs)


def create_loss(loss_name: str, **kwargs) -> BaseLoss:
    """Factory function to create loss functions by name.

    Args:
        loss_name: Name of the loss to create.
        **kwargs: Loss-specific parameters.

    Returns:
        Loss function instance.

    Raises:
        ValueError: If loss_name is not recognized.
    """
    # Import here to avoid circular imports
    from .losses import CenterLoss, ContrastiveLoss, TripletCenterLoss, TripletLoss

    losses = {
        "triplet": TripletLoss,
        "triplet_loss": TripletLoss,  # alias
        "contrastive": ContrastiveLoss,
        "contrastive_loss": ContrastiveLoss,  # alias
        "center": CenterLoss,
        "center_loss": CenterLoss,  # alias
        "triplet_center": TripletCenterLoss,
        "triplet_center_loss": TripletCenterLoss,  # alias
    }

    if loss_name not in losses:
        available = ", ".join(losses.keys())
        raise ValueError(f"Unknown loss: {loss_name}. Available: {available}")

    return losses[loss_name](**kwargs)
