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
    Supports both training mode (L2-normalized embeddings) and inference mode
    (can use cosine similarity head for classification).
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
            
        Note:
            Embeddings should be L2-normalized for metric learning with
            triplet/contrastive loss.
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
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward pass - extract embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.forward(x)


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
    from .unet import AttentionUNet, UNetEmbedding
    from .visual_trasformer import DeiTEmbedding, VisionTransformerEmbedding

    models = {
        # Basic CNN models
        "simple_cnn": SimpleEmbeddingModel,
        "simple_embedding": SimpleEmbeddingModel,  # alias
        # U-Net models
        "unet_embedding": UNetEmbedding,
        "attention_unet": AttentionUNet,
        # Vision Transformer models
        "vit_embedding": VisionTransformerEmbedding,
        "deit_embedding": DeiTEmbedding,
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
    from .losses import ContrastiveLoss, TripletLoss

    losses = {
        "triplet": TripletLoss,
        "triplet_loss": TripletLoss,  # alias
        "contrastive": ContrastiveLoss,
        "contrastive_loss": ContrastiveLoss,  # alias
    }

    if loss_name not in losses:
        available = ", ".join(losses.keys())
        raise ValueError(f"Unknown loss: {loss_name}. Available: {available}")

    return losses[loss_name](**kwargs)


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature.
    
    Used for validation and inference in OpenSet recognition.
    Computes cosine similarity between embeddings and class prototypes,
    scaled by a learnable temperature parameter.
    
    For OpenSet:
    - During validation: classify to known classes with temperature scaling
    - During inference: compute similarity to prototypes + threshold for rejection
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        """Initialize cosine classifier.
        
        Args:
            embedding_dim: Dimension of input embeddings.
            num_classes: Number of classes (known finger classes).
            temperature: Initial temperature value (lower = sharper distribution).
            learnable_temperature: If True, temperature is learned during training.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Learnable class prototypes (will be normalized)
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Temperature parameter
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        self.learnable_temperature = learnable_temperature
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarities scaled by temperature.
        
        Args:
            embeddings: L2-normalized embeddings of shape (batch_size, embedding_dim).
            
        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # Normalize weight vectors (class prototypes)
        normalized_weight = nn.functional.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        # embeddings: (batch_size, embedding_dim)
        # normalized_weight.T: (embedding_dim, num_classes)
        # Result: (batch_size, num_classes)
        cosine_sim = torch.matmul(embeddings, normalized_weight.t())
        
        # Scale by temperature
        logits = cosine_sim / self.temperature
        
        return logits
    
    def get_similarities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get raw cosine similarities without temperature scaling.
        
        Useful for threshold-based rejection in OpenSet inference.
        
        Args:
            embeddings: L2-normalized embeddings of shape (batch_size, embedding_dim).
            
        Returns:
            Cosine similarities of shape (batch_size, num_classes).
        """
        normalized_weight = nn.functional.normalize(self.weight, p=2, dim=1)
        return torch.matmul(embeddings, normalized_weight.t())
