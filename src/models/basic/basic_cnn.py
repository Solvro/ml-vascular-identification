"""
Basic CNN architectures for embedding learning.

This module contains concrete implementations of embedding models and loss functions
based on simple CNN architectures suitable for vascular identification tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEmbeddingModel


class SimpleEmbeddingModel(BaseEmbeddingModel):
    """Simple CNN for embedding learning.

    A basic convolutional neural network that extracts features from input images
    and produces normalized embedding vectors for similarity learning.
    """

    def __init__(self, embedding_dim: int = 256, input_channels: int = 3):
        super().__init__(embedding_dim)

        self.backbone = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns L2-normalized embeddings.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization


# Model configurations
MODEL_CONFIGS = {
    "simple_cnn": {
        "class": SimpleEmbeddingModel,
        "default_params": {"embedding_dim": 256, "input_channels": 3},
        "description": "Simple CNN with 3 conv blocks and global average pooling",
    }
}
