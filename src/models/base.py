from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class BaseEmbeddingModel(nn.Module, ABC):
    """Base contract for embedding models.

    Subclasses should set `embedding_dim` and implement `forward` to return
    a tensor of shape (B, embedding_dim).
    """

    embedding_dim: int

    @abstractmethod
    def forward(self, x: Any) -> torch.Tensor:  # (B, D)
        """Compute embeddings for input batch and return (B, D) tensor."""
        raise NotImplementedError

    def get_embedding(self, x: Any) -> torch.Tensor:
        """Alias for forward; override if different behavior is needed."""
        return self.forward(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters for bookkeeping/logging."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

