
from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn

class BaseEmbeddingModel(nn.Module, ABC):
    """
    Abstract base class for embedding models.
    All embedding models should inherit from this class and implement the required methods.
    """
    @abstractmethod
    def forward(self, x: Any) -> Any:
        """
        Forward pass for the model.
        Args:
            x: Input tensor or data.
        Returns:
            Output tensor or embedding.
        """
        pass

    @abstractmethod
    def get_embedding(self, x: Any) -> Any:
        """
        Returns the embedding for the given input.
        Args:
            x: Input tensor or data.
        Returns:
            Embedding tensor.
        """
        pass

class BaseFactory(ABC):
    """
    Abstract factory for creating embedding models.
    """
    @abstractmethod
    def create_model(self) -> BaseEmbeddingModel:
        """
        Creates and returns an instance of BaseEmbeddingModel.
        Returns:
            BaseEmbeddingModel: An embedding model instance.
        """
        pass
