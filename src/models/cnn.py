import torch
from torch import nn, Tensor
from src.models.base import BaseEmbeddingModel



class SimpleCNN(BaseEmbeddingModel):
    """
    Simple CNN model implementing the BaseEmbeddingModel interface, returns an embedding from an image.
    Assumes grayscale input (e.g., MNIST-style images).
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        """
        Args:
            embedding_dim (int): Dimension of the output embedding. Defaults to 128.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Assumes grayscale input
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the CNN encoder and projector.
        Args:
            x (Tensor): Input image tensor.
        Returns:
            Tensor: Embedding tensor.
        """
        x = self.encoder(x)
        x = self.projector(x)
        return x

    def get_embedding(self, x: Tensor) -> Tensor:
        """
        Returns the embedding for the given input.
        Args:
            x (Tensor): Input image tensor.
        Returns:
            Tensor: Embedding tensor.
        """
        return self.forward(x)
