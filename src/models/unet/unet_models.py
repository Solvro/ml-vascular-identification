"""
U-Net embedding models for vascular identification.

This module contains U-Net architectures adapted for embedding learning.
U-Net is typically used for dense prediction tasks, but here we adapt it
for feature extraction and embedding learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEmbeddingModel


class ConvBlock(nn.Module):
    """Double convolution block used in U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block with convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x


class UNetEmbedding(BaseEmbeddingModel):
    """U-Net architecture adapted for embedding learning.

    Uses the encoder part of U-Net for feature extraction,
    followed by global pooling and a projection head for embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        input_channels: int = 3,
        encoder_depth: int = 5,
        base_channels: int = 64,
    ):
        super().__init__(embedding_dim)

        self.encoder_depth = encoder_depth
        self.channels = [base_channels * (2**i) for i in range(encoder_depth)]

        # Encoder blocks
        self.encoders = nn.ModuleList()
        in_ch = input_channels

        for out_ch in self.channels[:-1]:
            self.encoders.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch

        # Bottom block (no pooling)
        self.bottom = ConvBlock(in_ch, self.channels[-1])

        # Global pooling and projection head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(self.channels[-1], self.channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.channels[-1] // 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        # Encoder path
        for encoder in self.encoders:
            x, _ = encoder(x)  # We don't need skip connections for embedding

        # Bottom
        x = self.bottom(x)

        # Global pooling and projection
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return F.normalize(x, p=2, dim=1)


class AttentionUNet(BaseEmbeddingModel):
    """U-Net with attention mechanism for embedding learning."""

    def __init__(
        self,
        embedding_dim: int = 512,
        input_channels: int = 3,
        encoder_depth: int = 4,
        base_channels: int = 64,
    ):
        super().__init__(embedding_dim)

        self.encoder_depth = encoder_depth
        self.channels = [base_channels * (2**i) for i in range(encoder_depth)]

        # Encoder blocks
        self.encoders = nn.ModuleList()
        in_ch = input_channels

        for out_ch in self.channels[:-1]:
            self.encoders.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch

        # Bottom block
        self.bottom = ConvBlock(in_ch, self.channels[-1])

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(self.channels[-1], self.channels[-1] // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[-1] // 4, 1, 1),
            nn.Sigmoid(),
        )

        # Global pooling and projection head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(self.channels[-1], embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        # Encoder path
        for encoder in self.encoders:
            x, _ = encoder(x)

        # Bottom with attention
        x = self.bottom(x)
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Global pooling and projection
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return F.normalize(x, p=2, dim=1)


if __name__ == "__main__":
    # Test the models
    print("🔍 Testing U-Net models...")

    # Test UNetEmbedding
    model = UNetEmbedding(embedding_dim=256, encoder_depth=4)
    print(f"📊 UNet model info: {model.get_model_info()}")

    # Test forward pass
    batch_size, channels, height, width = 4, 3, 256, 256
    x = torch.randn(batch_size, channels, height, width)
    embeddings = model(x)

    print(f"🖼️  Input shape: {x.shape}")
    print(f"🎯 Output shape: {embeddings.shape}")
    print(f"📏 Embedding norm: {embeddings.norm(dim=1).mean():.4f}")

    # Test AttentionUNet
    attention_model = AttentionUNet(embedding_dim=128)
    print(f"📊 Attention UNet model info: {attention_model.get_model_info()}")

    embeddings2 = attention_model(x)
    print(f"🎯 Attention UNet output shape: {embeddings2.shape}")
