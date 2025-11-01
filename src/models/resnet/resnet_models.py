"""
ResNet embedding models for vascular identification.

This module contains ResNet architectures adapted for embedding learning.
ResNet uses residual connections for better gradient flow and deeper networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEmbeddingModel


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ResNetEmbedding(BaseEmbeddingModel):
    """ResNet for embedding learning.

    A residual network that extracts features from input images
    and produces normalized embedding vectors for similarity learning.
    """

    def __init__(self, embedding_dim: int = 256, input_channels: int = 3, num_blocks: int = 16):
        super().__init__(embedding_dim)
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        # num_blocks should be divisible by 3 (for 3 stages)
        blocks_per_stage = num_blocks // 3
        
        self.layer1 = self._make_layer(64, 64, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks_per_stage, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding head
        self.head = nn.Linear(256, embedding_dim)
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns L2-normalized embeddings.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        # Initial layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling and embedding
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.head(x)
        
        return F.normalize(x, p=2, dim=1)  # L2 normalization


class ResNet50Embedding(BaseEmbeddingModel):
    """ResNet-50 adapted for embedding learning."""

    def __init__(self, embedding_dim: int = 256, input_channels: int = 3):
        super().__init__(embedding_dim)
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers (ResNet50 has 3,4,6,3 blocks)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding head
        self.head = nn.Linear(256, embedding_dim)
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns L2-normalized embeddings."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        
        return F.normalize(x, p=2, dim=1)


if __name__ == "__main__":
    # Test the models
    print("🔍 Testing ResNet models...")

    # Test ResNet16
    model = ResNetEmbedding(embedding_dim=256, num_blocks=16)
    print(f"📊 ResNet16 model info: {model.get_model_info()}")

    # Test forward pass
    batch_size, channels, height, width = 4, 3, 256, 256
    x = torch.randn(batch_size, channels, height, width)
    embeddings = model(x)

    print(f"🖼️  Input shape: {x.shape}")
    print(f"🎯 Output shape: {embeddings.shape}")
    print(f"📏 Embedding norm: {embeddings.norm(dim=1).mean():.4f}")

    # Test ResNet50
    model50 = ResNet50Embedding(embedding_dim=256)
    print(f"\n📊 ResNet50 model info: {model50.get_model_info()}")
    embeddings50 = model50(x)
    print(f"🎯 ResNet50 output shape: {embeddings50.shape}")
