"""
Basic CNN architectures for embedding learning in OpenSet recognition.

This module contains concrete implementations of embedding models
based on simple CNN architectures optimized for vascular finger vein identification.

Architecture optimized for:
- 512-dimensional embeddings (increased from 256 for better discrimination)
- L2-normalization for metric learning
- P=16, K=4 batch sampling (64 images per batch)
- Input size: 224x224 RGB images
- Residual connections for better gradient flow
- Attention mechanism for feature refinement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEmbeddingModel


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention module to emphasize important feature channels."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * out


class SimpleEmbeddingModel(BaseEmbeddingModel):
    """Improved CNN for embedding learning with residual connections and attention.

    Enhanced convolutional neural network with:
    - Residual connections for better gradient flow and deeper networks
    - Channel attention for feature refinement
    - 512-dimensional embeddings (default, configurable)
    - L2-normalization for metric learning
    
    Optimized for OpenSet recognition with:
    - Better discriminative power for 420+ classes
    - Residual blocks to prevent degradation
    - Attention mechanism to focus on vascular patterns
    - Compatible with P=16, K=4 batch sampling
    
    Architecture:
        Input: (batch, 3, 224, 224)
        Initial conv: 3 → 64
        Residual stage 1: 64 → 128 (×2 blocks)
        Residual stage 2: 128 → 256 (×2 blocks)
        Residual stage 3: 256 → 512 (×2 blocks)
        Residual stage 4: 512 → 1024 (×2 blocks)
        Channel attention
        Global average pooling
        Projection head: 1024 → embedding_dim
        L2-normalization
        Output: (batch, embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int = 512,  # Increased from 256
        input_channels: int = 3,
        dropout: float = 0.3,  # Increased from 0.0
        use_attention: bool = True,
    ):
        """Initialize ImprovedEmbeddingModel.
        
        Args:
            embedding_dim: Dimension of output embeddings (default 512, increased from 256).
            input_channels: Number of input channels (default 3 for RGB).
            dropout: Dropout rate before final projection (default 0.3).
            use_attention: Whether to use channel attention (default True).
        """
        super().__init__(embedding_dim)
        self.input_channels = input_channels
        self.dropout = dropout
        self.use_attention = use_attention

        # Initial convolution: 224x224 -> 112x112
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 112x112 -> 56x56
        )
        
        # Residual stages with increasing depth
        # Stage 1: 56x56, 64 → 128
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
        )
        
        # Stage 2: 56x56 → 28x28, 128 → 256
        self.stage2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        )
        
        # Stage 3: 28x28 → 14x14, 256 → 512
        self.stage3 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
        )
        
        # Stage 4: 14x14 → 7x7, 512 → 1024
        self.stage4 = nn.Sequential(
            ResidualBlock(512, 1024, stride=2),
            ResidualBlock(1024, 1024, stride=1),
        )
        
        # Channel attention (optional but recommended)
        self.attention = ChannelAttention(1024, reduction=16) if use_attention else nn.Identity()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)

        # Projection head with bottleneck: 1024 → 512 → embedding_dim
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),  # Lighter dropout in bottleneck
            nn.Linear(512, embedding_dim, bias=False)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns L2-normalized embeddings.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
               Expected: (batch_size, 3, 224, 224)

        Returns:
            L2-normalized embedding tensor of shape (batch_size, embedding_dim).
            Values are in range [-1, 1] with L2 norm = 1.
            
        Note:
            Residual connections enable deeper networks without degradation.
            Attention mechanism helps focus on discriminative vascular patterns.
        """
        # Initial convolution
        x = self.conv1(x)  # (batch, 64, 56, 56)
        
        # Residual stages
        x = self.stage1(x)  # (batch, 128, 56, 56)
        x = self.stage2(x)  # (batch, 256, 28, 28)
        x = self.stage3(x)  # (batch, 512, 14, 14)
        x = self.stage4(x)  # (batch, 1024, 7, 7)
        
        # Apply channel attention
        x = self.attention(x)  # (batch, 1024, 7, 7)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, 1024, 1, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 1024)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Project to embedding dimension
        x = self.head(x)  # (batch, embedding_dim)
        
        # L2 normalization: critical for metric learning
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def get_model_info(self):
        """Get detailed model information."""
        info = super().get_model_info()
        info['input_channels'] = self.input_channels
        info['dropout'] = self.dropout
        info['use_attention'] = self.use_attention
        info['architecture'] = 'ImprovedCNN-1024-ResNet-Attention'
        return info
