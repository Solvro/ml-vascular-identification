"""
ResNet-50 with CBAM (Convolutional Block Attention Module) for vascular biometrics.

CBAM combines:
- Channel Attention: learns which feature maps are important
- Spatial Attention: learns which spatial regions have discriminative patterns

Optimal for finger vein recognition where both vessel features and local bifurcations matter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEmbeddingModel


class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).
    
    Sequential application of channel and spatial attention.
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    """

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class BasicBlock(nn.Module):
    """ResNet basic block (for shallow networks)."""

    expansion = 1

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, downsample=None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet bottleneck block (for deeper networks like ResNet-50)."""

    expansion = 4

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, downsample=None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetCBAM(BaseEmbeddingModel):
    """ResNet with CBAM (Channel and Spatial Attention) for vascular biometrics.
    
    Combines ResNet backbone with CBAM attention for optimal finger vein recognition:
    - Channel attention: selects vessel-relevant feature maps
    - Spatial attention: highlights bifurcations and local landmarks
    
    Supports ResNet-18, 34, 50, 101, 152.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        depth: int = 50,
        input_channels: int = 3,
        dropout: float = 0.1,
        use_cbam: bool = True,
    ):
        """Initialize ResNet with CBAM.
        
        Args:
            embedding_dim: Output embedding dimension (default 256).
            depth: ResNet depth - 18, 34, 50, 101, 152 (default 50).
            input_channels: Number of input channels (default 3 for RGB).
            dropout: Dropout probability (default 0.1).
            use_cbam: Whether to use CBAM attention (default True).
        """
        super().__init__(embedding_dim)
        self.depth = depth
        self.use_cbam = use_cbam

        # ResNet configuration
        if depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
            self.base_channels = 64
        elif depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
            self.base_channels = 64
        elif depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
            self.base_channels = 64
        elif depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
            self.base_channels = 64
        elif depth == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
            self.base_channels = 64
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        self.in_channels = self.base_channels

        # Initial convolution
        self.conv1 = nn.Conv2d(
            input_channels, self.base_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, self.base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, self.base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base_channels * 8, layers[3], stride=2)

        # CBAM attention blocks - dynamically apply to final layer
        if self.use_cbam:
            self.cbam_final = CBAM(self.base_channels * 8 * block.expansion)

        # Global pooling and embedding head
        final_channels = self.base_channels * 8 * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_head = nn.Sequential(
            nn.Linear(final_channels, final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_channels // 2, embedding_dim),
        )

    def _make_layer(self, block, channels: int, blocks: int, stride: int = 1):
        """Build a residual layer."""
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with CBAM attention.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            L2-normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply CBAM attention to final layer
        if self.use_cbam:
            x = self.cbam_final(x)

        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Embedding head
        x = self.embedding_head(x)

        # L2 normalization
        return F.normalize(x, p=2, dim=1)


class ResNet50CBAM(ResNetCBAM):
    """ResNet-50 with CBAM - optimal for vascular biometrics."""

    def __init__(
        self,
        embedding_dim: int = 256,
        input_channels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            depth=50,
            input_channels=input_channels,
            dropout=dropout,
            use_cbam=True,
        )
