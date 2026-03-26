"""
ResNet with CBAM (Convolutional Block Attention Module) models.

Combines ResNet backbone with channel and spatial attention for optimal
finger vein recognition and other vascular biometrics.
"""
from .resnet_cbam_models import ResNet50CBAM, ResNetCBAM

__all__ = [
    "ResNetCBAM",
    "ResNet50CBAM",
]
