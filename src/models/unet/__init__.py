"""
U-Net models package.

Contains U-Net architectures for dense prediction tasks
and adaptations for embedding learning.
"""
from .unet_models import AttentionUNet, UNetEmbedding

__all__ = [
    "UNetEmbedding",
    "AttentionUNet",
]
