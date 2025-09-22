"""
Vision Transformer models for vascular identification.

This module contains Vision Transformer (ViT) architectures adapted for embedding learning.
ViTs process images as sequences of patches and use self-attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEmbeddingModel


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""

    def __init__(
        self,
        input_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        image_size: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input image to patch embeddings.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim).
        """
        # x: (B, C, H, W) -> (B, embed_dim, H//P, W//P)
        x = self.projection(x)
        # Flatten spatial dimensions: (B, embed_dim, H//P, W//P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEmbedding(BaseEmbeddingModel):
    """Vision Transformer for embedding learning.

    Processes images as sequences of patches using transformer architecture.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        input_channels: int = 3,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 3072,
        image_size: int = 224,
        dropout: float = 0.1,
    ):
        super().__init__(embedding_dim)

        self.patch_embedding = PatchEmbedding(
            input_channels, patch_size, embedding_dim, image_size
        )

        num_patches = self.patch_embedding.num_patches

        # Class token and positional embeddings
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Vision Transformer.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        B = x.shape[0]

        # Convert to patch embeddings
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)

        # Add class token
        class_tokens = self.class_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([class_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer:
            x = transformer_block(x)

        x = self.norm(x)

        # Use class token for final embedding
        cls_token = x[:, 0]  # (B, embed_dim)

        return F.normalize(cls_token, p=2, dim=1)


class DeiTEmbedding(VisionTransformerEmbedding):
    """Data-efficient Image Transformer (DeiT) for embedding learning.

    A more efficient version of ViT with knowledge distillation capabilities.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        input_channels: int = 3,
        patch_size: int = 16,
        num_layers: int = 6,
        num_heads: int = 6,
        hidden_dim: int = 1536,
        image_size: int = 224,
        dropout: float = 0.1,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            image_size=image_size,
            dropout=dropout,
        )
