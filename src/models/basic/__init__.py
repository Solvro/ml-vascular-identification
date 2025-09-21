"""
Basic CNN models package.

Contains simple convolutional neural network architectures
for embedding learning in vascular identification.
"""
from .basic_cnn import MODEL_CONFIGS, SimpleEmbeddingModel

__all__ = [
    "SimpleEmbeddingModel",
    "MODEL_CONFIGS",
]
