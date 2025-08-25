from typing import Callable, Dict, Optional

from .base import BaseEmbeddingModel
from .cnn.simple import SimpleCNNEmbeddingModel
from .cnn.resnet import TorchvisionResNetEmbeddingModel


_REGISTRY: Dict[str, Callable[..., BaseEmbeddingModel]] = {}


def register_model(name: str):
	"""Decorator to register a model builder under a name."""
	def _wrap(fn: Callable[..., BaseEmbeddingModel]):
		_REGISTRY[name] = fn
		return fn
	return _wrap


# Note: timm-based models are not registered here to keep dependencies minimal.


def create_model(name: str, **kwargs) -> BaseEmbeddingModel:
	"""Create a model by name, using the registry.

	Example:
		create_model("timm", backbone="resnet50", embed_dim=256)
	"""
	if name not in _REGISTRY:
		raise KeyError(f"Unknown model name '{name}'. Available: {sorted(_REGISTRY.keys())}")
	return _REGISTRY[name](**kwargs)


@register_model("simple_cnn")
def build_simple_cnn(**kwargs) -> BaseEmbeddingModel:
	return SimpleCNNEmbeddingModel(**kwargs)


@register_model("resnet50")
def build_resnet50(**kwargs) -> BaseEmbeddingModel:
	return TorchvisionResNetEmbeddingModel(variant="resnet50", **kwargs)


@register_model("resnet18")
def build_resnet18(**kwargs) -> BaseEmbeddingModel:
	return TorchvisionResNetEmbeddingModel(variant="resnet18", **kwargs)
