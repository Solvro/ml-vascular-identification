import importlib
import torch

from src.models.model_factory import create_model


def test_simple_cnn_factory_forward():
	model = create_model(
		"simple_cnn",
		in_chans=3,
		embed_dim=64,
		width=16,
		normalize=True,
	)
	x = torch.randn(2, 3, 128, 128)
	with torch.inference_mode():
		z = model(x)
	assert z.shape == (2, 64)


def test_resnet50_factory_forward():
	# Skip if torchvision is not available
	import importlib
	if importlib.util.find_spec("torchvision") is None:
		return
	model = create_model(
		"resnet50",
		pretrained=False,
		in_chans=3,
		embed_dim=16,
		normalize=True,
	)
	x = torch.randn(2, 3, 224, 224)
	with torch.inference_mode():
		z = model(x)
	assert z.shape == (2, 16)