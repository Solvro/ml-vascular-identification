from typing import Any
import torch
import torch.nn as nn

from ..base import BaseEmbeddingModel


class TorchvisionResNetEmbeddingModel(BaseEmbeddingModel):
    """Wrap torchvision ResNet backbones (default: resnet50) to produce embeddings.

    Supports changing `in_chans` and replacing the final `fc` with a Linear to `embed_dim`.
    """

    def __init__(
        self,
        variant: str = "resnet50",
        pretrained: bool = True,
        in_chans: int = 3,
        embed_dim: int = 256,
        normalize: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embed_dim)
        self.normalize = bool(normalize)

        try:
            import torchvision.models as tvm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("torchvision is required for TorchvisionResNetEmbeddingModel") from e

        # Select constructor and weights by variant
        variant = str(variant).lower()
        if variant == "resnet50":
            ctor = tvm.resnet50
            try:
                # torchvision >=0.13
                weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            except Exception:
                weights = "IMAGENET1K_V2" if pretrained else None
        elif variant == "resnet18":
            ctor = tvm.resnet18
            try:
                weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            except Exception:
                weights = "IMAGENET1K_V1" if pretrained else None
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        self.backbone = ctor(weights=weights)

        # Adjust first conv if in_chans != 3
        if in_chans != 3:
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(in_chans, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                if pretrained and old_conv.weight.shape[1] == 3:
                    # Adapt weights: average or repeat across channels
                    if in_chans < 3:
                        w = old_conv.weight[:, :in_chans, :, :].clone()
                        # simple average projection to fewer channels
                        scale = 3.0 / float(in_chans)
                        new_conv.weight.copy_(w * scale)
                    else:
                        # repeat weights across new channels and scale
                        reps = (in_chans + 2) // 3
                        w = old_conv.weight.repeat(1, reps, 1, 1)[:, :in_chans, :, :].clone()
                        scale = 3.0 / float(in_chans)
                        new_conv.weight.copy_(w * scale)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            self.backbone.conv1 = new_conv

        # Replace classifier head to output embed_dim
        in_features = self.backbone.fc.in_features
        head = [nn.Linear(in_features, self.embedding_dim)]
        if dropout and dropout > 0:
            head = [nn.Dropout(p=float(dropout)), *head]
        self.backbone.fc = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if self.normalize:
            z = nn.functional.normalize(z, p=2, dim=1)
        return z
