from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEmbeddingModel


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SimpleCNNEmbeddingModel(BaseEmbeddingModel):
    """A compact CNN that outputs a fixed-size embedding.

    Architecture: stem -> 3 conv stages with downsampling -> GAP -> Linear -> (optional L2 norm)
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 256,
        width: int = 32,
        normalize: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embed_dim)
        self.normalize = bool(normalize)

        c1, c2, c3, c4 = width, width * 2, width * 4, width * 4
        self.stem = ConvBNReLU(in_chans, c1, k=7, s=2, p=3)
        self.layer1 = nn.Sequential(ConvBNReLU(c1, c1), ConvBNReLU(c1, c1))
        self.down1 = ConvBNReLU(c1, c2, s=2)
        self.layer2 = nn.Sequential(ConvBNReLU(c2, c2), ConvBNReLU(c2, c2))
        self.down2 = ConvBNReLU(c2, c3, s=2)
        self.layer3 = nn.Sequential(ConvBNReLU(c3, c3), ConvBNReLU(c3, c4))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.head = nn.Linear(c4, self.embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        z = self.head(x)
        if self.normalize:
            z = F.normalize(z, p=2, dim=1)
        return z
