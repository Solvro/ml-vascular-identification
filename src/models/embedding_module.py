import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingModule(nn.Module):
    """Backbone + linear head to produce embeddings."""
    
    def __init__(self, backbone, in_dim, embed_dim=256, bn=True, dropout=0.0, normalize=True):
        """Configure head and normalization."""
        super().__init__()
        head = [nn.Linear(in_dim, embed_dim, bias=not bn)]
        if bn: head.append(nn.BatchNorm1d(embed_dim))
        if dropout and dropout > 0: head.append(nn.Dropout(dropout))
        
        self.backbone = backbone
        self.head = nn.Sequential(*head)
        self.normalize = normalize

    def forward(self, x):
        """Return embeddings (L2-normalized if enabled)."""
        z = self.backbone(x)
        if isinstance(z, (list, tuple)): z = z[-1]  # handle backbones returning (feat, ...)
        z = self.head(z)
        if self.normalize:
            z = F.normalize(z, p=2, dim=1)
        return z
