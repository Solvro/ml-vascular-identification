from __future__ import annotations
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from .model_factory import create_model
from src.losses.factory import get_loss


class LitEmbeddingModel(pl.LightningModule):
    """LightningModule: backbone + embedding head + metric loss.

    Uses the model factory to build the embedding network and a loss from the loss factory.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])  # keep cfg outside hparams serialization
        self.cfg = cfg

        # Build backbone model via factory
        # From conf/model/<name>.yaml we expect keys in cfg.model
        model_name = str(cfg.model.backbone) if hasattr(cfg.model, "backbone") else str(cfg.model.name)
        # Map common names to our registry if needed
        name_map = {"resnet50": "resnet50", "resnet18": "resnet18", "simple": "simple_cnn", "simple_cnn": "simple_cnn"}
        reg_name = name_map.get(model_name, model_name)

        kwargs = dict(
            in_chans=3,
            embed_dim=int(getattr(cfg.model, "embed_dim", 256)),
            normalize=bool(getattr(cfg.model, "normalize", True)),
            dropout=float(getattr(getattr(cfg.model, "neck", {}), "dropout", 0.0)),
            pretrained=bool(getattr(cfg.model, "pretrained", True)),
        )
        self.net = create_model(reg_name, **kwargs)

        # Loss
        self.loss_fn = get_loss(cfg)

        # Validation buffers for simple recall@k over batch
        self._val_embeds = []
        self._val_labels = []
        self._ks = (1, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_embeds = []
        self._val_labels = []

    def validation_step(self, batch, _):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, y)
        self.log("val/loss", loss, prog_bar=True)
        self._val_embeds.append(z.detach().float().cpu())
        self._val_labels.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_embeds:
            return
        emb = torch.cat(self._val_embeds, 0)
        lab = torch.cat(self._val_labels, 0)
        emb = F.normalize(emb, p=2, dim=1)
        sim = emb @ emb.t()
        sim.fill_diagonal_(-1e9)
        max_k = 5
        topk = sim.topk(max_k, dim=1).indices
        correct = (lab[topk] == lab.unsqueeze(1))
        for k in self._ks:
            r = correct[:, :k].any(dim=1).float().mean().item()
            self.log(f"val/R@{k}", r, prog_bar=True)

    def configure_optimizers(self):
        opt_name = str(getattr(self.cfg.train.optimizer, "name", "adamw")).lower()
        lr = float(getattr(self.cfg.train.optimizer, "lr", 3e-4))
        wd = float(getattr(self.cfg.train.optimizer, "weight_decay", 1e-4))

        if opt_name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        sch_name = str(getattr(self.cfg.train.scheduler, "name", "cosine")).lower()
        if sch_name == "cosine":
            t_max = int(getattr(self.cfg.train.scheduler, "t_max", self.cfg.train.max_epochs))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

        return {"optimizer": opt}
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