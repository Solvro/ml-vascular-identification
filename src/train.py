import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import hydra
from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn.functional as F

from data.datamodule import VascularDataModule
from models.factory import get_backbone
from models.embedding_module import EmbeddingModule
from losses.factory import get_loss


class LitMetric(pl.LightningModule):
    """LightningModule: backbone + embedding head + metric loss."""
    
    def __init__(self, cfg: DictConfig):
        """Save cfg, build network and loss."""
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        bb, in_dim = get_backbone(cfg.model.backbone, cfg.model.pretrained, in_chans=3)
        self.net = EmbeddingModule(bb, in_dim, cfg.model.embed_dim,
                                   bn=cfg.model.neck.bn, dropout=cfg.model.neck.dropout,
                                   normalize=cfg.model.normalize)
        self.loss_fn = get_loss(cfg)
        self.cfg = cfg

        # R@1/R@5 validation
        self._val_embeds = []
        self._val_labels = []
        self._ks = tuple(getattr(getattr(cfg, "eval", {}), "recall_at", (1, 5)))


    def forward(self, x): return self.net(x)
    """Forward to embeddings."""

    def training_step(self, batch, _):
        """Compute train loss and log it."""
        x, y = batch
        z = self.net(x)
        loss = self.loss_fn(z, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    # R@1/R@5 validation
    def on_validation_epoch_start(self):
        self._val_embeds = []
        self._val_labels = []


    def validation_step(self, batch, _):
        """Compute val loss and log it."""
        x, y = batch
        z = self.net(x)
        loss = self.loss_fn(z, y)
        self.log("val/loss", loss, prog_bar=True)
        
        self.log("val_loss", loss, prog_bar=True) 
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
        max_k = int(max(self._ks))
        topk = sim.topk(max_k, dim=1).indices
        correct = (lab[topk] == lab.unsqueeze(1))
        for k in self._ks:
            r = correct[:, :int(k)].any(dim=1).float().mean().item()
            if k == 1: self.log("val_R1", r, prog_bar=True)
            elif k == 5: self.log("val_R5", r, prog_bar=True)
            else: self.log(f"val_R{int(k)}", r, prog_bar=True)


    def configure_optimizers(self):
        """Create optimizer (and optional LR scheduler)."""
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.cfg.train.optimizer.lr,
                                weight_decay=self.cfg.train.optimizer.weight_decay)
        name = str(self.cfg.train.scheduler.name)
        if name == "cosine":
            # Default T_max to max_epochs if not provided.
            t_max = int(getattr(self.cfg.train.scheduler, "t_max", self.cfg.train.max_epochs))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
        return {"optimizer": opt}

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed, workers=True)

    dm = VascularDataModule(cfg)
    model = LitMetric(cfg)

    logger = TensorBoardLogger("outputs/runs", name=f"{cfg.data.name}-{cfg.model.backbone}")

    ckpt_loss = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename=f"{cfg.data.name}-{cfg.model.backbone}" + "-loss-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    ckpt_r1 = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename=f"{cfg.data.name}-{cfg.model.backbone}" + "-r1-{epoch:02d}-{val_R1:.4f}",
        monitor="val_R1",
        mode="max",
        save_top_k=1,
        save_last=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        logger=logger,
        callbacks=[ckpt_loss, ckpt_r1],
        default_root_dir="outputs/checkpoints",
        log_every_n_steps=1,
        num_sanity_val_steps=1,
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
