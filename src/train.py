import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import hydra
from lightning.pytorch.loggers import TensorBoardLogger

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

    def forward(self, x): return self.net(x)
    """Forward to embeddings."""

    def training_step(self, batch, _):
        """Compute train loss and log it."""
        x, y = batch
        z = self.net(x)
        loss = self.loss_fn(z, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        """Compute val loss and log it."""
        x, y = batch
        z = self.net(x)
        loss = self.loss_fn(z, y)
        self.log("val/loss", loss, prog_bar=True)

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
    """Set seed, build datamodule/model, and train."""
    pl.seed_everything(cfg.train.seed, workers=True)
    dm = VascularDataModule(cfg)
    model = LitMetric(cfg)
    logger = TensorBoardLogger("outputs/runs", name=f"{cfg.data.name}-{cfg.model.backbone}")
    
    ckpt_cb = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename=f"{cfg.data.name}-{cfg.model.backbone}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    
    trainer = pl.Trainer(
        # For quick smoke tests, you can limit batches:
        # limit_train_batches=5, limit_val_batches=1,
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        logger=logger,
        callbacks=[ckpt_cb],
        default_root_dir="outputs/checkpoints",
        log_every_n_steps=10
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
