from omegaconf import DictConfig
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.datamodule import VascularDataModule
from src.models.embedding_module import LitEmbeddingModel

import hydra


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed, workers=True)

    dm = VascularDataModule(cfg); dm.setup()
    model = LitEmbeddingModel(cfg)

    logger = TensorBoardLogger("outputs/runs", name=f"{cfg.data.name}-{cfg.model.backbone}")

    ckpt_loss = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename=f"{cfg.data.name}-{cfg.model.backbone}" + "-loss-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    ckpt_r1 = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename=f"{cfg.data.name}-{cfg.model.backbone}" + "-r1-{epoch:02d}-{val/R@1:.4f}",
        monitor="val/R@1",
        mode="max",
        save_top_k=1,
        save_last=False,
    )

    trainer = pl.Trainer(
        accelerator="auto",
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
