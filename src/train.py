from omegaconf import DictConfig
import lightning.pytorch as pl
from src.data.datamodule import VascularDataModule

import hydra


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed, workers=True)

    dm = VascularDataModule(cfg)


if __name__ == "__main__":
    main()
