from typing import Any, Dict, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torchmetrics import MeanMetric
import lightning as L



class EmbeddingLitModule(L.LightningModule):
    """
    LightningModule for embedding tasks using triplet loss.
    """

    def __init__(
        self,
        net: Module,
        optimizer: Optimizer,
        scheduler: Any = None,
        compile: bool = False,
    ) -> None:
        """
        Args:
            net (Module): The embedding model.
            optimizer (Optimizer): Optimizer for training.
            scheduler (Any, optional): Learning rate scheduler. Defaults to None.
            compile (bool, optional): Whether to use torch.compile. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0)

        # Track losses
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to get embedding.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Embedding tensor.
        """
        return self.net.get_embedding(x)

    def on_train_start(self) -> None:
        """Reset validation loss at the start of training."""
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Performs a step for triplet loss.
        Args:
            batch (Tuple[Tensor, Tensor, Tensor]): (anchor, positive, negative)
        Returns:
            Tuple containing loss and embeddings for anchor, positive, negative.
        """
        anchor, positive, negative = batch
        anchor_emb = self.net.get_embedding(anchor)
        positive_emb = self.net.get_embedding(positive)
        negative_emb = self.net.get_embedding(negative)

        loss = self.criterion(anchor_emb, positive_emb, negative_emb)
        return loss, anchor_emb, positive_emb, negative_emb

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for triplet loss.
        """
        loss, *_ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """
        Validation step for triplet loss.
        """
        loss, *_ = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Log validation loss at the end of the epoch.
        """
        self.log("val/loss_epoch", self.val_loss.compute(), prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """
        Test step for triplet loss.
        """
        loss, *_ = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """
        Log test loss at the end of the epoch.
        """
        self.log("test/loss_epoch", self.test_loss.compute(), prog_bar=True)

    def setup(self, stage: str) -> None:
        """
        Setup model for training or evaluation.
        Args:
            stage (str): Stage of training ('fit', 'validate', etc.).
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        Returns:
            Dict[str, Any]: Optimizer and scheduler configuration.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # Example usage
    _ = EmbeddingLitModule(None, None, None, None)
