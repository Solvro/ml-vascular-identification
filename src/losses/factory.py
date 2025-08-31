from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch

from .triplet import batch_hard_triplet_loss


@dataclass
class LossWrapper:
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    def __call__(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.fn(emb, labels)


def get_loss(cfg) -> LossWrapper:
    """Return a loss function wrapper based on config.

    Expects cfg.loss.type == "triplet" and cfg.loss.mining == "batch_hard".
    """
    typ = str(cfg.loss.type)
    if typ == "triplet":
        mining = str(cfg.loss.mining)
        margin = float(cfg.loss.margin)
        if mining == "batch_hard":
            return LossWrapper(lambda emb, y: batch_hard_triplet_loss(emb, y, margin=margin))
        else:
            raise ValueError(f"Unsupported triplet mining: {mining}")
    raise ValueError(f"Unsupported loss type: {typ}")
from .triplet import batch_hard_triplet

def get_loss(cfg):
    """Return loss function based on cfg.loss.type."""
    if cfg.loss.type == "triplet":
        def _loss(emb, labels):
            # Batch-hard triplet: hardest pos/neg within the batch.
            return batch_hard_triplet(emb, labels, margin=cfg.loss.margin)
        return _loss
    
    raise ValueError(cfg.loss.type)
