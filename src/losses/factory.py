from .triplet import batch_hard_triplet

def get_loss(cfg):
    """Return loss function based on cfg.loss.type."""
    if cfg.loss.type == "triplet":
        def _loss(emb, labels):
            # Batch-hard triplet: hardest pos/neg within the batch.
            return batch_hard_triplet(emb, labels, margin=cfg.loss.margin)
        return _loss
    
    raise ValueError(cfg.loss.type)
