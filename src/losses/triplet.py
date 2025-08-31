from __future__ import annotations
import torch
import torch.nn.functional as F


def pairwise_distances(x: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Compute pairwise Euclidean distances between rows of x.

    Args:
        x: (B, D) tensor.
        squared: if True, return squared distances.
    Returns:
        (B, B) distance matrix.
    """
    # x2: (B, 1), y2 (1, B), xy: (B, B)
    x_norm = (x * x).sum(dim=1, keepdim=True)
    d2 = x_norm + x_norm.t() - 2.0 * (x @ x.t())
    d2 = torch.clamp(d2, min=0.0)
    if squared:
        return d2
    # Numerical stability: sqrt only on positive
    mask = (d2 == 0.0).to(x.dtype)
    d = d2 + mask * 1e-16
    d = torch.sqrt(d)
    d *= (1.0 - mask)
    return d


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Batch-hard triplet loss (Schroff et al.).

    For each anchor, selects the hardest positive and hardest negative within the batch.
    Assumes labels are 1D with class indices, and emb shape is (B, D).
    """
    if emb.ndim != 2:
        raise ValueError("emb must be 2D (B, D)")
    if labels.ndim != 1 or labels.shape[0] != emb.shape[0]:
        raise ValueError("labels must be 1D with same B as emb")

    # Optionally normalize to make metric consistent
    emb = F.normalize(emb, p=2, dim=1)
    d = pairwise_distances(emb, squared=False)  # (B, B)

    labels = labels.view(-1, 1)
    eq = torch.eq(labels, labels.t())  # (B, B)

    # For positives, ignore self (diagonal) by setting diag to False
    pos_mask = eq & (~torch.eye(eq.shape[0], dtype=torch.bool, device=eq.device))
    neg_mask = ~eq

    # Hardest positive: max distance among positives; if no positive exists, distance=0
    d_pos = d.clone()
    d_pos[~pos_mask] = -1e9
    hardest_pos, _ = d_pos.max(dim=1)
    hardest_pos[hardest_pos < 0] = 0.0  # no positives edge-case

    # Hardest negative: min distance among negatives; if none, set large
    d_neg = d.clone()
    d_neg[~neg_mask] = 1e9
    hardest_neg, _ = d_neg.min(dim=1)

    loss = F.relu(margin + hardest_pos - hardest_neg)
    return loss.mean()
import torch
import torch.nn.functional as F

def pairwise_dist(x):
    """Return NxN matrix of squared Euclidean distances between rows of x."""
    xx = (x * x).sum(dim=1, keepdim=True)   # ||x_i||^2
    dist = xx + xx.t() - 2.0 * (x @ x.t())  # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    return torch.clamp(dist, min=0.0)   # numerical floor at 0

def batch_hard_triplet(emb, labels, margin=0.2):
    """Batch-hard triplet loss (squared Euclidean).

    Args:
        emb: (N, D) embeddings.
        labels: (N,) int class labels.
        margin: triplet margin.

    Returns:
        Scalar loss tensor.
    """
    d = pairwise_dist(emb)
    N = d.size(0)
    
    labels = labels.unsqueeze(1)    # (N,1) for broadcasting
    mask_pos = (labels == labels.t()) & (~torch.eye(N, dtype=torch.bool, device=emb.device))    # same class, exclude self
    mask_neg = labels != labels.t() # different classes
    
    pos = d.clone(); pos[~mask_pos] = -1e6  # so max() picks only valid positives
    neg = d.clone(); neg[~mask_neg] = 1e6   # so min() picks only valid negatives
    
    hardest_pos = pos.max(dim=1)[0]
    hardest_neg = neg.min(dim=1)[0]
    
    loss = F.relu(hardest_pos - hardest_neg + margin).mean()
    return loss
