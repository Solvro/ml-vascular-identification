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
