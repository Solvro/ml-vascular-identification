import torch
import torch.nn.functional as F

from .base import BaseLoss


class TripletLoss(BaseLoss):
    """Triplet loss with hard negative mining.

    Implements triplet loss using batch hard negative mining strategy.
    For each anchor, finds the hardest positive (most distant same-class sample)
    and hardest negative (closest different-class sample) within the batch.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute batch hard triplet loss.

        Args:
            embeddings: L2-normalized embedding vectors of shape (batch_size, embedding_dim).
            labels: Target labels of shape (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        # Compute pairwise L2 distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()

        # Remove diagonal (distance to self)
        pos_mask.fill_diagonal_(0)

        # Hard positive and hard negative mining
        # For positives: find the maximum distance (hardest positive)
        pos_dist = distances * pos_mask
        # Add large value to positive pairs when computing hard negatives
        neg_dist = distances * neg_mask + 1e6 * pos_mask

        # Get hardest positive and hardest negative for each sample
        hard_pos = pos_dist.max(dim=1)[0]
        hard_neg = neg_dist.min(dim=1)[0]

        # Triplet loss: max(0, hard_positive - hard_negative + margin)
        loss = F.relu(hard_pos - hard_neg + self.margin)
        return loss.mean()

    def get_loss_info(self):
        """Get loss function information."""
        info = super().get_loss_info()
        info["margin"] = self.margin
        return info


class ContrastiveLoss(BaseLoss):
    """Contrastive loss for pair-wise learning.

    Alternative loss function that works with pairs of samples,
    pushing similar samples together and dissimilar samples apart.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using all pairs in the batch.

        Args:
            embeddings: L2-normalized embedding vectors of shape (batch_size, embedding_dim).
            labels: Target labels of shape (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create labels for pairs
        labels = labels.unsqueeze(0)
        same_class = (labels == labels.t()).float()

        # Remove diagonal
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
        distances = distances[mask]
        same_class = same_class[mask]

        # Contrastive loss
        pos_loss = same_class * distances.pow(2)
        neg_loss = (1 - same_class) * F.relu(self.margin - distances).pow(2)

        return (pos_loss + neg_loss).mean()
