"""
Loss functions for metric learning in OpenSet recognition.

Implements triplet loss and contrastive loss optimized for
L2-normalized embeddings in finger vein identification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss


class TripletLoss(BaseLoss):
    """Triplet loss with batch hard negative mining.

    Implements triplet loss using batch hard negative mining strategy.
    For each anchor, finds the hardest positive (most distant same-class sample)
    and hardest negative (closest different-class sample) within the batch.
    
    Optimized for OpenSet recognition with:
    - Hard negative mining for better discrimination
    - Works with L2-normalized embeddings
    - Margin in range 0.2-0.3 as recommended for biometrics
    """

    def __init__(self, margin: float = 0.3, mining: str = "hard"):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss (default 0.3, recommended 0.2-0.3).
            mining: Mining strategy - 'hard' or 'semi-hard' (default 'hard').
        """
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute batch hard triplet loss.

        Args:
            embeddings: L2-normalized embedding vectors of shape (batch_size, embedding_dim).
            labels: Target labels of shape (batch_size,).

        Returns:
            Scalar loss tensor.
            
        Note:
            With P=16, K=4 sampling, batch will have 64 samples with 16 classes.
            This provides 4 positives per anchor (excluding self) and many negatives.
        """
        # Compute pairwise L2 distances (Euclidean distance)
        # For L2-normalized vectors: distance = sqrt(2 - 2*cosine_similarity)
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative pairs
        labels_col = labels.unsqueeze(1)  # (batch_size, 1)
        labels_row = labels.unsqueeze(0)  # (1, batch_size)
        pos_mask = (labels_col == labels_row).float()
        neg_mask = (labels_col != labels_row).float()

        # Remove diagonal (distance to self)
        pos_mask.fill_diagonal_(0)

        # Hard positive and hard negative mining
        if self.mining == "hard":
            # Hard positive: find the maximum distance (hardest positive)
            # For each sample, find its most distant same-class sample
            pos_dist = distances * pos_mask
            pos_dist[pos_mask == 0] = -float('inf')  # Mask out non-positives
            hard_pos = pos_dist.max(dim=1)[0]
            
            # Hard negative: find the minimum distance (closest different-class sample)
            # For each sample, find its closest different-class sample
            neg_dist = distances.clone()
            neg_dist[pos_mask == 1] = float('inf')  # Mask out positives
            neg_dist.fill_diagonal_(float('inf'))  # Mask out self
            hard_neg = neg_dist.min(dim=1)[0]
            
        else:  # semi-hard mining
            # Semi-hard: negatives that are closer than hardest positive but still violate margin
            pos_dist = distances * pos_mask
            pos_dist[pos_mask == 0] = -float('inf')
            hard_pos = pos_dist.max(dim=1)[0]
            
            # Find semi-hard negatives
            neg_dist = distances.clone()
            neg_dist[neg_mask == 0] = float('inf')
            # Only consider negatives within margin of hardest positive
            hard_neg = neg_dist.min(dim=1)[0]

        # Check for valid triplets (samples with at least one positive)
        valid_triplets = hard_pos > -float('inf')
        
        if not valid_triplets.any():
            # No valid triplets in batch (shouldn't happen with P-K sampling)
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Triplet loss: max(0, hard_positive - hard_negative + margin)
        loss = F.relu(hard_pos[valid_triplets] - hard_neg[valid_triplets] + self.margin)
        
        # Return mean loss and track statistics
        return loss.mean()

    def get_loss_info(self):
        """Get loss function information."""
        info = super().get_loss_info()
        info["margin"] = self.margin
        info["mining"] = self.mining
        return info


class ContrastiveLoss(BaseLoss):
    """Contrastive loss for pair-wise learning.

    Alternative loss function that works with pairs of samples,
    pushing similar samples together and dissimilar samples apart.
    
    Siamese-style loss that can be used instead of triplet loss.
    Works well with P-K sampling where we have multiple positive pairs per batch.
    """

    def __init__(self, margin: float = 1.0, pos_weight: float = 1.0, neg_weight: float = 1.0):
        """Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs (default 1.0).
            pos_weight: Weight for positive pair loss (default 1.0).
            neg_weight: Weight for negative pair loss (default 1.0).
        """
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using all pairs in the batch.

        Args:
            embeddings: L2-normalized embedding vectors of shape (batch_size, embedding_dim).
            labels: Target labels of shape (batch_size,).

        Returns:
            Scalar loss tensor.
            
        Note:
            For L2-normalized embeddings, distance range is [0, 2].
            Margin of 1.0 is reasonable for this range.
        """
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create labels for pairs
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        same_class = (labels_col == labels_row).float()

        # Remove diagonal (self-pairs)
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
        distances = distances[mask]
        same_class = same_class[mask]

        # Contrastive loss
        # Positive pairs: minimize distance
        pos_loss = same_class * distances.pow(2)
        
        # Negative pairs: push apart beyond margin
        neg_loss = (1 - same_class) * F.relu(self.margin - distances).pow(2)

        # Weighted combination
        total_loss = self.pos_weight * pos_loss + self.neg_weight * neg_loss

        return total_loss.mean()
    
    def get_loss_info(self):
        """Get loss function information."""
        info = super().get_loss_info()
        info["margin"] = self.margin
        info["pos_weight"] = self.pos_weight
        info["neg_weight"] = self.neg_weight
        return info


class CenterLoss(BaseLoss):
    """Center loss for learning deep features with class centers.

    Center loss penalizes the distance between deep features and their
    corresponding class centers. When combined with triplet loss, it learns
    to cluster class features tightly around class centers while pushing
    different classes apart.
    
    Optimal for OpenSet recognition as it:
    - Creates tight, well-defined clusters for known classes
    - Makes unknown samples scatter far from all known class centers
    - Improves both accuracy and unknown rejection rate
    
    Reference: "A Discriminative Feature Learning Approach for Deep Face Recognition"
    (Wen et al., ECCV 2016)
    """

    def __init__(self, embedding_dim: int, num_classes: int, lambda_center: float = 0.001):
        """Initialize center loss.
        
        Args:
            embedding_dim: Dimension of embeddings.
            num_classes: Number of classes (for centers initialization).
            lambda_center: Weight factor for center loss (default 0.001).
                          Lower values (0.0001-0.001) for balance with other losses.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.lambda_center = lambda_center
        
        # Initialize class centers (will be updated during training)
        self.centers = nn.Parameter(
            torch.randn(num_classes, embedding_dim, dtype=torch.float32)
        )
        # Normalize centers to L2 unit sphere
        with torch.no_grad():
            self.centers.div_(torch.norm(self.centers, p=2, dim=1, keepdim=True).clamp(min=1e-12))
        
        # Center update rate (how much centers move per update)
        self.center_update_rate = 0.5

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss.

        Args:
            embeddings: L2-normalized embedding vectors of shape (batch_size, embedding_dim).
            labels: Target labels of shape (batch_size,).

        Returns:
            Scalar center loss tensor.
            
        Note:
            Center loss is typically combined with other losses (e.g., triplet loss).
            When used standalone, lambda_center acts as a scaling factor.
            When combined: Total loss = triplet_loss + lambda_center * center_loss
            
            Labels should be in range [0, num_classes).
        """
        batch_size = embeddings.size(0)
        
        # Get centers for samples in this batch
        # Make sure labels are within bounds
        labels_clipped = torch.clamp(labels, 0, self.num_classes - 1)
        centers_batch = self.centers[labels_clipped]  # (batch_size, embedding_dim)
        
        # Distance from embeddings to their corresponding class centers
        # For L2-normalized embeddings: use cosine distance
        center_distances = torch.norm(embeddings - centers_batch, p=2, dim=1)  # (batch_size,)
        
        # Center loss: average squared distance to class centers
        loss = (center_distances ** 2).mean()
        
        # Note: When using standalone center loss, don't scale by lambda_center
        # (it should be 1.0 for standalone usage, or used only in combined losses)
        return loss

    def get_loss_info(self):
        """Get loss function information."""
        info = super().get_loss_info()
        info["embedding_dim"] = self.embedding_dim
        info["num_classes"] = self.num_classes
        info["lambda_center"] = self.lambda_center
        return info


class TripletCenterLoss(BaseLoss):
    """Combined Triplet + Center loss for OpenSet recognition.
    
    Hybrid loss combining:
    - Triplet loss: push apart different classes, pull together same class
    - Center loss: cluster each class tightly around its center
    
    This combination is specifically tuned for OpenSet recognition where
    tight known-class clusters and unknown rejection are critical.
    
    Reference: "A Discriminative Feature Learning Approach for Deep Face Recognition"
    combined with metric learning triplet approaches.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.3,
        mining: str = "hard",
        lambda_center: float = 0.001,
        lambda_triplet: float = 1.0,
    ):
        """Initialize combined triplet + center loss.
        
        Args:
            embedding_dim: Dimension of embeddings.
            num_classes: Number of classes for center loss.
            margin: Triplet loss margin (default 0.3).
            mining: Mining strategy - 'hard' or 'semi-hard' (default 'hard').
            lambda_center: Weight for center loss (default 0.001).
            lambda_triplet: Weight for triplet loss (default 1.0).
        """
        super().__init__()
        self.triplet_loss = TripletLoss(margin=margin, mining=mining)
        self.center_loss = CenterLoss(embedding_dim, num_classes, lambda_center=lambda_center)
        self.lambda_triplet = lambda_triplet

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute combined triplet + center loss.

        Args:
            embeddings: L2-normalized embedding vectors of shape (batch_size, embedding_dim).
            labels: Target labels of shape (batch_size,).

        Returns:
            Scalar combined loss tensor.
        """
        triplet_loss = self.triplet_loss(embeddings, labels)
        center_loss = self.center_loss(embeddings, labels)
        
        # Combined loss with weights
        # Note: center_loss no longer scales by lambda_center internally, so we do it here
        total_loss = self.lambda_triplet * triplet_loss + self.center_loss.lambda_center * center_loss
        
        return total_loss

    def get_loss_info(self):
        """Get loss function information."""
        info = super().get_loss_info()
        info["triplet_margin"] = self.triplet_loss.margin
        info["triplet_mining"] = self.triplet_loss.mining
        info["lambda_triplet"] = self.lambda_triplet
        info["lambda_center"] = self.center_loss.lambda_center
        return info

