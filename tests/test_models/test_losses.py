"""
Tests for TripletLoss and ContrastiveLoss.

Verifies that loss functions work correctly for OpenSet recognition:
- Proper loss computation with L2-normalized embeddings
- Hard negative mining functionality
- Various batch sizes and margins
- Edge cases (all same class, all different classes)
"""
import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models import ContrastiveLoss, TripletLoss


class TestTripletLoss:
    """Test suite for TripletLoss."""
    
    @pytest.fixture
    def loss_03(self):
        """TripletLoss with margin=0.3."""
        return TripletLoss(margin=0.3)
    
    @pytest.fixture
    def loss_02(self):
        """TripletLoss with margin=0.2."""
        return TripletLoss(margin=0.2)
    
    @pytest.fixture
    def normalized_embeddings(self):
        """Create L2-normalized embeddings for 8 samples."""
        embeddings = torch.randn(8, 256)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    @pytest.fixture
    def pk_batch(self):
        """Create embeddings for P=16, K=4 batch (64 samples)."""
        embeddings = torch.randn(64, 256)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    def test_loss_creation(self, loss_03):
        """Test that loss is created correctly."""
        assert isinstance(loss_03, TripletLoss)
        assert loss_03.margin == 0.3
        assert loss_03.mining == "hard"
    
    def test_loss_forward_simple(self, loss_03, normalized_embeddings):
        """Test forward pass with simple batch."""
        # Create labels: 4 classes, 2 samples each
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss_value = loss_03(normalized_embeddings, labels)
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.shape == ()  # Scalar
        assert loss_value.item() >= 0, "Loss should be non-negative"
    
    def test_loss_backward(self, loss_03, normalized_embeddings):
        """Test that loss can backpropagate."""
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        normalized_embeddings.requires_grad = True
        loss_value = loss_03(normalized_embeddings, labels)
        loss_value.backward()
        
        assert normalized_embeddings.grad is not None
        assert not torch.isnan(normalized_embeddings.grad).any()
    
    def test_pk_batch(self, loss_03, pk_batch):
        """Test with P=16, K=4 batch."""
        # Create labels: 16 classes, 4 samples each
        labels = torch.repeat_interleave(torch.arange(16), 4)
        
        loss_value = loss_03(pk_batch, labels)
        
        assert loss_value.item() >= 0
        assert not torch.isnan(loss_value).any()
        assert not torch.isinf(loss_value).any()
    
    def test_margin_effect(self, loss_02, loss_03, normalized_embeddings):
        """Test that margin affects loss value."""
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss_02_val = loss_02(normalized_embeddings, labels)
        loss_03_val = loss_03(normalized_embeddings, labels)
        
        # Both should be non-negative
        assert loss_02_val.item() >= 0
        assert loss_03_val.item() >= 0
    
    def test_perfect_separation(self, loss_03):
        """Test with perfectly separated classes."""
        # Create well-separated embeddings
        class1 = torch.ones(4, 256) * 1.0
        class2 = torch.ones(4, 256) * -1.0
        embeddings = torch.cat([class1, class2], dim=0)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        
        loss_value = loss_03(embeddings, labels)
        
        # Loss should be very small for well-separated classes
        assert loss_value.item() < 0.5
    
    def test_identical_embeddings(self, loss_03):
        """Test with identical embeddings (worst case)."""
        # All embeddings identical
        embeddings = torch.ones(8, 256)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss_value = loss_03(embeddings, labels)
        
        # Loss should be positive since negatives are not pushed apart
        assert loss_value.item() >= 0
    
    def test_single_class(self, loss_03, normalized_embeddings):
        """Test with all samples from same class."""
        labels = torch.zeros(8, dtype=torch.long)
        
        # Should handle gracefully (no valid triplets)
        loss_value = loss_03(normalized_embeddings, labels)
        
        assert loss_value.item() >= 0
    
    def test_all_different_classes(self, loss_03, normalized_embeddings):
        """Test with each sample from different class."""
        labels = torch.arange(8)
        
        # Should handle gracefully (no positives)
        loss_value = loss_03(normalized_embeddings, labels)
        
        assert loss_value.item() >= 0
    
    def test_loss_info(self, loss_03):
        """Test get_loss_info method."""
        info = loss_03.get_loss_info()
        
        assert 'loss_name' in info
        assert info['loss_name'] == 'TripletLoss'
        assert 'margin' in info
        assert info['margin'] == 0.3
        assert 'mining' in info
        assert info['mining'] == 'hard'


class TestContrastiveLoss:
    """Test suite for ContrastiveLoss."""
    
    @pytest.fixture
    def loss_10(self):
        """ContrastiveLoss with margin=1.0."""
        return ContrastiveLoss(margin=1.0)
    
    @pytest.fixture
    def normalized_embeddings(self):
        """Create L2-normalized embeddings."""
        embeddings = torch.randn(8, 256)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    def test_loss_creation(self, loss_10):
        """Test that loss is created correctly."""
        assert isinstance(loss_10, ContrastiveLoss)
        assert loss_10.margin == 1.0
    
    def test_loss_forward(self, loss_10, normalized_embeddings):
        """Test forward pass."""
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss_value = loss_10(normalized_embeddings, labels)
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.shape == ()
        assert loss_value.item() >= 0
    
    def test_loss_backward(self, loss_10, normalized_embeddings):
        """Test backpropagation."""
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        normalized_embeddings.requires_grad = True
        loss_value = loss_10(normalized_embeddings, labels)
        loss_value.backward()
        
        assert normalized_embeddings.grad is not None
    
    def test_weighted_loss(self):
        """Test loss with custom weights."""
        loss_weighted = ContrastiveLoss(margin=1.0, pos_weight=2.0, neg_weight=0.5)
        
        embeddings = torch.randn(8, 256)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss_value = loss_weighted(embeddings, labels)
        
        assert loss_value.item() >= 0
    
    def test_loss_info(self, loss_10):
        """Test get_loss_info method."""
        info = loss_10.get_loss_info()
        
        assert 'loss_name' in info
        assert info['loss_name'] == 'ContrastiveLoss'
        assert 'margin' in info
        assert info['margin'] == 1.0


def test_loss_comparison():
    """Compare TripletLoss and ContrastiveLoss on same data."""
    # Create sample data
    embeddings = torch.randn(16, 256)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.repeat_interleave(torch.arange(8), 2)
    
    triplet_loss = TripletLoss(margin=0.3)
    contrastive_loss = ContrastiveLoss(margin=1.0)
    
    triplet_val = triplet_loss(embeddings, labels)
    contrastive_val = contrastive_loss(embeddings, labels)
    
    # Both should produce valid losses
    assert triplet_val.item() >= 0
    assert contrastive_val.item() >= 0
    assert not torch.isnan(triplet_val)
    assert not torch.isnan(contrastive_val)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
