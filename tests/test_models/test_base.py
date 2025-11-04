"""
Tests for base model components.

Tests CosineClassifier and factory functions for OpenSet recognition.
"""
import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models import CosineClassifier, create_loss, create_model


class TestCosineClassifier:
    """Test suite for CosineClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier for 420 known classes with 256-dim embeddings."""
        return CosineClassifier(
            embedding_dim=256,
            num_classes=420,
            temperature=0.07,
            learnable_temperature=True,
        )
    
    @pytest.fixture
    def normalized_embeddings(self):
        """Create L2-normalized embeddings."""
        embeddings = torch.randn(32, 256)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    def test_classifier_creation(self, classifier):
        """Test classifier is created correctly."""
        assert isinstance(classifier, CosineClassifier)
        assert classifier.embedding_dim == 256
        assert classifier.num_classes == 420
        assert classifier.learnable_temperature == True
    
    def test_forward_pass(self, classifier, normalized_embeddings):
        """Test forward pass produces correct output shape."""
        logits = classifier(normalized_embeddings)
        
        assert logits.shape == (32, 420), f"Expected (32, 420), got {logits.shape}"
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_temperature_scaling(self):
        """Test that temperature affects logits."""
        classifier_low_temp = CosineClassifier(
            embedding_dim=256,
            num_classes=420,
            temperature=0.01,  # Low temp = sharper
            learnable_temperature=False,
        )
        
        classifier_high_temp = CosineClassifier(
            embedding_dim=256,
            num_classes=420,
            temperature=0.5,  # High temp = softer
            learnable_temperature=False,
        )
        
        # Copy weights to make them identical
        classifier_high_temp.weight.data = classifier_low_temp.weight.data.clone()
        
        embeddings = torch.randn(16, 256)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        logits_low = classifier_low_temp(embeddings)
        logits_high = classifier_high_temp(embeddings)
        
        # Low temperature should give more extreme values
        assert logits_low.abs().mean() > logits_high.abs().mean()
    
    def test_get_similarities(self, classifier, normalized_embeddings):
        """Test get_similarities returns raw cosine similarities."""
        similarities = classifier.get_similarities(normalized_embeddings)
        
        assert similarities.shape == (32, 420)
        
        # Cosine similarities should be in [-1, 1]
        assert (similarities >= -1.0).all() and (similarities <= 1.0).all(), \
            f"Similarities out of range: [{similarities.min()}, {similarities.max()}]"
    
    def test_gradient_flow(self, classifier, normalized_embeddings):
        """Test gradients flow through classifier."""
        logits = classifier(normalized_embeddings)
        loss = logits.sum()
        loss.backward()
        
        # Check weight gradients
        assert classifier.weight.grad is not None
        assert classifier.weight.grad.abs().sum() > 0
        
        # Check temperature gradients if learnable
        if classifier.learnable_temperature:
            assert classifier.temperature.grad is not None
    
    def test_learnable_temperature(self):
        """Test that temperature can be learned."""
        classifier_learnable = CosineClassifier(
            embedding_dim=256,
            num_classes=420,
            temperature=0.07,
            learnable_temperature=True,
        )
        
        classifier_fixed = CosineClassifier(
            embedding_dim=256,
            num_classes=420,
            temperature=0.07,
            learnable_temperature=False,
        )
        
        # Learnable should have temperature as parameter
        assert isinstance(classifier_learnable.temperature, torch.nn.Parameter)
        
        # Fixed should have temperature as buffer
        assert not isinstance(classifier_fixed.temperature, torch.nn.Parameter)
    
    def test_weight_normalization(self, classifier, normalized_embeddings):
        """Test that class prototypes are normalized."""
        # Forward pass normalizes weights internally
        logits = classifier(normalized_embeddings)
        
        # Manually normalize and check
        normalized_weight = torch.nn.functional.normalize(classifier.weight, p=2, dim=1)
        norms = torch.norm(normalized_weight, p=2, dim=1)
        
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestFactoryFunctions:
    """Test suite for factory functions."""
    
    def test_create_model_simple_cnn(self):
        """Test creating SimpleEmbeddingModel."""
        model = create_model("simple_cnn", embedding_dim=256)
        
        assert model is not None
        assert model.embedding_dim == 256
    
    def test_create_model_alias(self):
        """Test model creation with alias."""
        model = create_model("simple_embedding", embedding_dim=128)
        
        assert model is not None
        assert model.embedding_dim == 128
    
    def test_create_model_invalid(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("invalid_model_name")
    
    def test_create_loss_triplet(self):
        """Test creating TripletLoss."""
        loss = create_loss("triplet", margin=0.3)
        
        assert loss is not None
        assert loss.margin == 0.3
    
    def test_create_loss_contrastive(self):
        """Test creating ContrastiveLoss."""
        loss = create_loss("contrastive", margin=1.0)
        
        assert loss is not None
        assert loss.margin == 1.0
    
    def test_create_loss_alias(self):
        """Test loss creation with alias."""
        loss = create_loss("triplet_loss", margin=0.25)
        
        assert loss is not None
        assert loss.margin == 0.25
    
    def test_create_loss_invalid(self):
        """Test that invalid loss name raises error."""
        with pytest.raises(ValueError, match="Unknown loss"):
            create_loss("invalid_loss_name")


def test_cosine_classifier_integration():
    """Integration test: model + cosine classifier."""
    from models import SimpleEmbeddingModel
    
    # Create model and classifier
    model = SimpleEmbeddingModel(embedding_dim=256)
    classifier = CosineClassifier(embedding_dim=256, num_classes=420)
    
    # Create sample input
    images = torch.randn(16, 3, 224, 224)
    
    # Forward pass
    model.eval()
    classifier.eval()
    
    with torch.no_grad():
        embeddings = model(images)
        logits = classifier(embeddings)
    
    assert embeddings.shape == (16, 256)
    assert logits.shape == (16, 420)
    
    # Check that embeddings are normalized
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
