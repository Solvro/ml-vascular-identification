"""
Tests for training functions in src/train.py.

Tests cover:
- train_epoch: Training loop with various batch sizes
- evaluate_metric_learning: Validation/test evaluation
- compute_prototypes: Prototype extraction from enrollment set
- evaluate_openset: OpenSet recognition evaluation
"""
import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import CosineClassifier, SimpleEmbeddingModel, TripletLoss
from train import (
    compute_prototypes,
    evaluate_metric_learning,
    evaluate_openset,
    train_epoch,
)


class TestTrainEpoch:
    """Tests for train_epoch function."""
    
    @pytest.fixture
    def setup(self):
        """Create model, optimizer, loss, and fake data loader."""
        model = SimpleEmbeddingModel(embedding_dim=128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = TripletLoss(margin=0.3)
        
        # Create fake data loader
        def fake_loader():
            for _ in range(3):  # 3 batches
                images = torch.randn(16, 3, 224, 224)
                labels = torch.randint(0, 4, (16,))  # 4 classes
                metadata = {}
                yield images, labels, metadata
        
        return model, optimizer, criterion, list(fake_loader())
    
    def test_train_epoch_basic(self, setup):
        """Test basic training epoch."""
        model, optimizer, criterion, data = setup
        
        # Create data loader
        train_loader = data
        device = torch.device('cpu')
        
        # Train for one epoch
        avg_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch=1, log_interval=1
        )
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0.0
        assert not torch.isnan(torch.tensor(avg_loss))
    
    def test_train_epoch_updates_parameters(self, setup):
        """Test that training actually updates model parameters."""
        model, optimizer, criterion, data = setup
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        train_loader = data
        device = torch.device('cpu')
        
        # Train
        train_epoch(model, train_loader, optimizer, criterion, device, epoch=1, log_interval=10)
        
        # Check that parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break
        
        assert params_changed, "Parameters should be updated during training"
    
    def test_train_epoch_with_cuda(self, setup):
        """Test training with CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model, optimizer, criterion, data = setup
        
        device = torch.device('cuda')
        model = model.to(device)
        criterion = criterion.to(device)
        
        # Move data to CUDA
        train_loader = [
            (imgs.to(device), lbls.to(device), meta) 
            for imgs, lbls, meta in data
        ]
        
        avg_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch=1, log_interval=10
        )
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0.0
    
    def test_train_epoch_gradient_flow(self, setup):
        """Test that gradients are computed and cleared properly."""
        model, optimizer, criterion, data = setup
        
        train_loader = data
        device = torch.device('cpu')
        
        # Before training, no gradients
        for p in model.parameters():
            assert p.grad is None
        
        # Train one batch manually
        images, labels, _ = train_loader[0]
        
        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        
        # After backward, gradients should exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads
        
        optimizer.step()
        optimizer.zero_grad()
        
        # After zero_grad, gradients should be None or zero
        all_zero = all(
            p.grad is None or p.grad.abs().sum() == 0 
            for p in model.parameters()
        )
        assert all_zero


class TestEvaluateMetricLearning:
    """Tests for evaluate_metric_learning function."""
    
    @pytest.fixture
    def setup(self):
        """Create model, loss, and fake data loader."""
        model = SimpleEmbeddingModel(embedding_dim=128)
        criterion = TripletLoss(margin=0.3)
        
        # Create fake validation data
        def fake_loader():
            for _ in range(2):  # 2 batches
                images = torch.randn(16, 3, 224, 224)
                labels = torch.randint(0, 4, (16,))
                metadata = {}
                yield images, labels, metadata
        
        return model, criterion, list(fake_loader())
    
    def test_evaluate_basic(self, setup):
        """Test basic evaluation."""
        model, criterion, data = setup
        
        val_loader = data
        device = torch.device('cpu')
        
        avg_loss = evaluate_metric_learning(model, val_loader, criterion, device)
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0.0
        assert not torch.isnan(torch.tensor(avg_loss))
    
    def test_evaluate_no_gradient(self, setup):
        """Test that evaluation doesn't compute gradients."""
        model, criterion, data = setup
        
        val_loader = data
        device = torch.device('cpu')
        
        # Set model to train mode to ensure eval mode is set inside function
        model.train()
        
        evaluate_metric_learning(model, val_loader, criterion, device)
        
        # Check no gradients were accumulated
        for p in model.parameters():
            assert p.grad is None
    
    def test_evaluate_deterministic(self, setup):
        """Test that evaluation is deterministic."""
        model, criterion, data = setup
        
        val_loader = data
        device = torch.device('cpu')
        
        # Run evaluation twice
        loss1 = evaluate_metric_learning(model, val_loader, criterion, device)
        loss2 = evaluate_metric_learning(model, val_loader, criterion, device)
        
        assert loss1 == loss2, "Evaluation should be deterministic"


class TestComputePrototypes:
    """Tests for compute_prototypes function."""
    
    def test_compute_prototypes_basic(self):
        """Test basic prototype computation."""
        model = SimpleEmbeddingModel(embedding_dim=128)
        device = torch.device('cpu')
        
        # Create fake enrollment data
        # 4 classes, 5 samples each
        def fake_loader():
            for class_id in range(4):
                for _ in range(5):
                    images = torch.randn(1, 3, 224, 224)
                    labels = torch.tensor([class_id])
                    metadata = {}
                    yield images, labels, metadata
        
        enrollment_loader = list(fake_loader())
        
        prototypes, class_to_idx = compute_prototypes(
            model, enrollment_loader, num_classes=4, embedding_dim=128, device=device
        )
        
        assert prototypes.shape == (4, 128)
        assert len(class_to_idx) == 4
        
        # Check L2 normalization
        norms = torch.norm(prototypes, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)
    
    def test_compute_prototypes_unbalanced(self):
        """Test prototype computation with unbalanced classes."""
        model = SimpleEmbeddingModel(embedding_dim=64)
        device = torch.device('cpu')
        
        # Create unbalanced enrollment data
        # Class 0: 10 samples, Class 1: 3 samples, Class 2: 7 samples
        samples_per_class = [10, 3, 7]
        
        def fake_loader():
            for class_id, n_samples in enumerate(samples_per_class):
                for _ in range(n_samples):
                    images = torch.randn(1, 3, 224, 224)
                    labels = torch.tensor([class_id])
                    metadata = {}
                    yield images, labels, metadata
        
        enrollment_loader = list(fake_loader())
        
        prototypes, class_to_idx = compute_prototypes(
            model, enrollment_loader, num_classes=3, embedding_dim=64, device=device
        )
        
        assert prototypes.shape == (3, 64)
        assert len(class_to_idx) == 3
        
        # All prototypes should be normalized
        norms = torch.norm(prototypes, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)
    
    def test_compute_prototypes_batch_processing(self):
        """Test prototype computation with batched data."""
        model = SimpleEmbeddingModel(embedding_dim=128)
        device = torch.device('cpu')
        
        # Create batched enrollment data
        def fake_loader():
            for _ in range(3):  # 3 batches
                images = torch.randn(8, 3, 224, 224)
                labels = torch.randint(0, 4, (8,))  # 4 classes
                metadata = {}
                yield images, labels, metadata
        
        enrollment_loader = list(fake_loader())
        
        prototypes, class_to_idx = compute_prototypes(
            model, enrollment_loader, num_classes=4, embedding_dim=128, device=device
        )
        
        assert prototypes.shape == (4, 128)
        assert len(class_to_idx) == 4


class TestEvaluateOpenSet:
    """Tests for evaluate_openset function."""
    
    @pytest.fixture
    def setup(self):
        """Create model, prototypes, and fake data loaders."""
        model = SimpleEmbeddingModel(embedding_dim=128)
        device = torch.device('cpu')
        
        # Create random prototypes (4 classes)
        prototypes = torch.randn(4, 128, device=device)
        prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)
        
        # Create fake test data
        def fake_known_loader():
            for _ in range(2):
                images = torch.randn(8, 3, 224, 224)
                labels = torch.randint(0, 4, (8,))  # Known classes: 0-3
                metadata = {}
                yield images, labels, metadata
        
        def fake_unknown_loader():
            for _ in range(2):
                images = torch.randn(8, 3, 224, 224)
                labels = torch.randint(4, 8, (8,))  # Unknown classes: 4-7
                metadata = {}
                yield images, labels, metadata
        
        return model, prototypes, list(fake_known_loader()), list(fake_unknown_loader()), device
    
    def test_evaluate_openset_basic(self, setup):
        """Test basic OpenSet evaluation."""
        model, prototypes, known_loader, unknown_loader, device = setup
        
        metrics = evaluate_openset(
            model, known_loader, unknown_loader, prototypes, device, 
            class_to_idx=None, threshold=0.5
        )
        
        # Check that all expected metrics are present
        assert 'eer' in metrics
        assert 'auroc' in metrics
        assert 'oscr' in metrics
        assert 'tpr_at_fpr_001' in metrics
        assert 'tpr_at_fpr_01' in metrics
        assert 'known_accuracy' in metrics
        assert 'unknown_rejection' in metrics
        assert 'threshold' in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics['eer'] <= 100.0  # EER is in percentage
        assert 0.0 <= metrics['auroc'] <= 1.0
        assert 0.0 <= metrics['known_accuracy'] <= 1.0
        assert 0.0 <= metrics['unknown_rejection'] <= 1.0
    
    def test_evaluate_openset_different_thresholds(self, setup):
        """Test OpenSet evaluation with different thresholds."""
        model, prototypes, known_loader, unknown_loader, device = setup
        
        thresholds = [0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            metrics = evaluate_openset(
                model, known_loader, unknown_loader, prototypes, device, 
                class_to_idx=None, threshold=threshold
            )
            
            assert metrics['threshold'] == threshold
            assert 'known_accuracy' in metrics
            assert 'unknown_rejection' in metrics
    
    def test_evaluate_openset_no_unknown(self, setup):
        """Test OpenSet evaluation with only known samples."""
        model, prototypes, known_loader, _, device = setup
        
        # Create empty unknown loader
        empty_unknown_loader = []
        
        metrics = evaluate_openset(
            model, known_loader, empty_unknown_loader, prototypes, device, 
            class_to_idx=None, threshold=0.5
        )
        
        # Should still compute metrics (unknown_rejection will be 0 or undefined)
        assert 'eer' in metrics
        assert 'auroc' in metrics


class TestIntegration:
    """Integration tests combining training components."""
    
    @pytest.mark.slow
    def test_full_training_pipeline(self):
        """Test complete training pipeline: train → validate → compute prototypes → evaluate."""
        # Setup
        model = SimpleEmbeddingModel(embedding_dim=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = TripletLoss(margin=0.3)
        device = torch.device('cpu')
        
        # Create fake data
        def create_loader(num_batches=2, batch_size=8, num_classes=4):
            data = []
            for _ in range(num_batches):
                images = torch.randn(batch_size, 3, 224, 224)
                labels = torch.randint(0, num_classes, (batch_size,))
                metadata = {}
                data.append((images, labels, metadata))
            return data
        
        train_loader = create_loader(num_batches=3)
        val_loader = create_loader(num_batches=2)
        enrollment_loader = create_loader(num_batches=5, batch_size=4)  # For prototypes
        test_known_loader = create_loader(num_batches=2)
        test_unknown_loader = create_loader(num_batches=2, num_classes=8)  # Different classes
        
        # Step 1: Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch=1, log_interval=10
        )
        assert train_loss >= 0.0
        
        # Step 2: Validate
        val_loss = evaluate_metric_learning(model, val_loader, criterion, device)
        assert val_loss >= 0.0
        
        # Step 3: Compute prototypes
        prototypes, class_to_idx = compute_prototypes(
            model, enrollment_loader, num_classes=4, embedding_dim=64, device=device
        )
        assert prototypes.shape == (4, 64)
        assert len(class_to_idx) == 4
        
        # Step 4: Evaluate OpenSet
        metrics = evaluate_openset(
            model, test_known_loader, test_unknown_loader, prototypes, device, 
            class_to_idx, threshold=0.5
        )
        assert 'eer' in metrics
        assert 'auroc' in metrics
        assert 'known_accuracy' in metrics


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short", "-k", "not slow"])
