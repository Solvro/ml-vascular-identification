"""
Tests for SimpleEmbeddingModel (basic_cnn.py).

Verifies that the model meets all OpenSet requirements:
- 256-dimensional embeddings
- L2-normalization (unit norm)
- Correct forward pass with various batch sizes
- Compatible with P=16, K=4 sampling (batch_size=64)
- Proper initialization and parameter counting
"""
import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models import SimpleEmbeddingModel


class TestSimpleEmbeddingModel:
    """Test suite for SimpleEmbeddingModel."""
    
    @pytest.fixture
    def model_256(self):
        """Create model with default 256-dim embeddings."""
        return SimpleEmbeddingModel(embedding_dim=256)
    
    @pytest.fixture
    def model_128(self):
        """Create model with 128-dim embeddings."""
        return SimpleEmbeddingModel(embedding_dim=128)
    
    @pytest.fixture
    def sample_input_224(self):
        """Create sample input for 224x224 images."""
        return torch.randn(4, 3, 224, 224)
    
    @pytest.fixture
    def sample_batch_pk(self):
        """Create sample batch matching P=16, K=4 (batch_size=64)."""
        return torch.randn(64, 3, 224, 224)
    
    def test_model_creation(self, model_256):
        """Test that model is created correctly."""
        assert isinstance(model_256, SimpleEmbeddingModel)
        assert model_256.embedding_dim == 256
        assert model_256.input_channels == 3
    
    def test_embedding_dimension(self, model_256, model_128, sample_input_224):
        """Test that output has correct embedding dimension."""
        output_256 = model_256(sample_input_224)
        output_128 = model_128(sample_input_224)
        
        assert output_256.shape == (4, 256), f"Expected (4, 256), got {output_256.shape}"
        assert output_128.shape == (4, 128), f"Expected (4, 128), got {output_128.shape}"
    
    def test_l2_normalization(self, model_256, sample_input_224):
        """Test that embeddings are L2-normalized (unit norm)."""
        embeddings = model_256(sample_input_224)
        
        # Compute L2 norms
        norms = torch.norm(embeddings, p=2, dim=1)
        
        # Check all norms are approximately 1.0
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), \
            f"Expected unit norms, got {norms}"
    
    def test_forward_pass_224(self, model_256, sample_input_224):
        """Test forward pass with 224x224 images."""
        model_256.eval()
        with torch.no_grad():
            embeddings = model_256(sample_input_224)
        
        assert embeddings.shape == (4, 256)
        assert not torch.isnan(embeddings).any(), "Embeddings contain NaN"
        assert not torch.isinf(embeddings).any(), "Embeddings contain Inf"
    
    def test_batch_size_flexibility(self, model_256):
        """Test model works with different batch sizes."""
        batch_sizes = [1, 4, 16, 32, 64, 128]
        
        model_256.eval()
        with torch.no_grad():
            for batch_size in batch_sizes:
                input_tensor = torch.randn(batch_size, 3, 224, 224)
                embeddings = model_256(input_tensor)
                
                assert embeddings.shape == (batch_size, 256), \
                    f"Failed for batch_size={batch_size}"
                
                # Verify L2 normalization
                norms = torch.norm(embeddings, p=2, dim=1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_pk_sampling_batch(self, model_256, sample_batch_pk):
        """Test model with P=16, K=4 batch (64 samples)."""
        model_256.eval()
        with torch.no_grad():
            embeddings = model_256(sample_batch_pk)
        
        assert embeddings.shape == (64, 256)
        
        # Verify L2 normalization for large batch
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_gradient_flow(self, model_256, sample_input_224):
        """Test that gradients flow through the model."""
        model_256.train()
        
        embeddings = model_256(sample_input_224)
        loss = embeddings.sum()
        loss.backward()
        
        # Check that gradients exist for backbone and head
        has_gradients = False
        for param in model_256.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in model"
    
    def test_reproducibility(self, model_256, sample_input_224):
        """Test that model produces same output for same input (eval mode)."""
        model_256.eval()
        
        with torch.no_grad():
            output1 = model_256(sample_input_224)
            output2 = model_256(sample_input_224)
        
        assert torch.allclose(output1, output2), \
            "Model not reproducible in eval mode"
    
    def test_different_inputs(self, model_256):
        """Test that different inputs produce different embeddings."""
        model_256.eval()
        
        input1 = torch.randn(4, 3, 224, 224)
        input2 = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            emb1 = model_256(input1)
            emb2 = model_256(input2)
        
        # Embeddings should be different (with high probability)
        assert not torch.allclose(emb1, emb2, atol=1e-3), \
            "Different inputs produced identical embeddings"
    
    def test_cosine_similarity_range(self, model_256):
        """Test that cosine similarity between embeddings is in valid range."""
        model_256.eval()
        
        input_tensor = torch.randn(10, 3, 224, 224)
        
        with torch.no_grad():
            embeddings = model_256(input_tensor)
        
        # Compute pairwise cosine similarities (dot product for normalized vectors)
        cosine_sim = torch.matmul(embeddings, embeddings.t())
        
        # Cosine similarity should be in [-1, 1] (allow small numerical errors)
        assert (cosine_sim >= -1.01).all() and (cosine_sim <= 1.01).all(), \
            f"Cosine similarities out of range: [{cosine_sim.min()}, {cosine_sim.max()}]"
        
        # Diagonal should be 1.0 (self-similarity)
        diagonal = torch.diag(cosine_sim)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5), \
            f"Self-similarity not 1.0: {diagonal}"
    
    def test_model_info(self, model_256):
        """Test get_model_info method."""
        info = model_256.get_model_info()
        
        assert 'model_name' in info
        assert info['model_name'] == 'SimpleEmbeddingModel'
        assert info['embedding_dim'] == 256
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['trainable_parameters'] <= info['total_parameters']
    
    def test_dropout(self):
        """Test model with dropout enabled."""
        model_with_dropout = SimpleEmbeddingModel(embedding_dim=256, dropout=0.5)
        
        input_tensor = torch.randn(4, 3, 224, 224)
        
        # In train mode, dropout should cause variation
        model_with_dropout.train()
        output1 = model_with_dropout(input_tensor)
        output2 = model_with_dropout(input_tensor)
        
        # Outputs should differ due to dropout (with high probability)
        assert not torch.allclose(output1, output2, atol=1e-3), \
            "Dropout not working in train mode"
        
        # In eval mode, should be deterministic
        model_with_dropout.eval()
        with torch.no_grad():
            output3 = model_with_dropout(input_tensor)
            output4 = model_with_dropout(input_tensor)
        
        assert torch.allclose(output3, output4), \
            "Model not deterministic in eval mode with dropout"
    
    def test_encode_alias(self, model_256, sample_input_224):
        """Test that encode() is an alias for forward()."""
        model_256.eval()
        
        with torch.no_grad():
            output_forward = model_256.forward(sample_input_224)
            output_encode = model_256.encode(sample_input_224)
        
        assert torch.allclose(output_forward, output_encode), \
            "encode() and forward() produce different results"
    
    def test_parameter_count(self, model_256):
        """Test that model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in model_256.parameters())
        
        # Model with improvements can have up to 50M parameters
        # (Increased from 10M due to ResNet blocks + attention)
        assert total_params < 50_000_000, \
            f"Model too large: {total_params:,} parameters"
        
        # Should have at least some parameters
        assert total_params > 100_000, \
            f"Model too small: {total_params:,} parameters"


def test_model_on_gpu():
    """Test model can be moved to GPU (if available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = SimpleEmbeddingModel(embedding_dim=256)
    model = model.cuda()
    
    input_tensor = torch.randn(4, 3, 224, 224).cuda()
    
    model.eval()
    with torch.no_grad():
        embeddings = model(input_tensor)
    
    assert embeddings.is_cuda, "Output not on GPU"
    assert embeddings.shape == (4, 256)
    
    # Verify L2 normalization on GPU
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
