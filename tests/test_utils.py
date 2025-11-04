"""
Tests for utility functions (visualization, protocol saving, timing).
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils import (
    InferenceTimer,
    plot_det_curve,
    plot_roc_curve,
    save_evaluation_protocol,
)


class TestInferenceTimer:
    """Test timing utilities."""
    
    def test_timer_basic(self):
        """Test basic timer functionality."""
        import time
        
        with InferenceTimer("Test operation") as timer:
            time.sleep(0.01)  # Sleep for 10ms
        
        elapsed_ms = timer.get_elapsed_ms()
        assert elapsed_ms >= 10.0, "Timer should measure at least 10ms"
        assert elapsed_ms < 100.0, "Timer should be reasonably accurate"
    
    def test_timer_nested(self):
        """Test nested timers."""
        import time
        
        with InferenceTimer("Outer") as outer_timer:
            time.sleep(0.01)
            with InferenceTimer("Inner") as inner_timer:
                time.sleep(0.01)
        
        outer_ms = outer_timer.get_elapsed_ms()
        inner_ms = inner_timer.get_elapsed_ms()
        
        assert outer_ms > inner_ms, "Outer timer should be longer"
        assert outer_ms >= 20.0, "Outer should be at least 20ms"


class TestPlotting:
    """Test visualization functions."""
    
    def test_plot_det_curve(self):
        """Test DET curve plotting."""
        # Create synthetic data
        genuine_scores = np.random.uniform(0.6, 1.0, 100)
        impostor_scores = np.random.uniform(0.0, 0.5, 100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "det_test.png")
            plot_det_curve(genuine_scores, impostor_scores, save_path)
            
            assert os.path.exists(save_path), "DET curve should be saved"
            assert os.path.getsize(save_path) > 0, "DET curve file should not be empty"
    
    def test_plot_roc_curve(self):
        """Test ROC curve plotting."""
        genuine_scores = np.random.uniform(0.6, 1.0, 100)
        impostor_scores = np.random.uniform(0.0, 0.5, 100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "roc_test.png")
            plot_roc_curve(genuine_scores, impostor_scores, save_path, auroc=0.95)
            
            assert os.path.exists(save_path), "ROC curve should be saved"
            assert os.path.getsize(save_path) > 0, "ROC curve file should not be empty"
    
    def test_plot_with_overlapping_scores(self):
        """Test plotting with overlapping score distributions."""
        # More challenging case with overlap
        genuine_scores = np.random.uniform(0.4, 0.9, 100)
        impostor_scores = np.random.uniform(0.2, 0.7, 100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            det_path = os.path.join(tmpdir, "det_overlap.png")
            roc_path = os.path.join(tmpdir, "roc_overlap.png")
            
            plot_det_curve(genuine_scores, impostor_scores, det_path)
            plot_roc_curve(genuine_scores, impostor_scores, roc_path)
            
            assert os.path.exists(det_path)
            assert os.path.exists(roc_path)


class TestProtocolSaving:
    """Test evaluation protocol saving."""
    
    def test_save_protocol_basic(self):
        """Test basic protocol saving."""
        metrics = {
            'eer': 0.05,
            'auroc': 0.95,
            'oscr': 0.90,
        }
        
        config = {
            'dataset_name': 'test_dataset',
            'known_classes': 420,
            'unknown_classes': 180,
            'model_name': 'test_model',
            'embedding_dim': 256,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "protocol.json")
            save_evaluation_protocol(metrics, config, save_path)
            
            assert os.path.exists(save_path), "Protocol should be saved"
            
            # Load and verify
            with open(save_path, 'r') as f:
                protocol = json.load(f)
            
            assert 'timestamp' in protocol
            assert protocol['dataset']['name'] == 'test_dataset'
            assert protocol['dataset']['known_classes'] == 420
            assert protocol['metrics']['eer'] == 0.05
    
    def test_save_protocol_complete(self):
        """Test protocol saving with all fields."""
        metrics = {
            'eer': 0.05,
            'auroc': 0.95,
            'oscr': 0.90,
            'cmc_rank1': 0.85,
            'cmc_rank5': 0.92,
            'known_accuracy': 0.88,
            'unknown_rejection': 0.87,
        }
        
        config = {
            'dataset_name': 'mmcbnu',
            'total_classes': 600,
            'known_classes': 420,
            'unknown_classes': 180,
            'subject_disjoint': True,
            'train_size': 1000,
            'val_size': 300,
            'test_known_size': 400,
            'test_unknown_size': 200,
            'enrollment_size': 300,
            'model_name': 'simple_cnn',
            'embedding_dim': 256,
            'total_parameters': 100000,
            'loss_name': 'triplet',
            'margin': 0.3,
            'optimizer': 'AdamW',
            'learning_rate': 0.0003,
            'epochs': 50,
            'best_epoch': 48,
            'best_val_loss': 0.068,
            'P': 8,
            'K': 4,
            'k': 1,
            'threshold': 0.5,
            'threshold_optimization': 'oscr',
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "protocol_complete.json")
            save_evaluation_protocol(metrics, config, save_path)
            
            with open(save_path, 'r') as f:
                protocol = json.load(f)
            
            # Verify structure
            assert 'timestamp' in protocol
            assert 'dataset' in protocol
            assert 'splits' in protocol
            assert 'model' in protocol
            assert 'training' in protocol
            assert 'inference' in protocol
            assert 'metrics' in protocol
            
            # Verify content
            assert protocol['dataset']['known_classes'] == 420
            assert protocol['training']['P'] == 8
            assert protocol['inference']['k'] == 1
            assert protocol['metrics']['auroc'] == 0.95
    
    def test_protocol_json_format(self):
        """Test that protocol is valid JSON and properly formatted."""
        metrics = {'eer': 0.05}
        config = {'dataset_name': 'test'}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "protocol.json")
            save_evaluation_protocol(metrics, config, save_path)
            
            # Verify JSON is valid and formatted
            with open(save_path, 'r') as f:
                content = f.read()
                protocol = json.loads(content)  # Should not raise
            
            # Check indentation (should be pretty-printed)
            assert '\n' in content, "JSON should be formatted with newlines"
            assert '  ' in content, "JSON should have indentation"
    
    def test_protocol_numpy_types(self):
        """Test that numpy types are properly converted to JSON-serializable types."""
        metrics = {
            'eer': np.float32(0.1248),
            'auroc': np.float64(0.9274),
            'cmc_rank1': np.float32(0.1493),
            'cmc_curve': [np.float32(0.1), np.float32(0.2)],
            'total_known': np.int64(1000),
            'total_unknown': np.int32(500),
        }
        
        config = {
            'dataset_name': 'mmcbnu',
            'known_classes': np.int32(420),
            'learning_rate': np.float64(0.0003),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "protocol_numpy.json")
            save_evaluation_protocol(metrics, config, save_path)
            
            # Should not raise TypeError
            with open(save_path, 'r') as f:
                protocol = json.load(f)
            
            # Verify types are converted
            assert isinstance(protocol['metrics']['eer'], float)
            assert isinstance(protocol['metrics']['total_known'], int)
            assert isinstance(protocol['metrics']['cmc_curve'], list)
            assert isinstance(protocol['dataset']['known_classes'], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
