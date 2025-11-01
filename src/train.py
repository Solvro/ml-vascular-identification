#!/usr/bin/env python3
"""
Simple training script using Hydra config and PyTorch DataLoaders.

Example usage:
    python train.py                     # Use default dorsal config
    python train.py data=mmcbnu         # Use MMCBNU dataset
    python train.py sampler.P=4 sampler.K=8  # Override sampling params
"""
import hydra
import torch
import torch.optim as optim
from models import create_loss, create_model
from omegaconf import DictConfig

from data import create_data_loaders_from_config
from sklearn.metrics import euclidean_distances
import numpy as np

def compute_biometric_metrics(embeddings, labels, threshold=None):
    """Compute biometric authentication metrics.
    
    Returns:
        - TAR@FAR: True Accept Rate at 0.1% False Accept Rate
        - FAR: False Accept Rate
        - FRR: False Rejection Rate
        - Precision@1: Precision of top-1 match
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    distances = euclidean_distances(embeddings_np)
    
    # Compute all match scores (genuine and impostor)
    genuine_distances = []
    impostor_distances = []
    
    for i in range(len(labels_np)):
        for j in range(i + 1, len(labels_np)):
            dist = distances[i, j]
            if labels_np[i] == labels_np[j]:
                genuine_distances.append(dist)
            else:
                impostor_distances.append(dist)
    
    genuine_distances = np.array(genuine_distances)
    impostor_distances = np.array(impostor_distances)
    
    # Find threshold for 0.1% FAR (False Accept Rate)
    if len(impostor_distances) > 0:
        target_far = 0.001  # 0.1% FAR
        n_impostor_thresh = max(1, int(len(impostor_distances) * target_far))
        threshold = np.sort(impostor_distances)[n_impostor_thresh - 1]
        
        # Calculate TAR at this threshold
        tar = np.mean(genuine_distances <= threshold) if len(genuine_distances) > 0 else 0.0
        far = np.mean(impostor_distances <= threshold)
    else:
        tar = 0.0
        far = 0.0
        threshold = None
    
    # Compute Precision@1 (True Positive Rate for top-1)
    tp = 0  # True positives
    total = 0  # Total predictions
    
    for i in range(len(labels_np)):
        nearest_idx = distances[i].argsort()[1]  # Nearest neighbor (excluding self)
        if labels_np[i] == labels_np[nearest_idx]:
            tp += 1
        total += 1
    
    precision_at_1 = tp / total if total > 0 else 0.0
    
    # FRR at same threshold (False Rejection Rate)
    frr = np.mean(genuine_distances > threshold) if threshold is not None and len(genuine_distances) > 0 else 0.0
    
    return {
        "tar_at_far_01": tar,
        "far_at_far_01": far,
        "frr_at_far_01": frr,
        "precision_at_1": precision_at_1,
    }


def compute_recall_at_k(embeddings, labels, k=1):
    """Compute Recall@K metric for embeddings.
    
    For each query sample, we check if any of the k nearest neighbors
    share the same label (excluding the query itself).
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels
        k: Number of nearest neighbors to check
    
    Returns:
        Recall@K score (0-1)
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    distances = euclidean_distances(embeddings_np)
    
    correct = 0
    for i in range(len(labels_np)):
        # Get k+1 nearest neighbors (including self), then exclude self
        nearest_indices = distances[i].argsort()[:k+1]
        # Remove self (distance 0)
        nearest_indices = nearest_indices[1:k+1]
        
        # Check if any of the k nearest neighbors has the same label
        if any(labels_np[idx] == labels_np[i] for idx in nearest_indices):
            correct += 1
    
    return correct / len(labels_np)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, loader_name="Validation"):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_embeddings = []
    all_labels = []
    num_batches = len(data_loader)

    for images, labels, _ in data_loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        total_loss += loss.item()
        
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    # Concatenate all batches
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    avg_loss = total_loss / num_batches
    
    # Compute metrics
    recall_at_1 = compute_recall_at_k(all_embeddings, all_labels, k=1)
    biometric_metrics = compute_biometric_metrics(all_embeddings, all_labels)
    
    return avg_loss, recall_at_1, biometric_metrics


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    print("\n" + "="*70)
    print("🚀 Starting training...")
    print("="*70)
    print(f"📁 Dataset: {cfg.name}")
    print(f"🎯 Sampler: P={cfg.loader.sampler.P}, K={cfg.loader.sampler.K}")
    print(f"🖼️  Image size: {cfg.transforms.img_size}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader, test_loader, info = create_data_loaders_from_config(cfg)

    print("✅ Data loaded:")
    print(f"   📊 Splits: {info['splits']}")
    print(f"   🎯 Sampling: {info['sampling']}")

    # Create model
    model_name = getattr(cfg.model, "name", "simple_cnn")
    embedding_dim = getattr(cfg.model, "embedding_dim", 256)
    input_channels = getattr(cfg.model, "input_channels", 3)
    image_size = getattr(cfg.transforms, "img_size", 256)
    
    # Pass image_size only to models that support it (ViT models)
    model_kwargs = {
        "embedding_dim": embedding_dim,
        "input_channels": input_channels
    }
    if "vit" in model_name.lower() or "deit" in model_name.lower():
        model_kwargs["image_size"] = image_size
    
    model = create_model(model_name, **model_kwargs).to(device)

    # Loss and optimizer
    loss_config = getattr(cfg.model, "loss", {})
    loss_name = getattr(loss_config, "name", "triplet")
    margin = getattr(loss_config, "margin", 0.3)

    optimizer_config = getattr(cfg.model, "optimizer", {})
    learning_rate = getattr(optimizer_config, "lr", 0.001)
    weight_decay = getattr(optimizer_config, "weight_decay", 0.0001)

    criterion = create_loss(loss_name, margin=margin).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Training parameters
    epochs = getattr(cfg.trainer, "epochs", 10)

    model_info = model.get_model_info()
    print(f"\n🏗️  Model: {model_info['model_name']}")
    print(f"   Parameters: {model_info['total_parameters']:,}")
    print(f"   Embedding dim: {model_info['embedding_dim']}")
    print(f"\n⚙️  Training config:")
    print(f"   Loss: {loss_name} (margin={margin})")
    print(f"   Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"   Epochs: {epochs}")
    print("="*70 + "\n")

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = -float("inf")  # Biometric score (TAR - penalty*FAR)
    best_tar = 0.0
    best_far = 1.0
    best_epoch = 0
    
    # Early stopping parameters
    patience = 5 # Stop if no improvement for 10 epochs
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_loss, val_recall, val_metrics = evaluate(model, val_loader, criterion, device, loader_name="Validation")

        # Compute biometric score: TAR - penalty for FAR
        # TAR@FAR=0.1% is good (want to maximize)
        # FAR@FAR=0.1% is bad (want to minimize) - penalize 10x more
        far_penalty_weight = 10.0
        biometric_score = val_metrics['tar_at_far_01'] - far_penalty_weight * val_metrics['far_at_far_01']

        # Print epoch summary with biometric metrics (only validation metrics, not train R@1)
        print(f"Epoch {epoch:3d}/{epochs} │ "
              f"Train Loss: {train_loss:.4f} │ "
              f"Val Loss: {val_loss:.4f} R@1: {val_recall:.4f} Prec@1: {val_metrics['precision_at_1']:.4f} │ "
              f"TAR: {val_metrics['tar_at_far_01']:.4f} FAR: {val_metrics['far_at_far_01']:.6f} Score: {biometric_score:.4f}")

        # Save best model based on biometric score (TAR penalized by FAR)
        if biometric_score > best_val_acc:
            best_val_acc = biometric_score
            best_val_loss = val_loss
            best_tar = val_metrics['tar_at_far_01']
            best_far = val_metrics['far_at_far_01']
            best_epoch = epoch
            patience_counter = 0  # Reset patience counter
            
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_recall": val_recall,
                    "val_metrics": val_metrics,
                    "biometric_score": biometric_score,
                    "config": cfg,
                },
                f"best_model_{cfg.name}.pt",
            )
            print(f"                   💾 Saved best model (Score: {biometric_score:.4f}, TAR: {best_tar:.4f}, FAR: {best_far:.6f})")
        else:
            patience_counter += 1
            if patience_counter % 5 == 0:  # Print every 5 epochs without improvement
                print(f"                   ⏳ No improvement for {patience_counter}/{patience} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
            print(f"   Best epoch was {best_epoch} with Score: {best_val_acc:.4f}")
            break

    # Test only after complete training
    print("\n" + "-"*70)
    print("📋 Final evaluation on test set...")
    test_loss, test_recall, test_metrics = evaluate(model, test_loader, criterion, device, loader_name="Test")
    test_score = test_metrics['tar_at_far_01'] - far_penalty_weight * test_metrics['far_at_far_01']
    
    print(f"\n🎯 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Recall@1: {test_recall:.4f}")
    print(f"   Precision@1: {test_metrics['precision_at_1']:.4f}")
    print(f"   ---")
    print(f"   TAR@FAR=0.1%: {test_metrics['tar_at_far_01']:.4f} ✅ (lepiej)")
    print(f"   FAR@FAR=0.1%: {test_metrics['far_at_far_01']:.6f} ⚠️ (gorzej)")
    print(f"   FRR@FAR=0.1%: {test_metrics['frr_at_far_01']:.4f}")
    print(f"   ---")
    print(f"   Biometric Score: {test_score:.4f}")

    print("\n" + "="*70)
    print(f"✅ Training completed!")
    print(f"   Best validation Biometric Score: {best_val_acc:.4f}")
    print(f"   Best validation TAR@FAR=0.1%: {best_tar:.4f}")
    print(f"   Best validation FAR@FAR=0.1%: {best_far:.6f}")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    train()
