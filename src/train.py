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
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from data import create_data_loaders_from_config


class SimpleEmbeddingModel(nn.Module):
    """Simple CNN for embedding learning."""

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return nn.functional.normalize(x, p=2, dim=1)


class TripletLoss(nn.Module):
    """Simple triplet loss."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """Compute batch hard triplet loss."""
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks
        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()

        # Remove diagonal
        pos_mask.fill_diagonal_(0)

        # Hard positive and hard negative mining
        pos_dist = distances * pos_mask
        neg_dist = (
            distances * neg_mask + 1e6 * pos_mask
        )  # Add large value to positive pairs

        hard_pos = pos_dist.max(dim=1)[0]
        hard_neg = neg_dist.min(dim=1)[0]

        # Triplet loss
        loss = torch.relu(hard_pos - hard_neg + self.margin)
        return loss.mean()


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0

    for images, labels, _ in val_loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    print("🚀 Starting training...")
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
    embedding_dim = getattr(cfg.trainer, "embedding_dim", 256)
    model = SimpleEmbeddingModel(embedding_dim=embedding_dim).to(device)

    # Loss and optimizer
    criterion = TripletLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    epochs = getattr(cfg.trainer, "epochs", 10)

    print(f"🏗️  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🔄 Training for {epochs} epochs...")

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                },
                f"best_model_{cfg.name}.pt",
            )
            print(f"💾 Saved best model (val_loss: {val_loss:.4f})")

        print("-" * 50)

    print(f"✅ Training completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
