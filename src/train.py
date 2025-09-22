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
    model_name = getattr(cfg.model, "name", "simple_cnn")
    embedding_dim = getattr(cfg.model, "embedding_dim", 256)
    input_channels = getattr(cfg.model, "input_channels", 3)
    model = create_model(
        model_name, embedding_dim=embedding_dim, input_channels=input_channels
    ).to(device)

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
    print(
        f"🏗️  Model: {model_info['model_name']} ({model_info['total_parameters']:,} parameters)"
    )
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
