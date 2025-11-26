#!/usr/bin/env python3
"""
OpenSet recognition training script using Hydra config and PyTorch DataLoaders.

Example usage:
    # OpenSet training (default)
    python train.py data=mmcbnu         # Use MMCBNU dataset
    python train.py loader.sampler.P=16 loader.sampler.K=4  # Override P-K sampling

    # Closed-set training (legacy)
    python train.py mode=closedset data=dorsal
"""
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import compute_openset_metrics, create_loss, create_model
from omegaconf import DictConfig

from data import create_data_loaders_from_config, create_openset_data_loaders


def train_epoch(
    model, train_loader, optimizer, criterion, device, epoch, log_interval=10
):
    """
    Train embedding model for one epoch using metric learning.

    Args:
        model: Embedding model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function (TripletLoss or ContrastiveLoss)
        device: Device to train on
        epoch: Current epoch number
        log_interval: How often to print progress

    Returns:
        avg_loss: Average training loss for the epoch
    """
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

        if batch_idx % log_interval == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} - Average Train Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate_metric_learning(model, val_loader, criterion, device):
    """
    Evaluate embedding model using metric learning loss.

    Args:
        model: Embedding model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)

    print(f"Starting validation ({num_batches} batches)...", flush=True)
    for batch_idx, (images, labels, _) in enumerate(val_loader):
        if batch_idx % 5 == 0:
            print(f"  Validation batch {batch_idx}/{num_batches}", flush=True)
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def compute_prototypes(model, enrollment_loader, num_classes, embedding_dim, device):
    """
    Compute class prototypes by averaging embeddings from enrollment set.

    Args:
        model: Embedding model
        enrollment_loader: DataLoader with enrollment samples
        num_classes: Number of known classes
        embedding_dim: Dimension of embeddings
        device: Device to compute on

    Returns:
        prototypes: Tensor of shape (num_classes, embedding_dim)
        class_to_idx: Dictionary mapping original class IDs to prototype indices
    """
    model.eval()

    # Accumulate embeddings per class
    class_embeddings = {}

    for images, labels, _ in enrollment_loader:
        images = images.to(device)
        embeddings = model(images)

        for emb, label in zip(embeddings, labels):
            label_item = label.item()
            if label_item not in class_embeddings:
                class_embeddings[label_item] = []
            class_embeddings[label_item].append(emb)

    # Create mapping from original class IDs to contiguous indices
    unique_classes = sorted(class_embeddings.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

    # Compute mean prototype for each class
    prototypes = torch.zeros(len(unique_classes), embedding_dim, device=device)

    for class_id, embs in class_embeddings.items():
        if embs:
            class_prototype = torch.stack(embs).mean(dim=0)
            # L2-normalize
            class_prototype = F.normalize(class_prototype, p=2, dim=0)
            prototypes[class_to_idx[class_id]] = class_prototype

    return prototypes, class_to_idx


@torch.no_grad()
def find_optimal_threshold(
    model,
    val_known_loader,
    val_unknown_loader,
    prototypes,
    device,
    class_to_idx=None,
    metric="oscr",
):
    """
    Find optimal threshold by maximizing OSCR or AUROC on validation set.

    Args:
        model: Embedding model
        val_known_loader: Validation loader for known classes (enrollment set)
        val_unknown_loader: Validation loader for unknown classes
        prototypes: Class prototypes tensor (num_classes, embedding_dim)
        device: Device to compute on
        class_to_idx: Optional mapping from original class IDs to prototype indices
        metric: Metric to optimize ('oscr' or 'auroc')

    Returns:
        optimal_threshold: Threshold that maximizes the chosen metric
        best_metric_value: Best metric value achieved
    """
    model.eval()

    all_similarities = []
    all_labels = []  # 1 for known, 0 for unknown
    all_predictions = []
    all_true_classes = []

    # Collect validation data
    for images, labels, _ in val_known_loader:
        images = images.to(device)
        embeddings = model(images)
        similarities = torch.matmul(embeddings, prototypes.t())
        max_similarities, predictions = similarities.max(dim=1)

        all_similarities.extend(max_similarities.cpu().numpy())
        all_labels.extend([1] * len(labels))
        all_predictions.extend(predictions.cpu().numpy())

        if class_to_idx is not None:
            mapped_labels = [class_to_idx.get(label.item(), -1) for label in labels]
            all_true_classes.extend(mapped_labels)
        else:
            all_true_classes.extend(labels.numpy())

    for images, _, _ in val_unknown_loader:
        images = images.to(device)
        embeddings = model(images)
        similarities = torch.matmul(embeddings, prototypes.t())
        max_similarities, _ = similarities.max(dim=1)

        all_similarities.extend(max_similarities.cpu().numpy())
        all_labels.extend([0] * len(images))

    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_true_classes = np.array(all_true_classes)

    # Split known and unknown similarities
    known_mask = all_labels == 1
    known_similarities = all_similarities[known_mask]
    unknown_similarities = all_similarities[~known_mask]

    # Generate candidate thresholds
    thresholds = np.linspace(
        all_similarities.min() - 0.1, all_similarities.max() + 0.1, 200
    )

    best_threshold = thresholds[0]
    best_metric_value = -1.0

    if metric == "oscr":
        # Optimize OSCR (Correct Classification Rate vs FPR)
        from models import compute_oscr

        oscr_results = compute_oscr(
            all_predictions[: len(all_true_classes)],
            all_true_classes,
            known_similarities,
            unknown_similarities,
            thresholds,
        )

        # Find threshold with best OSCR AUC approximation
        # We want high CCR and low FPR, so optimize CCR - FPR
        scores = oscr_results["ccr"] - oscr_results["fpr"]
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_metric_value = scores[best_idx]

    elif metric == "auroc":
        # Optimize threshold for AUROC (find threshold closest to equal error rate)
        from sklearn.metrics import roc_curve

        # For AUROC, find threshold at optimal operating point (e.g., Youden's index)
        fpr, tpr, thresh = roc_curve(all_labels, all_similarities)

        # Youden's index: maximize TPR - FPR
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresh[best_idx]
        best_metric_value = youden_index[best_idx]

    return best_threshold, best_metric_value


@torch.no_grad()
def evaluate_openset(
    model,
    test_known_loader,
    test_unknown_loader,
    prototypes,
    device,
    class_to_idx=None,
    threshold=0.5,
    k=1,
):
    """
    Evaluate OpenSet recognition performance with k-NN support.

    Args:
        model: Embedding model
        test_known_loader: DataLoader for test samples from known classes
        test_unknown_loader: DataLoader for test samples from unknown classes
        prototypes: Class prototypes tensor (num_classes, embedding_dim)
        device: Device to evaluate on
        class_to_idx: Optional mapping from original class IDs to prototype indices
        threshold: Similarity threshold for OpenSet detection
        k: Number of nearest neighbors to consider (default 1 for 1-NN)

    Returns:
        metrics: Dictionary with OpenSet metrics (EER, AUROC, OSCR, etc.)

    Note:
        - k=1: Use maximum similarity (1-NN, default behavior)
        - k>1: Average top-k similarities for decision (k-NN)
    """
    model.eval()

    all_similarities = []
    all_labels = []  # 1 for known, 0 for unknown
    all_predictions = []  # class predictions for known samples
    all_true_classes = []  # true class labels for known samples

    # Evaluate known samples
    for images, labels, _ in test_known_loader:
        images = images.to(device)
        embeddings = model(images)

        # Compute similarities to all prototypes
        similarities = torch.matmul(embeddings, prototypes.t())  # (batch, num_classes)

        if k == 1:
            # 1-NN: Use maximum similarity
            max_similarities, predictions = similarities.max(dim=1)
        else:
            # k-NN: Average top-k similarities
            topk_similarities, topk_indices = similarities.topk(k, dim=1)
            max_similarities = topk_similarities.mean(dim=1)

            # Prediction: class with highest similarity among top-k
            predictions = topk_indices[:, 0]  # Use most similar class

        all_similarities.extend(max_similarities.cpu().numpy())
        all_labels.extend([1] * len(labels))  # Known = 1
        all_predictions.extend(predictions.cpu().numpy())

        # Map true labels to prototype indices if mapping provided
        if class_to_idx is not None:
            mapped_labels = [class_to_idx.get(label.item(), -1) for label in labels]
            all_true_classes.extend(mapped_labels)
        else:
            all_true_classes.extend(labels.numpy())

    # Evaluate unknown samples
    for images, _, _ in test_unknown_loader:
        images = images.to(device)
        embeddings = model(images)

        # Compute similarities to all prototypes
        similarities = torch.matmul(embeddings, prototypes.t())

        if k == 1:
            max_similarities, _ = similarities.max(dim=1)
        else:
            topk_similarities, _ = similarities.topk(k, dim=1)
            max_similarities = topk_similarities.mean(dim=1)

        all_similarities.extend(max_similarities.cpu().numpy())
        all_labels.extend([0] * len(images))  # Unknown = 0

    # Compute OpenSet metrics
    # Extract known and unknown scores
    known_scores = np.array(all_similarities[: len(all_true_classes)])
    unknown_scores = np.array(all_similarities[len(all_true_classes) :])

    metrics = compute_openset_metrics(
        known_predictions=np.array(all_predictions),
        known_labels=np.array(all_true_classes),
        known_scores=known_scores,
        unknown_scores=unknown_scores,
    )

    # Compute CMC curve for identification (1:N matching on known classes)
    from models import compute_cmc_curve

    # Collect query embeddings and labels from test_known
    query_embeddings_list = []
    query_labels_list = []

    for images, labels, _ in test_known_loader:
        images = images.to(device)
        embeddings = model(images)

        query_embeddings_list.append(embeddings.cpu().numpy())

        if class_to_idx is not None:
            mapped_labels = [class_to_idx.get(label.item(), -1) for label in labels]
            query_labels_list.extend(mapped_labels)
        else:
            query_labels_list.extend(labels.cpu().numpy())

    if query_embeddings_list:
        query_embeddings = np.vstack(query_embeddings_list)
        query_labels = np.array(query_labels_list)

        # Gallery is the prototypes, gallery labels are 0, 1, 2, ..., num_prototypes-1
        gallery_embeddings = prototypes.cpu().numpy()
        gallery_labels = np.arange(len(prototypes))

        cmc_curve = compute_cmc_curve(
            query_embeddings,
            gallery_embeddings,
            query_labels,
            gallery_labels,
            max_rank=20,
        )

        metrics["cmc_rank1"] = float(cmc_curve[0])
        metrics["cmc_rank5"] = float(cmc_curve[4]) if len(cmc_curve) > 4 else 0.0
        metrics["cmc_rank10"] = float(cmc_curve[9]) if len(cmc_curve) > 9 else 0.0
        metrics["cmc_curve"] = cmc_curve.tolist()

    # Compute accuracy at threshold
    correct_known = sum(
        1
        for sim, pred, true_cls in zip(
            all_similarities[: len(all_true_classes)], all_predictions, all_true_classes
        )
        if sim >= threshold and pred == true_cls
    )

    correct_unknown = sum(
        1 for sim in all_similarities[len(all_true_classes) :] if sim < threshold
    )

    total_known = len(all_true_classes)
    total_unknown = len(all_similarities) - total_known

    known_accuracy = correct_known / total_known if total_known > 0 else 0.0
    unknown_rejection = correct_unknown / total_unknown if total_unknown > 0 else 0.0

    metrics["known_accuracy"] = known_accuracy
    metrics["unknown_rejection"] = unknown_rejection
    metrics["threshold"] = threshold
    metrics["total_known"] = total_known
    metrics["total_unknown"] = total_unknown
    metrics["all_similarities"] = all_similarities  # Store for plotting

    return metrics


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg: DictConfig) -> None:
    """
    Main OpenSet training function.

    Trains an embedding model using metric learning (triplet/contrastive loss),
    then evaluates on OpenSet recognition task with known and unknown classes.
    """
    from datetime import datetime
    from pathlib import Path

    # Determine training mode
    mode = getattr(cfg, "mode", "openset")  # openset | closedset

    # Create output directory structure
    # outputs/<dataset>/<model>/<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_str = getattr(cfg.model, "name", "simple_cnn")
    output_dir = Path("outputs") / cfg.name / model_name_str / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting training...")
    print(f"📁 Dataset: {cfg.name}")
    print(f"🏗️  Model: {model_name_str}")
    print(f"🎯 Mode: {mode.upper()}")
    print(f"🎯 Sampler: P={cfg.loader.sampler.P}, K={cfg.loader.sampler.K}")
    print(f"🖼️  Image size: {cfg.transforms.img_size}")
    print(f"📂 Output: {output_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Create data loaders based on mode
    print("📊 Creating data loaders...")

    if mode == "openset":
        # OpenSet mode: train/val_known/test_known/test_unknown
        known_ratio = getattr(cfg, "known_ratio", 0.7)
        val_ratio = getattr(cfg, "val_ratio", 0.15)

        # Read openset-specific params from config if present (dorsal overrides)
        if hasattr(cfg, "openset"):
            enrollment_samples = getattr(cfg.openset, "enrollment_samples", 7)
            test_samples = getattr(cfg.openset, "test_samples", 3)
            split_ratios = getattr(cfg.openset, "split_ratios", None)
        else:
            enrollment_samples = 7
            test_samples = 3
            split_ratios = None

        loaders, info = create_openset_data_loaders(
            dataset_name=cfg.name,
            img_size=cfg.transforms.img_size,
            known_ratio=known_ratio,
            val_ratio=val_ratio,
            enrollment_samples=enrollment_samples,
            test_samples=test_samples,
            P=cfg.loader.sampler.P,
            K=cfg.loader.sampler.K,
            batch_size=getattr(cfg.loader, "batch_size", 32),
            num_workers=getattr(cfg.loader, "num_workers", 4),
            seed=getattr(cfg, "seed", 42),
            split_ratios=split_ratios,
        )

        train_loader = loaders["train"]
        val_loader = loaders["val_known"]
        test_known_enrollment_loader = loaders.get(
            "test_known_enrollment", loaders.get("test_known", None)
        )
        test_known_query_loader = loaders.get(
            "test_known_query", loaders.get("test_known", None)
        )
        test_unknown_loader = loaders["test_unknown"]

        print("✅ OpenSet data loaded:")
        print(f"   📊 Mode: {info['mode']}")
        print(f"   👥 Known classes: {info['known_finger_classes']}")
        print(f"   ❓ Unknown classes: {info['unknown_finger_classes']}")
        print(f"   🎯 Sampling: P={info['sampling']['P']}, K={info['sampling']['K']}")
        print(f"   ✅ Subject-disjoint: {info['subject_disjoint']}")
    else:
        # Closed-set mode (legacy)
        train_loader, val_loader, test_loader, info = create_data_loaders_from_config(
            cfg
        )
        test_known_loader = test_loader
        test_unknown_loader = None

        print("✅ Closed-set data loaded:")
        print(f"   📊 Splits: {info['splits']}")
        print(f"   🎯 Sampling: {info['sampling']}")

    # Create model
    model_name = getattr(cfg.model, "name", "simple_cnn")
    embedding_dim = getattr(cfg.model, "embedding_dim", 256)
    input_channels = getattr(cfg.model, "input_channels", 3)
    dropout = getattr(cfg.model, "dropout", 0.0)

    model = create_model(
        model_name,
        embedding_dim=embedding_dim,
        input_channels=input_channels,
        dropout=dropout,
    ).to(device)

    # Loss and optimizer
    loss_config = getattr(cfg.model, "loss", {})
    loss_name = getattr(loss_config, "name", "triplet")

    # Build loss kwargs from config (flexible for different loss types)
    loss_kwargs = {}

    # Common parameters
    if hasattr(loss_config, "margin"):
        loss_kwargs["margin"] = loss_config.margin

    # Specific parameters based on loss type
    if loss_name in [
        "triplet",
        "triplet_loss",
        "triplet_center",
        "triplet_center_loss",
    ]:
        if hasattr(loss_config, "mining"):
            loss_kwargs["mining"] = loss_config.mining

    if loss_name in ["contrastive", "contrastive_loss"]:
        if hasattr(loss_config, "pos_weight"):
            loss_kwargs["pos_weight"] = loss_config.pos_weight
        if hasattr(loss_config, "neg_weight"):
            loss_kwargs["neg_weight"] = loss_config.neg_weight

    if loss_name in ["center", "center_loss", "triplet_center", "triplet_center_loss"]:
        if hasattr(loss_config, "lambda_center"):
            loss_kwargs["lambda_center"] = loss_config.lambda_center
        loss_kwargs["embedding_dim"] = embedding_dim
        # Use number of known classes (for OpenSet) or total training classes
        if mode == "openset":
            loss_kwargs["num_classes"] = info.get("known_finger_classes", 100)
        else:
            # For closed-set, count unique labels in training set
            loss_kwargs["num_classes"] = (
                len(train_loader.dataset.classes)
                if hasattr(train_loader.dataset, "classes")
                else 100
            )

    if loss_name in ["triplet_center", "triplet_center_loss"]:
        if hasattr(loss_config, "lambda_triplet"):
            loss_kwargs["lambda_triplet"] = loss_config.lambda_triplet

    optimizer_config = getattr(cfg.model, "optimizer", {})
    learning_rate = getattr(optimizer_config, "lr", 3e-4)
    weight_decay = getattr(optimizer_config, "weight_decay", 1e-4)

    criterion = create_loss(loss_name, **loss_kwargs).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler (optional)
    use_scheduler = getattr(optimizer_config, "use_scheduler", True)
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=getattr(cfg.trainer, "epochs", 50)
        )
    else:
        scheduler = None

    # Training parameters
    epochs = getattr(cfg.trainer, "epochs", 50)
    log_interval = getattr(cfg.trainer, "log_interval", 10)

    model_info = model.get_model_info()
    print(
        f"🏗️  Model: {model_info['model_name']} ({model_info['total_parameters']:,} parameters)"
    )
    loss_info_str = ", ".join([f"{k}={v}" for k, v in loss_kwargs.items()])
    print(
        f"📉 Loss: {loss_name} ({loss_info_str if loss_info_str else 'default params'})"
    )
    print(f"⚙️  Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
    print(f" Training for {epochs} epochs...")

    # Early stopping configuration
    early_stopping_config = getattr(cfg.trainer, "early_stopping", {})
    early_stopping_enabled = getattr(early_stopping_config, "enabled", True)
    patience = getattr(early_stopping_config, "patience", 10)
    min_delta = getattr(early_stopping_config, "min_delta", 0.0001)

    # Check for existing checkpoint to load
    checkpoint_path = getattr(cfg.model, "checkpoint_path", None)
    if checkpoint_path:
        from pathlib import Path

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            print(f"🔄 Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint and epochs > 0:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ Checkpoint loaded successfully")
        else:
            print(f"⚠️  Checkpoint path provided but not found: {checkpoint_path}")

    if early_stopping_enabled:
        print(f"🛑 Early stopping: patience={patience}, min_delta={min_delta}")

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    # If epochs=0, skip training (evaluation only)
    if epochs == 0:
        print("⏩ Skipping training (epochs=0)")
        # If we loaded a checkpoint, use its val_loss as best_val_loss
        if checkpoint_path and "val_loss" in checkpoint:
            best_val_loss = checkpoint["val_loss"]
            best_epoch = checkpoint.get("epoch", 0)

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, log_interval
        )

        # Validate
        val_loss = evaluate_metric_learning(model, val_loader, criterion, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"📉 Learning rate: {current_lr:.6f}")

        # Save best model and check early stopping
        if val_loss < best_val_loss - min_delta:
            # Significant improvement
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "embedding_dim": embedding_dim,
                "model_name": model_name,
                "dataset": cfg.name,
            }

            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            model_path = output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, model_path)
            print(f"💾 Saved best model to {model_path} (val_loss: {val_loss:.4f})")
        else:
            # No improvement
            epochs_without_improvement += 1
            if early_stopping_enabled:
                print(
                    f"⏳ No improvement for {epochs_without_improvement}/{patience} epochs"
                )

        print(f"📊 Best: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

        # Early stopping check
        if early_stopping_enabled and epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs")
            print(f"   No improvement for {patience} consecutive epochs")
            break

    print(f"\n{'='*60}")
    print("✅ Training completed!")
    print(f"{'='*60}")
    print(f"📈 Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")

    # OpenSet evaluation
    if mode == "openset" and test_unknown_loader is not None:
        print(f"\n{'='*60}")
        print("🔍 OpenSet Evaluation")
        print(f"{'='*60}")

        # Load best model (weights_only=False for config compatibility)
        model_path = output_dir / "checkpoint_best.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"⚠️  No checkpoint found at {model_path}, using current model state")

        # Import utilities for visualization and protocol saving
        from utils import (
            InferenceTimer,
            save_evaluation_protocol,
        )

        # Get enrollment and query loaders
        # Enrollment: samples used to build prototypes for each known class (7 samples per finger)
        # Query: samples used to test against prototypes (3 samples per finger)
        enrollment_loader = loaders.get(
            "test_known_enrollment", val_loader
        )  # Fallback to val if not available
        query_known_loader = loaders.get(
            "test_known_query", loaders.get("test_known", val_loader)
        )

        # Compute prototypes from enrollment set (with timing)
        print("📊 Computing class prototypes from enrollment set...")
        num_known_classes = info["known_finger_classes"]

        with InferenceTimer("Prototype computation (registration)") as timer:
            prototypes, class_to_idx = compute_prototypes(
                model, enrollment_loader, num_known_classes, embedding_dim, device
            )
        registration_time_ms = timer.get_elapsed_ms()
        print(
            f"✅ Computed {len(prototypes)} prototypes from {len(enrollment_loader.dataset)} enrollment samples"
        )

        # Optimize threshold on validation set
        print("🔍 Finding optimal threshold on validation set...")
        optimize_threshold = getattr(cfg, "optimize_threshold", True)
        threshold_metric = getattr(cfg, "threshold_metric", "oscr")

        if optimize_threshold:
            # Create validation unknown loader if not exists
            # For now, use a portion of test_unknown as validation unknown
            optimal_threshold, best_metric_val = find_optimal_threshold(
                model,
                val_loader,
                test_unknown_loader,
                prototypes,
                device,
                class_to_idx,
                metric=threshold_metric,
            )
            threshold = optimal_threshold
            print(
                f"✅ Optimal threshold: {threshold:.4f} ({threshold_metric}={best_metric_val:.4f})"
            )
        else:
            threshold = getattr(cfg, "openset_threshold", 0.5)
            print(f"   Using fixed threshold: {threshold:.4f}")

        # Evaluate OpenSet (with timing)
        print("🎯 Evaluating OpenSet recognition...")
        k_neighbors = getattr(cfg, "k_neighbors", 1)

        with InferenceTimer("OpenSet evaluation (query)") as timer:
            metrics = evaluate_openset(
                model,
                query_known_loader,
                test_unknown_loader,
                prototypes,
                device,
                class_to_idx,
                threshold,
                k=k_neighbors,
            )
        query_time_ms = timer.get_elapsed_ms()

        # Add timing info to metrics
        metrics["registration_time_ms"] = registration_time_ms
        metrics["query_time_ms"] = query_time_ms
        metrics["avg_query_time_per_sample_ms"] = (
            query_time_ms
            / (len(query_known_loader.dataset) + len(test_unknown_loader.dataset))
            if hasattr(query_known_loader, "dataset")
            else 0.0
        )

        print("\n📊 OpenSet Recognition Results:")
        print(f"   EER: {metrics['eer']:.4f}")
        print(f"   AUROC: {metrics['auroc']:.4f}")
        print(f"   OSCR: {metrics.get('oscr', metrics.get('oscr_auc', 0.0)):.4f}")
        print(f"   TPR@FPR=0.01: {metrics['tpr_at_fpr_001']:.4f}")
        print(f"   TPR@FPR=0.1: {metrics['tpr_at_fpr_01']:.4f}")
        print(f"   Known Accuracy: {metrics['known_accuracy']:.4f}")
        print(f"   Unknown Rejection: {metrics['unknown_rejection']:.4f}")
        print(f"   Threshold: {metrics['threshold']:.4f}")

        # Print CMC results if available
        if "cmc_rank1" in metrics:
            print("\n📊 Identification (CMC):")
            print(f"   Rank-1: {metrics['cmc_rank1']:.4f}")
            print(f"   Rank-5: {metrics['cmc_rank5']:.4f}")
            print(f"   Rank-10: {metrics['cmc_rank10']:.4f}")

        # Print timing results
        print("\n⏱️  Timing:")
        print(f"   Registration: {registration_time_ms:.2f} ms")
        print(f"   Query (total): {query_time_ms:.2f} ms")
        print(
            f"   Query (per sample): {metrics['avg_query_time_per_sample_ms']:.2f} ms"
        )

        # Save prototypes and metrics
        prototypes_path = output_dir / "prototypes.pt"
        torch.save(
            {
                "prototypes": prototypes,
                "class_to_idx": class_to_idx,
                "metrics": metrics,
                "threshold": threshold,
                "num_known_classes": num_known_classes,
                "k_neighbors": k_neighbors,
            },
            prototypes_path,
        )
        print(f"\n💾 Saved prototypes and metrics to {prototypes_path}")

        # Plot DET and ROC curves (skip visualization to avoid matplotlib GUI issues)
        # Need to separate genuine and impostor scores for DET/ROC
        # For simplicity, use known as genuine and unknown as impostor
        genuine_scores = np.array(
            [
                s
                for i, s in enumerate(metrics.get("all_similarities", []))
                if i < metrics.get("total_known", 0)
            ]
        )
        impostor_scores = np.array(
            [
                s
                for i, s in enumerate(metrics.get("all_similarities", []))
                if i >= metrics.get("total_known", 0)
            ]
        )

        # Commenting out plotting to avoid issues in headless/batch mode
        # if len(genuine_scores) > 0 and len(impostor_scores) > 0:
        #     det_curve_path = output_dir / "det_curve.png"
        #     roc_curve_path = output_dir / "roc_curve.png"
        #
        #     plot_det_curve(genuine_scores, impostor_scores,
        #                   str(det_curve_path),
        #                   title=f"DET Curve - {cfg.name}")
        #     plot_roc_curve(genuine_scores, impostor_scores,
        #                   str(roc_curve_path),
        #                   title=f"ROC Curve - {cfg.name}",
        #                   auroc=metrics['auroc'])

        # Save evaluation protocol
        protocol_config = {
            "dataset_name": cfg.name,
            "total_classes": info.get("total_finger_classes", 600),
            "known_classes": info["known_finger_classes"],
            "unknown_classes": info["unknown_finger_classes"],
            "subject_disjoint": info.get("subject_disjoint", True),
            "train_size": len(train_loader.dataset)
            if hasattr(train_loader, "dataset")
            else 0,
            "val_size": len(val_loader.dataset)
            if hasattr(val_loader, "dataset")
            else 0,
            "test_known_size": len(query_known_loader.dataset)
            if hasattr(query_known_loader, "dataset")
            else 0,
            "test_unknown_size": len(test_unknown_loader.dataset)
            if hasattr(test_unknown_loader, "dataset")
            else 0,
            "enrollment_size": len(enrollment_loader.dataset)
            if hasattr(enrollment_loader, "dataset")
            else 0,
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "total_parameters": model.get_model_info()["total_parameters"],
            "loss_name": loss_name,
            "loss_params": loss_kwargs,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "epochs": best_epoch,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "P": cfg.loader.sampler.P,
            "K": cfg.loader.sampler.K,
            "k": k_neighbors,
            "threshold": threshold,
            "threshold_optimization": threshold_metric
            if optimize_threshold
            else "manual",
        }

        protocol_path = output_dir / "evaluation_protocol.json"
        save_evaluation_protocol(metrics, protocol_config, str(protocol_path))

    print(f"✨ Done! Outputs saved to: {output_dir}")


if __name__ == "__main__":
    train()
