"""
Custom data samplers for balanced batch sampling.

Implements PK-sampling where each batch contains P classes with K samples each.
"""
import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Yield batches of size P*K: P classes, K samples per class.

    This is useful for metric learning where you want each batch to contain
    multiple examples from the same classes for computing triplet/contrastive losses.
    """

    def __init__(self, labels: List[int], P: int, K: int):
        """Initialize sampler.

        Args:
            labels: List of integer class labels for all samples.
            P: Number of distinct classes per batch.
            K: Number of samples per class in each batch.

        Note:
            If a class has fewer than K samples, sampling will be done with replacement.
        """
        self.labels = [int(y) for y in labels]
        self.P, self.K = int(P), int(K)

        # Map class -> list of sample indices
        self.index_by_class = defaultdict(list)
        for i, y in enumerate(self.labels):
            self.index_by_class[y].append(i)

        self.classes = list(self.index_by_class.keys())

        if len(self.classes) < self.P:
            raise ValueError(f"Not enough classes ({len(self.classes)}) for P={self.P}")

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches of indices.

        Yields:
            List of indices for each batch of size P*K.
        """
        classes = self.classes.copy()
        random.shuffle(classes)

        i = 0
        while i + self.P <= len(classes):
            batch_classes = classes[i : i + self.P]
            i += self.P

            batch_indices = []
            for class_label in batch_classes:
                class_indices = self.index_by_class[class_label]

                if len(class_indices) >= self.K:
                    # Sample without replacement
                    selected = random.sample(class_indices, self.K)
                else:
                    # Sample with replacement if not enough samples
                    selected = [random.choice(class_indices) for _ in range(self.K)]

                batch_indices.extend(selected)

            # Shuffle indices within batch to mix classes
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        """Number of batches per epoch.

        Returns:
            Number of complete batches that can be formed.
        """
        return len(self.classes) // self.P

    def get_stats(self) -> dict:
        """Get statistics about the sampler.

        Returns:
            Dictionary with sampler statistics.
        """
        class_sizes = {
            cls: len(indices) for cls, indices in self.index_by_class.items()
        }

        return {
            "total_classes": len(self.classes),
            "total_samples": len(self.labels),
            "P": self.P,
            "K": self.K,
            "batch_size": self.P * self.K,
            "batches_per_epoch": len(self),
            "min_class_size": min(class_sizes.values()),
            "max_class_size": max(class_sizes.values()),
            "avg_class_size": sum(class_sizes.values()) / len(class_sizes),
            "classes_with_replacement": sum(
                1 for size in class_sizes.values() if size < self.K
            ),
        }
