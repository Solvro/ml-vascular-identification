import random
from collections import defaultdict
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    """Yield index lists of size P*K: P classes, K samples per class."""
    
    def __init__(self, labels, P, K):
        """Args:
        labels: iterable of int class labels.
        P: number of distinct classes per batch.
        K: number of samples per class.
        """
        self.labels = [int(y) for y in labels]
        self.P, self.K = int(P), int(K)
        
        # Map class -> list of sample indices.
        self.index_by_class = defaultdict(list)
        for i, y in enumerate(self.labels):
            self.index_by_class[y].append(i)
        self.classes = list(self.index_by_class.keys())

    def __iter__(self):
        """Iterate over one pass of shuffled classes, yielding batches."""
        classes = self.classes.copy()
        random.shuffle(classes)
        
        i = 0
        while i + self.P <= len(classes):
            batch_classes = classes[i:i+self.P]
            i += self.P
            
            batch_idx = []
            for c in batch_classes:
                pool = self.index_by_class[c]
                if len(pool) >= self.K:
                    idxs = random.sample(pool, self.K)  # without replacement
                else:
                    idxs = [random.choice(pool) for _ in range(self.K)]  # with replacement
                batch_idx.extend(idxs)
                
            random.shuffle(batch_idx)   # mix classes within batch
            yield batch_idx 

    def __len__(self):
        """Number of batches per epoch (one sweep over classes)."""
        return len(self.classes) // self.P
