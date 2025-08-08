import os
import lightning as pl
from torch.utils.data import DataLoader
from .scanners import build_manifest
from .splits import make_patient_split
from .datasets import ImageDataset
from .samplers import BalancedBatchSampler
from .transforms import build_transforms

class VascularDataModule(pl.LightningDataModule):
    """Handles datasets, transforms, and loaders for training/validation/test."""
    
    def __init__(self, cfg):
        """Initialize with a config object (e.g., OmegaConf/Hydra)."""
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        """Build manifest, split by patient, and construct datasets."""
        df = build_manifest(self.cfg.data.name, self.cfg.data.root)
        df = make_patient_split(df, **self.cfg.data.split)
        
        self.df_train = df[df.split=="train"].copy()
        self.df_val   = df[df.split=="val"].copy()
        self.df_test  = df[df.split=="test"].copy()

        # Train: augmentations; Eval: deterministic
        t_train = build_transforms(self.cfg.transforms.img_size, True, self.cfg.transforms.hflip_p)
        t_eval  = build_transforms(self.cfg.transforms.img_size, False)

        self.train_ds = ImageDataset(self.df_train, t_train, label_encoder=None)
        self.le = self.train_ds.le  # Cache: patient_id -> int label.
        
        self.val_ds   = ImageDataset(self.df_val, t_eval, label_encoder=None)
        self.test_ds  = ImageDataset(self.df_test, t_eval, label_encoder=None)
        
        # Sanity check: disjoint patients across splits.
        print("Patients:", len(self.df_train.patient_id.unique()), len(self.df_val.patient_id.unique()), len(self.df_test.patient_id.unique()))
        assert set(self.df_train.patient_id.unique()).isdisjoint(self.df_val.patient_id.unique())
        assert set(self.df_train.patient_id.unique()).isdisjoint(self.df_test.patient_id.unique())
        assert set(self.df_val.patient_id.unique()).isdisjoint(self.df_test.patient_id.unique())

    def train_dataloader(self):
        """Return PK-sampled training DataLoader."""
        # Encode patient_ids to contiguous labels for the sampler.
        labels = [self.le[pid] for pid in self.df_train["patient_id"].tolist()]
        
        P = int(self.cfg.data.sampler.P); K = int(self.cfg.data.sampler.K)
        batch_sampler = BalancedBatchSampler(labels, P, K)  # P ids × K samples.
        
        return DataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,    # Mutually exclusive with batch_size/shuffle.
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.data.num_workers > 0,
        )


    def val_dataloader(self):
        """Return validation DataLoader (no shuffle)."""
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.data.num_workers > 0,
        )

    def test_dataloader(self):
        """Return test DataLoader (no shuffle)."""
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.data.num_workers > 0,
        )
