from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    """Reads images listed in a DataFrame and encodes patient_id labels."""
    
    def __init__(self, df, transform=None, label_encoder=None):
        """Initialize dataset.

        Args:
            df: Pandas DataFrame with at least ["path", "patient_id"] columns.
            transform: Optional callable applied to PIL image.
            label_encoder: Optional dict {patient_id: int}; built if None.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        if label_encoder is None:
            pids = sorted(self.df["patient_id"].unique().tolist())
            self.le = {pid:i for i, pid in enumerate(pids)} # contiguous labels
        else:
            self.le = label_encoder

    def __len__(self):
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load item at index: (image, label)."""
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("L")  # force grayscale
        if self.transform:
            img = self.transform(img)
        label = self.le[row["patient_id"]]
        return img, label
