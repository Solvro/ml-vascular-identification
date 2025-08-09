import os, glob
import numpy as np
import torch
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.datamodule import VascularDataModule
from src.data.datasets import ImageDataset
from src.data.transforms import build_transforms
from src.models.factory import get_backbone
from src.models.embedding_module import EmbeddingModule
from src.index.faiss_store import build_index, save_index, timestamp

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Encode split → build FAISS index → save index + metadata."""
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pl.seed_everything(cfg.train.seed, workers=True)

    # Model + device
    bb, in_dim = get_backbone(cfg.model.backbone, cfg.model.pretrained, in_chans=3)
    device = "cuda" if (getattr(getattr(cfg, "index", {}), "device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
    net = EmbeddingModule(bb, in_dim, cfg.model.embed_dim,
                          bn=cfg.model.neck.bn, dropout=cfg.model.neck.dropout,
                          normalize=True).eval().to(device)

    use_fp16 = bool(getattr(getattr(cfg, "index", {}), "fp16", True)) and device == "cuda"
    if use_fp16:
        net = net.half()

    # Checkpoint policy
    ckpt_dir = "outputs/checkpoints"
    policy = getattr(cfg.eval, "ckpt", "last")
    ckpt_path = None
    if os.path.isfile(policy):
        ckpt_path = policy
    elif policy == "best_r1":
        c = sorted(glob.glob(os.path.join(ckpt_dir, "*-r1-*-*.ckpt")), reverse=True)
        ckpt_path = c[0] if c else None
    elif policy == "best_loss":
        c = sorted(glob.glob(os.path.join(ckpt_dir, "*-loss-*-*.ckpt")), reverse=True)
        ckpt_path = c[0] if c else None
    else:
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Strip Lightning's "net." prefix.
        state = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
        net.load_state_dict(state, strict=True)
        print(f"Loaded ckpt: {ckpt_path}")
    else:
        print("WARNING: no checkpoint found; using ImageNet-pretrained weights.")

    # Data to index
    split = getattr(getattr(cfg, "index", {}), "split", "train")
    metric = getattr(getattr(cfg, "index", {}), "metric", "cosine")
    save_dir = getattr(getattr(cfg, "index", {}), "save_dir", "outputs/index")

    dm = VascularDataModule(cfg); dm.setup()
    df = {"train": dm.df_train, "val": dm.df_val, "test": dm.df_test}[split]
    t_eval = build_transforms(cfg.transforms.img_size, train=False)
    ds = ImageDataset(df, t_eval, label_encoder=None)
    
    idx_bs = int(getattr(getattr(cfg, "index", {}), "batch_size", min(32, cfg.data.batch_size)))
    loader = DataLoader(
        ds, batch_size=idx_bs, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True, persistent_workers=cfg.data.num_workers > 0
    )

    # Encode all images
    zs = []
    with torch.inference_mode():
        for x, _ in loader:
            if device == "cuda":
                x = x.to(device, non_blocking=True)
                if use_fp16:
                    x = x.half()
            z = net(x).detach().to("cpu", dtype=torch.float32)
            zs.append(z)
            
            if device == "cuda":
                torch.cuda.empty_cache()
                
    emb = torch.cat(zs, 0).numpy()
    meta_rows = df[["patient_id", "path"]].to_dict(orient="records")

    # Build and save FAISS index
    index = build_index(emb, metric=metric)
    base = f"{cfg.data.name}-{cfg.model.backbone}-{metric}-{split}-{timestamp()}"
    meta = {
        "dataset": cfg.data.name,
        "backbone": cfg.model.backbone,
        "metric": metric,
        "split": split,
        "dim": int(emb.shape[1]),
        "count": int(emb.shape[0]),
        "items": meta_rows,
    }
    idx_path, meta_path = save_index(index, meta, save_dir, base)
    print("Saved:", idx_path)
    print("Saved:", meta_path)

if __name__ == "__main__":
    main()
