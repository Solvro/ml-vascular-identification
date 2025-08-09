import torch
import glob
import os 
import numpy as np
from omegaconf import DictConfig
import hydra
import lightning.pytorch as pl


from data.datamodule import VascularDataModule
from models.factory import get_backbone
from models.embedding_module import EmbeddingModule

def recall_at_k(emb, labels, ks=(1,5)):
    """Compute Recall@K for L2-normalized embeddings."""
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    sim = emb @ emb.t()
    N = sim.size(0)
    sim.fill_diagonal_(-1e9)    # ignore self-matches
    idx = sim.topk(max(ks), dim=1).indices.cpu().numpy()
    
    y = labels.cpu().numpy()
    recalls = {}
    for k in ks:
        ok = 0
        for i in range(N):
            if any(y[j]==y[i] for j in idx[i,:k]): ok += 1
        recalls[f"R@{k}"] = ok / N
    return recalls

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Load model/ckpt, embed test set, print Recall@K."""
    pl.seed_everything(cfg.train.seed, workers=True)
    
    dm = VascularDataModule(cfg); dm.setup()
    
    bb, in_dim = get_backbone(cfg.model.backbone, cfg.model.pretrained, in_chans=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = EmbeddingModule(bb, in_dim, cfg.model.embed_dim,
                          bn=cfg.model.neck.bn, dropout=cfg.model.neck.dropout,
                          normalize=True).eval().to(device)
    
    # Pick best checkpoint by validation loss (fallback to 'last.ckpt').
    ckpt_dir = "outputs/checkpoints"
    policy = getattr(cfg.eval, "ckpt", "best_r1")  # best_r1 | best_loss | last | <pełna_ścieżka>

    ckpt_path = None
    if os.path.isfile(policy):
        ckpt_path = policy
    elif policy == "best_r1":
        candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*-r1-*-*.ckpt")), reverse=True)
        ckpt_path = candidates[0] if candidates else None
    elif policy == "best_loss":
        candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*-loss-*-*.ckpt")), reverse=True)
        ckpt_path = candidates[0] if candidates else None
    elif policy == "last":
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
        net.load_state_dict(state, strict=True)
    else:
        print("WARNING: no checkpoint found, evaluating ImageNet-pretrained weights.")
    
    xs, ys = [], []
    loader = dm.test_dataloader()
    with torch.no_grad():
        for x, y in loader:
            z = net(x.to(device)).cpu()
            xs.append(z); ys.append(y)
            
    emb = torch.cat(xs, 0); labels = torch.cat(ys, 0)
    r = recall_at_k(emb, labels, ks=tuple(cfg.eval.recall_at))
    for k,v in r.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
