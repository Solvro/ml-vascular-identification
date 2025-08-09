import os, json, argparse, torch, numpy as np
import lightning.pytorch as pl
from PIL import Image

from src.data.transforms import build_transforms
from src.models.factory import get_backbone
from src.models.embedding_module import EmbeddingModule
from src.index.faiss_store import load_index, query as faiss_query

def load_net(cfg, device="cuda"):
    """Build embedding model from cfg and move to device (eval mode)."""
    bb, in_dim = get_backbone(cfg.model.backbone, cfg.model.pretrained, in_chans=3)
    net = EmbeddingModule(bb, in_dim, cfg.model.embed_dim,
                          bn=cfg.model.neck.bn, dropout=cfg.model.neck.dropout,
                          normalize=True).eval().to(device)
    return net

def main():
    """Load index, embed the query image, search FAISS, print results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="path to .index file")
    parser.add_argument("--image", required=True, help="path to query image")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=256)
    args, unknown = parser.parse_known_args()


    # Minimal inline cfg (avoids depending on Hydra at runtime)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class C: pass
    
    cfg = C(); cfg.model = C(); cfg.model.backbone = "resnet50"; cfg.model.pretrained = True
    cfg.model.embed_dim = 256; cfg.model.neck = C(); cfg.model.neck.bn = True; cfg.model.neck.dropout = 0.1

    net = load_net(cfg, device=device)
    use_fp16 = (device == "cuda")
    if use_fp16:
        net = net.half()    # faster/lighter inference on GPU
        
    # Try to load latest training checkpoint (optional).
    ckpt_last = "outputs/checkpoints/last.ckpt"
    if os.path.exists(ckpt_last):
        ckpt = torch.load(ckpt_last, map_location="cpu")
        state = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
        try:
            net.load_state_dict(state, strict=True)
            print(f"Loaded ckpt: {ckpt_last}")
        except Exception as e:
            print("No compatible checkpoint, using pretrained weights.")

    index, meta = load_index(args.index)
    metric = meta.get("metric", "cosine") if meta else "cosine"

    # Preprocess the query image.
    tfm = build_transforms(args.img_size, train=False)
    img = Image.open(args.image).convert("L")
    x = tfm(img).unsqueeze(0)
    
    x = x.to(device, non_blocking=True)
    if use_fp16:
        x = x.half()

    # Embed and move vector to CPU float32 for FAISS.
    with torch.inference_mode():
        z = net(x).detach().to("cpu", dtype=torch.float32).numpy()[0]

    # Search top-K neighbors.
    scores, ids = faiss_query(index, z, topk=args.topk, metric=metric)

    # Pretty print with optional metadata (patient_id, path).
    items = meta["items"] if meta and "items" in meta else []
    for rank, (i, s) in enumerate(zip(ids.tolist(), scores.tolist()), start=1):
        item = items[i] if 0 <= i < len(items) else {}
        print(f"#{rank}: score={s:.4f} patient_id={item.get('patient_id')} path={item.get('path')}")

if __name__ == "__main__":
    main()
