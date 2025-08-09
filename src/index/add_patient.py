import os, glob, json, argparse, torch
import numpy as np
from PIL import Image
from src.data.transforms import build_transforms
from src.models.factory import get_backbone
from src.models.embedding_module import EmbeddingModule
from src.index.faiss_store import load_index, save_index, _normalize, timestamp
import faiss

def load_net(device="cuda"):
    """Build embedding model and (optionally) load last checkpoint."""
    bb, in_dim = get_backbone("resnet50", True, in_chans=3)
    from types import SimpleNamespace
    neck = SimpleNamespace(bn=True, dropout=0.1)
    net = EmbeddingModule(bb, in_dim, 256, bn=neck.bn, dropout=neck.dropout, normalize=True).eval().to(device)
    
    # Load weights if a Lightning checkpoint exists (ignore if keys don't match).
    ckpt_last = "outputs/checkpoints/last.ckpt"
    if os.path.exists(ckpt_last):
        ckpt = torch.load(ckpt_last, map_location="cpu")
        state = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
        try: net.load_state_dict(state, strict=True)
        except: pass
    return net

def main():
    """Load an index, embed images from a glob, add vectors, and re-save."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="path to existing .index")
    ap.add_argument("--patient-id", required=True)
    ap.add_argument("--images-glob", required=True, help='e.g. "C:/path/patientX/*.png"')
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    index, meta = load_index(args.index)    # load FAISS index + sidecar metadata
    if meta is None:
        meta = {}
    meta.setdefault("items", [])
        
    metric = meta.get("metric", "cosine") if meta else "cosine"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")   # half precision only on CUDA

    net = load_net(device=device)
    if use_fp16:
        net = net.half()
        
    tfm = build_transforms(args.img_size, train=False)  # eval transforms

    paths = sorted(glob.glob(args.images_glob))
    zs = []
    with torch.inference_mode():    # no grads while encoding
        for p in paths:
            img = Image.open(p).convert("L")    # force grayscale input
            x = tfm(img).unsqueeze(0).to(device, non_blocking=True)
            if use_fp16:
                x = x.half()
            # Move embedding to CPU as float32 for FAISS.
            z = net(x).detach().to("cpu", dtype=torch.float32).numpy()[0]
            zs.append(z)
            
    if not zs:
        print("No images found.")
        return

    vecs = np.stack(zs).astype("float32")
    if metric == "cosine":
        vecs = _normalize(vecs) # cosine needs L2-normalized vectors
    index.add(vecs) # append to FAISS index

    # Update metadata with new items and total count.
    meta["items"].extend([{"patient_id": args.patient_id, "path": p} for p in paths])
    meta["count"] = int(index.ntotal)

    # Save as a new versioned index (keep original intact).
    base_old = os.path.splitext(os.path.basename(args.index))[0]
    out_dir = os.path.dirname(args.index)
    base_new = base_old + f"-added-{args.patient_id}-{timestamp()}"
    idx_path, meta_path = save_index(index, meta, out_dir, base_new)
    print("Saved:", idx_path)
    print("Saved:", meta_path)

if __name__ == "__main__":
    main()
