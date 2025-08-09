import os, json, time
import numpy as np
import faiss

def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize rows (float32)."""
    v = v.astype("float32", copy=False)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def build_index(vectors: np.ndarray, metric: str = "cosine") -> faiss.Index:
    """Create index (cosine via IP on normalized vecs, or L2) and add data."""
    d = int(vectors.shape[1])
    if metric == "cosine":
        vectors = _normalize(vectors)
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(metric)
    index.add(vectors.astype("float32"))    # FAISS expects float32
    return index

def query(index: faiss.Index, vector: np.ndarray, topk: int = 5, metric: str = "cosine"):
    """Search top-k; returns (distances, indices)."""
    q = vector.astype("float32")[None, :]
    if metric == "cosine":
        q = _normalize(q)
    D, I = index.search(q, topk)
    return D[0], I[0]

def save_index(index: faiss.Index, meta: dict, out_dir: str, base: str):
    """Write index to .index and metadata to .meta.json; return paths."""
    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, base + ".index")
    meta_path = os.path.join(out_dir, base + ".meta.json")
    faiss.write_index(index, idx_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return idx_path, meta_path

def load_index(idx_path: str):
    """Read index and optional sidecar metadata."""
    index = faiss.read_index(idx_path)
    meta_path = os.path.splitext(idx_path)[0] + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return index, meta

def timestamp() -> str:
    """Return YYYYMMDD-HHMMSS string."""
    return time.strftime("%Y%m%d-%H%M%S")
