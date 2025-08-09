FAISS “VOLATILE” INDEXING — QUICK GUIDE (Windows, PowerShell)
Project root: C:\Users\pawel\Desktop\Vain\ml-vascular-identification

============================================================
0) PREREQUISITES (run once)
============================================================
# If you ever see an OpenMP runtime error (OMP: Error #15), set:
setx KMP_DUPLICATE_LIB_OK TRUE
setx OMP_NUM_THREADS 1
# (Restart the terminal after setx.)

Tip: Close GPU-hungry apps (browser/VS Code hardware acceleration, Discord, Spotify) during indexing.

============================================================
1) BUILD AN INDEX FROM A DATASET SPLIT
============================================================
What it does:
- Loads your trained model checkpoint (e.g., last.ckpt),
- embeds all images from a chosen split (train/val/test),
- builds a FAISS index and saves:
  - outputs\index\<name>.index
  - outputs\index\<name>.meta.json (paths + patient_id list)

Command (Dorsal, TRAIN split, using the “last” checkpoint):
uv run python -m src.index.build_index data=dorsal eval.ckpt=last index.split=train

Useful overrides (reduce GPU load if needed):
uv run python -m src.index.build_index data=dorsal eval.ckpt=last index.split=train index.batch_size=16 transforms.img_size=224

Build for MMCBNU:
uv run python -m src.index.build_index data=mmcbnu eval.ckpt=last index.split=train index.batch_size=16 transforms.img_size=224

Expected console output:
Saved: outputs\index\dorsal-resnet50-cosine-<split>-YYYYMMDD-HHMMSS.index
Saved: outputs\index\dorsal-resnet50-cosine-<split>-YYYYMMDD-HHMMSS.meta.json

Notes:
- eval.ckpt can be: last | best_r1 | best_loss | <full_path_to_ckpt>
- You can also build for val/test splits: index.split=val or index.split=test

============================================================
2) QUERY THE INDEX (TOP-K NEIGHBORS FOR ONE IMAGE)
============================================================
What it does:
- Embeds a single image,
- searches the given FAISS index,
- prints the top-K nearest entries (score, patient_id, path).

Command (query a Dorsal image):
uv run python -m src.index.query --index "outputs\index\dorsal-resnet50-cosine-train-20250809-193459.index" --image "C:\Users\pawel\Desktop\Vain\Dorsal_Hand_Vein\P001\Left\P001_Left_L1.png" --topk 5 --img_size 256

Example output:
Loaded ckpt: outputs/checkpoints/last.ckpt
#1: score=1.0000 patient_id=001 path=C:\...\P001_Left_L1.png
#2: score=0.8710 patient_id=001 path=C:\...\P001_Left_L2.png
#3: score=0.8265 patient_id=001 path=C:\...\P001_Left_L3.png
#4: score=0.8106 patient_id=001 path=C:\...\P001_Left_L4.png
#5: score=0.7789 patient_id=001 path=C:\...\P001_Right_R4.png

Notes:
- Self-match (the exact same file) is allowed by default; can be filtered later.

============================================================
3) ADD A NEW PATIENT TO AN EXISTING INDEX (NO RETRAINING)
============================================================
What it does:
- Embeds images matching --images-glob,
- appends them to an existing FAISS index,
- saves a NEW index file with suffix: -added-<patient>-<timestamp>

Command (simulate a “new” patient using a TEST patient):
uv run python -m src.index.add_patient --index "outputs\index\dorsal-resnet50-cosine-train-20250809-193459.index" --patient-id 201 --images-glob "C:\Users\pawel\Desktop\Vain\Dorsal_Hand_Vein\P201\Left\*.png"

Expected output (new file paths):
Saved: outputs\index\dorsal-resnet50-cosine-train-20250809-193459-added-201-YYYYMMDD-HHMMSS.index
Saved: outputs\index\dorsal-resnet50-cosine-train-20250809-193459-added-201-YYYYMMDD-HHMMSS.meta.json

(Optional) Add right-hand images too — IMPORTANT: use the NEW index from above:
uv run python -m src.index.add_patient --index "outputs\index\dorsal-resnet50-cosine-train-20250809-193459-added-201-YYYYMMDD-HHMMSS.index" --patient-id 201 --images-glob "C:\Users\pawel\Desktop\Vain\Dorsal_Hand_Vein\P201\Right\*.png"

Query the updated index:
uv run python -m src.index.query --index "outputs\index\dorsal-resnet50-cosine-train-20250809-193459-added-201-YYYYMMDD-HHMMSS.index" --image "C:\Users\pawel\Desktop\Vain\Dorsal_Hand_Vein\P201\Left\P201_Left_L1.png" --topk 5 --img_size 256

Expected: patient_id=201 appears in the top results (often #1).

============================================================
4) WHAT EACH SCRIPT DOES
============================================================
src/index/build_index.py
- Embeds a chosen split (train/val/test) with your model and builds a FAISS index.
- Inputs: Hydra args (data=..., eval.ckpt=..., index.split=...).
- Outputs: .index + .meta.json under outputs\index\...
- Prints the saved file paths.

src/index/query.py
- Embeds one query image and retrieves top-K nearest neighbors from a FAISS index.
- Inputs: --index, --image, --topk, --img_size
- Prints: score, patient_id, path for each neighbor.

src/index/add_patient.py
- Embeds images (glob) for a new patient and appends them to an existing index.
- Inputs: --index, --patient-id, --images-glob, --img_size
- Outputs: a NEW .index + .meta.json with “-added-<patient>-<timestamp>”.

src/index/faiss_store.py (utility)
- build_index(vectors, metric): create FAISS index (cosine or L2).
- query(index, vector, topk, metric): return (scores, indices).
- save_index(index, meta, out_dir, base): write .index + .meta.json.
- load_index(path): load both index and metadata.

============================================================
5) CHECKPOINT SELECTION & COMMON OVERRIDES
============================================================
- Default policy in build/eval can be controlled via Hydra:
  eval.ckpt=last | best_r1 | best_loss | <full_ckpt_path>

Examples:
uv run python -m src.index.build_index data=dorsal eval.ckpt=best_r1 index.split=test
uv run python -m src.eval data=dorsal eval.ckpt=last

Reduce GPU load during indexing:
- Smaller batch: index.batch_size=8 or 16
- Smaller input: transforms.img_size=224
Example:
uv run python -m src.index.build_index data=dorsal eval.ckpt=last index.split=train index.batch_size=16 transforms.img_size=224

Force CPU for indexing (slow but safest on Windows):
uv run python -m src.index.build_index data=dorsal eval.ckpt=last index.split=train index.device=cpu

============================================================
6) TROUBLESHOOTING
============================================================
OpenMP conflict (OMP: Error #15):
- Set both:
  setx KMP_DUPLICATE_LIB_OK TRUE
  setx OMP_NUM_THREADS 1
- Open a new terminal and retry.

CUDA driver “black screen” / hangs:
- Use smaller index.batch_size and/or transforms.img_size=224,
- Close other GPU apps, or build on CPU (index.device=cpu).

Self-match in query (score=1.0 for the same file):
- This is expected if the exact same file is in the index.
- We can add a filter in query.py later to skip the same path.

============================================================
END
============================================================
