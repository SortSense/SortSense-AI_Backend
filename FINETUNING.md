# SortSense-AI_Backend — YOLO26 Fine-Tuning Guide & Pipeline Documentation

This document records the complete end-to-end dataset preparation, cleaning, stratified splitting, Google Colab GPU training, and local deployment workflow used to fine-tune **Ultralytics YOLO26 Nano (`yolo26n.pt`)** for the SortSense Smart Waste Bin.

---

## 1. Why Fine-Tune `yolo26n.pt`?

* The base Ultralytics **`yolo26n.pt`** checkpoint (5.3 MB, powered by `ultralytics`) is pretrained on the 80-class **COCO** dataset, which does not include dedicated recyclable categories (`paper`, `cardboard`, `metal` cans, or general waste).
* For example, testing untuned COCO `yolo26n.pt` on a Pepsi can (`test_image.jpg`) predicted `bottle` at `0.486` confidence (below our `0.50` `CONFIDENCE_THRESHOLD`).
* Fine-tuning adapts the YOLO26 Nano detection head specifically to SortSense's **4 standardized bin classes** while maintaining low-latency CPU inference on the local LAN server.

| Class ID | SortSense Category | Source Materials Included |
|:---:|:---:|---|
| `0` | `plastic` | Plastic bottles, cups, bags, wrappers |
| `1` | `paper` | Paper sheets, cardboard, cartons, boxes |
| `2` | `metal` | Aluminum soda cans, tin cans, metal foil |
| `3` | `general` | Biodegradable/food waste and general non-recyclables |
| `-1` | `none` | Empty camera view or detections below `CONFIDENCE_THRESHOLD` (`0.50`) |

---

## 2. Raw Dataset Inspection & Deduplication

Four Roboflow YOLO26 exports were placed in `dataset/raw/`:

| Raw Archive (`dataset/raw/`) | Images | Original Source |
|---|---:|---|
| `GARBAGE CLASSIFICATION 3.v2-gc1.yolo26.zip` | 10,464 | `material-identification/garbage-classification-3` |
| `GARBAGE CLASSIFICATION.v1-garbage-classification.yolo26.zip` | 7,327 | `material-identification/garbage-classification-pfla2` |
| `METAL.v1-metal.yolo26.zip` | 1,248 | `material-identification/metal-mubv0` |
| `PAPER.v1-paper.yolo26.zip` | 1,746 | `material-identification/paper-rjcig` |

### Why Only `GARBAGE CLASSIFICATION 3` Was Used
Inspection of filename stems and MD5 hashes across all 4 zips showed that all 4 datasets came from the same Roboflow workspace (`material-identification`), and `GARBAGE CLASSIFICATION 3.v2-gc1.yolo26.zip` already contained:
* `1,248 / 1,248` (**100%**) of `METAL.v1`
* `1,746 / 1,746` (**100%**) of `PAPER.v1`
* `7,326 / 7,327` (**99.99%**) of `GARBAGE CLASSIFICATION.v1`

Merging all 4 zips naively would have duplicated ~10,320 images across `train`/`val`/`test` splits and caused **data leakage** (artificially inflated validation scores). Using **`GARBAGE CLASSIFICATION 3.v2-gc1.yolo26.zip`** alone eliminated duplicates while keeping every unique sample.

---

## 3. Class Remapping & Separating `GLASS`

### Class Mapping
`GARBAGE CLASSIFICATION 3` uses 6 raw classes (`nc: 6`):
* Raw `5` (`PLASTIC`) → **`0` (`plastic`)**
* Raw `1` (`CARDBOARD`) + Raw `4` (`PAPER`) → **`1` (`paper`)**
* Raw `3` (`METAL`) → **`2` (`metal`)**
* Raw `0` (`BIODEGRADABLE`) → **`3` (`general`)**
* Raw `2` (`GLASS`) → **Moved to `dataset/glass_archive/`**

### How `GLASS` Was Handled
Because SortSense currently deploys 4 bins (`plastic`, `paper`, `metal`, `general`) and campus bins rarely receive glass:
* All **2,684 images containing `GLASS`** (`1,821` glass-only images + `863` mixed-class images containing glass) were separated into **`dataset/glass_archive/`** with labels remapped to single-class `0: glass` and a dedicated `dataset/glass_archive/dataset.yaml`.
* Keeping those 863 mixed-class images out of the 4-class training set ensures YOLO never sees unlabeled glass bottles next to labeled plastic bottles (avoiding confusion between clear plastic and glass).
* Only the **7,780 pure non-glass images** were kept in the main 4-class training dataset (`dataset/`).

---

## 4. Stratified Re-Splitting & Label Validation

### Fixing Roboflow's Imbalanced `val`/`test` Split
In the raw export, Roboflow's default split was heavily skewed across splits (for example, `valid` contained only **9 `plastic` images** out of 1,293, while `test` had **354 `plastic` images** and only **8 `general` images**). Because YOLO selects `best.pt` using `val` metrics, an imbalanced `val` split hides `plastic` accuracy.

All **7,780 non-glass images** (and the **2,684 `glass_archive` images**) were re-split using a deterministic stratified split (`random.Random(42)`) at **75% `train` / 15% `val` / 10% `test`**:

#### Main 4-Class Dataset (`dataset/`)
| Split | Total Images | `plastic (0)` | `paper (1)` | `metal (2)` | `general (3)` | Label Validation (`validate_yolo_labels`) |
|---|---:|---:|---:|---:|---:|---|
| `train` | **5,833** | 903 | 2,422 | 981 | 1,661 | 5,833 valid / **0 invalid** (47,487 boxes) |
| `val` | **1,164** | 180 | 486 | 197 | 332 | 1,164 valid / **0 invalid** (8,191 boxes) |
| `test` | **783** | 122 | 320 | 133 | 227 | 783 valid / **0 invalid** (7,125 boxes) |
| **Total** | **7,780** | **1,205** | **3,228** | **1,311** | **2,220** | **7,780 valid / 0 invalid (62,803 boxes)** |

#### Separated Glass Archive (`dataset/glass_archive/`)
* `train`: **2,013** images | `val`: **402** images | `test`: **269** images (all `0: glass`)

### Directory Layout (`dataset/`)
```text
dataset/                          # Gitignored (.gitignore)
├── dataset.yaml                  # Main 4-class YOLO config
├── colab_dataset.zip             # Clean upload bundle (15,561 files, 151.99 MB)
├── images/
│   ├── train/                    # 5,833 images
│   ├── val/                      # 1,164 images
│   └── test/                     # 783 images
├── labels/
│   ├── train/                    # 5,833 .txt files
│   ├── val/                      # 1,164 .txt files
│   └── test/                     # 783 .txt files
├── glass_archive/                # Separated 2,684 glass images (0: glass)
│   ├── dataset.yaml
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── raw/                          # Original downloaded Roboflow .zip archives
```

---

## 5. Training on Google Colab (GPU)

Because local PyTorch on the development laptop is CPU-only (`torch.cuda.is_available() == False`), training is offloaded to **Google Colab's free T4 GPU** using the lightweight upload bundle `dataset/colab_dataset.zip` (**15,561 files, 151.99 MB** — excluding `raw/` and `glass_archive/`).

### Colab Execution Steps

1. **Select GPU Runtime:**
   In Google Colab: **Runtime → Change runtime type → T4 GPU → Save**.

2. **Mount Google Drive & Unzip to Local SSD (`/content/dataset`):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !mkdir -p /content/dataset
   !unzip -q "/content/drive/MyDrive/colab_dataset.zip" -d /content/dataset
   !ls /content/dataset
   ```
   > **Important:** `drive.mount(...)` must always point to the mount folder `/content/drive` (never directly to a `.zip` file). Unzipping onto `/content/dataset` avoids reading 7,780 files slowly over the Drive network mount.

3. **Fix `dataset.yaml` Path & Train `yolo26n.pt`:**
   Ultralytics resolves relative `path: .` against Colab's working directory (`/content`) rather than the folder containing `dataset.yaml` (`/content/dataset`). Updating `path:` to `/content/dataset` resolves this cleanly:
   ```python
   !pip install -q ultralytics

   !sed -i 's|^path: .*|path: /content/dataset|' /content/dataset/dataset.yaml
   !cat /content/dataset/dataset.yaml

   from ultralytics import YOLO

   model = YOLO("yolo26n.pt")
   model.train(
       data="/content/dataset/dataset.yaml",
       epochs=50,
       imgsz=640,        # Matches ESP32-S3 OV5640 640x480 frames
       batch=16,
       device=0,         # T4 GPU
       patience=10,      # Early stopping if val mAP plateaus
       project="/content/runs",
       name="sortsense_yolo26",
   )
   ```

4. **Evaluate on the Held-Out `test` Split & Save `best.pt`:**
   ```python
   import glob, os, shutil
   from ultralytics import YOLO
   from google.colab import files

   found = sorted(glob.glob("/content/**/weights/best.pt", recursive=True), key=os.path.getmtime)
   best_path = found[-1]

   best_model = YOLO(best_path)
   metrics = best_model.val(data="/content/dataset/dataset.yaml", split="test")

   # Back up to Google Drive and download to PC
   shutil.copy(best_path, "/content/drive/MyDrive/sortsense_best.pt")
   files.download(best_path)
   ```

   **Verified Baseline Test Metrics (783 test images, 7,125 instances):**
   | Class | Images | Instances | Precision | Recall | `mAP50` | `mAP50-95` |
   |---|---:|---:|---:|---:|---:|---:|
   | **`all`** | 783 | 7,125 | 0.766 | 0.656 | **0.736** | 0.511 |
   | `plastic` | 122 | 449 | 0.715 | 0.677 | **0.762** | 0.544 |
   | `paper` | 320 | 952 | 0.732 | 0.581 | **0.680** | 0.501 |
   | `metal` | 133 | 436 | 0.806 | 0.764 | **0.807** | 0.622 |
   | `general` | 227 | 5,288 | 0.810 | 0.602 | **0.696** | 0.377 |

---

## 6. Deploying Trained Weights in the Backend

1. **Install `ultralytics` locally (if not already installed in `.venv`):**
   ```powershell
   cd D:\SortSense-AI_Backend
   .venv\Scripts\pip install ultralytics
   ```
   *(Kept outside base `requirements.txt` so CI and minimal builds remain lightweight.)*

2. **Place & Rename Trained Weights:**
   Move your downloaded `best.pt` (from Colab) or `sortsense_best.pt` (from Google Drive) into `weights/` and **rename it to `yolo26n.pt`**:
   ```text
   D:\SortSense-AI_Backend\weights\yolo26n.pt
   ```
   This matches the default configuration in `.env` and `.env.example`:
   ```env
   MODEL_BACKEND=yolov26
   MODEL_WEIGHTS_PATH=weights/yolo26n.pt
   CONFIDENCE_THRESHOLD=0.50
   ```
   *(Note: `.pt` files inside `weights/` are ignored by `.gitignore`, so keep your trained weights backed up on Google Drive.)*

3. **Empty-Frame / Low-Confidence Handling (`app/detector.py`):**
   * When trained weights are loaded (`self.model is not None`), if YOLO detects **0 objects** or detections below `CONFIDENCE_THRESHOLD` (`0.50`), `YOLOv26Detector.classify_frame()` returns `{"category": "none", "bin_index": -1, "confidence": ...}` so blank bin backgrounds do not trigger false LEDs.
   * When no `.pt` weights are loaded (`self.model is None`), it falls back to `_heuristic_edge_classify()`.

4. **Verify Locally:**
   *(Uvicorn `--reload` watches `.py` files, not `.pt` weight files — press `Ctrl + C` and restart if the server was already running.)*
   ```powershell
   cd D:\SortSense-AI_Backend
   .venv\Scripts\activate
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Confirm startup logs print `Loaded YOLO model weights from weights/yolo26n.pt`, then test live WebSocket classification in a second terminal:
   ```powershell
   cd D:\SortSense-AI_Backend
   .venv\Scripts\python websocket.py
   ```
   *(Note: if replacing `test_image.jpg`, ensure it is a true JPEG file and not a `.webp` image renamed to `.jpg`.)*

---

## 7. Future Expansion: Adding `glass` (5th Bin)

To incorporate glass in the future:
1. Combine `dataset/` (`0: plastic`, `1: paper`, `2: metal`, `3: general`) with `dataset/glass_archive/` (remapping `0: glass` → `4: glass`), or re-process `dataset/raw/GARBAGE CLASSIFICATION 3.v2-gc1.yolo26.zip` with `nc: 5`.
2. Re-train `yolo26n.pt` on Colab.
3. Update `WasteCategory`, `BIN_INDEX_TO_CATEGORY`, `CATEGORY_TO_BIN_INDEX`, and `VerifyDropRequest` (`le=4`) in [app/schemas.py](file:///d:/SortSense-AI_Backend/app/schemas.py) and `LABEL_TO_CATEGORY_MAP` in [app/detector.py](file:///d:/SortSense-AI_Backend/app/detector.py), plus add the 5th IR sensor/LED on `SortSense-Hardware` and `"glass"` support on `SortSense-Website`.
